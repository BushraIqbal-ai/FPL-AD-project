import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np

from models.backbone import WideResNetFeatureExtractor
from models.adapter import FeatureAdapter
from models.mfgf import MFGF
from models.discriminator import Discriminator
from datasets.anomaly_dataset import AnomalyDataset
from utils.pro_metric import compute_pro


def main():
    DATASET_NAME = "mvtec_ad"
    CATEGORY = "bottle"
    BATCH_SIZE = 8
    TOP_K = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_ds = AnomalyDataset(DATASET_NAME, CATEGORY, split="test",
                              transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Evaluating (FPL-AD) on {DATASET_NAME}/{CATEGORY}: {len(test_ds)} test images")

    backbone = WideResNetFeatureExtractor().to(DEVICE)
    adapter = FeatureAdapter(in_channels=1536, out_channels=1536).to(DEVICE)
    mfgf = MFGF(channels=1536).to(DEVICE)
    discriminator = Discriminator(in_channels=1536).to(DEVICE)

    ckpt = torch.load(f"checkpoint_{DATASET_NAME}_{CATEGORY}_fpl_ad.pth", map_location=DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    mfgf.load_state_dict(ckpt["mfgf"])
    discriminator.load_state_dict(ckpt["discriminator"])

    adapter.eval()
    mfgf.eval()
    discriminator.eval()

    all_image_scores = []
    all_image_labels = []
    all_pixel_scores_flat = []
    all_pixel_labels_flat = []
    all_amaps_full = []
    all_masks_full = []

    with torch.no_grad():
        for imgs, masks, labels in test_loader:
            imgs = imgs.to(DEVICE)

            feats = backbone(imgs)
            x_adapted = adapter(feats)
            x_prime = mfgf(x_adapted)   # BAPS disabled at inference (design intent)
            logits = discriminator(x_prime)
            anomaly_map = torch.sigmoid(logits)

            anomaly_map_up = F.interpolate(anomaly_map, size=(224, 224),
                                            mode="bilinear", align_corners=False)

            b = anomaly_map_up.shape[0]
            flat = anomaly_map_up.view(b, -1)
            topk_vals, _ = torch.topk(flat, k=TOP_K, dim=1)
            image_scores = topk_vals.mean(dim=1)

            all_image_scores.extend(image_scores.cpu().numpy().tolist())
            all_image_labels.extend(labels.numpy().tolist())

            all_pixel_scores_flat.append(anomaly_map_up.cpu().numpy().reshape(-1))
            all_pixel_labels_flat.append(masks.numpy().reshape(-1))

            amap_np = anomaly_map_up.cpu().numpy()[:, 0, :, :]
            mask_np = masks.numpy()[:, 0, :, :]
            all_amaps_full.append(amap_np)
            all_masks_full.append(mask_np)

    i_auroc = roc_auc_score(all_image_labels, all_image_scores)

    pixel_scores = np.concatenate(all_pixel_scores_flat)
    pixel_labels = np.concatenate(all_pixel_labels_flat)
    pixel_labels_bin = (pixel_labels > 0.5).astype(int)
    p_auroc = roc_auc_score(pixel_labels_bin, pixel_scores)

    amaps_full = np.concatenate(all_amaps_full, axis=0)
    masks_full = np.concatenate(all_masks_full, axis=0)
    masks_full_bin = (masks_full > 0.5).astype(np.uint8)
    pro = compute_pro(masks_full_bin, amaps_full, num_thresholds=100)

    print(f"\nResults (Full FPL-AD) for {DATASET_NAME}/{CATEGORY}:")
    print(f"I-AUROC: {i_auroc*100:.2f}")
    print(f"P-AUROC: {p_auroc*100:.2f}")
    print(f"PRO:     {pro*100:.2f}")

    print(f"\n--- Full Comparison ---")
    print(f"Baseline:  I-AUROC 100.00 / P-AUROC 97.04 / PRO 87.99")
    print(f"MFGF only: I-AUROC 100.00 / P-AUROC 97.01 / PRO 88.96")
    print(f"Full FPL-AD: I-AUROC {i_auroc*100:.2f} / P-AUROC {p_auroc*100:.2f} / PRO {pro*100:.2f}")


if __name__ == "__main__":
    main()