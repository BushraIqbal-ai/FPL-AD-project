import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np

from models.build import build_models
from datasets.anomaly_dataset import AnomalyDataset
from utils.pro_metric import compute_pro


def evaluate_one_category(dataset_name, category, device, state_dict,
                           batch_size=8, top_k=20, pro_thresholds=100):
    """
    Evaluates a trained FPL-AD model on a single category's test set.
    state_dict: dict with keys 'adapter', 'mfgf', 'discriminator' (from train_one_category).
    Returns: dict with 'i_auroc', 'p_auroc', 'pro'.
    """
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

    test_ds = AnomalyDataset(dataset_name, category, split="test",
                              transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    backbone, adapter, mfgf, baps, discriminator = build_models(device)
    adapter.load_state_dict(state_dict["adapter"])
    mfgf.load_state_dict(state_dict["mfgf"])
    discriminator.load_state_dict(state_dict["discriminator"])

    adapter.eval()
    mfgf.eval()
    discriminator.eval()

    all_image_scores, all_image_labels = [], []
    all_pixel_scores_flat, all_pixel_labels_flat = [], []
    all_amaps_full, all_masks_full = [], []

    with torch.no_grad():
        for imgs, masks, labels in test_loader:
            imgs = imgs.to(device)

            feats = backbone(imgs)
            x_adapted = adapter(feats)
            x_prime = mfgf(x_adapted)  # BAPS disabled at inference
            logits = discriminator(x_prime)
            anomaly_map = torch.sigmoid(logits)

            anomaly_map_up = F.interpolate(anomaly_map, size=(224, 224),
                                            mode="bilinear", align_corners=False)

            b = anomaly_map_up.shape[0]
            flat = anomaly_map_up.view(b, -1)
            k = min(top_k, flat.shape[1])
            topk_vals, _ = torch.topk(flat, k=k, dim=1)
            image_scores = topk_vals.mean(dim=1)

            all_image_scores.extend(image_scores.cpu().numpy().tolist())
            all_image_labels.extend(labels.numpy().tolist())

            all_pixel_scores_flat.append(anomaly_map_up.cpu().numpy().reshape(-1))
            all_pixel_labels_flat.append(masks.numpy().reshape(-1))

            amap_np = anomaly_map_up.cpu().numpy()[:, 0, :, :]
            mask_np = masks.numpy()[:, 0, :, :]
            all_amaps_full.append(amap_np)
            all_masks_full.append(mask_np)

    # Guard: I-AUROC needs both classes present
    if len(set(all_image_labels)) < 2:
        i_auroc = float("nan")
    else:
        i_auroc = roc_auc_score(all_image_labels, all_image_scores)

    pixel_scores = np.concatenate(all_pixel_scores_flat)
    pixel_labels = np.concatenate(all_pixel_labels_flat)
    pixel_labels_bin = (pixel_labels > 0.5).astype(int)
    if len(set(pixel_labels_bin.tolist())) < 2:
        p_auroc = float("nan")
    else:
        p_auroc = roc_auc_score(pixel_labels_bin, pixel_scores)

    amaps_full = np.concatenate(all_amaps_full, axis=0)
    masks_full = np.concatenate(all_masks_full, axis=0)
    masks_full_bin = (masks_full > 0.5).astype(np.uint8)
    pro = compute_pro(masks_full_bin, amaps_full, num_thresholds=pro_thresholds)

    return {
        "i_auroc": i_auroc * 100 if not np.isnan(i_auroc) else float("nan"),
        "p_auroc": p_auroc * 100 if not np.isnan(p_auroc) else float("nan"),
        "pro": pro * 100,
    }



if __name__ == "__main__":
    import torch
    from train_utils import train_one_category

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = train_one_category("mvtec_ad", "bottle", device, epochs=3, verbose=False)
    metrics = evaluate_one_category("mvtec_ad", "bottle", device, state)
    print("Metrics:", metrics)
    