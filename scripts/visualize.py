import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from train_utils import train_one_category
from models.build import build_models
from datasets.anomaly_dataset import AnomalyDataset

DATASET_NAME = "mvtec_ad"
CATEGORY = "screw"          # change this to inspect other categories
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 6             # how many test images to visualize
OUTPUT_PATH = f"viz_{DATASET_NAME}_{CATEGORY}.png"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    """img_tensor: [3, H, W] normalized -> [H, W, 3] in [0,1] for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print(f"Training on {DATASET_NAME}/{CATEGORY} for visualization...")
    state = train_one_category(DATASET_NAME, CATEGORY, DEVICE, epochs=20, verbose=False)

    test_ds = AnomalyDataset(DATASET_NAME, CATEGORY, split="test",
                              transform=transform, mask_transform=mask_transform)

    backbone, adapter, mfgf, baps, discriminator = build_models(DEVICE)
    adapter.load_state_dict(state["adapter"])
    mfgf.load_state_dict(state["mfgf"])
    discriminator.load_state_dict(state["discriminator"])
    adapter.eval()
    mfgf.eval()
    discriminator.eval()

    # Pick a mix: prioritize anomalous samples, include a couple normal ones
    anomalous_idx = [i for i in range(len(test_ds)) if test_ds.labels[i] == 1]
    normal_idx = [i for i in range(len(test_ds)) if test_ds.labels[i] == 0]

    n_anom = min(NUM_SAMPLES - 1, len(anomalous_idx))
    n_norm = NUM_SAMPLES - n_anom
    chosen_idx = anomalous_idx[:n_anom] + normal_idx[:n_norm]

    fig, axes = plt.subplots(3, len(chosen_idx), figsize=(3 * len(chosen_idx), 9))
    if len(chosen_idx) == 1:
        axes = axes.reshape(3, 1)

    with torch.no_grad():
        for col, idx in enumerate(chosen_idx):
            img, mask, label = test_ds[idx]
            img_batch = img.unsqueeze(0).to(DEVICE)

            feats = backbone(img_batch)
            x_adapted = adapter(feats)
            x_prime = mfgf(x_adapted)
            logits = discriminator(x_prime)
            amap = torch.sigmoid(logits)
            amap_up = F.interpolate(amap, size=(224, 224), mode="bilinear", align_corners=False)
            amap_np = amap_up[0, 0].cpu().numpy()

            img_disp = denormalize(img)
            mask_disp = mask[0].cpu().numpy()

            axes[0, col].imshow(img_disp)
            axes[0, col].set_title(f"Input ({'anomaly' if label == 1 else 'normal'})", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(mask_disp, cmap="gray")
            axes[1, col].set_title("Ground Truth", fontsize=9)
            axes[1, col].axis("off")

            axes[2, col].imshow(img_disp)
            axes[2, col].imshow(amap_np, cmap="jet", alpha=0.5)
            axes[2, col].set_title(f"Predicted Heatmap\n(score={amap_np.max():.3f})", fontsize=9)
            axes[2, col].axis("off")

    plt.suptitle(f"{DATASET_NAME}/{CATEGORY} — FPL-AD predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()