import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from contextlib import redirect_stdout

from train_utils import train_one_category
from models.build import build_models
from datasets.anomaly_dataset import AnomalyDataset

# ---- The 4 remaining, undiagnosed collapse cases ----
CASES = [
    ("mvtec_ad", "screw"),
    ("mvtec_ad", "cable"),
    ("mpdd", "bracket_black"),
    ("btad", "03"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "diagnostics"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class Tee:
    """Writes to both the real stdout and a log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def save_heatmap_visualization(dataset_name, category, state, num_samples=6):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    ])

    test_ds = AnomalyDataset(dataset_name, category, split="test",
                              transform=transform, mask_transform=mask_transform)

    backbone, adapter, mfgf, baps, discriminator = build_models(DEVICE)
    adapter.load_state_dict(state["adapter"])
    mfgf.load_state_dict(state["mfgf"])
    discriminator.load_state_dict(state["discriminator"])
    adapter.eval(); mfgf.eval(); discriminator.eval()

    anomalous_idx = [i for i in range(len(test_ds)) if test_ds.labels[i] == 1]
    normal_idx = [i for i in range(len(test_ds)) if test_ds.labels[i] == 0]
    n_anom = min(num_samples - 1, len(anomalous_idx))
    n_norm = num_samples - n_anom
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

    plt.suptitle(f"{dataset_name}/{category} — FPL-AD predictions", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{category}_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset_name, category in CASES:
        print(f"\n{'='*60}")
        print(f"DIAGNOSING: {dataset_name}/{category}")
        print(f"{'='*60}")

        log_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{category}_loss.txt")
        with open(log_path, "w") as log_file:
            tee = Tee(sys.stdout, log_file)
            with redirect_stdout(tee):
                state = train_one_category(dataset_name, category, DEVICE,
                                            epochs=20, verbose=True)

        print(f"Loss log saved to {log_path}")
        save_heatmap_visualization(dataset_name, category, state)

    print(f"\n\nAll diagnostics complete. Check the '{OUTPUT_DIR}' folder for:")
    for dataset_name, category in CASES:
        print(f"  - {dataset_name}_{category}_loss.txt")
        print(f"  - {dataset_name}_{category}_heatmap.png")


if __name__ == "__main__":
    main()