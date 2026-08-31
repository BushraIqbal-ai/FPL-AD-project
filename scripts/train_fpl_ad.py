import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.backbone import WideResNetFeatureExtractor
from models.adapter import FeatureAdapter
from models.mfgf import MFGF
from models.baps import BAPS
from models.discriminator import Discriminator
from datasets.anomaly_dataset import AnomalyDataset


def main():
    DATASET_NAME = "mvtec_ad"
    CATEGORY = "bottle"
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AnomalyDataset(DATASET_NAME, CATEGORY, split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    print(f"Training (FPL-AD: MFGF+BAPS) on {DATASET_NAME}/{CATEGORY}: {len(train_ds)} normal images")

    backbone = WideResNetFeatureExtractor().to(DEVICE)
    adapter = FeatureAdapter(in_channels=1536, out_channels=1536).to(DEVICE)
    mfgf = MFGF(channels=1536).to(DEVICE)
    baps = BAPS(channels=1536).to(DEVICE)
    discriminator = Discriminator(in_channels=1536).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(mfgf.parameters()) + list(discriminator.parameters()),
        lr=LR
    )
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for imgs, masks, labels in train_loader:
            imgs = imgs.to(DEVICE)

            with torch.no_grad():
                feats = backbone(imgs)

            x_adapted = adapter(feats)
            mfgf.update_ema(x_adapted)
            x_prime = mfgf(x_adapted)  # enhanced normal features

            baps.update_ema(x_prime)

            # Get discriminator's view of normal features FIRST (needed for boundary feedback)
            logits_normal = discriminator(x_prime)

            # Generate pseudo-anomalies using BAPS, informed by discriminator feedback
            x_tilde = baps(x_prime, logits_normal.detach())  # detach: don't backprop boundary signal into x_prime twice

            logits_pseudo = discriminator(x_tilde)

            target_normal = torch.zeros_like(logits_normal)
            target_pseudo = torch.ones_like(logits_pseudo)

            loss = bce_loss(logits_normal, target_normal) + bce_loss(logits_pseudo, target_pseudo)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f}")

    print("Training complete.")

    torch.save({
        "adapter": adapter.state_dict(),
        "mfgf": mfgf.state_dict(),
        "discriminator": discriminator.state_dict(),
    }, f"checkpoint_{DATASET_NAME}_{CATEGORY}_fpl_ad.pth")
    print("Checkpoint saved.")


if __name__ == "__main__":
    main()