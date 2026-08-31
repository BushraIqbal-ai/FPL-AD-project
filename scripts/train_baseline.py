import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.backbone import WideResNetFeatureExtractor
from models.adapter import FeatureAdapter
from models.discriminator import Discriminator
from datasets.anomaly_dataset import AnomalyDataset


def main():
    # ---------------- Config ----------------
    DATASET_NAME = "mvtec_ad"
    CATEGORY = "bottle"
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-4
    NOISE_STD = 0.015
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Data ----------------
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AnomalyDataset(DATASET_NAME, CATEGORY, split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    print(f"Training on {DATASET_NAME}/{CATEGORY}: {len(train_ds)} normal images")

    # ---------------- Models ----------------
    backbone = WideResNetFeatureExtractor().to(DEVICE)
    adapter = FeatureAdapter(in_channels=1536, out_channels=1536).to(DEVICE)
    discriminator = Discriminator(in_channels=1536).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(discriminator.parameters()), lr=LR
    )
    bce_loss = nn.BCEWithLogitsLoss()

    # ---------------- Training loop ----------------
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for imgs, masks, labels in train_loader:
            imgs = imgs.to(DEVICE)

            with torch.no_grad():
                feats = backbone(imgs)

            x_prime = adapter(feats)

            noise = torch.randn_like(x_prime) * NOISE_STD
            x_pseudo = x_prime + noise

            logits_normal = discriminator(x_prime)
            logits_pseudo = discriminator(x_pseudo)

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
        "discriminator": discriminator.state_dict(),
    }, f"checkpoint_{DATASET_NAME}_{CATEGORY}_baseline.pth")
    print("Checkpoint saved.")


if __name__ == "__main__":
    main()