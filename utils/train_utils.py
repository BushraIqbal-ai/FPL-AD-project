import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.build import build_models
from datasets.anomaly_dataset import AnomalyDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Force deterministic cuDNN behavior. Without this, convolution operations
    # on GPU can pick different (faster but non-reproducible) algorithms across
    # runs, causing results to diverge even with the same seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_category(dataset_name, category, device,
                        batch_size=8, epochs=20, lr=1e-4,
                        lambda_h=0.1,        # entropy regularizer weight
                        target_entropy=0.5,  # H_hat_0: target entropy threshold
                        lambda_w=1e-5,       # L2 weight decay coefficient
                        seed=42,
                        verbose=True):
    """
    Trains FPL-AD (MFGF + BAPS) on a single category, using the full
    loss from Eq. 17: two BCE terms + entropy regularizer + L2 weight decay.
    Seeded for reproducibility (including deterministic cuDNN).
    """
    set_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AnomalyDataset(dataset_name, category, split="train", transform=transform)

    # Use a seeded generator for the DataLoader's shuffling too, so batch order
    # is also reproducible across runs, not just weight initialization.
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True, generator=g)

    if len(train_loader) == 0:
        raise ValueError(f"{dataset_name}/{category}: not enough training images "
                          f"for batch_size={batch_size} (have {len(train_ds)})")

    backbone, adapter, mfgf, baps, discriminator = build_models(device)

    optimizer = torch.optim.Adam(
        list(mfgf.parameters()) + list(discriminator.parameters()),
        lr=lr, weight_decay=lambda_w
    )
    optimizer_adapter = torch.optim.Adam(adapter.parameters(), lr=lr)

    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_entropy_reg = 0.0

        for imgs, masks, labels in train_loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                feats = backbone(imgs)

            x_adapted = adapter(feats)
            mfgf.update_ema(x_adapted)

            x_prime, h_map, u_map = mfgf(x_adapted, return_aux=True)

            baps.update_ema(x_prime)
            logits_normal = discriminator(x_prime)
            x_tilde = baps(x_prime, logits_normal.detach())
            logits_pseudo = discriminator(x_tilde)

            target_normal = torch.zeros_like(logits_normal)
            target_pseudo = torch.ones_like(logits_pseudo)

            bce = bce_loss(logits_normal, target_normal) + bce_loss(logits_pseudo, target_pseudo)

            entropy_reg = (lambda_h * ((h_map - target_entropy) ** 2) * (1 - u_map)).mean()

            loss = bce + entropy_reg

            optimizer.zero_grad()
            optimizer_adapter.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_adapter.step()

            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_entropy_reg += entropy_reg.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_bce = epoch_bce / len(train_loader)
        avg_ereg = epoch_entropy_reg / len(train_loader)
        if verbose:
            print(f"  [{dataset_name}/{category}] Epoch {epoch+1}/{epochs} - "
                  f"total: {avg_loss:.4f} | bce: {avg_bce:.4f} | entropy_reg: {avg_ereg:.4f}")

    return {
        "adapter": adapter.state_dict(),
        "mfgf": mfgf.state_dict(),
        "discriminator": discriminator.state_dict(),
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = train_one_category("mvtec_ad", "bottle", device, epochs=3)
    print("Returned keys:", list(state.keys()))