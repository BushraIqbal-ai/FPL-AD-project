import torch
from models.backbone import WideResNetFeatureExtractor
from models.adapter import FeatureAdapter
from models.mfgf import MFGF
from models.baps import BAPS
from models.discriminator import Discriminator


def build_models(device, channels=1536):
    """
    Creates and returns all model components needed for FPL-AD.
    Backbone is frozen; others are trainable.
    """
    backbone = WideResNetFeatureExtractor().to(device)
    adapter = FeatureAdapter(in_channels=channels, out_channels=channels).to(device)
    mfgf = MFGF(channels=channels).to(device)
    baps = BAPS(channels=channels).to(device)
    discriminator = Discriminator(in_channels=channels).to(device)
    return backbone, adapter, mfgf, baps, discriminator


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, adapter, mfgf, baps, discriminator = build_models(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        feats = backbone(dummy)
    x_adapted = adapter(feats)
    mfgf.update_ema(x_adapted)
    x_prime = mfgf(x_adapted)
    baps.update_ema(x_prime)
    logits = discriminator(x_prime)
    x_tilde = baps(x_prime, logits)
    print("feats:", feats.shape)
    print("x_prime:", x_prime.shape)
    print("logits:", logits.shape)
    print("x_tilde:", x_tilde.shape)