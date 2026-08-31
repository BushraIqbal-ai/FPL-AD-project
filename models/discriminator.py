import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Lightweight discriminator head. Operates per-pixel (per-patch)
    on the feature map, outputting a single logit per spatial location.
    Follows SimpleNet-style design: a small stack of 1x1 convs.
    """
    def __init__(self, in_channels=1536, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, C, H, W] -> logits: [B, 1, H, W]
        return self.net(x)


if __name__ == "__main__":
    disc = Discriminator().cuda()
    dummy = torch.randn(2, 1536, 28, 28).cuda()
    out = disc(dummy)
    print(out.shape)  # expect torch.Size([2, 1, 28, 28])