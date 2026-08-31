import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Lightweight feature adapter: a linear projection applied per-pixel
    (i.e., a 1x1 conv) to align/normalize the channel scale of the
    concatenated backbone features before further processing.
    """
    def __init__(self, in_channels=1536, out_channels=1536):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        return self.proj(x)


if __name__ == "__main__":
    adapter = FeatureAdapter().cuda()
    dummy = torch.randn(2, 1536, 28, 28).cuda()
    out = adapter(dummy)
    print(out.shape)  # expect torch.Size([2, 1536, 28, 28])