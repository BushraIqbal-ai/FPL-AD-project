import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class WideResNetFeatureExtractor(nn.Module):
    def __init__(self, layers=("layer2", "layer3"), out_size=28):
        super().__init__()
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layers = layers
        self.out_size = out_size
        self.model = backbone
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self._features = {}
        for name in layers:
            layer = dict(self.model.named_children())[name]
            layer.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        _ = self.model(x)
        feats = [self._features[name] for name in self.layers]
        # resize all feature maps to the same spatial size, then concat channel-wise
        resized = [F.interpolate(f, size=(self.out_size, self.out_size),
                                  mode="bilinear", align_corners=False) for f in feats]
        x_cat = torch.cat(resized, dim=1)  # [B, 1536, 28, 28]
        return x_cat
    




if __name__ == "__main__":
    model = WideResNetFeatureExtractor().cuda()
    dummy = torch.randn(2, 3, 224, 224).cuda()
    out = model(dummy)
    print(out.shape)  # expect torch.Size([2, 1536, 28, 28])