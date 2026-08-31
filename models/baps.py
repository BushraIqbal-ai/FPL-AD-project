import torch
import torch.nn as nn



#step 1 EMA-Based Feature Memory (BAPS-specific, separate from MFGF's)


class BAPSFeatureMemory(nn.Module):
    """
    Eq. 15: EMA-based feature memory, independent from MFGF's EMA stats.
    Tracks per-channel mean and std of normal features for BAPS.
    """
    def __init__(self, channels=1536, momentum=0.9, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_std", torch.ones(channels))
        self.register_buffer("initialized", torch.tensor(False))

    def update_ema(self, x):
        """
        x: [B, C, H, W] -- normal (enhanced) features, called during training.
        """
        with torch.no_grad():
            batch_mean = x.mean(dim=[0, 2, 3])                          # [C]
            batch_std = x.std(dim=[0, 2, 3], unbiased=False)            # [C]

            if not self.initialized:
                self.running_mean.copy_(batch_mean)
                self.running_std.copy_(batch_std)
                self.initialized.fill_(True)
            else:
                self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.running_std.mul_(self.momentum).add_(batch_std, alpha=1 - self.momentum)

    def get_stats(self):
        """Returns (mean, std) reshaped for broadcasting against [B, C, H, W]."""
        mean = self.running_mean.view(1, self.channels, 1, 1)
        std = self.running_std.view(1, self.channels, 1, 1)
        return mean, std




#step 2 Feature-Aware Perturbation
class FeatureAwarePerturbation(nn.Module):
    """
    Eq. 12-13: importance_i(x) = sigmoid(|x_i - mu_i^B| / (sigma_i^B + eps))
               s_i(x) = gamma * (alpha + beta * importance_i(x))
    """
    def __init__(self, gamma=0.02, alpha=0.5, beta=0.5, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, x, mean, std):
        # x: [B, C, H, W], mean/std: [1, C, 1, 1] (broadcastable)
        importance = torch.sigmoid(torch.abs(x - mean) / (std + self.eps))  # [B, C, H, W]
        s = self.gamma * (self.alpha + self.beta * importance)                # [B, C, H, W]
        return s


#step 3 Boundary-Aware Perturbation (Eq. 14)
class BoundaryAwarePerturbation(nn.Module):
    """
    Eq. 14: b(x) = 1 + exp(-|D(x)| / tau)
    Amplifies perturbation strength for samples near the decision boundary,
    using the discriminator's own logit output as feedback.
    """
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def forward(self, discriminator_logits):
        # discriminator_logits: [B, 1, H, W] (D(x) from the discriminator)
        b = 1 + torch.exp(-torch.abs(discriminator_logits) / self.tau)
        return b  # [B, 1, H, W]


#step 4 Full BAPS Module (combines everything, Eq. 16)
class BAPS(nn.Module):
    """
    Full Boundary-Aware Pseudo-Anomaly Synthesis module (Section 3.2, Fig. 4).
    Training-only: generates pseudo-anomalous features from normal features.
    """
    def __init__(self, channels=1536, momentum=0.9, gamma=0.02,
                 alpha=0.5, beta=0.5, tau=0.2, eps=1e-6):
        super().__init__()
        self.memory = BAPSFeatureMemory(channels, momentum, eps)
        self.feature_aware = FeatureAwarePerturbation(gamma, alpha, beta, eps)
        self.boundary_aware = BoundaryAwarePerturbation(tau)

    def update_ema(self, x):
        """Call during training with normal (enhanced) features."""
        self.memory.update_ema(x)

    def forward(self, x, discriminator_logits):
        """
        x: [B, C, H, W] -- normal enhanced features (x')
        discriminator_logits: [B, 1, H, W] -- D(x') from discriminator's
            forward pass on the SAME normal features, used for boundary feedback
        Returns: x_tilde, the synthesized pseudo-anomalous features
        """
        mean, std = self.memory.get_stats()
        s = self.feature_aware(x, mean, std)               # [B, C, H, W]
        b = self.boundary_aware(discriminator_logits)        # [B, 1, H, W]

        eta = torch.randn_like(x)                             # [B, C, H, W], standard Gaussian
        # b broadcasts from [B,1,H,W] to [B,C,H,W] automatically
        x_tilde = x + (s * b) * eta
        return x_tilde



if __name__ == "__main__":
    dummy = torch.randn(2, 1536, 28, 28).cuda()

    mem = BAPSFeatureMemory(channels=1536).cuda()
    mem.update_ema(dummy)
    mean, std = mem.get_stats()
    print("BAPS EMA mean shape:", mean.shape)
    print("BAPS EMA std shape:", std.shape)

    fap = FeatureAwarePerturbation(gamma=0.02, alpha=0.5, beta=0.5).cuda()
    s = fap(dummy, mean, std)
    print("Feature-aware perturbation (s) shape:", s.shape)
    print("s value range:", s.min().item(), "to", s.max().item())

    bap = BoundaryAwarePerturbation(tau=0.2).cuda()
    fake_logits = torch.randn(2, 1, 28, 28).cuda()
    b = bap(fake_logits)
    print("Boundary-aware perturbation (b) shape:", b.shape)
    print("b value range:", b.min().item(), "to", b.max().item())

    print("\n--- Full BAPS module ---")
    baps = BAPS(channels=1536).cuda()
    baps.update_ema(dummy)
    x_tilde = baps(dummy, fake_logits)
    print("BAPS output (x_tilde) shape:", x_tilde.shape)  # expect [2, 1536, 28, 28]
    print("Difference from input (should be small, nonzero):",
          (x_tilde - dummy).abs().mean().item())