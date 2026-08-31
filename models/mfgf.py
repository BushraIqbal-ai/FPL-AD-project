import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Eq. 1: z = GAP(x), a_c = sigmoid(W2 . ReLU(W1 . z))
    """
    def __init__(self, channels=1536, reduction_ratio=8):
        super().__init__()
        hidden = channels // reduction_ratio
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.gap(x).view(b, c)
        z = self.relu(self.fc1(z))
        a_c = self.sigmoid(self.fc2(z))
        a_c = a_c.view(b, c, 1, 1)
        return a_c


class FeatureResponseModulation(nn.Module):
    """
    Eq. 2: x_enhanced = sigmoid(W4 . ReLU(W3 . x))
    """
    def __init__(self, channels=1536, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = channels
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.sigmoid(self.conv2(out))
        return out


class UncertaintyEstimation(nn.Module):
    """
    Eq. 3-5: combines a learned energy score with deviation from
    EMA-tracked normal feature statistics (mean, variance) per channel.
    """
    def __init__(self, channels=1536, momentum=0.9, lam=0.5, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.momentum = momentum
        self.lam = lam
        self.eps = eps

        self.energy_mlp = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, 1),
        )

        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("initialized", torch.tensor(False))

    def update_ema(self, x):
        with torch.no_grad():
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            if not self.initialized:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
                self.initialized.fill_(True)
            else:
                self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.running_var.mul_(self.momentum).add_(batch_var, alpha=1 - self.momentum)

    def forward(self, x):
        b, c, h, w = x.shape

        x_perm = x.permute(0, 2, 3, 1).reshape(-1, c)
        energy = self.energy_mlp(x_perm).view(b, 1, h, w)
        energy_norm = (energy - energy.mean()) / (energy.std() + self.eps)

        mean = self.running_mean.view(1, c, 1, 1)
        std = torch.sqrt(self.running_var.view(1, c, 1, 1) + self.eps)
        dev = torch.abs((x - mean) / std).mean(dim=1, keepdim=True)

        uncertainty = energy_norm + self.lam * dev
        return uncertainty


class EntropySuppression(nn.Module):
    """
    Eq. 6-7: H(x) = -sum(p_i * log(p_i + eps)), entropy_factor = exp(-beta_h * H(x))
    """
    def __init__(self, beta_h=0.1, eps=1e-8):
        super().__init__()
        self.beta_h = beta_h
        self.eps = eps

    def forward(self, x):
        p = torch.softmax(x, dim=1)
        entropy = -(p * torch.log(p + self.eps)).sum(dim=1, keepdim=True)
        entropy_factor = torch.exp(-self.beta_h * entropy)
        return entropy_factor


class GatedFusionUnit(nn.Module):
    """
    Eq. 8-9: gate g(x, U(x)) computed from concatenated [x; U(x)]
    via a lightweight 2-layer perceptron (implemented as 1x1 convs).
    """
    def __init__(self, channels=1536, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = channels // 4
        self.conv1 = nn.Conv2d(channels + 1, hidden_dim, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, u):
        z = torch.cat([x, u], dim=1)
        g = self.leaky_relu(self.conv1(z))
        g = self.sigmoid(self.conv2(g))
        return g


class MFGF(nn.Module):
    """
    Full Multi-Factor Gated Fusion module (Section 3.1, Fig. 3).
    Combines channel attention, feature response modulation,
    uncertainty estimation, entropy suppression, and gated fusion.
    """
    def __init__(self, channels=1536, reduction_ratio=8, momentum=0.9,
                 lam=0.5, beta_h=0.1, lambda_a=0.5):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.feature_response_mod = FeatureResponseModulation(channels)
        self.uncertainty_est = UncertaintyEstimation(channels, momentum, lam)
        self.entropy_suppress = EntropySuppression(beta_h)
        self.gate = GatedFusionUnit(channels)
        self.lambda_a = lambda_a

    def update_ema(self, x):
        """Call this during training with normal-only features."""
        self.uncertainty_est.update_ema(x)

    def forward(self, x, return_aux=False):
        a_c = self.channel_attention(x)
        x_enhanced = self.feature_response_mod(x)
        u_raw = self.uncertainty_est(x)
        sigmoid_u = torch.sigmoid(u_raw)

        p = torch.softmax(x, dim=1)
        entropy_map = -(p * torch.log(p + 1e-8)).sum(dim=1, keepdim=True)
        entropy_factor = torch.exp(-self.entropy_suppress.beta_h * entropy_map)

        g = self.gate(x, u_raw)

        x_out = a_c * x_enhanced * sigmoid_u * entropy_factor * g
        x_prime = x + self.lambda_a * x_out

        if return_aux:
            max_entropy = torch.log(torch.tensor(float(x.shape[1]), device=x.device))
            entropy_norm = entropy_map / (max_entropy + 1e-8)
            entropy_norm = torch.clamp(entropy_norm, 0.0, 1.0)
            uncertainty_norm = sigmoid_u
            return x_prime, entropy_norm, uncertainty_norm

        return x_prime


if __name__ == "__main__":
    dummy = torch.randn(2, 1536, 28, 28).cuda()

    mfgf = MFGF(channels=1536).cuda()
    mfgf.update_ema(dummy)

    x_prime = mfgf(dummy)
    print("Standard forward (no aux) shape:", x_prime.shape)

    x_prime2, h, u = mfgf(dummy, return_aux=True)
    print("With aux -- x_prime:", x_prime2.shape, "| h (entropy):", h.shape, "| u (uncertainty):", u.shape)
    print("h range:", h.min().item(), "to", h.max().item())
    print("u range:", u.min().item(), "to", u.max().item())