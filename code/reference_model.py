"""State + trend temporal summaries followed by a single spatial U-Net.

Implements the user's proposed architecture; no pretrained weights.
Dates are days since 2018-09-01, divided by the SAME 365-day scale for all samples.
"""
import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        )


class SpatialUNet(nn.Module):
    def __init__(self, widths=(32, 64, 96, 128), classes=20):
        super().__init__()
        self.encoders = nn.ModuleList([DoubleConv(32, widths[0])] + [
            DoubleConv(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(a, b, 2, stride=2)
            for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.decoders = nn.ModuleList([
            DoubleConv(2*c, c) for c in widths[-2::-1]])
        self.head = nn.Conv2d(widths[0], classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoders):
            x = block(self.pool(x) if i else x)
            skips.append(x)
        for up, block, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class StateTrendUNet(nn.Module):
    def __init__(self, features=16, windows=6, widths=(32, 64, 96, 128),
                 time_scale=365.0, sigma_days=60.0, ridge=1e-3):
        super().__init__()
        self.features = features
        self.windows = windows
        self.time_scale = time_scale
        self.sigma_days = sigma_days
        self.ridge = ridge
        # Endpoints use TRAINING metadata only; fixed for all splits and samples.
        self.register_buffer('centers_days', torch.linspace(16., 421., windows))
        self.spectral = nn.Sequential(nn.Conv2d(10, features, 1), nn.GELU())
        self.score = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, groups=features),
            nn.GELU(), nn.Conv2d(features, windows, 1))
        self.compress = nn.Sequential(
            nn.Conv2d(windows * 2 * features, 32, 1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True))
        self.spatial = SpatialUNet(widths)
        self.apply(self._init)
        # Start with the fixed Gaussian prior, then learn content modulation.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def summarize(self, features, content_scores, dates, valid, return_weights=False):
        """Return B,K,C,H,W states and regularized slopes.

        No B,K,T,C,H,W tensor is materialized. The centered regression numerator
        is evaluated as sum(alpha*dt*F) - M*sum(alpha*dt), algebraically identical
        to sum(alpha*dt*(F-M)). FP32 avoids BF16 cancellation in the regression.
        """
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            tau = dates.float() / self.time_scale
            center = self.centers_days / self.time_scale
            sigma = self.sigma_days / self.time_scale
            log_prior = -.5 * ((tau[:, None, :] - center[None, :, None]) / sigma)**2
            scores = content_scores.float().permute(0, 2, 1, 3, 4)
            logits = scores + log_prior[..., None, None]
            logits = logits.masked_fill(~valid[:, None, :, None, None], -torch.inf)
            alpha = logits.softmax(dim=2)
            mean = torch.einsum('bkthw,btchw->bkchw', alpha, features)
            tau_bar = (alpha * tau[:, None, :, None, None]).sum(dim=2)
            dt = tau[:, None, :, None, None] - tau_bar[:, :, None]
            coeff = alpha * dt
            covariance = torch.einsum('bkthw,btchw->bkchw', coeff, features)
            covariance = covariance - mean * coeff.sum(dim=2)[:, :, None]
            variance = (alpha * dt.square()).sum(dim=2)
            trend = covariance / (variance[:, :, None] + self.ridge)
        return (mean, trend, alpha) if return_weights else (mean, trend)

    def forward(self, x, batch_positions, valid_mask):
        b, t, _, h, w = x.shape
        frame_features = self.spectral(x.reshape(b*t, 10, h, w))
        scores = self.score(frame_features).reshape(b, t, self.windows, h, w)
        frame_features = frame_features.reshape(b, t, self.features, h, w)
        mean, trend = self.summarize(frame_features, scores, batch_positions, valid_mask)
        # Window-major ordering: [M1,D1, M2,D2, ..., M6,D6], 16 channels each.
        summary = torch.cat([mean, trend], dim=2).reshape(b, 2*self.windows*self.features, h, w)
        return self.spatial(self.compress(summary))
