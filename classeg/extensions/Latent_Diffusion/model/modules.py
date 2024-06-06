import torch
import torch.fft as fft
import torch.nn as nn


class ScaleULayer(nn.Module):

    def __init__(self, in_channels, skipped_count=2) -> None:
        super().__init__()
        self.b_scaling = nn.Parameter(torch.zeros(in_channels))  # Main features are b
        self.skipped_scaling = nn.ParameterList()
        for _ in range(skipped_count):
            self.skipped_scaling.append(nn.Parameter(torch.zeros(1)))  # skipped features from other decoder are s

    def Fourier_filter(self, x_in, threshold, scale):
        x = x_in
        B, C, H, W = x.shape
        # Non-power of 2 images must be float32
        if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
            x = x.to(dtype=torch.float32)

        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W), device=x.device)

        crow, ccol = H // 2, W // 2
        mask[
        ...,
        crow - threshold: crow + threshold,
        ccol - threshold: ccol + threshold,
        ] = scale
        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered.to(dtype=x_in.dtype)

    def forward(self, x, *skipped_connections):
        skipped_connections = list(skipped_connections)
        b_s = torch.tanh(self.b_scaling) + 1
        thresholds = []
        for weight in self.skipped_scaling:
            thresholds.append(torch.tanh(weight) + 1)

        for i in range(len(skipped_connections)):
            skipped_connections[i] = self.Fourier_filter(skipped_connections[i], 1,
                                                         thresholds[i])  # scale the skip connection from encoder

        x = torch.einsum("bchw,c->bchw", x, b_s)  # scale the main features
        x = torch.cat([x, *skipped_connections], dim=1)
        return x
