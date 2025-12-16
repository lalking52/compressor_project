import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Lightweight residual block to improve reconstruction without changing latent size."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return self.act(out + x)


class FullyConvAE(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        # Strided conv encoder squeezes time/freq while expanding channels
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=5, stride=2, padding=2),  # accept [mag, phase]
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = ResBlock(channels * 8)
        # Mirror decoder upsamples back to two-channel STFT representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                channels * 8, channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                channels * 4, channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                channels * 2, channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 2, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        return z, out
