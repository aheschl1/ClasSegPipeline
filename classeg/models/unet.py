import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, depth, channel_growth_factor) -> None:
        super().__init__()
        encoder_layers = []
        channels = in_channels
        for _ in range(depth):
            layer = nn.Sequential(
                nn.Conv2d(channels, channels * channel_growth_factor, 3, padding=1),
                nn.BatchNorm2d(channels * channel_growth_factor),
                nn.ReLU(),
                nn.Conv2d(channels * channel_growth_factor, channels * channel_growth_factor, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            channels *= channel_growth_factor
            encoder_layers.append(layer)
        self.encoder = nn.ModuleList(encoder_layers)

    def forward(self, x):
        skipped = []
        for layer in self.encoder:
            x = layer(x)
            skipped.append(x)
        skipped.pop()
        return x, skipped


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, channel_growth_factor) -> None:
        super().__init__()
        decoder_layers = []

        channels = in_channels * (channel_growth_factor ** depth)
        for i in range(depth):
            layer = nn.Sequential(
                nn.Conv2d(channels, channels // channel_growth_factor, 3, 1, 1),
                nn.BatchNorm2d(channels // channel_growth_factor),
                nn.ReLU(),
                nn.ConvTranspose2d(channels // channel_growth_factor, channels // channel_growth_factor, 2, 2),
                nn.ReLU()
            )
            decoder_layers.append(layer)
            channels //= channel_growth_factor
        decoder_layers.append(nn.Conv2d(channels, out_channels, 3, 1, 1))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x, skipped):
        for i, layer in enumerate(self.decoder):
            if 0 < i < len(self.decoder)-1:
                x = layer(x + skipped[-i])
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """
    Each encoder layer is Conv Batch Relu Conv Relu Dropout
    Each decoder layer is Conv Batch Relu Transpose Relu

    Each layer increases channels by channel_growth_factor
    There are 'depth' layers

    Classification projection is two linear layers
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=2,
                 depth=3,
                 channel_growth_factor=2
                 ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            depth=depth,
            channel_growth_factor=channel_growth_factor
        )

        self.decoder = Decoder(
            in_channels=in_channels,
            depth=depth,
            out_channels=out_channels,
            channel_growth_factor=channel_growth_factor
        )

    def forward(self, x):
        x, skipped = self.encoder(x)
        return self.decoder(x, skipped)


if __name__ == "__main__":
    net = UNet(in_channels=3)
    x = torch.randn(1, 3, 256, 256)
    print(x.shape)
    print(net(x).shape)
