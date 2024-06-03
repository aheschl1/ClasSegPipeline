import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, depth, hidden_size, input_dimension, channel_growth_factor) -> None:
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
        encoder_layers.append(nn.Flatten())
        flattened_shape = ((input_dimension // (2 ** depth)) ** 2) * channels
        encoder_layers.append(nn.Linear(flattened_shape, hidden_size))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, depth, hidden_size, input_dimension, channel_growth_factor) -> None:
        super().__init__()
        decoder_layers = []

        channels = in_channels * (channel_growth_factor ** depth)
        flattened_shape = ((input_dimension // (2 ** depth)) ** 2) * channels
        pre_flattened_shape = (channels, (input_dimension // (2 ** depth)), (input_dimension // (2 ** depth)))

        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_size, flattened_shape))
        decoder_layers.append(nn.Unflatten(1, pre_flattened_shape))
        decoder_layers.append(nn.ReLU())
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
        decoder_layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    Autoencoder with optional classification head on embedding.

    Each encoder layer is Conv Batch Relu Conv Relu Dropout
    Each decoder layer is Conv Batch Relu Transpose Relu

    Each layer increases channels by channel_growth_factor
    There are 'depth' layers

    Classification projection is two linear layers
    """

    def __init__(self,
                 in_channels=1,
                 depth=3,
                 hidden_size=64,
                 input_dimension=256,
                 channel_growth_factor=2
                 ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            depth=depth,
            hidden_size=hidden_size,
            input_dimension=input_dimension,
            channel_growth_factor=channel_growth_factor
        )

        self.decoder = Decoder(
            in_channels=in_channels,
            depth=depth,
            hidden_size=hidden_size,
            input_dimension=input_dimension,
            channel_growth_factor=channel_growth_factor
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
