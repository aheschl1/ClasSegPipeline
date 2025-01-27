import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM

class ModelBrain(nn.Module):
    """
    Class for encapsulating a pretrained language model
    """
    def __init__(self, model_id="meta-llama/Llama-3.2-1B", device="cuda"):
        super(ModelBrain, self).__init__()
        self.model_id = model_id
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model = base.model
        self.model.embed_tokens = nn.Identity()

    def forward(self, x):
        """
        Returns the output of the model.
        """
        return self.model(x).last_hidden_state

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
    # [3, 128, 128] -> [6, 64, 64] -> [12, 32, 32] -> [24, 16, 16] 24*16*16 = 
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 depth=4,
                 channel_growth_factor=2,
                 llm_brain="meta-llama/Llama-3.2-1B"
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

        self.llm =nn.Sequential( 
            nn.Flatten(start_dim=2),
            nn.LazyLinear(2048),
            ModelBrain(model_id=llm_brain),
            nn.Linear(2048, 1024),
            nn.Unflatten(2, (16, 16))
        )

    def forward(self, x):
        x, skipped = self.encoder(x)
        x = self.llm(x)
        return self.decoder(x, skipped)


if __name__ == "__main__":
    net = UNet(in_channels=3)
    x = torch.randn(1, 3, 128, 128)
    print(x.shape)
    print(net(x).shape)
