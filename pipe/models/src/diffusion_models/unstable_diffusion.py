import sys

sys.path.append("/home/student/andrewheschl/Documents/diffusion")
import torch
import torch.nn as nn
from pipe.models.pipe.utils import my_import


class LateInitializationLayerNorm(nn.Module):
    def __init__(self, **kwargs):
        super(LateInitializationLayerNorm, self).__init__()
        self.ln = None

    def forward(self, x):
        if self.ln is None:
            if self.training:
                print(
                    "WARNING: LateInitializationLayerNorm is in training and is initializing itself now. "
                    "Concider running an input through first to complete initialization early"
                )
            self.ln = nn.LayerNorm(x.shape[1:]).to(x.device)
        return self.ln(x)


class TimeEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, t, n):
        # TODO remove arg
        return self.embedder(t)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=100,
        conv_op="Conv2d",
        norm_op="BatchNorm",
        non_lin=nn.SiLU,
        num_heads=4,
        num_layers=1,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=True,
        attention=False,
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        norm_op = my_import(norm_op)
        conv_op = my_import(conv_op)
        self.perform_attention = attention
        self.num_layers = num_layers
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    non_lin(),
                    conv_op(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.time_embedding_layer = nn.ModuleList(
            [TimeEmbedder(time_emb_dim, out_channels) for _ in range(num_layers)]
        )
        self.second_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    non_lin(),
                    conv_op(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        if self.perform_attention:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(8, num_channels=out_channels) for _ in range(num_layers)]
            )
            self.multihead_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        self.pointwise_convolution = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.downsample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(self, x, time_embedding):
        # print('DownBlock forward')
        # print('receive input x: ', x.shape)
        out = x
        # TODO check time embeedding domension stuff
        for layer in range(self.num_layers):
            res_input = out
            out = self.first_residual_convs[layer](out)
            out = (
                out
                + self.time_embedding_layer[layer](time_embedding, out.shape[0])[
                    :, :, None, None
                ]
            )
            out = self.second_residual_convs[layer](out)
            # Skipped connection
            out = out + self.pointwise_convolution[layer](res_input)

            if self.perform_attention:
                # Attention
                N, C, H, W = out.shape
                in_attn = out.reshape(N, C, H * W)
                in_attn = self.attention_norms[layer](in_attn).transpose(1, 2)
                out_attn, _ = self.multihead_attentions[layer](
                    in_attn, in_attn, in_attn
                )
                out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
                # Skipped
                # TODO maybe concat?
                out = out + out_attn
        out = self.downsample_conv(out)
        # print('output shape: ', out.shape)
        return out


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        time_emb_dim=100,
        conv_op="Conv2d",
        norm_op="BatchNorm",
        non_lin=nn.SiLU,
        num_heads=4,
        num_layers=1,
        kernel_size=3,
        stride=1,
        padding=1,
        attention=False,
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        norm_op = my_import(norm_op)
        conv_op = my_import(conv_op)
        self.perform_attention = attention
        self.num_layers = num_layers
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels),
                    non_lin(),
                    conv_op(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(num_layers + 1)
            ]
        )
        self.time_embedding_layer = nn.ModuleList(
            [TimeEmbedder(time_emb_dim, in_channels) for _ in range(num_layers + 1)]
        )
        self.second_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels),
                    non_lin(),
                    conv_op(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(num_layers + 1)
            ]
        )
        if self.perform_attention:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(8, num_channels=in_channels) for _ in range(num_layers)]
            )
            self.multihead_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        self.pointwise_convolution = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, time_embedding):
        # TODO maybe combine with the down class
        # print('MidBlock forward')
        # print('receive input x: ', x.shape)
        out = x

        resnet_in = out
        out = self.first_residual_convs[0](out)
        out = (
            out
            + self.time_embedding_layer[0](time_embedding, out.shape[0])[
                :, :, None, None
            ]
        )
        out = self.second_residual_convs[0](out)
        out = out + self.pointwise_convolution[0](resnet_in)

        for layer in range(self.num_layers):
            # Attention
            if self.perform_attention:
                N, C, H, W = out.shape
                in_attn = out.reshape(N, C, H * W)
                in_attn = self.attention_norms[layer](in_attn).transpose(1, 2)
                out_attn, _ = self.multihead_attentions[layer](
                    in_attn, in_attn, in_attn
                )
                out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
                # Skipped
                # TODO maybe concat?
                out = out + out_attn

            res_input = out
            out = self.first_residual_convs[layer + 1](out)
            out = (
                out
                + self.time_embedding_layer[layer + 1](time_embedding, out.shape[0])[
                    :, :, None, None
                ]
            )
            out = self.second_residual_convs[layer + 1](out)
            # Skipped connection
            out = out + self.pointwise_convolution[layer + 1](res_input)
        # print('output shape: ', out.shape)
        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=100,
        norm_op="BatchNorm",
        conv_op="Conv2d",
        non_lin=nn.SiLU,
        num_heads=4,
        num_layers=1,
        kernel_size=3,
        stride=1,
        padding=1,
        upsample=True,
        attention=False,
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        norm_op = my_import(norm_op)
        conv_op = my_import(conv_op)
        self.perform_attention = attention
        self.num_layers = num_layers
        self.upsample = upsample
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    non_lin(),
                    conv_op(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.time_embedding_layer = nn.ModuleList(
            [TimeEmbedder(time_emb_dim, out_channels) for _ in range(num_layers)]
        )
        self.second_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    non_lin(),
                    conv_op(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        if self.perform_attention:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(8, num_channels=out_channels) for _ in range(num_layers)]
            )
            self.multihead_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        self.pointwise_convolution = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.upsample_conv = (
            nn.ConvTranspose2d(
                in_channels * 2, in_channels, kernel_size=4, stride=2, padding=1
            )
            if upsample
            else nn.Identity()
        )

    def forward(self, x, skipped_connection, time_embedding):
        # print('UpBlock forward')
        # print('receive input x: ', x.shape)
        # print('receive skipped: ', skipped_connection.shape)
        # x = in_channels
        # x = in_channels//2
        x = torch.concat([x, skipped_connection], dim=1)
        x = self.upsample_conv(x)

        out = x
        # TODO check time embeedding domension stuff
        for layer in range(self.num_layers):
            res_input = out
            out = self.first_residual_convs[layer](out)
            out = (
                out
                + self.time_embedding_layer[layer](time_embedding, out.shape[0])[
                    :, :, None, None
                ]
            )
            out = self.second_residual_convs[layer](out)
            # Skipped connection
            out = out + self.pointwise_convolution[layer](res_input)

            if self.perform_attention:
                # Attention
                N, C, H, W = out.shape
                in_attn = out.reshape(N, C, H * W)
                in_attn = self.attention_norms[layer](in_attn).transpose(1, 2)
                out_attn, _ = self.multihead_attentions[layer](
                    in_attn, in_attn, in_attn
                )
                out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
                # Skipped
                # TODO maybe concat?
                out = out + out_attn
        # print('output shape: ', out.shape)
        return out


NORM, HIGH, NONE = "norm", "high", "none"


class UnstableDiffusion(nn.Module):
    def __init__(
        self,
        im_channels,
        layer_depth=2,
        channels=None,
        middle_layers_count=1,
        conv_op: str = "Conv2d",
        norm_op: str = "BatchNorm2d",
        time_emb_dim=100,
        attention="norm",
    ):
        super(UnstableDiffusion, self).__init__()
        assert attention in [
            NORM,
            HIGH,
            NONE,
        ], f"Invalid attention mode {attention}. Expecting one of {[NORM, HIGH, NONE]}"
        self.time_emb_dim = time_emb_dim
        norm_op = my_import(norm_op)
        if channels is None:
            channels = [16, 32, 64]
        layers = len(channels)

        # Sinusoidal embedding
        self.t_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = my_import(conv_op)(
            in_channels=im_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Encoder
        self.encoder_layers = nn.ModuleList()
        for layer in range(layers - 1):
            # We want to build an downblock here.
            in_channels = channels[layer]
            out_channels = channels[layer + 1]
            self.encoder_layers.append(
                DownBlock(
                    conv_op=conv_op,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    downsample=True,
                    num_layers=layer_depth,
                    attention=attention == HIGH,
                )
            )

        # Middle
        mid_channels = channels[-1]
        self.middle_layers = nn.ModuleList()
        for layer in range(middle_layers_count):
            self.middle_layers.append(
                MidBlock(
                    in_channels=mid_channels,
                    time_emb_dim=self.time_emb_dim,
                    num_layers=layer_depth,
                    conv_op=conv_op,
                    attention=attention != NONE,
                )
            )
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for layer in range(layers - 1, 0, -1):
            in_channels = channels[layer]
            out_channels = channels[layer - 1]
            self.decoder_layers.append(
                UpBlock(
                    conv_op=conv_op,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=True,
                    num_layers=layer_depth,
                    attention=attention == HIGH,
                )
            )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=im_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _sinusoidal_embedding(self, t):
        assert self.time_emb_dim % 2 == 0
        factor = 10000 ** (
            (
                torch.arange(
                    start=0,
                    end=self.time_emb_dim // 2,
                    dtype=torch.float32,
                    device=t.device,
                )
                / (self.time_emb_dim // 2)
            )
        )
        t_emb = t[:, None].repeat(1, self.time_emb_dim // 2) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb

    def forward(self, x, t):
        # print('Network forward')
        # print('receive input x: ', x.shape)

        out = self.conv_in(x)
        t = self._sinusoidal_embedding(t)
        t = self.t_proj(t)

        # print('after conv in: ', x.shape)
        skipped_connections = [out]
        for encoder in self.encoder_layers:
            out = encoder(out, t)
            skipped_connections.append(out)

        for mid in self.middle_layers:
            out = mid(out, t)

        for decoder in self.decoder_layers:
            down_out = skipped_connections.pop()
            out = decoder(out, down_out, t)
        # print("finishing decoders", out.shape)
        out = self.output_layer(out)
        # print("after output", out.shape)
        return out


if __name__ == "__main__":
    torch.cuda.empty_cache()
    in_shape = 128
    z = torch.randn(32, 3, in_shape, in_shape).cuda(1)
    unet = UnstableDiffusion(
        im_channels=3,
        channels=[32, 64, 128],
        norm_op="BatchNorm2d",
        middle_layers_count=2,
        layer_depth=2,
    ).cuda(1)
    # y = down(z, torch.randn(1, 8, 64*2, 64*2).cuda(), torch.ones(100).float().cuda())
    y = unet(z, torch.rand((32)).cuda(1))
