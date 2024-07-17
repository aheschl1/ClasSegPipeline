from typing import Tuple, List

from classeg.extensions.unstable_diffusion.utils.utils import make_zero_conv

import torch
import torch.nn as nn
from classeg.extensions.unstable_diffusion.model.modules import ScaleULayer


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
            non_lin=nn.SiLU,
            num_heads=4,
            num_layers=1,
            kernel_size=3,
            stride=1,
            padding=1,
            downsample=True,
            attention=False,
            apply_zero_conv=False,
            apply_scale_u=False
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.perform_attention = attention

        self.apply_zero_conv = apply_zero_conv
        self.apply_scale_u = apply_scale_u
        assert not self.apply_scale_u or not self.apply_zero_conv, "Cannot do both scaleu and zero conv"
        if self.apply_zero_conv:
            self.zero_conv = make_zero_conv(channels=in_channels, conv_op=nn.Conv2d)
        if self.apply_scale_u:
            self.scale_u = ScaleULayer(in_channels, skipped_count=1)
            self.scale_u_pointwise_convolution = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)

        self.num_layers = num_layers
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    non_lin(),
                    nn.Conv2d(
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
                    nn.Conv2d(
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

    def forward(self, x, time_embedding, residual_connection=None):
        out = x
        if residual_connection is not None:
            if self.apply_zero_conv:
                out = out + self.zero_conv(residual_connection)
            elif self.apply_scale_u:
                out = self.scale_u_pointwise_convolution(self.scale_u(x, residual_connection))
            else:
                out = out + residual_connection

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
            non_lin=nn.SiLU,
            num_heads=4,
            kernel_size=3,
            stride=1,
            padding=1
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.first_residual_convs = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(8, in_channels),
            non_lin(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        ) for _ in range(2)])

        self.time_embedding_layer = nn.ModuleList([TimeEmbedder(time_emb_dim, in_channels) for _ in range(2)])

        self.self_attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, num_channels=in_channels) for _ in range(1)]
        )
        self.self_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
                for _ in range(1)
            ]
        )
        self.pointwise_convolution = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, im, time_embedding):
        # ================ SELF IM ====================
        im_out = self.first_residual_convs[0](im)
        im_out = im_out + self.time_embedding_layer[0](time_embedding, im_out.shape[0])[:, :, None, None]

        N, C, H, W = im_out.shape
        in_attn = im_out.reshape(N, C, H * W)
        in_attn = self.self_attention_norms[0](in_attn).transpose(1, 2)
        out_attn, _ = self.self_attentions[0](
            in_attn, in_attn, in_attn
        )
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        im_out = im_out + out_attn

        return self.pointwise_convolution(im_out)


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
        self.perform_attention = attention
        self.num_layers = num_layers
        self.upsample = upsample
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    non_lin(),
                    nn.Conv2d(
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
                    nn.Conv2d(
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
        self.scale_u = ScaleULayer(in_channels, skipped_count=1)

    def forward(
            self, x, skipped_connection_encoder, time_embedding
    ):
        # print('UpBlock forward')
        # print('receive input x: ', x.shape)
        # print('receive skipped: ', skipped_connection.shape)
        # x = in_channels
        # x = in_channels//2
        x = self.scale_u(x, skipped_connection_encoder)
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


class ConcatDiffusion(nn.Module):
    def __init__(
            self,
            im_channels,
            seg_channels,
            layer_depth=2,
            channels=None,
            time_emb_dim=100,
            realfy=False,
            **kwargs
    ):

        super(ConcatDiffusion, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.im_channels = im_channels
        self.seg_channels = seg_channels
        if channels is None:
            channels = [16, 32, 64]
        layers = len(channels)

        self.layers = layers
        self.channels = channels
        self.time_emb_dim = time_emb_dim
        self.layer_depth = layer_depth

        # Sinusoidal embedding
        self.t_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.im_conv_in = nn.Conv2d(
            in_channels=im_channels+seg_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Encoder
        self.encoder_layers = self._generate_encoder()

        # Middle
        mid_channels = channels[-1]
        self.middle_layer = MidBlock(
            in_channels=mid_channels,
            time_emb_dim=self.time_emb_dim,
        )
        # Decoder IM
        self.im_decoder_layers = nn.ModuleList()
        for layer in range(layers - 1, 0, -1):
            in_channels = channels[layer]
            out_channels = channels[layer - 1]
            self.im_decoder_layers.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=True,
                    num_layers=layer_depth
                )
            )

        self.output_layer_im = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=im_channels+seg_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        if realfy:
            self.realfier = self._get_realfier()
        else:
            self.realfier = None

    def realfy(self, x, t):
        t = self._sinusoidal_embedding(t)
        t = self.t_proj(t)
        x = self.realfier[0](x)

        return self.realfier[2](self.realfier[1](x, t))

    def _get_realfier(self):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.im_channels,
                    out_channels=self.channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GroupNorm(8, self.channels[0]),
                nn.SiLU(),
            ),
            DownBlock(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                time_emb_dim=self.time_emb_dim,
                downsample=False,
                num_layers=self.layer_depth,
                apply_zero_conv=False,
                apply_scale_u=False
            ),
            nn.Sequential(
                nn.GroupNorm(8, self.channels[0]),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=self.channels[0],
                    out_channels=self.im_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
        ])

    def get_discriminator(self):
        return nn.ModuleList([
            nn.Conv2d(
                in_channels=self.im_channels,
                out_channels=self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            self._generate_encoder(),
            nn.Sequential(
                nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.channels[-1], self.channels[-1] // 2),
                nn.SiLU(),
                nn.Linear(self.channels[-1] // 2, 1)
            )
        ])

    def discriminate(self, discriminator: nn.Module, im, t):
        t = self._sinusoidal_embedding(t)
        t = self.t_proj(t)

        im = discriminator[0](im)
        for layer in discriminator[1]:
            im = layer(im, t, None)
        return discriminator[2](im)

    def _generate_encoder(self):
        encoder_layers = nn.ModuleList()
        for layer in range(self.layers - 1):
            # We want to build a downblock here.
            in_channels = self.channels[layer]
            out_channels = self.channels[layer + 1]
            encoder_layers.append(
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    downsample=True,
                    num_layers=self.layer_depth,
                    apply_zero_conv=False,
                    apply_scale_u=False
                )
            )
        return encoder_layers

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

    def _encode_forward(
            self, im_out, t
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Encodes im and seg, keeping in mind shared encoder vs not.
        If no shared encoder, turns on Residual connections between the two
        :param im_out:
        :param seg_out:
        :return:
        """
        skipped_connections_im = [im_out]
        # =========== SHARED ENCODER ===========
        for encoder in self.encoder_layers:
            im_out = encoder(im_out, t)
            skipped_connections_im.append(im_out)
        return im_out, skipped_connections_im
    def forward(self, im, seg, t):

        assert im.shape[2] == 128, "Only 128 resolution supported"
        # ======== TIME ========
        t = self._sinusoidal_embedding(t)
        t = self.t_proj(t)
        # ======== ENTRY ========
        im_out = self.im_conv_in(torch.concat([im, seg], dim=1))
        # ======== ENCODE ========
        im_out, skipped_connections_im = (
            self._encode_forward(im_out, t)
        )
        # ======== MIDDLE ========
        im_out = self.middle_layer(im_out, t)
        # ======== DECODE ========
        i = 0
        for im_decode in self.im_decoder_layers:
            i += 1
            im_out = im_decode(im_out, skipped_connections_im[-i], t)
        # ======== EXIT ========
        im_out = self.output_layer_im(im_out)
        return im_out[:, :self.im_channels], im_out[:, self.im_channels:]


# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     in_shape = 32
#     im = torch.randn(1, 3, in_shape, in_shape).float().cuda(0)
#     seg = torch.randn(1, 1, in_shape, in_shape).float().cuda(0)

#     unet = UnstableDiffusion(
#         im_channels=3,
#         seg_channels=1,
#         channels=[32, 64],
#         layer_depth=2
#     ).cuda(0)
#     # y = down(z, torch.randn(1, 8, 64*2, 64*2).cuda(), torch.ones(100).float().cuda())
#     y = unet(im, seg, torch.rand((1)).cuda(0))


if __name__ == "__main__":
    x = torch.zeros(2, 3, 128, 128)
    m = torch.zeros(2, 1, 128, 128)
    t = torch.zeros(2)
    model = ConcatDiffusion(3, 1, channels=[16, 32, 64], shared_encoder=True, realfy=True)
    im, seg = model(x, m, t)
    im = model.realfy(im, t)
    print(im.shape, seg.shape)
