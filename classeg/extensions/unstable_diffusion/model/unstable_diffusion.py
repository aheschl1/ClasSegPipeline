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
            apply_scale_u=False,
            takes_time=True,
            verbose=False
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.perform_attention = attention
        self.verbose = verbose
        self.takes_time = takes_time
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
        if verbose:
            print('Time embedding dim: ', time_emb_dim, out_channels)
        if self.takes_time:
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

    def forward(self, x, time_embedding=None, residual_connection=None):
        assert (time_embedding is not None) == self.takes_time, "Time embedding is required if takes_time is True"
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
            # Add 'time'
            if time_embedding is not None:
                out = out + self.time_embedding_layer[layer](time_embedding, out.shape[0])[:, :, None, None]
            
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
            [nn.GroupNorm(8, num_channels=in_channels) for _ in range(2)]
        )
        self.self_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
                for _ in range(2)
            ]
        )
        self.cross_attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, num_channels=in_channels) for _ in range(2)]
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
                for _ in range(2)
            ]
        )
        self.pointwise_convolution_im = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.pointwise_convolution_seg = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, im, seg, time_embedding):
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
        # ================ SELF SEG ====================
        seg_out = self.first_residual_convs[1](seg)
        seg_out = seg_out + self.time_embedding_layer[1](time_embedding, seg_out.shape[1])[:, :, None, None]

        N, C, H, W = seg_out.shape
        in_attn = seg_out.reshape(N, C, H * W)
        in_attn = self.self_attention_norms[1](in_attn).transpose(1, 2)
        out_attn, _ = self.self_attentions[1](
            in_attn, in_attn, in_attn
        )
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        seg_out = seg_out + out_attn
        # ================ CROSS ATTENTION SEG====================
        N, C, H, W = seg_out.shape
        in_attn = seg_out.reshape(N, C, H * W)
        in_attn = self.cross_attention_norms[1](in_attn).transpose(1, 2)

        kv_attn = im_out.reshape(N, C, H * W)
        kv_attn = self.cross_attention_norms[1](kv_attn).transpose(1, 2)

        out_attn, _ = self.cross_attentions[1](
            in_attn, kv_attn, kv_attn
        )
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        seg_out = seg_out + out_attn
        # ================ CROSS ATTENTION IM====================
        N, C, H, W = im_out.shape
        in_attn = seg_out.reshape(N, C, H * W)
        in_attn = self.cross_attention_norms[0](in_attn).transpose(1, 2)

        kv_attn = seg_out.reshape(N, C, H * W)
        kv_attn = self.cross_attention_norms[0](kv_attn).transpose(1, 2)

        out_attn, _ = self.cross_attentions[0](
            in_attn, kv_attn, kv_attn
        )
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        im_out = im_out + out_attn

        # FINAL CONV

        return self.pointwise_convolution_im(im_out), self.pointwise_convolution_seg(seg_out)


class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            time_emb_dim=100,
            norm_op="GroupNorm",
            conv_op="Conv2d",
            non_lin=nn.SiLU,
            num_heads=4,
            num_layers=1,
            kernel_size=3,
            stride=1,
            padding=1,
            upsample=True,
            attention=False,
            skipped=True,
            do_time_embedding=True
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.perform_attention = attention
        self.num_layers = num_layers
        self.upsample = upsample
        norm_op = nn.GroupNorm
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    norm_op(8, in_channels if i == 0 else out_channels),
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
        if do_time_embedding:
            self.time_embedding_layer = nn.ModuleList(
                [TimeEmbedder(time_emb_dim, out_channels) for _ in range(num_layers)]
            )
        self.second_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    norm_op(8, out_channels),
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
                [norm_op(8, num_channels=out_channels) for _ in range(num_layers)]
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
                in_channels * 3 if skipped else in_channels, in_channels, kernel_size=4, stride=2, padding=1
            )
            if upsample
            else nn.Identity()
        )
        self.scale_u = ScaleULayer(in_channels, skipped_count=2)

    def forward(
            self, x, skipped_connection_encoder=None, skipped_connection_decoder=None, time_embedding=None
    ):
        # print('UpBlock forward')
        # print('receive input x: ', x.shape)
        # print('receive skipped: ', skipped_connection.shape)
        # x = in_channels
        # x = in_channels//2
        if skipped_connection_encoder is not None:
            x = self.scale_u(x, skipped_connection_encoder, skipped_connection_decoder)
        x = self.upsample_conv(x)

        out = x
        # TODO check time embeedding domension stuff
        for layer in range(self.num_layers):
            res_input = out
            out = self.first_residual_convs[layer](out)
            if time_embedding is not None:
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

class ContextIntegrator(nn.Module):
    def __init__(self, channels, time_emb_dim, context_embedding_dim):
        super(ContextIntegrator, self).__init__()
        self.context_embedding_dim = context_embedding_dim
        self.time_emb_dim = time_emb_dim

        self.context_embedding_projector = nn.Sequential(
            nn.Linear(context_embedding_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        # self.time_embedding_layer = TimeEmbedder(time_emb_dim, channels)
        
        self.cross_attention_norm = nn.GroupNorm(8, num_channels=channels)
        self.context_norm = nn.GroupNorm(8, num_channels=channels)

        self.cross_attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x, t, context_embedding):
        """
        Integrate time as is normal, then do a cross attention with the context embedding, then do conv, then skip
        context_embedding: B x context_embedding_dim
        x: B x C x H x W
        """
        # out = x + self.time_embedding_layer(t, x.shape[0])[:, :, None, None]
        out = x
        context_embedding = self.context_embedding_projector(context_embedding)
        # Do cross attention betwene the image x ~ [N, C, H, W] and the context embedding ~ [N, C]
        N, C, H, W = out.shape
        in_attn = out.reshape(N, C, H * W)
        in_attn = self.cross_attention_norm(in_attn).transpose(1, 2)

        kv_attn = context_embedding.reshape(N, C, 1)
        kv_attn = self.context_norm(kv_attn).transpose(1, 2)

        out_attn, _ = self.cross_attention(
            in_attn, kv_attn, kv_attn
        )

        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W) # [N, C, H, W]
        out = x + out_attn

        return self.convolution(out)



class UnstableDiffusion(nn.Module):
    def __init__(
            self,
            im_channels,
            seg_channels,
            layer_depth=2,
            channels=None,
            time_emb_dim=100,
            context_embedding_dim=512,
            do_context_embedding=False,
            shared_encoder=False
):

        super(UnstableDiffusion, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.context_embedding_dim = context_embedding_dim
        self.do_context_embedding = do_context_embedding
        self.im_channels = im_channels
        self.seg_channels = seg_channels
        if channels is None:
            channels = [16, 32, 64]
        layers = len(channels)

        self.shared_encoder = shared_encoder
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
            in_channels=im_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.seg_conv_in = nn.Conv2d(
            in_channels=seg_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Encoder
        self.encoder_layers = self._generate_encoder()

        if not shared_encoder:
            self.encoder_layers_mask = self._generate_encoder()

        # Middle
        mid_channels = channels[-1]
        self.middle_layer = MidBlock(
            in_channels=mid_channels,
            time_emb_dim=self.time_emb_dim,
        )
        # Decoder IM
        self.im_decoder_layers = self._generate_decoder()

        # Decoder SEG
        self.seg_decoder_layers = self._generate_decoder()

        # Context Embedding
        if self.do_context_embedding:
            downsampled_dim = int(128 / (2 ** (len(self.channels) - 1)))
            self.context_embedding_generator = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(im_channels, channels[0], kernel_size=3, padding=1),
                    nn.ReLU(),
                    self._generate_encoder(take_time=False, sequential=True)
                ),
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(channels[-1], channels[-1]*2, kernel_size=3, padding=1, stride=2),
                    nn.Flatten(start_dim=1),
                    nn.Linear(channels[-1]*2*(downsampled_dim//2)*(downsampled_dim//2), context_embedding_dim)
                )
            ])

            self.image_context_integrator = ContextIntegrator(
                channels=channels[-1],
                time_emb_dim=self.time_emb_dim,
                context_embedding_dim=self.context_embedding_dim
            )

            self.seg_context_integrator = ContextIntegrator(
                channels=channels[-1],
                time_emb_dim=self.time_emb_dim,
                context_embedding_dim=self.context_embedding_dim
            )

            self.image_context_integrator_first = ContextIntegrator(
                channels=channels[-1],
                time_emb_dim=self.time_emb_dim,
                context_embedding_dim=self.context_embedding_dim
            )

            self.seg_context_integrator_first = ContextIntegrator(
                channels=channels[-1],
                time_emb_dim=self.time_emb_dim,
                context_embedding_dim=self.context_embedding_dim
            )


            self.image_context_decoder = nn.Sequential(
                self._generate_decoder(sequential=True, skipped=False)
            )

            self.output_layer_embed = nn.Sequential(
                nn.GroupNorm(8, channels[0]),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=im_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Sigmoid()
            )

        self.output_layer_im = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=im_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.output_layer_seg = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=seg_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _generate_encoder(self, take_time=True, sequential=False):
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
                    takes_time=take_time,
                    apply_scale_u=True
                )
            )
        if sequential:
            return nn.Sequential(*encoder_layers)
        return encoder_layers

    def _generate_decoder(self, sequential=False, skipped=True):
        decoder_layers = nn.ModuleList()
        for layer in range(self.layers - 1, 0, -1):
            in_channels = self.channels[layer]
            out_channels = self.channels[layer - 1]
            decoder_layers.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=True,
                    skipped=skipped,
                    num_layers=self.layer_depth
                )
            )
        if sequential:
            return nn.Sequential(*decoder_layers)
        return decoder_layers

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
            self, im_out, seg_out, t
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Encodes im and seg, keeping in mind shared encoder vs not.
        If no shared encoder, turns on Residual connections between the two
        :param im_out:
        :param seg_out:
        :return:
        """
        skipped_connections_im = [im_out]
        skipped_connections_seg = [seg_out]
        # =========== SHARED ENCODER ===========
        if self.shared_encoder:
            for encoder in self.encoder_layers:
                im_out = encoder(im_out, t)
                skipped_connections_im.append(im_out)
            for encoder in self.encoder_layers:
                seg_out = encoder(seg_out, t)
                skipped_connections_seg.append(seg_out)
            return im_out, seg_out, skipped_connections_im, skipped_connections_seg
        # =========== NOT SHARED ENCODER ===========
        for i, (im_encode, seg_encode) in enumerate(zip(self.encoder_layers, self.encoder_layers_mask)):
            if i == 0:
                im_out, seg_out = im_encode(im_out, t), seg_encode(seg_out, t)
            else:
                im_out, seg_out = (
                    im_encode(im_out, t, residual_connection=seg_out),
                    seg_encode(seg_out, t, residual_connection=im_out),
                )
            skipped_connections_im.append(im_out)
            skipped_connections_seg.append(seg_out)

        return im_out, seg_out, skipped_connections_im, skipped_connections_seg

    def get_discriminator(self):
        return nn.ModuleList([
            nn.Conv2d(
                in_channels=self.im_channels + self.seg_channels,
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
    
    def embed_image(self, im, recon=True):
        assert self.do_context_embedding, "Image embedding is not enabled"
        encoded = self.context_embedding_generator[0](im)
        code = self.context_embedding_generator[1](encoded)
        decoded = None
        if recon:
            decoded = self.image_context_decoder(encoded)
            decoded = self.output_layer_embed(decoded)
        
        return code, decoded

    def forward(self, im, seg, t, img_embedding=None):
        assert im.shape[2] == 128, "Only 128 resolution supported"
        assert (img_embedding is not None) == self.do_context_embedding, "context embedding is not enabled"

        # ======== TIME ========
        t = self._sinusoidal_embedding(t)
        t = self.t_proj(t)
        # ======== ENTRY ========
        im_out = self.im_conv_in(im)
        seg_out = self.seg_conv_in(seg)
        # ======== ENCODE ========
        im_out, seg_out, skipped_connections_im, skipped_connections_seg = (
            self._encode_forward(im_out, seg_out, t)
        )
        #1st
        if self.do_context_embedding:
            im_out = self.image_context_integrator_first(im_out, t, img_embedding)
            seg_out = self.seg_context_integrator_first(seg_out, t, img_embedding)
        # ======== MIDDLE ========
        im_out, seg_out = self.middle_layer(im_out, seg_out, t)
        # Raw image embedding for controllable dataset generation
        if self.do_context_embedding:
            im_out = self.image_context_integrator(im_out, t, img_embedding)
            seg_out = self.seg_context_integrator(seg_out, t, img_embedding)
        # ======== DECODE ========
        i = 0
        for im_decode, seg_decode in zip(
                self.im_decoder_layers, self.seg_decoder_layers
        ):
            i += 1
            im_out, seg_out = (
                im_decode(im_out, skipped_connections_im[-i], seg_out, t),
                seg_decode(seg_out, skipped_connections_seg[-i], im_out, t),
            )
        # ======== EXIT ========
        im_out = self.output_layer_im(im_out)
        seg_out = self.output_layer_seg(seg_out)
        return im_out, seg_out


if __name__ == "__main__":
    x = torch.zeros(2, 3, 128, 128).cuda()
    m = torch.zeros(2, 1, 128, 128).cuda()
    t = torch.zeros(2).cuda()
    model = UnstableDiffusion(3, 1, channels=[16, 32, 256], do_image_embedding=True).cuda()
    out = model(x, m, t, torch.zeros(2, 64).cuda())