import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, last_layer: bool = False,
                 pad="default", transpose_kernel=2, transpose_stride=2) -> None:
        super().__init__()

        conv_op = nn.Conv2d
        transp_op = nn.ConvTranspose2d
        norm_op = nn.InstanceNorm2d

        pad = (kernel_size - 1) // 2 * dilation if pad == 'default' else int(pad)
        self.conv1 = nn.Sequential(
            conv_op(
                in_channels=in_channels * 2,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad
            ),
            norm_op(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_op(
                in_channels=in_channels,
                out_channels=(out_channels if last_layer else in_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad
            )
        )
        if not last_layer:
            self.transpose = transp_op(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=transpose_stride,
                kernel_size=transpose_kernel
            )
            self.conv2.append(
                norm_op(num_features=out_channels)
            )
            self.conv2.append(
                nn.LeakyReLU(inplace=True)
            )

        self.last_layer = last_layer

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        if not self.last_layer:
            return self.transpose(x)
        return x