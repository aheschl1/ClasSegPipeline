import math
from typing import List

from einops.layers.torch import Reduce
import torch.nn as nn
import torch

from pipe.models.pipe.utils import my_import

CONCAT = 'concat'
ADD = 'add'
TWO_D = '2d'
THREE_D = '3d'
INSTANCE = 'instance'
BATCH = "batch"


class ModuleStateController:
    TWO_D = "2d"
    THREE_D = "3d"

    state = TWO_D

    def __init__(self):
        assert False, "Don't make this object......"

    @classmethod
    def conv_op(cls):
        if cls.state == cls.THREE_D:
            return nn.Conv3d
        else:
            return nn.Conv2d

    @classmethod
    def instance_norm_op(cls):
        if cls.state == cls.THREE_D:
            return nn.InstanceNorm3d
        else:
            return nn.InstanceNorm2d

    @classmethod
    def batch_norm_op(cls):
        if cls.state == cls.THREE_D:
            return nn.BatchNorm3d
        else:
            return nn.BatchNorm2d

    @classmethod
    def transp_op(cls):
        if cls.state == cls.THREE_D:
            return nn.ConvTranspose3d
        else:
            return nn.ConvTranspose2d

    @classmethod
    def set_state(cls, state: str):
        assert state in [cls.TWO_D, cls.THREE_D], "Invalid state womp womp"
        cls.state = state

    @classmethod
    def avg_pool_op(cls):
        if cls.state == cls.THREE_D:
            return nn.AvgPool3d
        return nn.AvgPool2d


class ChannelAttentionCAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # performing pooling operations

        conv_op = ModuleStateController.conv_op()

        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        # convolutions
        self.conv1by1 = conv_op(channels, channels // 16, kernel_size=1)
        self.conv1by1_2 = conv_op(channels // 16, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxpooled = self.maxpooling(x)
        output_max_pooling = self.conv1by1(maxpooled)
        output_max_pooling = self.relu(output_max_pooling)
        output_max_pooling = self.conv1by1_2(output_max_pooling)

        avgpooled = self.avgpooling(x)
        output_avg_pooling = self.conv1by1(avgpooled)
        output_avg_pooling = self.relu(output_avg_pooling)
        output_avg_pooling = self.conv1by1_2(output_avg_pooling)

        # element wise summation
        output_feature_map = output_max_pooling + output_avg_pooling
        ftr_map = self.sigmoid(output_feature_map)
        ftr = ftr_map * x
        return ftr


class SpatialAttentionCAM(nn.Module):
    def __init__(self):
        super().__init__()

        conv_op = ModuleStateController.conv_op()
        # performing channel wise pooling
        self.spatialmaxpool = Reduce('b c h w -> b 1 h w', 'max')
        self.spatialavgpool = Reduce('b c h w -> b 1 h w', 'mean')
        # padding to keep the tensor shape same as input
        self.conv = conv_op(1, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxpooled = self.spatialmaxpool(x)
        # print(maxpooled.shape)
        avgpooled = self.spatialavgpool(x)
        # print(avgpooled.shape)
        # adding the tensors
        summed = maxpooled + avgpooled
        # print(summed.shape)
        convolved = self.conv(summed)
        # print(convolved.shape)
        ftr_map = self.sigmoid(convolved)
        # print(ftr_map.shape)
        ftr = ftr_map * x
        return ftr


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        conv_op = ModuleStateController.conv_op()
        norm_op = ModuleStateController.instance_norm_op()

        self.conv = conv_op(channels, channels, kernel_size=3, padding=1)
        self.bn = norm_op(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        # print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionCAM(channels)
        self.spatial = SpatialAttentionCAM()
        self.conv = ConvBlock(channels)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        x = self.conv(x)
        return x


class LearnableChannelAttentionCAM(nn.Module):
    def __init__(self, channels, mode=0):
        super().__init__()
        # performing pooling operations
        self.f_0 = nn.Sequential()
        self.f_1 = nn.Sequential()
        self.mode = mode

        for i in range(2):
            pool = nn.AdaptiveAvgPool2d(8) if i == 0 else nn.AdaptiveMaxPool2d(8)
            conv = DepthWiseSeparableConv(channels, channels, kernel_sizes=[8], pad=0, use_norm=False)

            for module in [pool, nn.ReLU(inplace=True), conv, nn.ReLU(inplace=True)]:
                if i == 0:
                    self.f_0.append(module)
                else:
                    self.f_1.append(module)

    def forward(self, x):
        output_0 = self.f_0(x)
        output_1 = self.f_1(x)

        if self.mode == 0:
            x = torch.add(output_0, output_1)
            return torch.mul(nn.Sigmoid()(x), x)
        else:
            y = torch.add(torch.mul(nn.Sigmoid()(output_0), x), torch.mul(nn.Sigmoid()(output_1), x))
            return torch.add(x, y)


class LearnableCAM(nn.Module):
    def __init__(self, channels, mode=0):
        super().__init__()
        self.channel = LearnableChannelAttentionCAM(channels, mode=mode)

    def forward(self, x):
        ax = self.channel(x)
        return torch.add(x, ax)


# CBAM start=================================
class LearnableChannelAttention(nn.Module):
    def __init__(self, channels, r, dimension):
        super().__init__()
        # performing pooling operations
        self.pool = nn.MaxPool2d(2)
        dimension //= 2  # Because pool
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(dimension, dimension),
                              groups=channels)
        # input the results of pooling to the 1 hidden layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        reduced = self.pool(x)
        convolved = self.conv(reduced)
        # squeeze to reduce dimension (n,c,1,1) to (n,c)
        convolved = torch.squeeze(convolved)
        output_feature_map = self.mlp(convolved)
        # element wise summation
        ftr_map = self.sigmoid(output_feature_map)
        # print(ftrMap.shape)
        # converting tension (n,c) to (n,c,w,h)
        ftr_map = ftr_map.unsqueeze(-1)
        ftr_map = ftr_map.unsqueeze(-1)
        # print(ftrMap.shape)
        ftr = ftr_map * x
        # print(ftr.shape)
        return ftr


class ChannelAttention(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        # performing pooling operations
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        # input the results of pooling to the 1 hidden layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxpooled = self.maxpooling(x)
        # squeeze to reduce dimension (n,c,1,1) to (n,c)
        maxpooled = torch.squeeze(maxpooled)
        # print(maxpooled.shape)
        avgpooled = self.avgpooling(x)
        # squeeze to reduce dimension (n,c,1,1) to (n,c)
        avgpooled = torch.squeeze(avgpooled)
        # print(avgpooled.shape)
        mlp_output_max_pooling = self.mlp(maxpooled)
        mlp_output_avg_pooling = self.mlp(avgpooled)
        # element wise summation
        output_feature_map = mlp_output_max_pooling + mlp_output_avg_pooling
        ftr_map = self.sigmoid(output_feature_map)
        # print(ftr_map.shape)
        # converting tension (n,c) to (n,c,w,h)
        ftr_map = ftr_map.unsqueeze(-1)
        ftr_map = ftr_map.unsqueeze(-1)
        # print(ftr_map.shape)
        ftr = ftr_map * x
        # print(ftr.shape)
        return ftr


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # performing channel wise pooling
        self.spatialmaxpool = Reduce('b c h w -> b 1 h w', 'max')
        self.spatialavgpool = Reduce('b c h w -> b 1 h w', 'mean')
        # padding to keep the tensor shape same as input
        self.conv1d = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxpooled = self.spatialmaxpool(x)
        # print(maxpooled.shape)
        avgpooled = self.spatialavgpool(x)
        # print(avgpooled.shape)
        # concatenating the tensors
        concat = torch.cat([maxpooled, avgpooled], dim=1)
        # print(concat.shape)
        convolved = self.conv1d(concat)
        # print(convolved.shape)
        ftr_map = self.sigmoid(convolved)
        # print(ftr_map.shape)
        ftr = ftr_map * x
        return ftr


class CBAM(nn.Module):
    def __init__(self, channels, r, stride=1, mode="regular", dimension=-1):
        super().__init__()
        assert mode in ["regular", "learnable"], "The two modes are 'learnable' and 'regular'."
        assert mode == 'regular' or dimension != -1, "If the mode is 'learnable' specify the dimension parameter."
        self.channel = ChannelAttention(channels, r) if mode == "regular" else LearnableChannelAttention(channels, r,
                                                                                                         dimension)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class UpsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 upscale_factor=2, mode='bilinear'):
        super(UpsamplingConv, self).__init__()
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode=mode, align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding)
        )

    def forward(self, x):
        return self.module(x)


# CBAM end===========================================================

class ConvPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 upscale_factor=2):
        super(ConvPixelShuffle, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * upscale_factor ** 2,
                      kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        return self.module(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, attention_channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.out = nn.Conv2d(attention_channels, in_channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x):
        batch_size, _, height, width = x.size()
        query = self.query(x).view(batch_size, self.num_heads, -1, height * width).permute(0, 2, 1, 3)
        key = self.key(x).view(batch_size, self.num_heads, -1, height * width)
        value = self.value(x).view(batch_size, self.num_heads, -1, height * width).permute(0, 2, 1, 3)

        attention_weights = torch.matmul(query, key) / math.sqrt(value.size(-2))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        attended_values = torch.matmul(attention_weights, value).permute(0, 2, 1, 3)
        attended_values = attended_values.contiguous().view(batch_size, -1, height, width)

        return self.out(attended_values) + x


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilations_dsc=None,
                 kernel_sizes_dsc=None, mode=CONCAT,
                 stride=1, padding='default', use_norm=False, **kwargs
                 ):
        if dilations_dsc is None:
            dilations_dsc = [1]
        if kernel_sizes_dsc is None:
            kernel_sizes_dsc = [3]
        if 'kernel_size' in kwargs:
            kernel_sizes_dsc = [kwargs['kernel_size']]

        assert len(dilations_dsc) == len(kernel_sizes_dsc)
        assert mode in ['concat', 'add']

        self.mode = mode
        # GET OPERATIONS
        norm = ModuleStateController.instance_norm_op()
        conv_op = ModuleStateController.conv_op()

        super(DepthWiseSeparableConv, self).__init__()
        self.branches = nn.ModuleList()
        for dilation, kernel_size in zip(dilations_dsc, kernel_sizes_dsc):
            pad = (kernel_size - 1) // 2 * dilation if padding == 'default' else int(padding)
            branch = nn.Sequential(
                conv_op(in_channels, in_channels, kernel_size=kernel_size, padding=pad,
                        dilation=dilation, groups=in_channels, stride=stride),
                conv_op(in_channels, out_channels, kernel_size=1),
            )

            if use_norm:
                branch.insert(1, norm(num_features=in_channels))

            self.branches.append(branch)

    def forward(self, x):
        results = []
        for branch in self.branches:
            results.append(branch(x))
        if self.mode == 'concat':
            return torch.concat(tuple(results), dim=1)
        return torch.sum(torch.stack(results), dim=0)


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()

        conv_op = ModuleStateController.conv_op()

        self.depthwise_conv = conv_op(in_channels, in_channels,
                                      kernel_size=1, groups=in_channels)
        self.pointwise_conv = conv_op(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.depthwise_conv(x)
        attention_map = self.pointwise_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        out = x * attention_map
        return out


# noinspection PyTypeChecker
class XModule(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, **kwargs):
        super(XModule, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [kwargs['kernel_size']]
        self.branches = nn.ModuleList()

        # Picl the norm op
        self.norm_op = nn.BatchNorm2d

        assert out_channels % len(kernel_sizes) == 0, f"Got out channels: {out_channels}"

        for k in kernel_sizes:
            pad = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, k), padding=(0, pad), groups=in_channels, stride=(1, stride)),
                nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1), padding=(pad, 0), groups=in_channels, stride=(stride, 1)),
            )
            self.branches.append(branch)

        self.pw = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels * len(kernel_sizes)),
            nn.Conv2d(in_channels=in_channels * len(kernel_sizes), out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        output = []
        for branch in self.branches:
            output.append(
                branch(x)
            )
        return self.pw(torch.concat(output, dim=1))


class PXModule(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, **kwargs):
        super(PXModule, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [kwargs['kernel_size']]
        self.branches = nn.ModuleList()

        # Picl the norm op
        self.norm_op = nn.BatchNorm2d

        assert out_channels % len(kernel_sizes) == 0, f"Got out channels: {out_channels}"

        self.x_coef = nn.ParameterList([
            nn.Parameter(torch.ones(1, )) for _ in range(len(kernel_sizes))
        ])
        for k in kernel_sizes:
            pad = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, k), padding=(0, pad), groups=in_channels, stride=(1, stride)),
                nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1), padding=(pad, 0), groups=in_channels, stride=(stride, 1)),
            )
            self.branches.append(branch)

        self.pw = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features=in_channels * len(kernel_sizes)),
            nn.Conv2d(in_channels=in_channels * len(kernel_sizes), out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        output = []
        for i, branch in enumerate(self.branches):
            output.append(
                branch(x) * self.x_coef[i]
            )
        return self.pw(torch.concat(output, dim=1))


class ChannelGroupAttention(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        assert num_channels % num_groups == 0
        self.scale_factor = num_channels // num_groups

        # G is a learnable parameter
        self.G = nn.Parameter(torch.rand(num_groups, num_groups))
        """ Element of real A x B, A = B = num groups. Should esentially represent that channel group R i
            s similar to channel group C.
             __________
            |
            |
            |
            |
        """

    def forward(self, x):
        # Expand G to build C, which is the expanded version of G with each
        # element of G being repeated num_channels % num_groups times
        with torch.no_grad():
            C = self.G.repeat_interleave(self.scale_factor, dim=0).repeat_interleave(self.scale_factor, dim=1)

        """ Element of real A x B, A = B = num channels. C is not learnable
             __________
            |
            |
            |
            |
        """
        # print(C.shape, C.requires_grad, self.G.shape) [[channels, channels], False, [Groups, Groups]]

        num_batches, num_channels, height, width = x.shape
        # Flatten the spatial dimensions of the input tensor
        x = x.view(num_batches, num_channels, -1)  # (B, C, H x W)
        # Now, transpose x to have dimensions (num_channels, -1, num_channels)
        x = x.transpose(1, 2)
        # Perform the matrix multiplication
        x = torch.matmul(x, C)
        """
        h x w = n
        [n x c][c x c] -> [n x c] ... x retains shape
        """
        # print(x.shape)
        # Finally, transpose back the output tensor to the original form
        x = x.transpose(1, 2)
        # Reshape x back to the original shape
        x = x.view(num_batches, num_channels, height, width)  # (b, c, h, w)
        return x


class SpatialGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=None, stride=1, kernel_size=1, dilation=1, padding=1,
                 **kwargs):
        super(SpatialGatedConv2d, self).__init__()

        self.conv_gate = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, stride=stride,
                                   dilation=dilation, padding=(kernel_size - 1) // 2)
        self.conv_values = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, stride=stride,
                                     dilation=dilation, padding=(kernel_size - 1) // 2)
        self.nonlinearity = nonlinearity

        # Create coordinate map as a constant parameter
        self.coord = None

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # Generate coordinate maps if not created yet.
        if self.coord is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
            grid_x = grid_x.to(x.device) / (width - 1)
            grid_y = grid_y.to(x.device) / (height - 1)
            coord_single = torch.concat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0).unsqueeze(0)
            coord = coord_single
            for _ in range(batch_size - 1):
                coord = torch.concat((coord, coord_single), dim=0)
            self.coord = coord
        # Concatenate coordinates with the input
        x_with_coords = torch.cat([x, self.coord], dim=1)

        gate = torch.sigmoid(self.conv_gate(x_with_coords))
        values = self.conv_values(x_with_coords)

        output = gate * values

        if self.nonlinearity is not None:
            output = self.nonlinearity(output)

        return output


class ReverseLinearBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 6, stride: int = 1):
        super().__init__()
        assert stride in [1, 2], "Stride needs to be 1 or 2."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.skip = stride == 1
        self.reducer = None

        if expansion > 1:
            self.operation = nn.Sequential(
                Conv(in_channels=in_channels, out_channels=in_channels * expansion, kernel_size=1),
                nn.BatchNorm2d(in_channels * expansion),
                nn.ReLU6(inplace=True),
                Conv(in_channels=in_channels * expansion,
                     out_channels=in_channels * expansion,
                     kernel_size=3,
                     stride=stride,
                     groups=in_channels * expansion
                     ),
                nn.BatchNorm2d(in_channels * expansion),
                nn.ReLU6(inplace=True),
                Conv(in_channels=in_channels * expansion, out_channels=out_channels, kernel_size=1)
            )
        else:
            self.operation = nn.Sequential(
                Conv(in_channels=in_channels, out_channels=in_channels * expansion, kernel_size=1, groups=in_channels),
                nn.BatchNorm2d(in_channels * expansion),
                nn.ReLU6(inplace=True),
                Conv(in_channels=in_channels * expansion, out_channels=out_channels, kernel_size=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip and self.reducer is None:
            self.reducer = XModule(
                in_channels=self.out_channels + self.in_channels,
                out_channels=self.out_channels,
                kernel_sizes=[(x.shape[1] if x.shape[1] % 2 != 0 else x.shape[1] - 1)]
            )

        if self.skip:
            return self.reducer(torch.concat((self.operation(x), x), dim=1))
        return self.operation(x)


class InstanceNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.norm = ModuleStateController.instance_norm_op()(num_features=num_features)

    def forward(self, x):
        return self.norm(x)


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.norm = ModuleStateController.batch_norm_op()(num_features=num_features)

    def forward(self, x):
        return self.norm(x)


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.transp_op = ModuleStateController.transp_op()(in_channels=in_channels, out_channels=out_channels,
                                                           kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.transp_op(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='auto',
                 groups: int = 1) -> None:
        super().__init__()
        padding = padding if isinstance(padding, int) else (kernel_size - 1) // 2
        self.conv = ModuleStateController.conv_op()(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    padding=padding,
                                                    stride=stride,
                                                    dilation=dilation,
                                                    kernel_size=kernel_size,
                                                    groups=groups
                                                    )

    def forward(self, x):
        return self.conv(x)


class AveragePool(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.pool = ModuleStateController.avg_pool_op()(kernel_size)

    def forward(self, x):
        return self.pool(x)


class PolyWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, order: List[int], stride: int = 1, conv_op='Conv', poly_mode='sum',
                 conv_args=None, **kwargs):
        super().__init__()
        if conv_args is None:
            conv_args = {}
        assert len(order) > 0, "Order must be a list of exponents with length > 0."
        assert poly_mode in ['sum', 'sumv2', 'concat']
        self.mode = poly_mode
        conv_op = my_import(conv_op)
        poly_block = PolyBlock if poly_mode in ['sum', 'concat'] else PolyBlockV2
        self.branches = nn.ModuleList([
            poly_block(in_channels, 
                       out_channels if self.mode in ['sum', 'sumv2'] else in_channels, o, stride, conv_op=conv_op, **conv_args) for o in order
        ])
        if self.mode == "concat":
            self.pointwise = nn.Conv2d(in_channels*len(order), out_channels, kernel_size=1)

    def forward(self, x):
        out = None
        for mod in self.branches:
            if out is None:
                out = mod(x)
                if self.mode == "concat":
                    out = [out]
            else:
                if self.mode in ["sum", "sumv2"]:
                    out = torch.add(out, mod(x))
                elif self.mode == "concat":
                    out.append(mod(x))
        if self.mode in ["sum", "sumv2"]:
            return out
        out = torch.concat(out, dim=1)
        return self.pointwise(out)
        


class MultiRoute(nn.Module):

    def __init__(self, in_channels, out_channels, routes: list, stride=1, conv_op='Conv', bias=False, **conv_args):
        super().__init__()
        assert isinstance(routes, list), "routes should be a list, where each entry is the number of convolutions on a path."
        self.bias = bias
        if bias:
            self.shifts = nn.ParameterList([nn.Parameter(torch.zeros(1, out_channels, 1, 1)) for _ in range(len(routes))])
        conv_op = my_import(conv_op)
        self.branches = nn.ModuleList()
        for route in routes:
            assert route > 0, "All route length must be greater than 0."
            steps = []
            for i in range(route):
                steps.append(conv_op(in_channels if i == 0 else out_channels,
                                     out_channels,
                                     kernel_size=conv_args.pop('kernel_size', 3),
                                     stride=stride if i == 0 else 1,
                                     **conv_args))
                if i != route - 1:
                    steps.append(nn.ReLU())
            self.branches.append(nn.Sequential(*steps))

    def forward(self, x):
        out = None
        for i, branch in enumerate(self.branches):
            if out is None:
                out = branch(x)
            else:
                y = branch(x)
                if self.bias:
                    y = y + self.shifts[i]
                out = torch.add(out, y)
        return out


class PolyBlockV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, order: int, stride: int = 1, conv_op=Conv, **conv_args):
        super().__init__()
        self.order = order
        self.conv = conv_op(in_channels, out_channels, kernel_size=conv_args.pop('kernel_size', 3), stride=stride,
                            **conv_args)
        self.ch_maxpool = nn.MaxPool3d((in_channels, 1, 1), stride=(in_channels, 1, 1))
        self.shift = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.order == 1:
            return self.conv(x)
        x_pow = torch.pow(x, self.order)
        norm = self.ch_maxpool(torch.abs(x_pow))
        x_normed = torch.div(x_pow, norm + 1e-7)
        out = self.relu(self.conv(x_normed) + self.shift)
        return out


class PolyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, order: int, stride: int = 1, conv_op=Conv, **conv_args):
        super().__init__()
        self.order = order
        self.conv = conv_op(in_channels, out_channels, kernel_size=conv_args.pop('kernel_size', 3), stride=stride,
                            **conv_args)
        self.ch_maxpool = nn.MaxPool3d((in_channels, 1, 1), stride=(in_channels, 1, 1))

    def forward(self, x):
        if self.order == 1:
            return self.conv(x)
        std = torch.std(x)
        x = torch.clip(x, -3 * std, 3 * std)
        x_pow = torch.pow(x, self.order)
        norm = self.ch_maxpool(torch.abs(x_pow))
        x_normed = torch.div(x_pow, norm + 1e-7)
        out = self.conv(x_normed)
        return out


class MultiBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.y = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.b = nn.Parameter(torch.zeros((1, num_features, 1, 1)))

        self.y2 = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.b2 = nn.Parameter(torch.zeros((1, num_features, 1, 1)))

        self.y3 = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.b3 = nn.Parameter(torch.zeros((1, num_features, 1, 1)))

        self.register_buffer('running_mean', torch.zeros((1, num_features, 1, 1)))
        self.register_buffer('running_std', torch.ones((1, num_features, 1, 1)))

        self.adaptive_avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def _train_forward(self, x):
        # x_b = self.adaptive_avg(x).squeeze(dim=(2, 3))
        # x_b = self.linear(x_b)
        # done here, doesn't seem to converge well

        mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        var = ((x-mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(var+1e-7)
        x = (x - mean) / std
        # calculate
        x_b = self.adaptive_avg(x).squeeze(dim=(2, 3))
        x_b = self.linear(x_b)
        # calculated here, seems to do fine
        # do da shift
        x = x_b[:, 0:1].unsqueeze(2).unsqueeze(3)*(self.y * x + self.b) + x_b[:, 1:2].unsqueeze(2).unsqueeze(3)*(self.y2 * x + self.b2) + x_b[:, 2:].unsqueeze(2).unsqueeze(3)*(self.y3 * x + self.b3)
        # update stats
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
        return x

    def _eval_forward(self, x):
        x = (x - self.running_mean) / self.running_std
        x_b = self.adaptive_avg(x).squeeze(dim=(2, 3))
        x_b = self.linear(x_b)
        # do da shift
        x = x_b[:, 0:1].unsqueeze(2).unsqueeze(3)*(self.y * x + self.b) + x_b[:, 1:2].unsqueeze(2).unsqueeze(3)*(self.y2 * x + self.b2) + x_b[:, 2:].unsqueeze(2).unsqueeze(3)*(self.y3 * x + self.b3)
        return x

    def forward(self, x):
        if self.training:
            return self._train_forward(x)
        return self._eval_forward(x)


class SpacialConv2d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.coords = None

        assert len(args) == 0, "use named arguments only"
        kwargs["in_channels"] += 2
        # grouped followed by pointwise
        self.conv = nn.Conv2d(**kwargs)

    def get_coords(self, im_shape, device="cpu"):
        x_coords = torch.zeros(im_shape[2:], device=device)
        xs = torch.arange(im_shape[2], device=device)

        x_coords = x_coords + xs
        y_coords = x_coords.permute(1, 0)

        # y, x
        coords = torch.stack([2*x_coords/torch.max(x_coords)-1, 2*y_coords/torch.max(y_coords)-1])
        coords.requires_grad = False
        # batch
        # coords = torch.stack([coords]*im_shape[0])

        return coords

    def build_spacial_tensor(self, x):
        new_shape = list(x.shape)
        if self.coords is None:
            # only needed to instantiate once
            self.coords = self.get_coords(x.shape, device=x.device)
            # self.coords.requires_grad = False

        # if new_shape[1] > 1:
        #     new_shape[1] **= 3
        # else:
        new_shape[1] += 2

        # for mixing
        new_tensor = torch.zeros(new_shape, device=x.device)
        new_tensor[:, 0: x.shape[1], :, :] = x
        new_tensor[:, x.shape[1]:, :, :] = torch.stack([self.coords]*x.shape[0])
        return new_tensor

    def forward(self, x):
        x = self.build_spacial_tensor(x)
        # return x
        return self.conv(x)

import torch.fft as fft


class ScaleULayer(nn.Module):

    def __init__(self, in_channels, skipped_count = 2) -> None:
        super().__init__()
        self.b_scaling = nn.Parameter(torch.zeros(in_channels))  # Main features are b
        self.skipped_scaling = nn.ParameterList()
        for _ in range(skipped_count):
            self.skipped_scaling.append(nn.Parameter(torch.zeros(1)))            # skipped features from other decoder are s

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
            crow - threshold : crow + threshold,
            ccol - threshold : ccol + threshold,
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
            skipped_connections[i] = self.Fourier_filter(skipped_connections[i], 1, thresholds[i])  # scale the skip connection from encoder

        x = torch.einsum("bchw,c->bchw", x, b_s)         # scale the main features
        x = torch.cat([x, *skipped_connections], dim=1)
        return x
