import importlib
import torchvision.models as models
from pipe.utils.constants import *


def my_import(class_name: str, dropout_package: str = "torch.nn"):
    """
    Returns a class based on a string name.
    :param class_name: The name of the object being searched for.
    :param dropout_package: The package to look into of the class isn't in the if/else statements
    :return: Any object defined in this long if/else sequence.
    """
    if class_name == "UnstableDiffusion":
        from pipe.models.src.diffusion_models.unstable_diffusion import UnstableDiffusion
        return UnstableDiffusion
    if class_name == "ExtraUnstableDiffusion":
        from pipe.models.src.diffusion_models.extra_unstable_diffusion import ExtraUnstableDiffusion
        return ExtraUnstableDiffusion
    if class_name == "PIMPNet":
        from pipe.models.src.diffusion_models.pimp_net import PIMPNet
        return PIMPNet
    if class_name == "mobilenet":
        from torchvision.models import mobilenet_v2
        return mobilenet_v2
    if class_name == "SpacialConv2d":
        from pipe.models.src.modules import SpacialConv2d
        return SpacialConv2d

    # Here checks backbones
    if class_name == ENB6_P:
        return models.efficientnet_b6
    elif class_name == ENV2_P:
        return models.efficientnet_v2_l
    elif class_name == ENB4_P:
        return models.efficientnet_b4
    elif class_name == ENB0_P:
        return models.efficientnet_b0
    elif class_name == ENB1_P:
        return models.efficientnet_b1
    # Here checks modules
    if class_name == "UpsamplingConv":
        from pipe.models.src.modules import UpsamplingConv
        return UpsamplingConv
    elif class_name == "ConvPixelShuffle":
        from pipe.models.src.modules import ConvPixelShuffle
        return ConvPixelShuffle
    elif class_name == "SelfAttention":
        from pipe.models.src.modules import SelfAttention
        return SelfAttention
    elif class_name == "SpatialAttentionModule":
        from pipe.models.src.modules import SpatialAttentionModule
        return SpatialAttentionModule
    elif class_name == "DepthWiseSeparableConv":
        from pipe.models.src.modules import DepthWiseSeparableConv
        return DepthWiseSeparableConv
    elif class_name == "XModule":
        from pipe.models.src.modules import XModule
        return XModule
    elif class_name == "PXModule":
        from pipe.models.src.modules import PXModule
        return PXModule
    elif class_name == "CBAM":
        from pipe.models.src.modules import CBAM
        return CBAM
    elif class_name == "CAM":
        from pipe.models.src.modules import CAM
        return CAM
    elif class_name == "LearnableCAM":
        from pipe.models.src.modules import LearnableCAM
        return LearnableCAM
    elif class_name == "InstanceNorm":
        from pipe.models.src.modules import InstanceNorm
        return InstanceNorm
    elif class_name == "BatchNorm":
        from pipe.models.src.modules import BatchNorm
        return BatchNorm
    elif class_name == "ConvTranspose":
        from pipe.models.src.modules import ConvTranspose
        return ConvTranspose
    elif class_name == "Conv":
        from pipe.models.src.modules import Conv
        return Conv
    elif class_name == "SpatialGatedConv2d":
        from pipe.models.src.modules import SpatialGatedConv2d
        return SpatialGatedConv2d
    elif class_name == "AveragePool":
        from pipe.models.src.modules import AveragePool
        return AveragePool
    elif class_name == "ReverseLinearBottleneck":
        from pipe.models.src.modules import ReverseLinearBottleneck
        return ReverseLinearBottleneck
    elif class_name == "MultiRoute":
        from pipe.models.src.modules import MultiRoute
        return MultiRoute
    elif class_name == "PolyBlockV2":
        from pipe.models.src.modules import PolyBlockV2
        return PolyBlockV2
    elif class_name == "PolyWrapper":
        from pipe.models.src.modules import PolyWrapper
        return PolyWrapper
    else:
        try:
            module = importlib.import_module(dropout_package)
            class_ = getattr(module, class_name)
            return class_
        except AttributeError:
            raise NotImplementedError(
                f"The requested module {class_name} has not been placed in my_import, and is "
                f"not in torch.nn."
            )


def make_zero_conv(channels, conv_op):
    zero_conv = conv_op(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    for p in zero_conv.parameters():
        p.detach().zero_()
    return zero_conv
