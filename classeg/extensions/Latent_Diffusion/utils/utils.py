from classeg.extensions.unstable_diffusion.forward_diffusers.diffusers import Diffuser, LinearDiffuser, CosDiffuser
import os
from classeg.utils.constants import AUTOENCODER
from omegaconf import OmegaConf
from classeg.extensions.Latent_Diffusion.model.autoencoder.autoencoder import VQModel
import torch

def get_forward_diffuser_from_config(config) -> Diffuser:
    min_beta = config.get("min_beta", 0.0001)
    max_beta = config.get("max_beta", 0.999)
    diffuser_mapping = {
        "linear": LinearDiffuser,
        "cos": CosDiffuser
    }
    assert config.get("diffuser", "cos") in ["linear", "cos"], \
        f"{config['diffuser']} is not a supported diffuser."
    return diffuser_mapping[config["diffuser"]](config["max_timestep"], min_beta, max_beta)


def get_autoencoder_from_config(config) -> VQModel:
    path = f'{AUTOENCODER}/{config.get("autoencoder", "vq-f8-n256")}'
    if not os.path.exists(path):
        raise ValueError #dont know what to raise
    config = OmegaConf.load(f'{path}/config.yaml')
    model = VQModel(**config.model.params)
    sd = torch.load(f'{path}/model.ckpt')["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


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
