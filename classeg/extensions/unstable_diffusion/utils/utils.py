from classeg.extensions.unstable_diffusion.forward_diffusers.diffusers import Diffuser, LinearDiffuser, CosDiffuser, LinearDDIM, CosDDIM


def get_forward_diffuser_from_config(config, ddim=False, timesteps=None) -> Diffuser:
    min_beta = config.get("min_beta", 0.0001)
    max_beta = config.get("max_beta", 0.999)
    diffuser_mapping = {
        "linear": LinearDiffuser,
        "cos": CosDiffuser
    }
    diffuser_mapping_ddim = {
        "linear": LinearDDIM,
        "cos": CosDDIM
    }
    assert config.get("diffuser", "cos") in ["linear", "cos"], \
        f"{config['diffuser']} is not a supported diffuser."
    if not ddim:
        return diffuser_mapping[config["diffuser"]](config["max_timestep"] if timesteps is None else timesteps, min_beta, max_beta)
    else:
        return diffuser_mapping_ddim[config["diffuser"]](config["max_timestep"] if timesteps is None else timesteps, min_beta, max_beta)


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
