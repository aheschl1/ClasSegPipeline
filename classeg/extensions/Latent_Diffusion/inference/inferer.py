import os
import shutil
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from PIL import Image
from classeg.extensions.Latent_Diffusion.model.autoencoder.autoencoder import VQModel

from classeg.dataloading.datapoint import Datapoint
from classeg.extensions.Latent_Diffusion.utils.utils import get_forward_diffuser_from_config, get_autoencoder_from_config
from classeg.extensions.Latent_Diffusion.model.latent_diffusion import LatentDiffusion
from classeg.inference.inferer import Inferer
from classeg.utils.utils import read_json
from classeg.utils.constants import RESULTS_ROOT

class LatentDiffusionInferer(Inferer):
    def __init__(self,
                 dataset_id: str,
                 fold: int,
                 name: str,
                 weights: str,
                 input_root: str,
                 late_model_instantiation=True,
                 model = None,
                 ae = None,
                 **kwargs):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_id, fold, name, weights, input_root, late_model_instantiation=late_model_instantiation)
        self.forward_diffuser = get_forward_diffuser_from_config(self.config)
        self.autoencoder = get_autoencoder_from_config(self.config, device=self.device) if ae is None else ae
        
        self.timesteps = self.config["max_timestep"]
        self.kwargs = kwargs
        self.model = model
        try:
            self.model_json = read_json(f"{self.lookup_root}/model.json")
        except:...

    def get_augmentations(self):
        ...

    def infer_single_sample(self, image: torch.Tensor, datapoint: Datapoint) -> None:
        """
        image: single sample batch which has gone through the augmentations

        handle the result in fields
        """
        ...

    def pre_infer(self) -> str:
        """
        Returns the output directory, and creates dataloader
        """
        folder_name = input("name the output folder")
        self.save_path = f"{self.lookup_root}/inference/{folder_name}"

        if not os.path.exists(f"{self.lookup_root}/inference"):
            os.mkdir(f"{self.lookup_root}/inference")
        self.model = self._get_model()
        return self.save_path

    def infer(self):
        self.pre_infer()
        to_folder = self.kwargs.get("to_folder", False) in [True, '1', 1, 't', 'T']
        if to_folder:
            self.infer_folder()
        else:
            self.infer_grid()

    def infer_folder(self):
        num_samples = int(self.kwargs.get("s", 100))

        self.model.eval()
        with torch.no_grad():
            xt_im, xt_seg = self.progressive_denoise(num_samples)
            os.mkdir(f'{self.save_path}/Images')
            os.mkdir(f'{self.save_path}/Masks')
            for i in range(xt_im.shape[0]):
                self.save_tensor(f'{self.save_path}/Images/x0_{i}', xt_im[i])
                self.save_tensor(f'{self.save_path}/Masks/x0_{i}', xt_seg[i])
        return xt_im, xt_seg
    
    def infer_grid(self):
        grid_size = int(self.kwargs.get("g", 1))
        grid_size = int(self.kwargs.get("grid_size", grid_size))
        
        self.model.eval()
        with torch.no_grad():
            xt_im, xt_seg = self.progressive_denoise(grid_size**2)
            grid_im = make_grid(xt_im, nrow=grid_size)
            grid_seg = make_grid(xt_seg, nrow=grid_size)
            self.save_tensor(f"{self.save_path}/Images.jpg", grid_im)
            self.save_tensor(f"{self.save_path}/Masks.jpg", grid_seg)
        return xt_im, xt_seg


    def progressive_denoise(self, num_samples):
        xt_im = torch.randn(
            (
                num_samples,
                *self.config["latent_size"],
            )
        )
        xt_seg = torch.randn(
            (
                num_samples,
                *self.config["latent_size"],
            )
        )
        xt_im = xt_im.to(self.device)
        xt_seg = xt_seg.to(self.device)
        for t in tqdm(range(self.timesteps - 1, -1, -1), desc="running inference"):
            time_tensor = (torch.ones(xt_im.shape[0]) * t).to(xt_im.device).long()

            noise_prediction_im, noise_prediciton_seg = self.model(
                xt_im, xt_seg, time_tensor
            )
            xt_im, xt_seg = self.forward_diffuser.inference_call(
                xt_im,
                xt_seg,
                noise_prediction_im,
                noise_prediciton_seg,
                t,
                clamp=False,
            )
        xt_im = self.autoencoder.decode(xt_im)
        xt_seg = self.autoencoder.decode(xt_seg)
        return xt_im, xt_seg

    
    def save_tensor(self, path, x):
        x = x.detach().cpu()
        x -= x.min()
        x *= 255 / x.max()
        x = x.permute(1,2,0).numpy()
        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        if not x.mode == "RGB":
          x = x.convert("RGB")
        x.save(path)
        return

    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        ...

    def _get_model(self):
        """
        Loads the model and weights.
        :return:
        """
        if self.model is not None:
            return self.model
        checkpoint = torch.load(
            f"{self.lookup_root}/{self.weights}.pth"
        )
        in_channels = self.config.get('latent_size')[0]
        layer_depth = self.config.get('layer_depth')
        channels = self.config.get('channels')
        time_emb_dim = self.config.get('time_emb_dim')
        apply_scale_u = self.config.get('apply_scale_u')
        apply_zero_conv = self.config.get('apply_zero_conv')
        shared_encoder = self.config.get('shared_encoder')

        model = LatentDiffusion( 
            in_channels,
            in_channels,
            layer_depth,
            channels,
            time_emb_dim,
            shared_encoder,
            apply_zero_conv,
            apply_scale_u
        )
        model.load_state_dict(checkpoint["weights"])
        return model.to(self.device)
    
