import os
import shutil
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

from classeg.dataloading.datapoint import Datapoint
from classeg.extensions.unstable_diffusion.utils.utils import get_forward_diffuser_from_config
from classeg.inference.inferer import Inferer
from classeg.utils.utils import read_json
from classeg.utils.constants import RESULTS_ROOT
from classeg.extensions.unstable_diffusion.model.unstable_diffusion import UnstableDiffusion

class UnstableDiffusionInferer(Inferer):
    def __init__(self,
                 dataset_id: str,
                 fold: int,
                 name: str,
                 weights: str,
                 input_root: str,
                 late_model_instantiation=True,
                 **kwargs):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_id, fold, name, weights, input_root, late_model_instantiation=late_model_instantiation)
        self.forward_diffuser = get_forward_diffuser_from_config(self.config)
        self.timesteps = self.config["max_timestep"]
        self.kwargs = kwargs
        
        # self.model_json = read_json(f"{self.lookup_root}/model.json")

    def get_augmentations(self):
        ...

    def _get_model(self):
        """
        Loads the model and weights.
        :return:
        """
        if self.model is not None:
            return self.model
        
        model = UnstableDiffusion(**self.config["model_args"])
        return model.to(self.device)

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
        save_path = f'{self.lookup_root}/inference'
        self.model = self._get_model().cpu()
        checkpoint = torch.load(
            f"{self.lookup_root}/{self.weights}.pth"
        )["weights"]
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        return save_path

    def infer(self):
        grid_size = int(self.kwargs.get("g", 1))
        grid_size = int(self.kwargs.get("grid_size", grid_size))

        save_process = self.kwargs.get("save_process", False) in [True, '1', 1, 't', 'T']

        save_path = self.pre_infer()
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        self.model.eval()
        with torch.no_grad():
            xt_im = torch.randn(
                (
                    grid_size ** 2,
                    self.config["model_args"]["im_channels"],
                    *self.config["target_size"],
                )
            )
            xt_seg = torch.randn(
                (
                    grid_size ** 2,
                    self.config["model_args"]["seg_channels"],
                    *self.config["target_size"],
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
                grid_im = make_grid(xt_im, nrow=grid_size)
                if save_process or t == 0:
                    grid_im = grid_im.cpu().permute(1, 2, 0).numpy()
                    grid_im -= grid_im.min()
                    grid_im *= 255 / grid_im.max()
                    grid_im = grid_im.astype(np.uint8)
                    plt.imsave(f"{save_path}/x0_{t}_im.png", grid_im)
                grid_seg = make_grid(xt_seg, nrow=grid_size)
                if save_process or t == 0:
                    grid_seg = grid_seg.cpu().permute(1, 2, 0).numpy()
                    grid_seg -= grid_seg.min()
                    grid_seg *= 255 / grid_seg.max()
                    grid_seg = grid_seg.astype(np.uint8)
                    plt.imsave(f"{save_path}/x0_{t}_seg.png", grid_seg)
        xt_im = xt_im.cpu()[0].permute(1, 2, 0).numpy()
        xt_seg = xt_seg.cpu()[0].permute(1, 2, 0).numpy()

        return xt_im, xt_seg

    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        ...
