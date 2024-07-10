import os
import shutil
from typing import Tuple

import pandas as pd
import cv2

import numpy as np
import torch
from tqdm import tqdm

from classeg.dataloading.datapoint import Datapoint
from classeg.extensions.unstable_diffusion.utils.utils import get_forward_diffuser_from_config
from classeg.extensions.unstable_diffusion.forward_diffusers.diffusers import LinearDiffuser
from classeg.inference.inferer import Inferer
from classeg.utils.utils import read_json
from classeg.utils.constants import RESULTS_ROOT
from classeg.extensions.unstable_diffusion.model.unstable_diffusion import UnstableDiffusion
from classeg.extensions.unstable_diffusion.preprocessing.bitifier import bitmask_to_label
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

    def get_augmentations(self):
        ...

    def _get_model(self):
        """
        Loads the model and weights.
        :return:
        """
        if self.model is not None:
            return self.model
        
        model = UnstableDiffusion(
            **self.config["model_args"], 
            super_resolution=self.config.get("super_resolution", False)
        )
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
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model = self._get_model().cpu()
        checkpoint = torch.load(
            f"{self.lookup_root}/{self.weights}.pth"
        )["weights"]
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        return save_path

    def infer(self):
        # To infer we need the number of samples to generate, and name of folder
        num_samples = int(self.kwargs.get("s", 1000))
        run_name  = self.kwargs.get("r", "Inference")

        # Inference generates folders with the csv file
        save_path = f'{self.pre_infer()}/{run_name}'
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        os.mkdir(save_path)
        os.mkdir(f'{save_path}/images')
        os.mkdir(f'{save_path}/masks')
        entries = []

        self.model.eval()
        in_shape = list(self.config["target_size"])
        if self.config.get("super_resolution", False):
            for i in range(len(in_shape)):
                in_shape[i] = in_shape[i] * 2
        
        batch_size = int(self.config.get("infer_batch_size", 32))
        case_num = 0
        with torch.no_grad():
            for runs in tqdm(range(0,int(np.ceil(num_samples/batch_size))), desc="running_inferences"):
                if ((num_samples - case_num) < batch_size):
                    batch_size = (num_samples - case_num)
                
                xt_im, xt_seg = self.progressive_denoise(batch_size, in_shape)
                # Binarize the mask
                xt_seg[xt_seg < 0.5] = 0
                xt_seg[xt_seg > 0] = 1

                xt_im = xt_im.detach().cpu()
                xt_seg = xt_seg.detach().cpu()
                xt_im = xt_im - xt_im.min(dim=1).values.min(dim=1).values.min(dim=1).values.reshape(-1,1,1,1)
                xt_seg = xt_seg - xt_seg.min(dim=1).values.min(dim=1).values.min(dim=1).values.reshape(-1,1,1,1)
                xt_im = xt_im * (255 / xt_im.max(dim=1).values.max(dim=1).values.max(dim=1).values).reshape(-1,1,1,1)
                xt_seg = xt_seg * (255 / xt_seg.max(dim=1).values.max(dim=1).values.max(dim=1).values).reshape(-1,1,1,1)
                
                xt_im = xt_im.permute(0,2,3,1).numpy().astype(np.uint8)
                xt_seg = xt_seg.permute(0,2,3,1).numpy().astype(np.uint8)
                for i in range(batch_size):
                    cv2.imwrite(f'{save_path}/images/case_{case_num}.jpg', xt_im[i])
                    cv2.imwrite(f'{save_path}/masks/case_{case_num}.jpg', xt_seg[i])
        return xt_im, xt_seg

    def progressive_denoise(self, batch_size, in_shape):
        xt_im = torch.randn(
            (
                batch_size,
                self.config["model_args"]["im_channels"],
                *in_shape,
            )
        )
        xt_seg = torch.randn(
           (
               batch_size,
               self.config["model_args"]["seg_channels"],
               *in_shape,
           )
        )
        xt_im = xt_im.to(self.device)
        xt_seg = xt_seg.to(self.device)
        # self.timesteps = 1000
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
        return xt_im, xt_seg
    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        ...