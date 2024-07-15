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
from classeg.extensions.super_resolution.inference.inferer import SuperResolutionInferer
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
        self.dataset_id = dataset_id
        self.name = name

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

    def pre_infer(self, build_model=True) -> str:
        """
        Returns the output directory, and creates dataloader
        """
        save_path = f'{self.lookup_root}/inference'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if build_model:
            self.model = self._get_model().cpu()
            checkpoint = torch.load(
                f"{self.lookup_root}/{self.weights}.pth"
            )["weights"]
            self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
        return save_path

    def infer(self, model=None, num_samples=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # To infer we need the number of samples to generate, and name of folder
        num_samples = num_samples if num_samples is not None else int(self.kwargs.get("s", 1000))
        run_name  = self.kwargs.get("r", "Inference")

        # Inference generates folders with the csv file
        save_path = f'{self.pre_infer(build_model=model is None)}/{run_name}'
        self.save_path = save_path
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        os.mkdir(save_path)
        os.mkdir(f'{save_path}/images')
        os.mkdir(f'{save_path}/masks')
        entries = []
        model = model if model is not None else self.model

        model.eval()
        in_shape = list(self.config["target_size"])
        batch_size = self.config.get("infer_batch_size", self.config["batch_size"])
        case_num = 0
        xt_im, xt_seg = None, None
        with torch.no_grad():
            for _ in tqdm(range(0,int(np.ceil(num_samples/batch_size))), desc="Running Inference"):
                if ((num_samples - case_num) < batch_size):
                    batch_size = (num_samples - case_num)
                
                xt_im, xt_seg = self.progressive_denoise(batch_size, in_shape, model=model)
                # Binarize the mask
                xt_im = xt_im.detach().cpu().permute(0,2,3,1)
                xt_seg = xt_seg.detach().cpu().permute(0,2,3,1)
                for i in range(batch_size):
                    im = xt_im[i]
                    seg = xt_seg[i].round()

                    im -= torch.min(im)
                    im *= (255 / torch.max(im))
                    seg *= 255

                    cv2.imwrite(f'{save_path}/images/case_{case_num}.jpg', cv2.cvtColor(im.to(torch.uint8).numpy(), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f'{save_path}/masks/case_{case_num}.jpg', seg.to(torch.uint8).numpy())
                    case_num += 1
        self.post_infer()
        return xt_im, xt_seg

    def progressive_denoise(self, batch_size, in_shape, model=None):
        if model is None:
            model = self.model
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
            noise_prediction_im, noise_prediciton_seg = model(
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
        # print("===============================super resolving===============================")
        # super_inferer = SuperResolutionInferer(self.dataset_id, self.fold, "super_resolution_v2", "latest", self.save_path, output_name=self.name)
        # super_inferer.infer()
        ...
        