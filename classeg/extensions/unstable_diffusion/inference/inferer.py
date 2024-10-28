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
from classeg.extensions.unstable_diffusion.model.concat_diffusion import ConcatDiffusion
import albumentations as A
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
                 infer_timesteps: int = 1000,
                 sr_timesteps = None,
                 training=False,
                 r="Inference",
                 **kwargs):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_id, fold, name, weights, input_root, late_model_instantiation=late_model_instantiation)
        self.timesteps = self.config["max_timestep"]
        self.infer_timesteps = int(infer_timesteps)
        self.training = training
        if sr_timesteps is None:
            self.sr_timesteps = self.infer_timesteps
        else:
            self.sr_timesteps = int(sr_timesteps)
        
        self.forward_diffuser = get_forward_diffuser_from_config(self.config, ddim=(self.infer_timesteps < 1000), timesteps=self.timesteps)
        self.kwargs = kwargs
        self.dataset_id = dataset_id
        self.name = name
        self.run_name = r

    def get_augmentations(self):
        import cv2
        print(self.config.get("target_size", [512, 512]))
        resize_image = A.Resize(*self.config.get("target_size", [512, 512]), interpolation=cv2.INTER_CUBIC)

        def my_resize(image=None, mask=None, **kwargs):
            if image is not None:
                return resize_image(image=image)["image"]

        val_transforms = A.Compose(
            [
                A.RandomCrop(width=512, height=512, p=1),
                A.Lambda(image=my_resize, p=1),
                A.ToFloat()
            ],
            is_check_shapes=False
        )
        return val_transforms

    def _get_model(self):
        """
        Loads the model and weights.
        :return:
        """
        if self.model is not None:
            return self.model
        
        mode = self.config["mode"]
        if mode == "concat":
            model = ConcatDiffusion(
                **self.config["model_args"]
            )    
        elif mode == "unstable":
            model = UnstableDiffusion(
                **self.config["model_args"],
                do_context_embedding=self.config.get("do_context_embedding", False)
            )
        else:
            raise ValueError("You must set mode to unstable or concat.")
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

    def infer(self, model=None, num_samples=None, embed_sample=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # To infer we need the number of samples to generate, and name of folder
        dataloader = None
        if embed_sample is None:
            dataloader = self.get_dataloader(batch_size=self.config['batch_size'])

        num_samples = num_samples if num_samples is not None else min(int(self.kwargs.get("s", 1000)), len(dataloader.dataset))
        run_name = self.run_name

        # Inference generates folders with the csv file
        save_path = f'{self.pre_infer(build_model=model is None)}/{run_name}'
        self.save_path = save_path
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        os.mkdir(save_path)
        os.mkdir(f'{save_path}/images')
        os.mkdir(f'{save_path}/masks')
        
        model = model if model is not None else self._get_model()

        model.eval()
        in_shape = list(self.config["target_size"])
        batch_size = self.config["batch_size"]

        
        case_num = 0
        xt_im, xt_seg = None, None
        with torch.no_grad():
            for _ in tqdm(range(0,int(np.ceil(num_samples/batch_size))), desc="Running Inference"):
                if ((num_samples - case_num) < batch_size):
                    batch_size = (num_samples - case_num)
                
                if dataloader is not None:
                    embed_sample,*_ = next(iter(dataloader))
                xt_im, xt_seg = self.progressive_denoise(batch_size, in_shape, model=model, embed_sample=embed_sample)
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

    def progressive_denoise(self, batch_size, in_shape, model=None, embed_sample=None):
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
        skip = self.timesteps // self.infer_timesteps
        seq = range(self.timesteps-1, -1, -skip)
        if embed_sample is not None:
            embed_sample = embed_sample.to(self.device)
        torch.save(embed_sample, "/home/student/andrewheschl/Documents/Diffusion_ClasSeg/embed_sample.pt")
        print("here")
        if embed_sample is not None and len(embed_sample.shape) > 2: # is it already embedded?
            # TODO this needs to be turned into an actual system
            embed_sample, recon = model.embed_image(embed_sample, recon=True)
            # save all recons and originals to disk
            torch.save(recon, "/home/student/andrewheschl/Documents/Diffusion_ClasSeg/recon.pt")
            # torch.save(embed_sample, "/home/student/andrewheschl/Documents/Diffusion_ClasSeg/embed_sample.pt")

            # torch.save(embed_sample, "/home/student/andrewheschl/Documents/Diffusion_ClasSeg/embed.pt")
        # randomly delete some features with dropout like behavior
        # embed_sample = torch.functional.F.dropout(embed_sample, p=self.context_dropout, training=True)

        for t in tqdm(seq, desc="Running Inference"):
            time_tensor = (torch.ones(xt_im.shape[0]) * t).to(xt_im.device).long()
            t_n = t - skip if t !=0 else -1
            if embed_sample is not None:
                noise_prediction_im, noise_prediciton_seg = model(
                    xt_im, xt_seg, time_tensor, embed_sample
                )
            else:
                noise_prediction_im, noise_prediciton_seg = model(
                    xt_im, xt_seg, time_tensor
                )
            xt_im, xt_seg = self.forward_diffuser.inference_call(
                xt_im,
                xt_seg,
                noise_prediction_im,
                noise_prediciton_seg,
                t,
                t_n,
            )            
        return xt_im, xt_seg

    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        if self.training:
            return 
        print("===============================super resolving with super_resolution_v3===============================")
        super_inferer = SuperResolutionInferer(self.dataset_id, self.fold, "super_resolution_v3", "latest", 
                                               self.save_path, output_name=f"{self.name}_{self.run_name}", infer_timesteps=self.sr_timesteps)
        super_inferer.infer()
        ...
        

def get_embed():
    mean_radius = 104
    dimensions = 128
    # sample from the n-sphere with radius mean_radius
    embed = torch.randn(1, dimensions)
    embed /= torch.norm(embed, dim=-1, keepdim=True)
    embed *= mean_radius
    print(embed.shape, torch.sqrt(torch.sum(embed**2)))

if __name__ == "__main__":
    inferer = UnstableDiffusionInferer(
        421,
        0,
        "embed_one_encoder",
        "best",
        None
    )
    embed = get_embed()
    inferer.infer(embed_sample=embed)