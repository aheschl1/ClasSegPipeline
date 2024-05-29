import sys

# Adds the source to path for imports and stuff
sys.path.append("/home/andrew.heschl/Documents/diffusion")
sys.path.append("/home/andrewheschl/PycharmProjects/diffusion")
sys.path.append("/home/student/andrew/Documents/diffusion")
sys.path.append("/home/student/andrewheschl/Documents/diffusion")
sys.path.append("/home/mauricio.murillo/diffusion")
sys.path.append("/Users/mauriciomurillogonzales/Documents/VisionResearchLab/diffusion")
from tqdm import tqdm
import numpy as np
import os
import shutil
import click
import torch
import torch.nn as nn
import glob
import torchvision
import matplotlib.pyplot as plt
from pipe.utils.constants import *
from pipe.utils.utils import get_dataset_name_from_id, get_forward_diffuser_from_config
from pipe.utils.utils import read_json
from torchvision.utils import make_grid
import cv2
import random
from torchvision import transforms


class Inferer:
    def __init__(
            self,
            dataset_id: str,
            fold: int,
            result_folder: str,
            weights: str,
    ):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param result_folder: The folder with the trained weights and config.
        :param weights: The name of the weights to load.
        """
        self.model = None
        self.dataset_name = get_dataset_name_from_id(dataset_id)
        self.fold = fold
        self.lookup_root = (
            f"{RESULTS_ROOT}/{self.dataset_name}/fold_{fold}/{result_folder}"
        )
        self.config = read_json(f"{self.lookup_root}/config.json")
        self.forward_diffuser = get_forward_diffuser_from_config(self.config)
        self.weights = weights
        self.device = "cuda"
        assert os.path.exists(self.lookup_root)
        assert torch.cuda.is_available(), "No gpu available."
        self.timesteps = self.config["max_timestep"]

    def _get_model(self) -> nn.Module:
        """
        Loads the model and weights.
        :return:
        """
        map_location = {"cuda:0": self.device}
        model = torch.load(
            f"{self.lookup_root}/{self.weights}.pth", map_location=map_location
        )
        return model

    def infer(self, grid_size: int = 1, save_process: bool = False):
        save_path = f'{self.lookup_root}/inference'
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        self.model = self._get_model()
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

                noise_prediction_im, noise_prediciton_seg = self.model(xt_im, xt_seg, time_tensor)
                xt_im, xt_seg = self.forward_diffuser.inference_call(
                    xt_im, xt_seg, noise_prediction_im, noise_prediciton_seg, t, clamp=False
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

    def process_image(self, image_path, scale=False):
        ext = image_path[-3:]
        print(ext)
        if ext == "npy":
            img = np.load(image_path)
            img = img.transpose((1,2,0))
        else:
            img = plt.imread(image_path)
            if scale:
                img = (img - img.min())/img.max()
                img = (img*2)-1
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        img = transforms.Compose(
            [
                transforms.FiveCrop((512, 512)),
                transforms.Lambda(lambda x: random.choice(x)),
                transforms.Resize(self.config.get("target_size", [512, 512]), antialias=False),
            ]
        )(img)
        img.unsqueeze(dim=0)
        return img

    def conditioned(self, image_path, mode):
        save_path = f"{self.lookup_root}/{mode}"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)

        self.model = self._get_model()
        self.model.eval()
        x_given = self.process_image(image_path, scale=mode=="seg")
        with torch.no_grad():
            if mode == "seg":
                xt_seg = torch.randn((1,self.config["model_args"]["seg_channels"],*self.config["target_size"],))
                xt_im = x_given
            elif mode == "gen":
                xt_im = torch.randn((1,self.config["model_args"]["im_channels"],*self.config["target_size"],))
                xt_seg = x_given
                
            print("Image attributes")
            print(xt_im.shape)
            print(xt_im.min(), xt_im.max())
            print("Mask attributes")
            print(xt_seg.shape)
            print(xt_seg.min(), xt_seg.max())

            noise_im = None
            noise_seg = None
            
            for t in tqdm(range(self.timesteps - 1, -1, -1), desc="running conditioned inference"):
                if mode == "seg":
                    noise_im,noise_seg, xt_im, _, _ = self.forward_diffuser(x_given, x_given, torch.tensor([t] * x_given.shape[0]).long(), noise_im, noise_seg)
                if mode == "gen":
                    noise_im,noise_seg,_, xt_seg, _ = self.forward_diffuser(x_given, x_given, torch.tensor([t] * x_given.shape[0]).long(), noise_im, noise_seg)
                xt_im = xt_im.to(self.device)
                xt_seg = xt_seg.to(self.device)
                time_tensor = (torch.ones(xt_im.shape[0]) * t).to(xt_im.device).long()

                noise_prediction_im, noise_prediciton_seg = self.model(xt_im, xt_seg, time_tensor)
                xt_im, xt_seg = self.forward_diffuser.inference_call(
                    xt_im, xt_seg, noise_prediction_im, noise_prediciton_seg, t, clamp=False
                )
                if t == 0:
                    xt_sa = xt_im.cpu()[0].permute(1, 2, 0).numpy()
                    xt_sa -= xt_sa.min()
                    xt_sa *= 255 / xt_sa.max()
                    xt_sa = xt_sa.astype(np.uint8)
                    plt.imsave(f"{save_path}/x0_{t}_im.png", xt_sa)
                    xt_sa = xt_seg.cpu()[0].permute(1, 2, 0).squeeze().numpy()
                    xt_sa -= xt_sa.min()
                    xt_sa *= 255 / xt_sa.max()
                    xt_sa = xt_sa.astype(np.uint8)
                    plt.imsave(f"{save_path}/x0_{t}_seg.png", xt_sa)
        xt_im = xt_im.cpu()[0].permute(1, 2, 0).numpy()
        xt_seg = xt_seg.cpu()[0].permute(1, 2, 0).numpy()
        return xt_im, xt_seg



@click.command()
@click.option("-dataset_id", "-d", required=True)  # 10
@click.option("-fold", "-f", required=True, type=int)  # 0
@click.option("-name", "-n", required=True)
@click.option("-weights", "-w", default="best")
@click.option("-grid_size", "-g", type=int, default=1)
@click.option("-segment", "-s", type=str, default=None)
@click.option("-generate", "-i", type=str, default=None)
@click.option("--save_process", "--s", help="Should all the inference process be saved", is_flag=True, type=bool)
def main(dataset_id: str, fold: int, name: str, weights: str, grid_size, segment, generate,
         save_process) -> None:
    inferer = Inferer(dataset_id, fold, name, weights)
    if segment is not None:
        inferer.conditioned(segment, "seg")
    elif generate is not None:
        inferer.conditioned(generate, "gen")
    else:
        print("normal inference")
        inferer.infer(grid_size, save_process)
    print("Completed inference!")


if __name__ == "__main__":
    main()
