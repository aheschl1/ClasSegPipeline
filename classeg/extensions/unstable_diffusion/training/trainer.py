import pdb
import sys
from typing import Tuple

import albumentations as A
import torch
import torch.nn as nn
from overrides import override
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from classeg.extensions.unstable_diffusion.forward_diffusers.scheduler import StepScheduler
from classeg.extensions.unstable_diffusion.inference.inferer import UnstableDiffusionInferer
from classeg.extensions.unstable_diffusion.model.unstable_diffusion import UnstableDiffusion
from classeg.training.trainer import Trainer, log
from classeg.extensions.unstable_diffusion.utils.utils import (
    get_forward_diffuser_from_config,
)
import torch.nn.functional as F


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class UnstableDiffusionTrainer(Trainer):
    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, world_size: int = 1, cache: bool = False):
        """
        Trainer class for training and checkpointing of networks.
        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param resume_training: None if we should train from scratch, otherwise the model weights that should be used.
        """
        super().__init__(dataset_name, fold, model_path, gpu_id, unique_folder_name, config_name, resume,
                         cache, world_size)
        self.dicriminator = None
        self.timesteps = self.config["max_timestep"]
        self.forward_diffuser = get_forward_diffuser_from_config(self.config)
        self.diffusion_schedule = StepScheduler(self.forward_diffuser, step_size=10, epochs_per_step=5)

        if resume and gpu_id in [0, "cpu"]:
            self.diffusion_schedule.load_state(torch.load(f"{self.output_dir}/latest.pth")["diffusion_schedule"])

        # self._instantiate_inferer(self.dataset_name, fold, unique_folder_name)
        self.infer_every: int = 10
        self.recon_loss, self.gan_loss = self.loss
        self.g_optim, self.d_optim = self.optim
        self.recon_weight = self.config.get("recon_weight", 0.5)
        self.gan_weight = self.config.get("gan_weight", 0.5)
        del self.optim, self.loss

    @override
    def get_augmentations(self) -> Tuple[A.Compose, A.Compose]:
        import cv2
        resize_image = A.Resize(*self.config.get("target_size", [512, 512]), interpolation=cv2.INTER_LINEAR)
        resize_mask = A.Resize(*self.config.get("target_size", [512, 512]), interpolation=cv2.INTER_NEAREST)

        def my_resize(image=None, mask=None, **kwargs):
            if mask is not None:
                return resize_mask(image=mask)["image"]
            if image is not None:
                return resize_image(image=image)["image"]

        train_transforms = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomCrop(width=512, height=512, p=1),
                A.Lambda(image=my_resize, mask=my_resize, p=1)
            ],
            is_check_shapes=False
        )
        val_transforms = A.Compose(
            [
                A.RandomCrop(width=512, height=512, p=1),
                A.Lambda(image=my_resize, mask=my_resize, p=1),
                A.ToFloat()
            ],
            is_check_shapes=False
        )
        return train_transforms, val_transforms

    def _instantiate_inferer(self, dataset_name, fold, result_folder):
        self._inferer = UnstableDiffusionInferer(dataset_name, fold, result_folder, "latest", None)

    def _save_model_weights(self, save_name: str) -> None:
        """
        Save the weights of the model, only if the current device is 0.

        :param save_name: The name of the checkpoint to save.
        """
        if self.device in [0, "cpu"]:
            checkpoint = {}
            path = f"{self.output_dir}/{save_name}.pth"
            if self.world_size > 1:
                checkpoint["weights"] = self.model.module.state_dict()
            else:
                checkpoint["weights"] = self.model.state_dict()
            checkpoint["optim"] = self.optim.state_dict()
            checkpoint["current_epoch"] = self._current_epoch
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
            checkpoint["diffusion_schedule"] = self.diffusion_schedule.get_state_dict()
            torch.save(checkpoint, path)

    @override
    def train_single_epoch(self, epoch) -> float:
        """
        The training of each epoch is done here.
        :return: The mean loss of the epoch.

        optimizer: self.optim
        loss: self.loss
        logger: self.log_helper
        model: self.model
        """
        self.dicriminator.train()
        running_loss = 0.0
        total_items = 0
        log_image = epoch % 10 == 0
        # ForkedPdb().set_trace()
        for images, segmentations, _ in tqdm(self.train_dataloader):
            self.g_optim.zero_grad()
            if log_image:
                self.log_helper.log_augmented_image(images[0], segmentations[0])
            images = images.to(self.device, non_blocking=True)
            segmentations = segmentations.to(self.device)

            im_noise, seg_noise, images, segmentations, t = self.forward_diffuser(images, segmentations)
            # do prediction and calculate loss
            predicted_noise_im, predicted_noise_seg = self.model(images, segmentations, t)
            gen_loss = self.recon_weight * self.recon_loss(predicted_noise_im, im_noise) + self.recon_weight * self.recon_loss(
                predicted_noise_seg, seg_noise)
            dis_loss = 0.0
            if self.gan_weight > 0:
                # convert x_t to x_{t-1} and descriminate the goods
                predicted_im, predicted_seg = self.forward_diffuser.inference_call(
                    images,
                    segmentations,
                    predicted_noise_im,
                    predicted_noise_seg, t, clamp=False, training_time=True
                )
                images, segmentations = self.forward_diffuser.inference_call(
                    images,
                    segmentations,
                    im_noise,
                    seg_noise, t, clamp=False, training_time=True
                )
                # Pass both im and seg together
                predicted_concat = torch.cat([predicted_im, predicted_seg], dim=1)
                real_concat = torch.cat([images, segmentations], dim=1)
                t -= 1
                # Labels for the discriminator
                fake_label = torch.zeros((images.shape[0],)).to(self.device)
                real_label = torch.ones((images.shape[0],)).to(self.device)
                # Fool the discriminator
                gen_loss += self.gan_weight * self.gan_loss(self.dicriminator(predicted_concat, t).squeeze(), real_label)
                # Train discriminator
                self.d_optim.zero_grad()
                real_loss = self.gan_loss(self.dicriminator(real_concat, t).squeeze(), real_label)
                fake_loss = self.gan_loss(self.dicriminator(predicted_concat.detach(), t).squeeze(), fake_label)
                # calculate the loss
                dis_loss = (real_loss + fake_loss) / 2
                dis_loss.backward()

            # update model
            gen_loss.backward()
            self.g_optim.step()
            self.d_optim.step()
            # gather data
            running_loss += (gen_loss.item()+dis_loss.item()) * images.shape[0]
            total_items += images.shape[0]

        return running_loss / total_items

    # noinspection PyTypeChecker
    @override
    def eval_single_epoch(self, epoch) -> float:
        """
        Runs evaluation for a single epoch.
        :return: The mean loss and mean accuracy respectively.

        optimizer: self.optim
        loss: self.loss
        logger: self.log_helper
        model: self.model
        """
        running_loss = 0.0
        total_items = 0
        # total_divergence = 0
        self.dicriminator.eval()
        for images, segmentations, _ in tqdm(self.val_dataloader):
            images = images.to(self.device, non_blocking=True)
            segmentations = segmentations.to(self.device, non_blocking=True)

            noise_im, noise_seg, images, segmentations, t = self.forward_diffuser(images, segmentations)

            predicted_noise_im, predicted_noise_seg = self.model(images, segmentations, t)
            gen_loss = self.recon_weight * self.recon_loss(predicted_noise_im, noise_im) + self.recon_weight * self.recon_loss(
                predicted_noise_seg, noise_seg)
            # convert x_t to x_{t-1} and descriminate the goods
            dis_loss = 0.0
            if self.gan_weight > 0:
                # convert x_t to x_{t-1} and descriminate the goods
                predicted_im, predicted_seg = self.forward_diffuser.inference_call(
                    images,
                    segmentations,
                    predicted_noise_im,
                    predicted_noise_seg, t, clamp=False, training_time=True
                )
                images, segmentations = self.forward_diffuser.inference_call(
                    images,
                    segmentations,
                    noise_im,
                    noise_seg, t, clamp=False, training_time=True
                )
                # Pass both im and seg together
                predicted_concat = torch.cat([predicted_im, predicted_seg], dim=1)
                real_concat = torch.cat([images, segmentations], dim=1)
                t -= 1
                # Labels for the discriminator
                fake_label = torch.zeros((images.shape[0],)).to(self.device)
                real_label = torch.ones((images.shape[0],)).to(self.device)
                # Fool the discriminator
                gen_loss += self.gan_weight * self.gan_loss(self.dicriminator(predicted_concat, t).squeeze(), real_label)
                # Train discriminator
                real_loss = self.gan_loss(self.dicriminator(real_concat, t).squeeze(), real_label)
                fake_loss = self.gan_loss(self.dicriminator(predicted_concat.detach(), t).squeeze(), fake_label)
                # calculate the loss
                dis_loss = (real_loss + fake_loss) / 2

            # gather data
            running_loss += (dis_loss.item() + gen_loss.item()) * images.shape[0]
            total_items += images.shape[0]

        # self.log_helper.log_scalar("Metrics/seg_divergence", total_divergence / len(self.val_dataloader), epoch)
        return running_loss / total_items

    @override
    def post_epoch(self, epoch: int) -> None:
        self.diffusion_schedule.step()
        return
        if epoch % self.infer_every == 0 and self.device == 0:
            print("Running inference to log")
            result_im, result_seg = self._inferer.infer()
            self.log_helper.log_image_infered(result_im.transpose(2, 0, 1), epoch, mask=result_seg.transpose(2, 0, 1))

    @override
    def get_model(self, path) -> nn.Module:
        model = UnstableDiffusion(**self.config["model_args"])
        self.dicriminator = model.get_discriminator()
        return model.to(self.device)

    def get_lr_scheduler(self):
        scheduler = StepLR(self.optim, step_size=100, gamma=0.9)
        if self.device == 0:
            log(f"Scheduler being used is {scheduler}")
        return scheduler

    def get_optim(self) -> torch.optim:
        """
        Instantiates and returns the optimizer.
        :return: Optimizer object.
        """
        from torch.optim import Adam

        g_optim = Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get('weight_decay', 0)
            # momentum=self.config.get('momentum', 0)
        )
        d_optim = Adam(
            self.dicriminator.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get('weight_decay', 0)
            # momentum=self.config.get('momentum', 0)
        )

        if self.device == 0:
            log(f"Optim being used is {g_optim}")
        return g_optim, d_optim

    class MSEWithKLDivergenceLoss(nn.Module):
        def __init__(self, kl_weight=0.1):
            super().__init__()
            self.kl_weight = kl_weight
            self.mse_loss = nn.MSELoss()
            self.kl_div_loss = nn.KLDivLoss()

        def forward(self, predicted, target):
            mse_loss = self.mse_loss(predicted, target)
            kl_div_loss = self.kl_div_loss(predicted, target)
            return mse_loss + self.kl_weight * kl_div_loss

    @override
    def get_loss(self) -> nn.Module:
        """
        Build the criterion object.
        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.MSELoss()")
        return (nn.MSELoss(), nn.BCEWithLogitsLoss())
