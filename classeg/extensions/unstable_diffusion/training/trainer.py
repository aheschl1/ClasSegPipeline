import os
import pdb
import sys
from typing import Tuple, Any

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from overrides import override
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from classeg.extensions.unstable_diffusion.forward_diffusers.scheduler import StepScheduler, VoidScheduler
from classeg.extensions.unstable_diffusion.inference.inferer import UnstableDiffusionInferer
from classeg.extensions.unstable_diffusion.model.concat_diffusion import ConcatDiffusion
from classeg.extensions.unstable_diffusion.model.unstable_diffusion import UnstableDiffusion
from classeg.extensions.unstable_diffusion.utils.utils import (
    get_forward_diffuser_from_config,
)
from classeg.extensions.unstable_diffusion.training.covariance_loss import CovarianceLoss
from classeg.training.trainer import Trainer, log


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
        if gpu_id != "cpu" and gpu_id > 1:
            raise NotImplementedError("Finihsh DDP implmentation of the dicriminator")
        self.dicriminator = None
        self.dicriminator_lr_schedule = None
        super().__init__(dataset_name, fold, model_path, gpu_id, unique_folder_name, config_name, resume,
                         cache, world_size)
        self.timesteps = self.config["max_timestep"]
        self.forward_diffuser = get_forward_diffuser_from_config(self.config)
        if self.config.get("diffusion_scheduler", None) in ["linear", "l"]:
            self.diffusion_schedule = StepScheduler(self.forward_diffuser, step_size=10, epochs_per_step=5, initial_max=10)
        else:
            self.diffusion_schedule = VoidScheduler(self.forward_diffuser)
        self.optim, self.d_optim = self.optim

        self.dicriminator_lr_schedule = self.get_lr_scheduler(self.d_optim)
        if resume:
            state = torch.load(f"{self.output_dir}/latest.pth")

            self.diffusion_schedule.load_state(state["diffusion_schedule"])
            self.dicriminator_lr_schedule.load_state_dict(state['dicriminator_lr_schedule'])
            self.d_optim.load_state_dict(state['d_optim'])
            if self.dicriminator is not None:
                self.dicriminator.load_state_dict(state['discriminator'])

        self._instantiate_inferer(self.dataset_name, fold, unique_folder_name)
        self.infer_every: int = 15
        self.recon_loss, self.gan_loss = self.loss
        self.recon_weight = self.config.get("recon_weight", 1)
        self.gan_weight = self.config.get("gan_weight", 0.5)
        
        self.covariance_weight = self.config.get("covariance_weight", 0)
        self.covariance_loss = CovarianceLoss()

        self.do_context_embedding = self.config.get("do_context_embedding", False)
        self.context_recon_weight = self.config.get("context_recon_weight", 0.5)

        del self.loss

    def load_checkpoint(self, weights_name) -> None:
        """
        Loads network checkpoint onto the DDP model.
        :param weights_name: The name of the weights to load in the form of *result folder*/*weight name*.pth
        :return: None
        """
        assert os.path.exists(f"{self.output_dir}/{weights_name}.pth")
        checkpoint = torch.load(f"{self.output_dir}/{weights_name}.pth")
        # Because we are saving during the current epoch, we need to increment the epoch by 1, to resume at the next
        # one.
        self._current_epoch = checkpoint["current_epoch"] + 1
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint["weights"])
        else:
            self.model.load_state_dict(checkpoint["weights"])
        self.optim[0].load_state_dict(checkpoint["optim"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self._best_val_loss = checkpoint["best_val_loss"]

    @override
    def get_augmentations(self) -> Tuple[A.Compose, A.Compose]:
        import cv2
        resize_image = A.Resize(*self.config.get("target_size", [512, 512]), interpolation=cv2.INTER_CUBIC)
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
        self._inferer = UnstableDiffusionInferer(dataset_name, fold, result_folder, "latest", None, training=True)

    def get_extra_checkpoint_data(self) -> torch.Dict[str, Any] | None:
        return {
            "diffusion_schedule": self.diffusion_schedule.state_dict(),
            "dicriminator_lr_schedule": self.dicriminator_lr_schedule.state_dict(),
            "d_optim": self.d_optim.state_dict(),
            "discriminator": self.dicriminator.state_dict()
        }

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
        print(f"Max t sample is {self.diffusion_schedule.compute_max_at_step(self.diffusion_schedule._step)}")
        # ForkedPdb().set_trace()
        for images, segmentations, _ in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            if log_image:
                self.logger.log_augmented_image(images[0], mask=segmentations[0].squeeze().numpy())
            images = images.to(self.device, non_blocking=True)
            segmentations = segmentations.to(self.device)

            images_original = images

            im_noise, seg_noise, images, segmentations, t = self.forward_diffuser(images, segmentations)
            # image emebdding
            context_embedding = None
            context_recon = None
            if self.do_context_embedding:
                context_embedding, context_recon = self.model.embed_image(images_original)
                if log_image:
                    self.logger.log_image("Recon", context_recon[0])
            # do prediction and calculate loss

            predicted_noise_im, predicted_noise_seg = self.model(images, segmentations, t, context_embedding)
            gen_loss = self.recon_loss(torch.concat([predicted_noise_im, predicted_noise_seg], dim=1),
                                        torch.concat([im_noise, seg_noise], dim=1))
            if self.do_context_embedding:
                gen_loss += self.context_recon_weight * self.recon_loss(context_recon, images_original)
                gen_loss += self.covariance_weight * self.covariance_loss(context_embedding)

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
                gen_loss = gen_loss * self.recon_weight
                gen_loss += self.gan_weight * self.gan_loss(self.model.discriminate(self.dicriminator, predicted_concat, t).squeeze(),
                                                            real_label)
                # Train discriminator
                self.d_optim.zero_grad()
                real_loss = self.gan_loss(self.model.discriminate(self.dicriminator, real_concat, t).squeeze(), real_label)
                fake_loss = self.gan_loss(self.model.discriminate(self.dicriminator, predicted_concat.detach(), t).squeeze(), fake_label)
                # calculate the loss
                dis_loss += real_loss + fake_loss
                # dis_loss = dis_loss*self.gan_weight
                dis_loss.backward()

            # update model
            # gen_loss = gen_loss*self.recon_weight
            gen_loss.backward()

            self.optim.step()
            self.d_optim.step()
            # self.diffusion_schedule.step()
            # self.dicriminator_lr_schedule.step()
            # self.lr_scheduler.step()

            # gather data
            running_loss += (gen_loss + dis_loss).item() * images.shape[0]
            total_items += images.shape[0]

        return running_loss / total_items

    def log_discriminator_progress(self, epoch, predicted_real, predicted_fake):
        labels = torch.cat([torch.ones_like(predicted_real), torch.zeros_like(predicted_fake)], dim=0)
        predictions = torch.cat([predicted_real, predicted_fake], dim=0)
        self.logger.log_scalar(self.gan_loss(predictions, labels), "Metrics/DisriminatorLoss")

        predictions = torch.sigmoid(predictions)
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        predictions[predictions > 0.5] = 1
        predictions[predictions != 1] = 0
        correct = (predictions == labels).sum()
        total = labels.shape[0]

        self.logger.log_scalar(correct / total, "Metrics/DisriminatorAccuracy")

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

        all_discriminator_predictions_real = []
        all_discriminator_predictions_fake = []

        for images, segmentations, _ in tqdm(self.val_dataloader):
            images = images.to(self.device, non_blocking=True)
            segmentations = segmentations.to(self.device, non_blocking=True)

            images_original = images

            noise_im, noise_seg, images, segmentations, t = self.forward_diffuser(images, segmentations)

            # image emebdding
            context_embedding = None
            context_recon = None
            if self.do_context_embedding:
                context_embedding, context_recon = self.model.embed_image(images_original)

            predicted_noise_im, predicted_noise_seg = self.model(images, segmentations, t, context_embedding)
            gen_loss = self.recon_loss(torch.concat([predicted_noise_im, predicted_noise_seg], dim=1),
                            torch.concat([noise_im, noise_seg], dim=1))
            if self.do_context_embedding:
                gen_loss += self.context_recon_weight * self.recon_loss(context_recon, images_original)
                gen_loss += self.covariance_weight * self.covariance_loss(context_embedding)
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
                gen_loss += self.gan_loss(self.model.discriminate(self.dicriminator, predicted_concat, t).squeeze(), real_label)
                # Train discriminator
                predicted_real = self.model.discriminate(self.dicriminator, real_concat, t).squeeze()
                predicted_fake = self.model.discriminate(self.dicriminator, predicted_concat.detach(), t).squeeze()
                all_discriminator_predictions_real.extend(predicted_real.tolist())
                all_discriminator_predictions_fake.extend(predicted_fake.tolist())

                real_loss = self.gan_loss(predicted_real, real_label)
                fake_loss = self.gan_loss(predicted_fake, fake_label)
                # calculate the loss
                dis_loss += real_loss + fake_loss

            running_loss += (self.gan_weight * dis_loss + self.recon_weight * gen_loss).item() * images.shape[0]
            total_items += images.shape[0]
        # gather data
        if self.gan_weight > 0:
            all_discriminator_predictions_real = torch.tensor(all_discriminator_predictions_real)
            all_discriminator_predictions_fake = torch.tensor(all_discriminator_predictions_fake)

            self.log_discriminator_progress(epoch, all_discriminator_predictions_real, all_discriminator_predictions_fake)
        # self.log_helper.log_scalar("Metrics/seg_divergence", total_divergence / len(self.val_dataloader), epoch)
        return running_loss / total_items

    @override
    def post_epoch(self, epoch: int) -> None:
        if epoch == 0:
            self.logger.log_net_structure(self.model)
        self.diffusion_schedule.step()
        self.dicriminator_lr_schedule.step()
        # self.lr_scheduler.step()

        if epoch % self.infer_every == 0 and epoch > 100:
            self._save_checkpoint(f"epoch_{epoch}")
        if epoch % self.infer_every == 0 and self.device == 0:
            print("Running inference to log")
            images_to_embed = None  # BxCxHxW
            if self.do_context_embedding:
                images_to_embed, *_ = next(iter(self.val_dataloader))

            result_im, result_seg = self._inferer.infer(model=self.model, num_samples=self.config["batch_size"], embed_sample=images_to_embed)
            data_for_hist_im_R = result_im[..., 0].flatten()
            data_for_hist_im_G = result_im[..., 1].flatten()
            data_for_hist_im_B = result_im[..., 2].flatten()

            self.logger.log_histogram(data_for_hist_im_R, "generated R distribution")
            self.logger.log_histogram(data_for_hist_im_G, "generated G distribution")
            self.logger.log_histogram(data_for_hist_im_B, "generated B distribution")

            result_im = result_im[0]
            result_seg = result_seg[0].round().squeeze()

            result_seg[result_seg > 0] = 1
            result_seg[result_seg != 1] = 0

            self.logger.log_image_infered(result_im.numpy().astype(np.float32), mask=result_seg.numpy().astype(np.float32))

    @override
    def get_model(self, path) -> nn.Module:
        mode = self.config["mode"]
        if mode == "concat":
            if self.config.get("do_context_embedding", False):
                raise "Context embedding is not supported in concat mode"
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
        self.dicriminator = model.get_discriminator().to(self.device)
        return model.to(self.device)

    def get_lr_scheduler(self, optim=None):
        if optim is None and isinstance(self.optim, tuple):
            optim = self.optim[0]
        # scheduler = StepLR(optim, step_size=120, gamma=0.9)
        # scheduler = CyclicLR(optim, self.config["lr"], self.config["lr"]*5, step_size_up=100, step_size_down=100, cycle_momentum=False)
        scheduler = MultiStepLR(optim, milestones=[100, 200, 300, 400, 500, 1000], gamma=0.9)

        if self.device == 0:
            log(f"Scheduler being used is {scheduler}")
        return scheduler

    def get_optim(self) -> torch.optim:
        """
        Instantiates and returns the optimizer.
        :return: Optimizer object.
        """
        from torch.optim import Adam

        optim = Adam(
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
            log(f"Optim being used is {optim}")
        # hacky fix
        self.optim = optim
        return optim, d_optim

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