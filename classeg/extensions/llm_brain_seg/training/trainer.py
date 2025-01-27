from typing import Tuple, Any

import torch
import torch.nn as nn
from monai.data import DataLoader

from classeg.training.trainer import Trainer, log
import albumentations as A
from monai.losses import DiceCELoss
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
import tqdm
import matplotlib.pyplot as plt

class SegmentationTrainer(Trainer):
    """
    This class is a subclass of the Trainer class and is used for training and checkpointing of networks.
    """

    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, cache: bool = False, world_size: int = 1):
        """
        Initializes the SegmentationTrainer object.

        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param unique_folder_name: Unique name for the folder.
        :param config_name: Name of the configuration.
        :param resume: Boolean indicating whether to resume training or not.
        :param cache: Boolean indicating whether to cache or not.
        :param world_size: Size of the world.
        """

        super().__init__(dataset_name, fold, model_path, gpu_id, unique_folder_name, config_name, resume, cache,
                         world_size)
        self._last_val_iou = 0.
        self._val_iou = 0.
        self._val_dice = 0.
        self._last_val_dice = 0.

        self.num_classes = None
        self.softmax = nn.Softmax(dim=1)

    def get_augmentations(self) -> Tuple[Any, Any]:
        """
        Returns the augmentations for training and validation.

        :return: Tuple containing the training and validation augmentations.
        """
        train_aug = A.Compose([
            A.Resize(*self.config.get('target_size', [512, 512])),
            A.VerticalFlip(p=0.25),
            A.HorizontalFlip(p=0.25),
        ])
        val_aug = A.Compose([
            A.Resize(*self.config.get('target_size', [512, 512]))
        ])

        return train_aug, val_aug

    def train_single_epoch(self, epoch) -> float:
        """
        Trains the model for a single epoch.

        :param epoch: The current epoch number.
        :return: The mean loss of the epoch.
        """
        running_loss = 0.
        total_items = 0
        # ForkedPdb().set_trace()
        log_image = epoch % 10 == 0
        for data, labels, _ in tqdm.tqdm(self.train_dataloader):
            self.optim.zero_grad(set_to_none=True)
            if log_image:
                self.logger.log_augmented_image(data[0], mask=labels[0].squeeze().cpu().numpy())
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            batch_size = data.shape[0]
            # ForkedPdb().set_trace()
            # do prediction and calculate loss
            predictions = self.model(data)
            self.num_classes = predictions.shape[1]
            loss = self.loss(predictions, labels)
            # update model
            loss.backward()
            self.optim.step()
            # gather data
            running_loss += loss.item() * batch_size
            total_items += batch_size

        return running_loss / total_items

    def post_epoch_log(self, epoch: int) -> Tuple:
        """
        Logs the information after each epoch.

        :param epoch: The current epoch number.
        :return: Tuple containing the log message.
        """
        message1 = f"Val Dice: {self._val_dice} --change-- {self._val_dice - self._last_val_dice}"
        message2 = f"Val IoU: {self._val_iou} --change-- {self._val_iou - self._last_val_iou}"

        self._last_val_dice = self._val_dice
        self._last_val_iou = self._val_iou

        return message1, message2

    # noinspection PyTypeChecker
    def eval_single_epoch(self, epoch) -> float:
        """
        Evaluates the model for a single epoch.

        :param epoch: The current epoch number.
        :return: The mean loss of the epoch.
        """

        running_loss = 0.

        dice_score = 0.
        iou_score = 0.

        total_items = 0
        i = 0

        dice_metric = GeneralizedDiceScore(1, include_background=False).to(self.device)
        iou_metric = MeanIoU(1, include_background=False).to(self.device)

        for data, labels, _ in tqdm.tqdm(self.val_dataloader):
            i += 1
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            if i == 1 and epoch % 10 == 0:
                self.logger.log_net_structure(self.model, data)
            batch_size = data.shape[0]
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            running_loss += loss.item() * batch_size
            # analyze
            predictions = torch.argmax(self.softmax(predictions), dim=1)

            if i == 1:
                self.logger.log_image_infered(data[0],
                                              ground_truth=labels[0].squeeze().cpu().numpy(),
                                              prediction=predictions[0].squeeze().cpu().numpy()
                                              )

            dice_score += dice_metric(predictions.squeeze().to(int), labels.squeeze().to(int)).cpu().item() * batch_size
            iou_score += iou_metric(predictions.squeeze().to(int), labels.squeeze().to(int)).cpu().item() * batch_size

            total_items += batch_size

        self.logger.log_metrics({
            "dice": dice_score / total_items,
            "iou": iou_score / total_items
        }, "val")
        self._val_dice = dice_score / total_items
        self._val_iou = iou_score / total_items
        return running_loss / total_items

    def get_loss(self) -> nn.Module:
        """
        Returns the loss function to be used.

        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.CrossEntropyLoss()")
        return DiceCELoss(
            include_background=False,
            softmax=True,
            to_onehot_y=True
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return super().get_dataloaders()

    def get_model(self, path: str) -> nn.Module:
        from classeg.extensions.llm_brain_seg.training.model.model import UNet
        return UNet(
            in_channels=3,
            use_weights=self.config.get('use_llm', True)
        )
