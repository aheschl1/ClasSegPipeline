import random
from typing import Tuple, Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from classeg.training.trainer import Trainer, log


class SelfSupervisedTrainer(Trainer):
    """
    The Self-Supervised Trainer is a class that is used to train models in a self-supervised manner.
    """
    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, cache: bool = False, world_size: int = 1):
        """
        Initializes the SelfSupervisedTrainer object.

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

    def get_augmentations(self) -> Tuple[Any, Any]:
        """
       Returns the augmentations for training and validation.

       :return: Tuple containing the training and validation augmentations.
       """
        train_transforms = transforms.Compose([
            transforms.Resize(self.config["target_size"]),
            transforms.RandomRotation(degrees=30),
            transforms.RandomVerticalFlip(p=0.25, ),
            transforms.RandomHorizontalFlip(p=0.25, )
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(self.config["target_size"]),
        ])
        return train_transforms, val_transforms

    def train_single_epoch(self, epoch) -> float:
        """
        Trains the model for a single epoch.

        :param epoch: The current epoch number.
        :return: The mean loss of the epoch.
        """
        log_image = epoch % 10 == 0
        running_loss = 0.
        total_items = 0.
        i = 0.
        for data, _, _ in tqdm.tqdm(self.train_dataloader):
            i += 1
            self.optim.zero_grad()
            if log_image:
                self.log_helper.log_augmented_image(data[0])

            data: torch.Tensor = data.to(self.device)
            og_data = data.detach().clone()

            batch_size = data.shape[0]

            predictions = self.model(data)
            loss = self.loss(predictions, og_data)
            # update model
            loss.backward()
            self.optim.step()
            # gather data
            running_loss += loss.item() * batch_size
            total_items += batch_size

        return running_loss / total_items

    # noinspection PyTypeChecker
    def eval_single_epoch(self, epoch) -> float:
        """
        Evaluates the model for a single epoch.

        :param epoch: The current epoch number.
        :return: The mean loss of the epoch.
        """

        running_loss = 0.
        total_items = 0
        for data, _, _ in tqdm.tqdm(self.val_dataloader):
            data = data.to(self.device)
            batch_size = data.shape[0]
            predictions = self.model(data)
            index = random.randint(0, data.shape[0]-1)
            self.log_helper.log_image_infered(predictions[index], epoch, Original=data[index])
            loss = self.loss(predictions, data)
            running_loss += loss.item() * batch_size
            total_items += batch_size

        return running_loss / total_items

    def get_loss(self) -> nn.Module:
        """
        Returns the loss function to be used.

        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.MSELoss()")
        return nn.MSELoss()

    def get_model(self, path: str) -> nn.Module:
        return super().get_model(path)
