from typing import Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classeg.training.trainer import Trainer, log
import torchvision.transforms as transforms
from classeg.utils.constants import PREPROCESSED_ROOT
from classeg.utils.utils import read_json
from tqdm import tqdm


class ClassificationTrainer(Trainer):
    """
    This class is a subclass of the Trainer class and is used for training and checkpointing of networks.
    """

    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, cache: bool = False, world_size: int = 1):
        """
         Initializes the ClassificationTrainer object.

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
        class_names = read_json(f"{PREPROCESSED_ROOT}/{self.dataset_name}/id_to_label.json")
        self.class_names = [i for i in sorted(class_names.values())]

        self._last_val_accuracy = 0.
        self._val_accuracy = 0.
        self._train_accuracy = 0.
        self._last_train_accuracy = 0.

        self.softmax = nn.Softmax(dim=1)

    def get_augmentations(self) -> Tuple[Any, Any]:
        """
        Returns the augmentations for training and validation.

        :return: Tuple containing the training and validation augmentations.
        """
        train_aug = transforms.Compose([
            transforms.Resize(self.config.get('target_size', [512, 512]), antialias=True),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAdjustSharpness(1.3),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
        ])
        val_aug = transforms.Compose([
            transforms.Resize(self.config.get('target_size', [512, 512]), antialias=True)
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
        correct_count = 0
        all_predictions, all_labels = [], []

        # ForkedPdb().set_trace()
        log_image = epoch % 10 == 0
        for data, labels, _ in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            if log_image:
                self.log_helper.log_augmented_image(data[0])
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            batch_size = data.shape[0]
            # ForkedPdb().set_trace()
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            # update model
            loss.backward()
            self.optim.step()
            predictions = torch.argmax(predictions, dim=1)
            correct_count += torch.sum(predictions == labels)
            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())
            # gather data
            running_loss += loss.item() * batch_size
            total_items += batch_size
        self.log_helper.plot_confusion_matrix(all_predictions, all_labels, self.class_names, set_name="train")
        self._train_accuracy = correct_count / total_items
        return running_loss / total_items

    def post_epoch_log(self, epoch: int) -> Tuple:
        """
        Logs the information after each epoch.

        :param epoch: The current epoch number.
        :return: Tuple containing the log message.
        """
        messagea = f"Val accuracy: {self._val_accuracy} --change-- {self._val_accuracy - self._last_val_accuracy}"
        messageb = f"Train accuracy: {self._train_accuracy} --change-- {self._train_accuracy - self._last_train_accuracy}"
        self._last_val_accuracy = self._val_accuracy
        self._last_train_accuracy = self._train_accuracy
        return messagea, messageb

    # noinspection PyTypeChecker
    def eval_single_epoch(self, epoch) -> float:
        """
        Evaluates the model for a single epoch.

        :param epoch: The current epoch number.
        :return: The mean loss of the epoch.
        """

        running_loss = 0.
        correct_count = 0.
        total_items = 0
        all_predictions, all_labels = [], []
        i = 0
        for data, labels, _ in tqdm(self.val_dataloader):
            i += 1
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            if i == 1:
                self.log_helper.log_net_structure(self.model, data)
            batch_size = data.shape[0]
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            running_loss += loss.item() * batch_size
            # analyze
            predictions = torch.argmax(self.softmax(predictions), dim=1)
            # labels = torch.argmax(labels, dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            correct_count += torch.sum(predictions == labels)
            total_items += batch_size
        self.log_helper.plot_confusion_matrix(all_predictions, all_labels, self.class_names, set_name="val")
        self._val_accuracy = correct_count / total_items
        return running_loss / total_items

    def get_loss(self) -> nn.Module:
        """
        Returns the loss function to be used.

        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.CrossEntropyLoss()")
        return nn.CrossEntropyLoss()

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return super().get_dataloaders()

    def get_model(self, path: str) -> nn.Module:
        return super().get_model(path)
