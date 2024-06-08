import logging
import os.path
import pdb
import shutil
import sys
import time
from abc import abstractmethod
from typing import Tuple, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from json_torch_models.model_factory import ModelFactory
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from classeg.logging.logging import LogHelper
from classeg.utils.constants import *
from classeg.utils.utils import get_dataloaders_from_fold, get_config_from_dataset, get_dataset_mode_from_name
from classeg.utils.utils import write_json


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        """
        Overrides the interaction method of pdb.Pdb to allow interaction from a forked multiprocessing child.
        """
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class Trainer:
    """
    Trainer class for training and checkpointing of networks.
    """

    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, cache: bool = False, world_size: int = 1):
        """
        Initializes the Trainer class.

        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param unique_folder_name: Unique name for the output directory.
        :param config_name: The name of the configuration to use.
        :param resume: None if we should train from scratch, otherwise the model weights that should be used.
        :param cache: If True, cache the dataset to memory.
        :param world_size: The number of processes for distributed training.
        """
        assert (
            torch.cuda.is_available()
        ), "This pipeline only supports GPU training. No GPU was detected, womp womp."

        self.cache = cache
        self.dataset_name = dataset_name
        self.mode = get_dataset_mode_from_name(self.dataset_name)
        self.fold = fold
        self.world_size = world_size
        self.device = gpu_id
        self.config_name = config_name
        self.output_dir = self._prepare_output_directory(unique_folder_name)
        self.log_helper = LogHelper(self.output_dir)
        self.config = get_config_from_dataset(dataset_name, config_name)
        self._assert_preprocess_ready_for_train()
        if gpu_id == 0:
            log("Config:", self.config)
        self.seperator = (
            "======================================================================="
        )
        # Start on important stuff here
        self.train_transforms, _ = self.get_augmentations()
        self.train_dataloader, self.val_dataloader = self._get_dataloaders()
        self._current_epoch = 0
        self.model_path = model_path
        self.model = self.get_model(model_path)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[gpu_id])
        self.loss = self.get_loss()
        self.optim: torch.optim = self.get_optim()
        self.lr_scheduler = self.get_lr_scheduler()
        if self.device == 0:
            log(f"Optim being used is {self.optim}")
        self._save_self_file()
        if resume:
            self._load_checkpoint("latest")
        log(f"Trainer finished initialization on rank {gpu_id}.")
        if self.world_size > 1:
            dist.barrier()

    def _assert_preprocess_ready_for_train(self) -> None:
        """
        Ensures that the preprocess folder exists for the current dataset,
        and that the fold specified has been processed.
        """
        preprocess_dir = f"{PREPROCESSED_ROOT}/{self.dataset_name}"
        assert os.path.exists(preprocess_dir), (
            f"Preprocess root for dataset {self.dataset_name} does not exist. "
            f"run src.preprocessing.preprocess_entry.py before training."
        )
        assert os.path.exists(
            f"{preprocess_dir}/fold_{self.fold}"
        ), f"The preprocessed data path for fold {self.fold} does not exist. womp womp"

    def _save_self_file(self):
        """
        Copies the current file and the model file to the output directory.
        Also writes the configuration to a json file in the output directory.
        """
        shutil.copy(__file__, f"{self.output_dir}/trainer_code.py")
        if os.path.exists(self.model_path):
            shutil.copy(self.model_path, f"{self.output_dir}/model.json")
        write_json(self.config, f"{self.output_dir}/config.json")

    def _prepare_output_directory(self, session_id: str) -> str:
        """
        Prepares the output directory, and sets up logging to it.

        :param session_id: Unique identifier for the session.
        :return: str which is the output directory.
        """
        output_dir = f"{RESULTS_ROOT}/{self.dataset_name}/fold_{self.fold}/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=f"{output_dir}/logs.txt", force=True)
        if self.device == 0:
            print(f"Sending logging and outputs to {output_dir}")
        return output_dir

    @abstractmethod
    def get_augmentations(self) -> Tuple[Any, Any]:
        """
        Returns augmentations to be used in the training

        :return: train transforms, val transforms
        """
        ...

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        This method is responsible for creating the augmentation and then fetching dataloaders.

        :return: Train and val dataloaders.
        """
        train_transforms, val_transforms = self.get_augmentations()
        return get_dataloaders_from_fold(
            self.dataset_name,
            self.fold,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            sampler=(None if self.world_size == 1 else DistributedSampler),
            cache=self.cache,
            rank=self.device,
            world_size=self.world_size,
            config_name=self.config_name
        )

    @abstractmethod
    def train_single_epoch(self, epoch: int) -> float:
        """
        The training of each epoch is done here.

        :param epoch: The current epoch number.
        :return: The mean loss of the epoch.
        """
        ...

    @abstractmethod
    def eval_single_epoch(self, epoch: int) -> float:
        """
        Runs evaluation for a single epoch.

        :param epoch: The current epoch number.
        :return: The mean loss and mean accuracy respectively.
        """
        ...

    def post_epoch(self, epoch: int) -> None:
        """
        Executed after each epoch

        :param epoch: The current epoch number.
        """
        ...

    def post_epoch_log(self, epoch: int) -> Tuple:
        """
        Executed after each default logging cycle

        :param epoch: The current epoch number.
        """
        ...

    def post_training(self) -> None:
        """
        Executed after training
        """
        ...

    def train(self) -> None:
        """
        Starts the training process.
        """
        epochs = self.config["epochs"]
        start_time = time.time()
        best_val_loss = 999999.999  # Arbitrary large number
        # last values to show change
        last_train_loss = 0
        last_val_loss = 0
        for epoch in list(range(self._current_epoch, epochs)):
            # epoch timing
            self._current_epoch = epoch
            epoch_start_time = time.time()
            if self.world_size > 1:
                self.train_dataloader.sampler.set_epoch(epoch)
                self.val_dataloader.sampler.set_epoch(epoch)
            if self.device == 0:
                log(self.seperator)
                log(f"Epoch {epoch}/{epochs - 1} running...")
                if epoch == 0 and self.world_size > 1:
                    log("First epochs will be slow due to loading forking workers in ddp.")
            self.model.train()
            mean_train_loss = self.train_single_epoch(epoch)
            self.model.eval()
            with torch.no_grad():
                mean_val_loss = self.eval_single_epoch(epoch)
            self._save_model_weights("latest")  # saving model every epoch
            if self.device == 0:
                log("Learning rate: ", self.lr_scheduler.optimizer.param_groups[0]["lr"])
                log(f"Train loss: {mean_train_loss} --change-- {mean_train_loss - last_train_loss}")
                log(f"Val loss: {mean_val_loss} --change-- {mean_val_loss - last_val_loss}")
                extra_messages = self.post_epoch_log(epoch)
                if extra_messages is not None:
                    log(*extra_messages)
            self.lr_scheduler.step()
            # update 'last' values
            last_train_loss = mean_train_loss
            last_val_loss = mean_val_loss
            # If best model, save!
            if mean_val_loss < best_val_loss:
                if self.device == 0:
                    log(BEST_EPOCH_CELEBRATION)
                best_val_loss = mean_val_loss
                self._save_model_weights("best")
            epoch_end_time = time.time()
            if self.device == 0:
                log(f"Process {self.device} took {epoch_end_time - epoch_start_time} seconds.")
                self.log_helper.epoch_end(
                    mean_train_loss,
                    mean_val_loss,
                    self.lr_scheduler.optimizer.param_groups[0]["lr"],
                    epoch_end_time - epoch_start_time,
                )
            self.post_epoch(epoch)
            if self.world_size > 1:
                dist.barrier()

        # Now training is completed, print some stuff
        if self.world_size > 1:
            dist.barrier()
        self._save_model_weights("final")  # save the final weights
        self.post_training()
        end_time = time.time()
        seconds_taken = end_time - start_time
        if self.device == 0:
            log(self.seperator)
            log(f"Finished training {epochs} epochs.")
            log(f"{seconds_taken} seconds")
            log(f"{seconds_taken / 60} minutes")
            log(f"{(seconds_taken / 3600)} hours")
            log(f"{(seconds_taken / 86400)} days")

    def _save_model_weights(self, save_name: str) -> None:
        """
        Save the weights of the model, only if the current device is 0.

        :param save_name: The name of the checkpoint to save.
        """
        if self.device == 0:
            checkpoint = {}
            path = f"{self.output_dir}/{save_name}.pth"
            if self.world_size > 1:
                checkpoint["weights"] = self.model.module.state_dict()
            else:
                checkpoint["weights"] = self.model.state_dict()
            checkpoint["optim"] = self.optim.state_dict()
            checkpoint["current_epoch"] = self._current_epoch
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
            torch.save(checkpoint, path)

    def get_lr_scheduler(self):
        """
        Creates and returns a learning rate scheduler.

        :return: Learning rate scheduler.
        """
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

        optim = Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get('weight_decay', 0)
            # momentum=self.config.get('momentum', 0)
        )
        return optim

    def get_model(self, path: str) -> nn.Module:
        """
        :param path: The path to the json architecture definition.
        :return: The pytorch network module.
        """
        if not os.path.exists(path):
            log(f"The model path {path} does not exist.")
            log("If you do not want to define models through json as described at "
                "https://github.com/aheschl1/JsonTorchModels"
                ", then you can override the get_model method in a custom trainer!")
            raise SystemExit
        factory = ModelFactory(path, lookup_packages=["classeg.models.autoencoder", "classeg.models.unet"])
        model = factory.get_model().to(self.device)
        log(factory.log_kwargs)

        if self.device == 0:
            log(f"Loaded model {path}")
            all_params = sum(param.numel() for param in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            log(f"Total parameters: {all_params}")
            log(f"Trainable params: {trainable_params}")
            self.log_helper.log_parameters(all_params, trainable_params)
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def _load_checkpoint(self, weights_name) -> None:
        """
        Loads network checkpoint onto the DDP model.
        :param weights_name: The name of the weights to load in the form of *result folder*/*weight name*.pth
        :return: None
        """
        assert os.path.exists(f"{self.output_dir}/{weights_name}.pth")
        checkpoint = torch.load(f"{self.output_dir}/{weights_name}.pth")
        self._current_epoch = checkpoint["current_epoch"]
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint["weights"])
        else:
            self.model.load_state_dict(checkpoint["weights"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        log(f"Successfully loaded epochs info on rank {self.device}. Starting at {self._current_epoch}")

    @abstractmethod
    def get_loss(self) -> nn.Module:
        """
        Build the criterion object.
        :return: The loss function to be used.
        """
        ...

    @property
    def data_shape(self) -> Tuple[int, int, int]:
        """
        Property which is the data shape we are training on.
        :return: Shape of data.
        """
        return self.train_dataloader.dataset.datapoints[0].get_data()[0].shape


def log(*messages):
    """
    Prints to screen and logs a message.
    :param messages: The messages to display and log.
    :return: None
    """
    print(*messages)
    for message in messages:
        logging.info(f"{message} ")
