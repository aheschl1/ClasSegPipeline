from typing import Tuple
import yaml, json
from classeg.training.trainer import Trainer
from classeg.utils.constants import RAW_ROOT, PREPROCESSED_ROOT
import os
from classeg.utils.utils import get_dataloaders_from_fold

class TamingTrainer(Trainer):
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
        self.train_taming(config_name, dataset_name, fold, world_size)
        exit()


    def train_taming(self, config_name: str, dataset_name: str, fold: int, world_size: int):
        config_path = f"{'/'.join(__file__.split('/')[:-1])}/taming-transformers/configs/{config_name}.yaml"
        with open(config_path) as f:
            config = yaml.load(f)
        train_files, val_files = self.get_training_image_list_file(dataset_name, fold)
        config['model']['data']['params']['train']['params']['training_images_list_file'] = train_files
        config['model']['data']['params']['validation']['params']['test_images_list_file'] = val_files

        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        os.system(f"python {'/'.join(__file__.split('/')[:-1])}/taming-transformers/main.py --base {config_path} -t True --gpus {'0,' if world_size == 1 else '0,1'} --name {dataset_name}_{fold}")
        
    def get_training_image_list_file(self, dataset_name: str, fold: int) -> Tuple[str, str]:
        trainset, valset = get_dataloaders_from_fold(dataset_name, fold, preprocessed_data=False)

        train_files = [f"{x.im_path}" for x in trainset.dataset]
        val_files = [f"{x.im_path}" for x in valset.dataset]

        return train_files, val_files
