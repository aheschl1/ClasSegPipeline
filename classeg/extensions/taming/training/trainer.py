from typing import Tuple, List
import yaml, json
from classeg.training.trainer import Trainer
from classeg.utils.constants import RAW_ROOT, PREPROCESSED_ROOT
import os
from classeg.training.trainer import log
from classeg.utils.utils import get_dataloaders_from_fold
from tqdm import tqdm


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
        self.dataset_name = dataset_name
        self.fold = fold
        self.config_name = config_name
        self.world_size = world_size
        self.device = gpu_id
        self.output_dir = super()._prepare_output_directory(unique_folder_name)

    def train_taming(self):
        config_path = f"{'/'.join(__file__.split('/')[:-1])}/taming-transformers/configs/{self.config_name}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        train_files, val_files = self.get_training_image_list_file(self.dataset_name, self.fold)
        log(train_files)
        log(val_files)
        with open(f"{self.output_dir}/train_files.txt", 'w') as f:
            # one path per line in txt file
            f.write("\n".join(train_files))
        with open(f"{self.output_dir}/val_files.txt", 'w') as f:
            # one path per line in txt file
            f.write("\n".join(val_files))
        config['data']['params']['train']['params']['training_images_list_file'] = f"{self.output_dir}/train_files.txt"
        config['data']['params']['validation']['params']['test_images_list_file'] = f"{self.output_dir}/val_files.txt"

        with open(config_path, 'w') as f:
            yaml.dump(config, f)
<<<<<<< HEAD
        
        os.system(f"conda activate taming; python {'/'.join(__file__.split('/')[:-1])}/taming-transformers/main.py --base {config_path} -t True --gpus {'0,' if world_size == 1 else '0,1'} --name {dataset_name}_{fold}")
        
    def get_training_image_list_file(self, dataset_name: str, fold: int) -> Tuple[str, str]:
=======

        os.system(
            f"python {'/'.join(__file__.split('/')[:-1])}/taming-transformers/main.py --base {config_path} -t True --gpus {'0,' if self.world_size == 1 else '0,1'} --name {self.dataset_name}_{self.fold}")

    def get_training_image_list_file(self, dataset_name: str, fold: int) -> Tuple[List[str], List[str]]:
>>>>>>> 851a2a3e5c0cb73acc879d1aae173bffcf5d33cc
        trainset, valset = get_dataloaders_from_fold(dataset_name, fold, preprocessed_data=False)
        train_files = [f"{x.im_path}" for x in tqdm(trainset.dataset.datapoints)]
        val_files = [f"{x.im_path}" for x in tqdm(valset.dataset.datapoints)]
        print(train_files)
        return train_files, val_files

    def train(self):
        self.train_taming()
