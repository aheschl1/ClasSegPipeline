from typing import Dict

import numpy as np
from PIL import ImageFile
from overrides import override

from classeg.preprocessing.preprocessor import Preprocessor
from classeg.preprocessing.splitting import Splitter
from classeg.utils.constants import *
import time
from tqdm import tqdm
from classeg.utils.utils import get_dataloaders_from_fold, get_case_name_from_number
from classeg.extensions.unstable_diffusion.preprocessing.bitifier import label_to_bitmask, bitmask_to_label
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch, cv2

"""
Extensions require to keep class name the same for proper loading
"""


class ExtensionPreprocessor(Preprocessor):
    def __init__(self, 
                 dataset_id: str, 
                 folds: int, 
                 processes: int, 
                 normalize: bool, 
                 dataset_desc: str = None, 
                 data_path=None, 
                 **kwargs):
        """
        :param folds: How many folds to generate.
        :param processes: How many processes should be used.
        :param normalize: Should normalized data be saved.
        :param dataset_id: The id of the dataset.

        This is the main driver for preprocessing.
        """
        super().__init__(dataset_id, folds, processes, normalize, dataset_desc, **kwargs)
        self.data_path = data_path

    def get_config(self) -> Dict:
        return {
            "batch_size": 64,
            "processes": 32,
            "lr": 0.0002,
            "epochs": 1000,
            "momentum": 0.9,
            "weight_decay": 0.00001,
            "target_size": [
                128,
                128
            ],
            "max_timestep": 1000,
            "diffuser": "linear",
            "diffusion_scheduler": "none",
            "min_beta": 0.0001,
            "max_beta": 0.02,
            "model_args": {
                "im_channels": 3,
                "seg_channels": 1,
                "layer_depth": 2,
                "channels": [
                    32,
                    64,
                    128,
                    256
                ],
                "shared_encoder": False,
                "time_emb_dim": 128
            },
            "mode": "unstable",
            "gan_weight": 0.25
        }


    def normalize_function(self, data: np.array) -> np.array:
        return data/255.0

    def post_preprocessing(self):
        """
        Called at the end of preprocessing
        """
        ...

    @override
    def pre_preprocessing(self):
        """
        Called before standard preprocessing flow
        """
        ...
        import glob, shutil
        cases = [x for x in glob.glob(f"{self.data_path}/**/*.png", recursive=True) if "_image.png" in x]
        
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/labelsTr", exist_ok=True)
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/imagesTr", exist_ok=True)
        case_max = -1
        for case in tqdm(cases, desc="Moving data"): 
            case_max += 1
            case_name = get_case_name_from_number(case_max)
            
            image = cv2.imread(case)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            mu, std = np.mean(image[:,:,0]), np.std(image[:,:,0])
            if mu < 50 or mu > 170 or std > 70:
                # too dark or too bright
                case_max-=1
                continue
            label = cv2.imread(case.replace("image", "label"))
            label = cv2.cvtColor(label, cv2.COLOR_BGRA2RGB)
            label = label[...,0]
            if np.sum(label) == 0:
                case_max-=1
                continue
            
            cv2.imwrite(f"{RAW_ROOT}/{self.dataset_name}/imagesTr/{case_name}.png", image)
            cv2.imwrite(f"{RAW_ROOT}/{self.dataset_name}/labelsTr/{case_name}.png", label)

    def process(self) -> None:
        super().process()

    def get_folds(self, k: int) -> Dict[int, Dict[str, list]]:
        """
        Gets random fold at 80/20 split. Returns in a map.
        :param k: How many folds for kfold cross validation.
        :return: Folds map
        """
        splitter = Splitter(self.datapoints, k)
        return splitter.get_split_map()

    def process_fold(self, fold: int) -> None:
        """
        Preprocesses a fold. This method indirectly triggers saving of metadata if necessary,
        writes data to proper folder, and will perform any other future preprocessing.
        :param fold: The fold that we are currently preprocessing.
        :return: Nothing.
        """
        print(f"Now starting with fold {fold}...")
        time.sleep(1)
        train_loader, val_loader = get_dataloaders_from_fold(
            self.dataset_name,
            fold,
            preprocessed_data=False,
            batch_size=1,
            shuffle=False,
            store_metadata=True,
            preload=False,
        )
        # prep dirs
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train/imagesTr")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val/imagesTr")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train/labelsTr")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val/labelsTr")
        # start saving preprocessed stuff
        for _set in ["train", "val"]:
            for images, masks, points in tqdm(
                    train_loader if _set == "train" else val_loader,
                    desc=f"Preprocessing {_set} set",
            ):
                point = points[0]
                writer = point.reader_writer
                images = images[0].float()  # Take 0 cause batched
                if images.shape[-1] == 3 and len(images.shape) == 3:
                    # move channel first
                    images = np.transpose(images, (2, 0, 1))
                if self.normalize:
                    images = self.normalize_function(images)
                writer.write(
                    images.to(torch.float32),
                    f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/imagesTr/{point.case_name}."
                    f"{point.extension if point.extension == 'nii.gz' else 'npy'}",
                )
                masks = masks[0]
                masks[masks != 0] = 1
                masks = torch.from_numpy(label_to_bitmask(masks.numpy()))
                writer.write(
                    masks.to(torch.int8),
                    f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/labelsTr/{point.case_name}."
                    f"{point.extension if point.extension == 'nii.gz' else 'npy'}",
                )
