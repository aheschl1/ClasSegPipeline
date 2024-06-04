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
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Extensions require to keep class name the same for proper loading
"""


class ExtensionPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, dataset_desc: str = None, data_path=None, **kwargs):
        """
        :param folds: How many folds to generate.
        :param processes: How many processes should be used.
        :param normalize: Should normalized data be saved.
        :param dataset_id: The id of the dataset.

        This is the main driver for preprocessing.
        """
        super().__init__(dataset_id, folds, processes, normalize, dataset_desc, **kwargs)
        self.data_path = f"{data_path}/1"

    def get_config(self) -> Dict:
        return {
            "batch_size": 32,
            "processes": self.processes,
            "lr": 0.0002,
            "epochs": 50,
            "momentum": 0,
            "weight_decay": 0.0001,
            "target_size": [224, 224],
            "max_timestep": 1000,
            "diffuser": "linear",
            "min_beta": 0.0001,
            "max_beta": 0.02,
        }

    def normalize_function(self, data: np.array) -> np.array:
        if self.datapoints[0].extension in ["JPEG", "jpg", "png", 'jpeg']:
            return ((data / 255) * 2) - 1
        print("Warn, skipping normalization")
        return data

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
        import glob, shutil
        cases = glob.glob(f"{self.data_path}/*")
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/labelsTr", exist_ok=True)
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/imagesTr", exist_ok=True)
        for case in tqdm(cases, desc="Moving data"):
            case_name = get_case_name_from_number(int(case.split("/")[-1]))
            shutil.copy(f"{case}/Mask.png", f"{RAW_ROOT}/{self.dataset_name}/labelsTr/{case_name}.jpg")
            shutil.copy(f"{case}/Image.jpg", f"{RAW_ROOT}/{self.dataset_name}/imagesTr/{case_name}.jpg")

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
        import torch, cv2
        def my_resize(image, mask=False):
            # Get the dimensions of the image
            image = np.array(image.permute(1, 2, 0))
            height, width = image.shape[:2]
            # Determine the scaling factor
            if height < width:
                scale_factor = 260 / height
            else:
                scale_factor = 260 / width

            # Compute the new dimensions
            new_dimensions = (int(width * scale_factor), int(height * scale_factor))

            # Resize the image
            resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
            if len(resized_image.shape) == 2:
                resized_image = resized_image[None].transpose((1, 2, 0))
            # Save the resized image
            return torch.from_numpy(resized_image).permute(2, 0, 1)
        for _set in ["train", "val"]:
            for images, masks, points in tqdm(
                    train_loader if _set == "train" else val_loader,
                    desc=f"Preprocessing {_set} set",
            ):
                if min(images[0, 0, :, :].shape) < 256:
                    images = my_resize(images[0]).unsqueeze(0)
                    print(masks.shape)
                    masks = my_resize(masks[0], mask=True).unsqueeze(0)
                    print(masks.shape, images.shape)
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
                writer.write(
                    masks.to(torch.float32),
                    f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/labelsTr/{point.case_name}."
                    f"{point.extension if point.extension == 'nii.gz' else 'npy'}",
                )
