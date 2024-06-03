import os
import shutil
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
from overrides import override
from tqdm import tqdm

from classeg.preprocessing.preprocessor import Preprocessor
from classeg.preprocessing.splitting import Splitter
from classeg.utils.constants import *
from classeg.utils.utils import get_case_name_from_number

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Extensions require to keep class name the same for proper loading
"""


class CocoPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool,
                 dataset_desc: str = None, coco_train_root=None, **kwargs):
        """
        :param folds: How many folds to generate.
        :param processes: How many processes should be used.
        :param normalize: Should normalized data be saved.
        :param dataset_id: The id of the dataset.

        This is the main driver for preprocessing.
        """
        super().__init__(dataset_id, folds, processes, normalize, dataset_desc, **kwargs)
        if coco_train_root is None:
            raise ValueError("Provide coco_train_root=<> argument for this extension.")
        self.coco_train_root = coco_train_root

    def get_config(self) -> Dict:
        return {
            "batch_size": 32,
            "processes": DEFAULT_PROCESSES,
            "lr": 0.001,
            "epochs": 50,
            "momentum": 0,
            "weight_decay": 0.0001,
            "target_size": [224, 224]
        }

    def normalize_function(self, data: np.array) -> np.array:
        """
        Perform normalization. z-score normalization will still always occur for classification and segmentation
        """
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
        # Build raw root
        from pycocotools.coco import COCO
        coco = COCO(f'{self.coco_train_root}/labels.json')
        img_dir = f'{self.coco_train_root}/data'

        cat_ids = coco.getCatIds(catNms=['person'])

        case_num = 0
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/imagesTr", exist_ok=True)
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/labelsTr", exist_ok=True)
        for id in tqdm(coco.imgs, desc="Building raw"):
            image = coco.imgs[id]
            path = os.path.join(img_dir, image['file_name'])
            anns_ids = coco.getAnnIds(imgIds=[id], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            mask = coco.annToMask(anns[0])
            for i in range(len(anns)):
                mask += coco.annToMask(anns[i])

            image = plt.imread(path)
            if image.shape[-1] != 3:
                image = image[None].transpose((1, 2, 0))
                image = np.stack([image, image, image], axis=2)
                plt.imsave(f"{RAW_ROOT}/{self.dataset_name}/imagesTr/{get_case_name_from_number(int(case_num))}.jpg", image)
            else:
                shutil.copy(path, f"{RAW_ROOT}/{self.dataset_name}/imagesTr/{get_case_name_from_number(int(case_num))}.jpg")
            mask[mask != 0] = 1
            cv2.imwrite(
                f"{RAW_ROOT}/{self.dataset_name}/labelsTr/{get_case_name_from_number(int(case_num))}.jpg",
                mask)
            case_num += 1

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
        super().process_fold(fold)
