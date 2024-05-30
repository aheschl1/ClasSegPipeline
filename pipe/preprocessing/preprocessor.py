import logging
import os.path
import time
import warnings
from abc import abstractmethod
from typing import Dict, Type, Union, Tuple
import click
import numpy as np
from PIL import ImageFile
from tqdm import tqdm

from pipe.preprocessing.splitting import Splitter
from pipe.preprocessing.utils import maybe_make_preprocessed
from pipe.utils.constants import *
from pipe.utils.utils import (
    write_json,
    get_dataset_name_from_id,
    check_raw_exists,
    get_raw_datapoints,
    get_dataloaders_from_fold,
    get_labels_from_raw,
    get_dataset_mode_from_name
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Preprocessor:
    def __init__(
            self,
            dataset_id: str,
            folds: int,
            processes: int,
            normalize: bool,
            dataset_desc: str = None,
            **kwargs
    ):
        """
        :param folds: How many folds to generate.
        :param processes: How many processes should be used.
        :param normalize: Should normalized data be saved.
        :param dataset_id: The id of the dataset.

        This is the main driver for preprocessing.
        """
        self.mode = None
        self.dataset_name = get_dataset_name_from_id(dataset_id, name=dataset_desc)
        self.processes = processes
        self.normalize = normalize
        self.datapoints = None
        self.folds = folds
        made_output = check_raw_exists(self.dataset_name)
        if made_output:
            print(f"Made folder {RAW_ROOT}/{self.dataset_name}")
        maybe_make_preprocessed(self.dataset_name, query_overwrite=True)
        # We export the config building to a new method
        self.build_config()

    def build_config(self) -> None:
        """
        Creates the config.json file that should contain training hyperparameters. Hardcode default values here.
        :return: None
        """
        config = {
            "batch_size": 64,
            "processes": self.processes,
            "lr": 0.0002,
            "epochs": 500,
            "momentum": 0,
            "weight_decay": 0,
            "target_size": [128, 128],
            "max_timestep": 1000,
            "diffuser": "cos",
            "min_beta": 0.0001,
            "max_beta": 0.999,
            "model_args": {
                "im_channels": 3,
                "layer_depth": 2,
                "middle_layers_count": 2,
                "channels": [32, 64, 128, 256],
                "time_emb_dim": 128,
                "norm_op": "GroupNorm",
                "conv_op": "Conv2d"
            }
        }
        write_json(config, f"{PREPROCESSED_ROOT}/{self.dataset_name}/config.json")

    def _self_supervised_normalize_function(self, data: np.array) -> np.array:
        if self.datapoints[0].extension in ["JPEG", "jpg", "png", 'jpeg']:
            return ((data / 255) * 2) - 1
        print("Warn, skipping normalization")
        return data

    def _preprocess_classification(self):
        def map_labels_to_id(return_inverse: bool = False) -> Union[
            Tuple[Dict[str, int], Dict[int, str]], Dict[str, int]]:
            """
            :param return_inverse: If true returns id:name as well as name:id mapping.
            :return: Dict that maps label name to id.
            """
            mapping = {}
            inverse = {}
            for i, label in enumerate(labels):
                mapping[label] = i
                inverse[i] = label
            if return_inverse:
                return mapping, inverse
            return mapping

        labels = get_labels_from_raw(self.dataset_name)
        assert len(labels) > 1, "We only found one label folder, maybe the folder structure is wrong."
        label_to_id_mapping, id_to_label_mapping = map_labels_to_id(return_inverse=True)
        write_json(label_to_id_mapping, f"{PREPROCESSED_ROOT}/{self.dataset_name}/label_to_id.json")
        write_json(id_to_label_mapping, f"{PREPROCESSED_ROOT}/{self.dataset_name}/id_to_label.json")
        # Label stuff done, start with fetching data. We will also save a case to label mapping.

    def _preprocess_segmentation(self):
        ...

    def _preprocess_diffusion(self):
        ...

    def post_preprocessing(self):
        """
        Called at the end of preprocessing
        """
        ...

    @abstractmethod
    def pre_preprocessing(self):
        """
        Called before standard preprocessing flow
        """
        ...

    def _ensure_raw_exists(self):
        if not os.path.exists(f"{RAW_ROOT}/{self.dataset_name}"):
            raise ValueError("The raw folder does not exist!")

    def process(self) -> None:
        # Here we will find what labels are present in the dataset. We will also map them to int labels, and save the
        # mappings.
        self.pre_preprocessing()
        self._ensure_raw_exists()
        self.mode = get_dataset_mode_from_name(self.dataset_name)
        print(f"Dataset mode detected through RAW_ROOT is {self.mode}.")
        {
            SEGMENTATION: self._preprocess_segmentation,
            SELF_SUPERVISED: self._preprocess_diffusion,
            CLASSIFICATION: self._preprocess_classification
        }[self.mode]()
        self.datapoints = get_raw_datapoints(self.dataset_name)
        assert len(self.datapoints) >= 1, f"There are no datapoints in {RAW_ROOT}/{self.dataset_name}"
        print(f"Found {len(self.datapoints)} samples.")
        # There is some circular dependency here regarding case to label mapping requiring datapoints
        # and datapoints requiring the result of _preprocess_classification
        # TODO move this hoe to _preprocess_classification
        if self.mode == CLASSIFICATION:
            def get_case_to_label_mapping() -> Dict[str, int]:
                """
                Given a list of datapoints, we create a mapping of label name to label id.
                :return:
                """
                mapping = {}
                for point in self.datapoints:
                    mapping[point.case_name] = point.label
                return mapping

            write_json(get_case_to_label_mapping(), f"{PREPROCESSED_ROOT}/{self.dataset_name}/case_label_mapping.json")

        # Label stuff done, start with fetching data. We will also save a case to label mapping.
        splits_map = self.get_folds(self.folds)
        write_json(splits_map, f"{PREPROCESSED_ROOT}/{self.dataset_name}/folds.json")
        self.verify_dataset_integrity()
        # We now have the folds: time to preprocess the data
        for fold_id in splits_map:
            self.process_fold(fold_id)
        self.post_preprocessing()

    def verify_dataset_integrity(self) -> None:
        """
        Ensures that all datapoints have the same shape.
        :return:
        """
        names = set()
        for point in tqdm(self.datapoints, desc="Verifying dataset integrity"):
            names_before = len(names)
            names.add(point.case_name)
            assert names_before < len(names), f"The name {point.case_name} is in the dataset at least 2 times :("

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
        if self.mode == SEGMENTATION:
            warnings.warn("Default preprocessor converts all segmentations to binary!")

        if self.mode == SEGMENTATION:
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train/imagesTr")
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val/imagesTr")
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train/labelsTr")
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val/labelsTr")
        # start saving preprocessed stuff
        if self.mode in [CLASSIFICATION, SEGMENTATION]:
            print("Performing z-score normalization")
            train_loader = train_loader.dataset[0][2].normalizer(train_loader, active=self.normalize)
            val_loader = val_loader.dataset[0][2].normalizer(val_loader, active=self.normalize, calculate_early=False)
            val_loader.sync(train_loader)
            if train_loader.mean is not None:
                mean_json = {
                    "mean": train_loader.mean.tolist(),
                    "std": train_loader.std.tolist()
                }
                write_json(mean_json, f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/mean_std.json")

        for _set in ["train", "val"]:
            for images, labels, points in tqdm(
                    train_loader if _set == "train" else val_loader,
                    desc=f"Preprocessing {_set} set",
            ):
                point = points[0]
                writer = point.reader_writer
                images = images[0].float()  # Take 0 cause batched
                if images.shape[-1] == 3 and len(images.shape) == 3:
                    # move channel first
                    images = np.transpose(images, (2, 0, 1))
                if self.normalize and self.mode in [SELF_SUPERVISED]:
                    images = self._self_supervised_normalize_function(images)
                image_path = f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/imagesTr/{point.case_name}.{point.extension if point.extension == 'nii.gz' else 'npy'}"
                if self.mode in [CLASSIFICATION, SELF_SUPERVISED]:
                    image_path = image_path.replace("/imagesTr", "")
                writer.write(
                    images,
                    image_path
                )
                if self.mode == SEGMENTATION:
                    labels[labels != 0] = 1
                    writer.write(
                        labels[0],
                        f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/labelsTr/{point.case_name}."
                        f"{point.extension if point.extension == 'nii.gz' else 'npy'}",
                    )
