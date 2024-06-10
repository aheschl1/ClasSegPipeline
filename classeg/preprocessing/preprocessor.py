import os.path
import os.path
import time
import warnings
from typing import Dict, Union, Tuple

import numpy as np
from PIL import ImageFile
from tqdm import tqdm

from classeg.preprocessing.splitting import Splitter
from classeg.preprocessing.utils import maybe_make_preprocessed
from classeg.utils.constants import *
from classeg.utils.utils import (
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
    """
    The Preprocessor class is responsible for preprocessing the dataset. It includes methods for normalizing data,
    building configuration, preprocessing classification, segmentation, and ssl, storing case label mapping for
    classification, verifying dataset integrity, getting folds, preparing fold directories, and processing folds.
    """
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
       Initialize the Preprocessor object.

       :param dataset_id: The id of the dataset.
       :param folds: How many folds to generate.
       :param processes: How many processes should be used.
       :param normalize: Should normalized data be saved.
       :param dataset_desc: The description of the dataset.
       """
        self.dataset_name = get_dataset_name_from_id(dataset_id, name=dataset_desc)
        self.mode = None  # late initialization in process
        self.processes = processes
        self.normalize = normalize
        self.datapoints = None
        self.folds = folds
        self.skip_zscore_norm = False
        made_output = check_raw_exists(self.dataset_name)
        if made_output:
            print(f"Made folder {RAW_ROOT}/{self.dataset_name}")
        maybe_make_preprocessed(self.dataset_name, query_overwrite=True)
        # We export the config building to a new method
        self._build_config()

    def get_config(self) -> Dict:
        """
        Get the configuration for the dataset.

        :return: A dictionary containing the configuration.
        """
        return {
            "batch_size": 32,
            "processes": DEFAULT_PROCESSES,
            "lr": 0.001,
            "epochs": 50,
            "momentum": 0,
            "weight_decay": 0.0001,
            "target_size": [224, 224]
        }

    def _build_config(self) -> None:
        """
        Creates the config.json file that should contain training hyperparameters. Hardcode default values here.
        """
        config = self.get_config()
        write_json(config, f"{PREPROCESSED_ROOT}/{self.dataset_name}/config.json")

    def normalize_function(self, data: np.array) -> np.array:
        """
        Perform normalization. z-score normalization will still always occur for classification and segmentation

        :param data: The data to be normalized.
        :return: The normalized data.
        """
        if self.mode == SELF_SUPERVISED:
            print("Default self supervised normalization scales to [-1, 1]")
            return ((data / 255) * 2) - 1
        return data

    def _preprocess_classification(self):
        """
       Preprocess the data for classification.
       """
        def map_labels_to_id(return_inverse: bool = False) -> Union[
                Tuple[Dict[str, int], Dict[int, str]],
                Dict[str, int]
        ]:
            """
            :param return_inverse: If true returns id:name as well as name:id mapping.
            :return: Dict that maps label name to id.
            """
            mapping = {}
            inverse = {}
            sorted_labels = sorted(labels)
            for i, label in enumerate(sorted_labels):
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

    def _store_case_label_mapping_for_classification(self):
        """
        Store the case to label mapping for classification.
        """
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

    def _preprocess_segmentation(self):
        """
        Preprocess the data for segmentation.
        """
        ...

    def _preprocess_ssl(self):
        """
        Preprocess the data for diffusion.
        """
        ...

    def post_preprocessing(self):
        """
        Called at the end of preprocessing
        """
        ...

    def pre_preprocessing(self):
        """
        Called before standard preprocessing flow
        """
        ...

    def _ensure_raw_exists(self):
        """
        Ensure the raw data exists in RAW_ROOT.
        """
        if not os.path.exists(f"{RAW_ROOT}/{self.dataset_name}"):
            raise ValueError("The raw folder does not exist!")

    def process(self) -> None:
        """
        Process the data.

        This is the entry point for the preprocessing pipeline.
        Override this method in subclasses to implement total custom preprocessing.
        """
        # Here we will find what labels are present in the dataset. We will also map them to int labels, and save the
        # mappings.
        self.pre_preprocessing()
        self._ensure_raw_exists()
        self.mode = get_dataset_mode_from_name(self.dataset_name)
        print(f"Dataset mode detected through RAW_ROOT is {self.mode}.")
        if self.mode == SEGMENTATION:
            warnings.warn("Default preprocessor converts all segmentations to binary when inside process_fold!")
        {
            SEGMENTATION: self._preprocess_segmentation,
            SELF_SUPERVISED: self._preprocess_ssl,
            CLASSIFICATION: self._preprocess_classification
        }[self.mode]()
        self.datapoints = get_raw_datapoints(self.dataset_name)
        assert len(self.datapoints) >= 1, f"There are no datapoints in {RAW_ROOT}/{self.dataset_name}"
        print(f"Found {len(self.datapoints)} samples.")
        # There is some circular dependency here regarding case to label mapping requiring datapoints
        # and datapoints requiring the result of _preprocess_classification
        if self.mode == CLASSIFICATION:
            self._store_case_label_mapping_for_classification()

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

        {
            0: {
                train: [case_xxxxx, ....],
                val: [case_xxxxx, ....]
            },
            1: {...},
            .
            .
            .
        }

        """
        splitter = Splitter(self.datapoints, k)
        return splitter.get_split_map()

    def _prep_fold_dirs(self, fold: int) -> None:
        """
        Prepares the folders for a fold.

        :param fold: The fold to prepare.
        """
        # prep dirs
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val")
        if self.mode == SEGMENTATION:
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train/imagesTr")
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val/imagesTr")
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train/labelsTr")
            os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val/labelsTr")

    def process_fold(self, fold: int) -> None:
        """
        Preprocesses a fold. This method indirectly triggers saving of metadata if necessary,
        writes data to proper folder, and will perform any other future preprocessing.

        :param fold: The fold that we are currently preprocessing.
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
            cache=False,
        )
        # prep dirs
        self._prep_fold_dirs(fold)
        # start saving preprocessed stuff
        if self.mode in [CLASSIFICATION, SEGMENTATION] and not self.skip_zscore_norm and self.normalize:

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
            loader = train_loader if _set == "train" else val_loader
            for images, labels, points in tqdm(loader, desc=f"Preprocessing {_set} set"):
                # Data is batched
                point = points[0]
                images = images[0]
                labels = labels[0]

                writer = point.reader_writer

                if self.normalize:
                    images = self.normalize_function(images)

                image_path = (f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/imagesTr/"
                              f"{point.case_name}.{point.extension if point.extension == 'nii.gz' else 'npy'}")
                if self.mode in [CLASSIFICATION, SELF_SUPERVISED]:
                    image_path = image_path.replace("/imagesTr", "")

                writer.write(
                    images,
                    image_path
                )
                if self.mode == SEGMENTATION:
                    labels[labels != 0] = 1
                    writer.write(
                        labels,
                        image_path.replace("imagesTr", "labelsTr")
                    )
