from typing import List, Callable, Tuple

import torch
from torch.utils.data import Dataset
from pipe.dataloading.datapoint import Datapoint
import glob
from pipe.utils.constants import RAW_ROOT, CLASSIFICATION, SEGMENTATION, SELF_SUPERVISED, PREPROCESSED_ROOT
import json

class PipelineDataset(Dataset):

    def __init__(self,
                 datapoints: List[Datapoint],
                 dataset_name: str,
                 transforms: Callable = None,
                 store_metadata: bool = False,
                 preload: bool = False,
                 dataset_type: str = "train"
                 ):
        """
        Custom dataset for this pipeline.
        :param datapoints: The list of datapoints for the dataset.
        :param transforms: The transforms to apply to the data.
        :param store_metadata: Whether this data requires metadata storage.
        """
        self.datapoints = datapoints
        self.transforms = transforms
        self.store_metadata = store_metadata
        self.dataset_name = dataset_name
        self.preload = preload
        self.dataset_type = dataset_type
        self.mode = self.datapoints[0].mode
        self._num_classes = None

    def _get_number_of_classes(self):
        """
        Checks how many classes there are based on classes of datapoints.
        :return: Number of classes in dataset.
        """
        if self.mode is not CLASSIFICATION:
            raise ValueError("Cannot query for number of classes outside of classification mode.")
        if self._num_classes is not None:
            return self._num_classes
        with open(f"{PREPROCESSED_ROOT}/{self.dataset_name}/case_label_mapping.json", "r") as f:
            mapping = json.load(f)
        num_classes = len(set(mapping.values()))
        self._num_classes = num_classes
        return num_classes

    def __len__(self):
        """
        Gets the length of dataset.
        :return: Length of datapoints list.
        """
        return len(self.datapoints)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Datapoint]:
        """
        Loads the data from the index and transforms it.
        :param idx: The data to grab
        :return: The loaded datapoint
        """

        point = self.datapoints[idx]
        image, label = point.get_data(store_metadata=self.store_metadata, )

        if self.transforms is not None:
            result = self.transforms(
                image=image.transpose((1, 2, 0)),
                mask=(label.transpose((1, 2, 0)) if self.mode == SEGMENTATION else None)
            )
            image = result["image"]
            if self.mode == SEGMENTATION:
                label = result["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)
            label = torch.from_numpy(label)

        if self.mode == CLASSIFICATION:
            point.set_num_classes(self._get_number_of_classes())
        return image, label, point
