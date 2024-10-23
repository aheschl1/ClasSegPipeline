import json
from typing import List, Callable, Tuple

import albumentations
import torch
from torch.utils.data import Dataset

from classeg.dataloading.datapoint import Datapoint
from classeg.utils.constants import CLASSIFICATION, PREPROCESSED_ROOT


class PipelineDataset(Dataset):

    def __init__(self,
                 datapoints: List[Datapoint],
                 dataset_name: str,
                 transforms: Callable = None,
                 store_metadata: bool = False,
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
        self.dataset_type = dataset_type
        self.mode = self.datapoints[0].mode
        self._num_classes = None

    def _get_number_of_classes(self):
        """
        Checks how many classes there are based on classes of datapoints.
        :return: Number of classes in dataset.
        """
        if self._num_classes is not None:
            return self._num_classes
        if self.mode is not CLASSIFICATION:
            raise ValueError("Cannot query for number of classes outside of classification mode.")
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
        # [C, ...]

        if self.transforms is not None:
            if isinstance(self.transforms, albumentations.Compose):
                # Albumentations takes [H, W, C]
                if label is not None:
                    result = self.transforms(
                        image=image.transpose((1, 2, 0)),
                        mask=label.transpose((1, 2, 0))
                    )
                else:
                    result = self.transforms(image=image.transpose((1, 2, 0)))
                image = result["image"].transpose((2, 0, 1))
                label = None if label is None else result["mask"].transpose((2, 0, 1))
            else:
                # Torchvision and monai transforms take [C, ...]
                image = self.transforms(torch.from_numpy(image))

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(label, torch.Tensor) and label is not None:
            label = torch.from_numpy(label)

        if self.mode == CLASSIFICATION:
            point.set_num_classes(self._get_number_of_classes())
        return image, label, point
