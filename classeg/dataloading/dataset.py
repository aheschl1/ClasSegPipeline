from typing import List, Callable, Tuple

import albumentations
import torch
from torch.utils.data import Dataset
from classeg.dataloading.datapoint import Datapoint
import glob
from classeg.utils.constants import RAW_ROOT, CLASSIFICATION, SEGMENTATION, SELF_SUPERVISED, PREPROCESSED_ROOT
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

        # TODO take a closer look at how we can handle dimension mismatch!
        # Preprocess saves arrays as channel first, but for inference, we don't really know
        # TODO Maybe for inference, we can preprocess first :thinking:
        if image.shape[-1] in [1, 3]:
            # perhaps it is still channel last?
            image = image.transpose((2, 0, 1))
        if label is not None and len(label.shape) > 1 and label.shape[-1] in [1]:
            label = label.transpose((2, 0, 1))

        if self.transforms is not None:
            if isinstance(self.transforms, albumentations.Compose):
                print(image.shape, label.shape)
                result = self.transforms(
                    image=image.transpose((1, 2, 0)),
                    mask=label.transpose((1, 2, 0))
                )
                image = result["image"]
                label = result["mask"]
            else:
                image = self.transforms(torch.from_numpy(image))

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)
        if not isinstance(label, torch.Tensor) and label is not None:
            label = torch.from_numpy(label)

        if self.mode == CLASSIFICATION:
            point.set_num_classes(self._get_number_of_classes())
        return image, label, point
