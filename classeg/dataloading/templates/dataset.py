import json
from typing import List, Callable, Tuple

import albumentations
import torch
from torch.utils.data import Dataset

from classeg.dataloading.datapoint import Datapoint
from classeg.dataloading.dataset import PipelineDataset as ParentDataset
from classeg.utils.constants import CLASSIFICATION, PREPROCESSED_ROOT


class PipelineDataset(ParentDataset):
    """
    It is recommended that you continue extending ParentDataset, but all you really need is __len__ and __getitem__

    If you start writing more complex flows, you may need to get into utils.py
    Specifically, get_dataset functions. Also watch out for the return values, regarding the trianing loop, and the batch_collate_fn
    If needed, modify utils batch_collate funciton
    """

    def __init__(self,
                 datapoints: List[Datapoint],
                 dataset_name: str,
                 transforms: Callable = None,
                 store_metadata: bool = False,
                 dataset_type: str = "train"
                 ):
        super().__init__(datapoints, dataset_name, transforms, store_metadata, dataset_type)

    def __len__(self):
        return len(super())

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Datapoint]:
        return super()[idx]
