from typing import Tuple, Union

import numpy as np
import torch

from classeg.dataloading.datapoint import Datapoint as ParentDatapoint


class Datapoint(ParentDatapoint):
    """
    It is recommended that you continue extending ParentDatapoint, but all you really need is get_data()

    If you start writing more complex flows, you may need to get into utils.py
    Specifically, get_datapoint functions
    """
    def __init__(self,
                 im_path: str,
                 label: Union[None, str, torch.Tensor, np.array],
                 dataset_name: str = None,
                 case_name: str = None,
                 writer: str = None,
                 cache: bool = False) -> None:
        super().__init__(im_path, label, dataset_name, case_name, writer, cache)

    @staticmethod
    def standardize(data: np.array) -> np.array:
        return ParentDatapoint.standardize(data)

    def get_data(self, **kwargs) -> Tuple[np.array, np.array]:
        return super().get_data(**kwargs)
