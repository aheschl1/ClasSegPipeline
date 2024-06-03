import torch

from classeg.dataloading.datapoint import Datapoint
from classeg.inference.inferer import Inferer


class SelfSupervisedInferer(Inferer):
    def __init__(self,
                 dataset_name: str,
                 fold: int,
                 name: str,
                 weights: str,
                 input_root: str):
        """
        Inferer for pipeline.
        :param dataset_name: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_name, fold, name, weights, input_root)

    def get_augmentations(self):
        ...

    def infer_single_sample(self, image: torch.Tensor, datapoint: Datapoint) -> None:
        """
        image: single sample batch which has gone through the augmentations

        handle the result in fields
        """
        ...

    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        ...

