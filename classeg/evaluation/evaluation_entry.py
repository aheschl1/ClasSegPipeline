
import sys

from tqdm import tqdm

from classeg.evaluation.utils import INCEPTION_SCORE
from classeg.inference.inference_entry import Inferer

# Adds the source to path for imports and stuff
sys.path.append("/home/andrew.heschl/Documents/diffusion")
sys.path.append("/home/andrewheschl/PycharmProjects/diffusion")
sys.path.append("/home/student/andrew/Documents/diffusion")
sys.path.append("/home/student/andrewheschl/Documents/diffusion")
import numpy as np
import os.path
import click
import torch
import torch.nn as nn

from classeg.utils.constants import *
from classeg.utils.utils import get_dataset_name_from_id, get_forward_diffuser_from_config
from classeg.utils.utils import read_json


class Evaluator:
    def __init__(self,
                 dataset_id: str,
                 fold: int,
                 result_folder: str,
                 weights: str
                 ):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param result_folder: The folder with the trained weights and config.
        :param weights: The name of the weights to load.
        """
        self.dataset_name = get_dataset_name_from_id(dataset_id)
        self.lookup_root = f"{RESULTS_ROOT}/{self.dataset_name}/fold_{fold}/{result_folder}"
        assert os.path.exists(self.lookup_root)
        assert torch.cuda.is_available(), "No gpu available."

        self.fold = fold
        self.config = read_json(f"{self.lookup_root}/config.json")
        self.weights = weights
        self.device = "cuda"
        self.infer = Inferer(dataset_id, fold, result_folder, weights)

        self.metrics = [INCEPTION_SCORE]


    def evaluate(self):
        self





@click.command()
@click.option('-dataset_id', '-d', required=True)  # 10
@click.option('-fold', '-f', required=True, type=int)  # 0
@click.option('-name', '-n', required=True)  # 2024_02_02_16_50_472684
@click.option('-weights', '-w', default='best')
def main(dataset_id: str, fold: int, name: str, weights: str) -> None:
    import matplotlib.pyplot as plt

    evaluator = Evaluator(dataset_id, fold, name, weights)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
