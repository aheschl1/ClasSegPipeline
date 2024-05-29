import os

from src.preprocessing.preprocess_entry import Preprocessor
from torchvision.datasets import MNIST
from PIL import Image
from src.utils.constants import RAW_ROOT
from src.utils.utils import get_case_name_from_number, get_dataset_name_from_id, check_raw_exists
import shutil
import tqdm


class MnistPreprocessor(Preprocessor):
    def __init__(
            self, dataset_id: str, folds: int, processes: int, normalize: bool, **kwargs
    ):
        super().__init__(dataset_id, folds, processes, normalize, **kwargs)
        self._setup_raw(dataset_id)

    def _setup_raw(self, dataset_id):
        check_raw_exists(self.dataset_name)

        dataset = MNIST(".", train=True, download=True)
        case_number = 0
        for image, _ in tqdm.tqdm(dataset, desc="Downloading MNIST"):
            image.save(f"{RAW_ROOT}/{self.dataset_name}/{get_case_name_from_number(case_number)}.jpg")
            case_number += 1
        shutil.rmtree("./MNIST")
