import shutil

import tqdm
from overrides import override
from torchvision.datasets import MNIST

from pipe.preprocessing.preprocess_entry import Preprocessor
from pipe.utils.constants import RAW_ROOT
from pipe.utils.utils import get_case_name_from_number, check_raw_exists


class MnistPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, **kwargs):
        super().__init__(dataset_id, folds, processes, normalize, **kwargs)

    @override
    def pre_preprocessing(self):
        check_raw_exists(self.dataset_name)
        dataset = MNIST(".", train=True, download=True)
        case_number = 0
        for image, _ in tqdm.tqdm(dataset, desc="Downloading MNIST"):
            image.save(f"{RAW_ROOT}/{self.dataset_name}/{get_case_name_from_number(case_number)}.jpg")
            case_number += 1
        shutil.rmtree("./MNIST")
