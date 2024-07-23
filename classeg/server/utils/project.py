import glob
import os

import numpy as np
from PIL import Image

from classeg.dataloading.datapoint import Datapoint
from classeg.utils.constants import *
from classeg.utils.utils import (
    get_dataset_mode_from_name, get_case_name_from_number, get_datapoint_from_dataset_and_case
)

class Project:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.mode = get_dataset_mode_from_name(dataset_name)

    def get_experiment_count(self):
        root = f"{RESULTS_ROOT}/{self.dataset_name}"
        if not os.path.exists(root):
            return 0
        folds = glob.glob(f"{root}/*")
        count = 0
        for fold in folds:
            results = glob.glob(f"{fold}/*")
            count += len(results)
        return count

    def get_sample_point(self, case: int, preprocessed: bool):
        return get_datapoint_from_dataset_and_case(self.dataset_name, case, preprocessed=preprocessed)

    def get_raw(self, case: int):
        return self.get_sample_point(case, preprocessed=False)

    def get_preprocessed(self, case: int):
        return self.get_sample_point(case, preprocessed=True)

    def to_dict(self):
        preprocessed_available = os.path.exists(f"{PREPROCESSED_ROOT}/{self.dataset_name}")
        return {
            "name": self.dataset_name,
            "type": self.mode,
            "preprocessed_available": preprocessed_available,
            "raw_available": os.path.exists(f"{RAW_ROOT}/{self.dataset_name}"),
            "experiment_count": self.get_experiment_count(),
        }
