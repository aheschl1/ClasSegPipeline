import glob

from classeg.utils.utils import (
    get_dataset_mode_from_name,
    get_config_from_dataset,
    get_folds_from_dataset
)
from classeg.utils.constants import *
import os


class Project:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

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

    def to_dict(self):
        preprocessed_available = os.path.exists(f"{PREPROCESSED_ROOT}/{self.dataset_name}")
        return {
            "name": self.dataset_name,
            "type": get_dataset_mode_from_name(self.dataset_name),
            "preprocessed_available": preprocessed_available,
            "raw_available": os.path.exists(f"{RAW_ROOT}/{self.dataset_name}"),
            "experiment_count": self.get_experiment_count(),
        }
