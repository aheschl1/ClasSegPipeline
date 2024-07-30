import glob
import os.path

from classeg.ui_server.utils.results import get_experiments_from_dataset, get_experiment_from_dataset
from classeg.utils.constants import *
from classeg.utils.utils import (
    get_dataset_mode_from_name, get_datapoint_from_dataset_and_case
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
            results = [x for x in glob.glob(f"{fold}/*") if os.path.isdir(x)]
            count += len(results)
        return count

    def get_experiments(self):
        return get_experiments_from_dataset(self.dataset_name)

    def get_experiment(self, experiment_id):
        return get_experiment_from_dataset(self.dataset_name, experiment_id)

    def get_sample_point(self, case: int, preprocessed: bool):
        return get_datapoint_from_dataset_and_case(self.dataset_name, case, preprocessed=preprocessed)

    def get_raw(self, case: int):
        return self.get_sample_point(case, preprocessed=False)

    def get_preprocessed(self, case: int):
        return self.get_sample_point(case, preprocessed=True)

    def get_configs(self):
        if not os.path.exists(f"{PREPROCESSED_ROOT}/{self.dataset_name}"):
            return []
        return [x.split('.')[0].split('/')[-1] for x in glob.glob(f"{PREPROCESSED_ROOT}/{self.dataset_name}/*") if
                ".json" in x and 'id_to_label' not in x and 'label_to_id' not in x and 'folds' not in x and 'case_label_mapping' not in x]

    def to_dict(self):
        preprocessed_available = os.path.exists(f"{PREPROCESSED_ROOT}/{self.dataset_name}")
        return {
            "name": self.dataset_name,
            "type": self.mode,
            "preprocessed_available": preprocessed_available,
            "raw_available": os.path.exists(f"{RAW_ROOT}/{self.dataset_name}"),
            "experiment_count": self.get_experiment_count(),
            "configs": self.get_configs()
        }
