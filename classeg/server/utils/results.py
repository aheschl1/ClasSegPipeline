import glob
import os

from classeg.utils.constants import RESULTS_ROOT


def get_experiments_from_dataset(dataset_name):
    root = f"{RESULTS_ROOT}/{dataset_name}"
    if not os.path.exists(root):
        return []
    folds = glob.glob(f"{root}/*")
    experiments = []
    for fold in folds:
        results = [x for x in glob.glob(f"{fold}/*") if os.path.isdir(x)]
        for result in results:
            experiment = {
                "name": result.split("/")[-1],
                "fold": fold.split("/")[-1],
                "checkpoints": [x.split('/')[-1].split('.')[0] for x in glob.glob(f"{result}/*") if '.pth' in x]
            }
            experiments.append(experiment)
    return experiments
