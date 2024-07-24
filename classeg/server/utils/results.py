import glob
import json
import os

from classeg.log_analysis.log_analyzer import LogAnalyzer
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


def get_experiment_from_dataset(dataset_name, experiment_name):
    root = f"{RESULTS_ROOT}/{dataset_name}"
    if not os.path.exists(root):
        raise ValueError(f"Dataset {dataset_name} does not exist")
    folds = glob.glob(f"{root}/*")
    for fold in folds:
        if os.path.exists(f"{fold}/{experiment_name}"):
            with open(f"{fold}/{experiment_name}/config.json") as f:
                config = json.load(f)

            with open(f"{fold}/{experiment_name}/logs.txt") as f:
                logs = f.read()

            model = None
            if os.path.exists(f"{fold}/{experiment_name}/model.json"):
                with open(f"{fold}/{experiment_name}/model.json") as f:
                    model = json.load(f)
            analyzer = LogAnalyzer(f"{fold}/{experiment_name}/logs.txt", experiment_name)
            experiment = {
                "name": experiment_name,
                "fold": fold.split("/")[-1],
                "checkpoints": [x.split('/')[-1].split('.')[0] for x in glob.glob(f"{fold}/{experiment_name}/*") if '.pth' in x],
                "config": config,
                "logs": logs,
                "model": model,
                "best_loss": analyzer.best_loss,
                "model_kwargs": analyzer.model_kwargs,
                "untrainable_params": analyzer.untrainable_params,
                "total_params": analyzer.total_params,
                "best_epoch": analyzer.best_epoch,
                "mean_time": analyzer.mean_time,
                "dataset": dataset_name,
                "latest_epoch": analyzer.latest_epoch,
                "train_losses": analyzer.train_losses,
                "val_losses": analyzer.val_losses,
            }
            return experiment
    raise ValueError(f"Experiment {experiment_name} does not exist in dataset {dataset_name}")
