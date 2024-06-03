import sys
sys.path.append("/home/student/andrewheschl/Documents/diffusion")
import click
from classeg.utils.constants import *
from classeg.utils.utils import get_dataset_name_from_id
import glob
from classeg.log_analysis.log_analyzer import LogAnalyzer
import pandas as pd


class Analysis:
    def __init__(self, folders, output_root: str):
        """
        Given a list of folders (expected to be folders inside a fold folder in result root),
        controls the analysis of logs.
        """
        self.output_root = output_root
        print("Reading logs...")
        self.logs = Analysis._get_logs(folders)
        grouped_analyzers = {}
        # Group the analyzers based on their discriminating_args
        for analyzer in self.logs:
            key = frozenset(analyzer.discriminating_args.keys())
            if key in grouped_analyzers:
                grouped_analyzers[key].append(analyzer)
            else:
                grouped_analyzers[key] = [analyzer]
        # Convert the dictionary values to a list of lists
        self.grouped_logs = list(grouped_analyzers.values())

    @staticmethod
    def _get_logs(folders):
        """
        Given a parent folder (fold_x folder), finds all log files recursively.
        Creates LogAnalyzer for each and saves to property. 
        """
        logs = []
        for folder in folders:
            time_stamp = folder.split('/')[-1]
            log_path = f"{folder}/logs.txt"
            if not os.path.exists(log_path):
                print(f"No log file found in {folder}.")
            try:
                logs.append(LogAnalyzer(log_path, time_stamp))
            except Exception as e:
                print(f"Skipping {log_path}. Likely incomplete log.")
        return logs

    def analyze(self):
        """
        Performs analysis on logs which which been processed from constructor.
        Saves to output folder.
        """
        frames = [Analysis._analyze_group(group) for group in self.grouped_logs]
        print(f"Saving results to {self.output_root}.")
        for i, frame in enumerate(frames):
            frame.to_csv(f"{self.output_root}/log_result_{i}.csv", index=False)

    @staticmethod
    def _analyze_group(logs):
        constant_columns = ['Best Accuracy', 'Best Loss', 'Best Epoch', 'Mean Time (S)',
                            'Total Params', 'Untrainable Params', 'Criterion', 'Optimizer', 'Timestamp', 'GPU Count']
        other_columns = list(logs[0].discriminating_args.keys())
        columns = constant_columns + other_columns
        df = pd.DataFrame(columns=columns)
        for i, log in enumerate(logs):
            row = {'Best Accuracy': log.best_accuracy,
                   'Best Loss': log.best_loss,
                   'Best Epoch': log.best_epoch,
                   'Mean Time (S)': log.mean_time,
                   'Total Params': log.total_params,
                   'Untrainable Params': log.untrainable_params,
                   'Criterion': log.loss_fn,
                   'Optimizer': log.optim,
                   'Timestamp': log.time_stamp,
                   'GPU Count': log.gpu_count
                   }
            row.update(log.discriminating_args)
            df.loc[i] = row
        return df


@click.command()
@click.option('-fold', '-f', help='Which fold to train.', type=int, required=True)
@click.option('-dataset_id', '-d', help='The dataset id to train.', type=str, required=True)
@click.option('-output_root', '-o',
              help='Root for saving results. If not specified, uses the result folder', type=str, required=False)
def main(fold, dataset_id, output_root):
    result_root = f"{RESULTS_ROOT}/{get_dataset_name_from_id(dataset_id)}/fold_{fold}"
    if output_root is None:
        output_root = result_root
    assert os.path.exists(output_root), "Output root does not exist."
    assert os.path.isdir(output_root), "Output root must be a folder not a file."
    trial_folders = [folder for folder in glob.glob(f"{result_root}/*") if os.path.isdir(folder)]
    analyzer = Analysis(trial_folders, output_root)
    analyzer.analyze()


if __name__ == "__main__":
    main()
