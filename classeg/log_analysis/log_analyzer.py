import os
from typing import TextIO
from classeg.log_analysis.utils import clean_line


class LogAnalyzer:
    def __init__(self, log_path: str, time_stamp: str):
        assert os.path.exists(log_path), "Log path does not exist"
        # not constant
        self.config = None
        self.model_kwargs = None
        # constant
        self.time_stamp = time_stamp
        self.untrainable_params = None
        self.total_params = None
        self.loss_fn = None
        self.optim = None
        self.gpu_count = None
        self.best_epoch = None
        self.mean_time = None
        self.best_loss = None
        self.best_accuracy = None
        with open(log_path) as log_file:
            self._parse(log_file)
        self.discriminating_args = {}
        # noinspection PyTypeChecker
        self.discriminating_args.update(self.model_kwargs)
        # noinspection PyTypeChecker
        self.discriminating_args.update(self.config)

    def _parse(self, log_file: TextIO):
        def readline():
            return clean_line(log_file.readline()).strip()

        # config
        log_file.readline()
        config = eval(readline())
        model_kwargs = config.pop("model_args")
        self.config = config
        self.model_kwargs = model_kwargs
        # path reading
        log_file.readline()
        log_file.readline()
        log_file.readline()
        # parameters
        total_params = int(readline().split(':')[-1].strip())
        trainable_params = int(readline().split(':')[-1].strip())
        self.total_params = total_params
        self.untrainable_params = total_params - trainable_params
        # loss
        loss_fn = readline().split(' ')[-1]
        self.loss_fn = loss_fn
        # optim
        optim = readline().split(' ')[-2]
        self.optim = optim
        # GPU count
        gpu_count = 1
        while("INFO") not in log_file.readline(): ...
        while '=' not in readline():
            gpu_count += 1
        self.gpu_count = gpu_count
        # header completed
        # start body
        log_lines = [clean_line(line).strip() for line in log_file.readlines()]
        curr_epoch, best_epoch = 0, 0
        best_loss = 90909090
        times = []
        for line in log_lines:
            if '...' in line:
                # epoch announcement line
                curr_epoch = int(line.split(' ')[-2].split('/')[0])
            elif 'Val loss' in line:
                val_loss = float(line.split(' ')[-3][0:7])
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = curr_epoch
            elif 'Process 0 took' in line:
                # epoch time
                time = float(line.split(' ')[-2][0:7])
                times.append(time)
        # parsing completed
        self.best_epoch = best_epoch
        self.mean_time = sum(times)/len(times)
        self.best_loss = best_loss


if __name__ == "__main__":
    assert False, "Enter the program through the 'analysis_entry.py' file."
