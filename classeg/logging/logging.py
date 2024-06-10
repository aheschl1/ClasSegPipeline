import logging
import os
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


class LogHelper:
    def __init__(self, output_dir: str, max_images: int = 10) -> None:
        """
        This class is for the storage and graphing of loss/accuracy data throughout training.
        :param output_dir: Folder on device where graphs should be output.
        """
        assert os.path.exists(
            output_dir
        ), f"Output directory {output_dir} doesn't exist."
        self.output_dir = output_dir
        self.losses_train, self.losses_val = [], []
        self.summary_writer = SummaryWriter(log_dir=f"{self.output_dir}/tensorboard")
        self.epoch = 0
        self.image_step = 0
        self.max_images = max_images

    def epoch_end(
            self, train_loss: float, val_loss: float, learning_rate: float, duration: float
    ) -> None:
        """
        Called at the end of an epoch and updates the lists of data.
        :param duration: duration of epoch
        :param learning_rate: Current learning rate
        :param train_loss: The train loss of the epoch
        :param val_loss: The validation loss from the epoch
        :return: Nothing
        """
        self.summary_writer.add_scalar("Loss/train", train_loss, self.epoch)
        self.summary_writer.add_scalar("Loss/val", val_loss, self.epoch)
        self.summary_writer.add_scalars(
            "Loss/both", {"train": train_loss, "val": val_loss}, self.epoch
        )

        self.summary_writer.add_scalar(
            "Metrics/learning_rate", learning_rate, self.epoch
        )
        self.summary_writer.add_scalar(
            "Peformance/epoch_duration", duration, self.epoch
        )
        self.epoch += 1
        self.summary_writer.flush()

    def plot_confusion_matrix(self, predictions: List, labels: List, class_names, set_name: str = "val"):
        cm = confusion_matrix(labels, predictions, normalize='true', labels=[i for i in range(len(class_names))])
        plt.figure(figsize=(11, 11))
        heatmap = sn.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fig = heatmap.get_figure()

        self.summary_writer.add_figure(f"Metrics/confusion/{set_name}", fig, self.epoch)
        plt.close(fig)
        del fig
        self.summary_writer.flush()

    def log_augmented_image(self, image: Any, mask: Any = None):
        self.summary_writer.add_image("Augmented Images", image, self.image_step)
        if mask is not None:
            self.summary_writer.add_image("Augmented Masks", mask, self.image_step)
        self.image_step += 1
        self.summary_writer.flush()

    def log_net_structure(self, net, *inputs):
        try:
            self.summary_writer.add_graph(net, list(inputs))
            logging.info(f"Logged network structure.")
            print("Logged network structure.")
        except Exception as e:
            logging.info(f"Failed to log network structure: {e}")
            print(f"Failed to log network structure")
        self.summary_writer.flush()

    def log_parameters(self, total, trainable):
        self.summary_writer.add_scalars(
            "Network/params", {"total": total, "trainable": trainable}
        )
        self.summary_writer.flush()

    def log_image_infered(self, image, epoch, **kwargs):
        self.summary_writer.add_image(
            "Infered Images",
            image,
            epoch
        )
        for key, value in kwargs.items():
            self.summary_writer.add_image(
                f"Infered {key}",
                value,
                epoch
            )
        self.summary_writer.flush()

    def log_image_mask(self, image, mask, epoch):
        self.summary_writer.add_image(
            "Image In",
            image,
            epoch
        )
        self.summary_writer.add_image(
            f"Infered mask",
            mask,
            epoch
        )
        self.summary_writer.flush()

    def __del__(self):
        self.summary_writer.flush()
        self.summary_writer.close()
