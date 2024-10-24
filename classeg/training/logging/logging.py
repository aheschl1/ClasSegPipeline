import logging
import os
from abc import abstractmethod
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import wandb
import socket
import uuid

from classeg.utils.constants import WANDB_ENTITY, WANDB_API_KEY


class Logger:
    def __init__(self, output_dir: str, current_epoch: int = 0) -> None:
        """
        This class is for the storage and graphing of loss/accuracy data throughout training.
        :param output_dir: Folder on device where graphs should be output.
        """
        assert os.path.exists(
            output_dir
        ), f"Output directory {output_dir} doesn't exist."
        self.output_dir = output_dir
        self.losses_train, self.losses_val = [], []
        self.epoch = current_epoch

    def set_current_epoch(self, epoch):
        self.epoch = epoch

    def epoch_end(self, train_loss: float, val_loss: float, learning_rate: float, duration: float) -> None:
        """
        Called at the end of an epoch and updates the lists of data.
        :param duration: duration of epoch
        :param learning_rate: Current learning rate
        :param train_loss: The train loss of the epoch
        :param val_loss: The validation loss from the epoch
        :return: Nothing
        """
        self.epoch += 1

    @abstractmethod
    def plot_confusion_matrix(self, predictions: List, labels: List, class_names, set_name: str = "val"):
        raise NotImplementedError("Method not implemented in parent class.")

    @staticmethod
    def build_confusion_matrix(predictions: List, labels: List, class_names) -> plt.figure:
        cm = confusion_matrix(labels, predictions, normalize='true', labels=[i for i in range(len(class_names))])
        plt.figure(figsize=(11, 11))
        heatmap = sn.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fig = heatmap.get_figure()
        return fig

    @abstractmethod
    def log_augmented_image(self, image: Any, mask: Any = None):
        raise NotImplementedError("Method not implemented in parent class.")

    @abstractmethod
    def log_net_structure(self, net, *inputs):
        raise NotImplementedError("Method not implemented in parent class.")

    @abstractmethod
    def log_parameters(self, total, trainable):
        raise NotImplementedError("Method not implemented in parent class.")

    @abstractmethod
    def log_graph(self, points: List[Tuple[float, float]], title="2D Graph"):
        raise NotImplementedError("Method not implemented in parent class.")

    @abstractmethod
    def log_image_infered(self, image, **masks):
        raise NotImplementedError("Method not implemented in parent class.")

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError("Method not implemented in parent class.")
    
    @abstractmethod
    def log_scalar(self, data, title):
        raise NotImplementedError("Method not implemented in parent class.")

    def __del__(self):
        self.cleanup()


class TensorboardLogger(Logger):
    def __init__(self, output_dir: str, max_images: int = 10, current_epoch=0) -> None:
        """
        This class is for the storage and graphing of loss/accuracy data throughout training.
        :param output_dir: Folder on device where graphs should be output.
        """
        super().__init__(output_dir, current_epoch)
        self.summary_writer = SummaryWriter(log_dir=f"{self.output_dir}/tensorboard")
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
        self.summary_writer.flush()

    def log_histogram(self, data, title):
        self.summary_writer.add_histogram(title, data, self.epoch)
        self.summary_writer.flush()

    def plot_confusion_matrix(self, predictions: List, labels: List, class_names, set_name: str = "val"):
        fig = Logger.build_confusion_matrix(predictions, labels, class_names)

        self.summary_writer.add_figure(f"Metrics/confusion/{set_name}", fig, self.epoch)
        plt.close(fig)
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

    def log_graph(self, points: List[Tuple[float, float]], title="2D Graph"):
        fig = plt.figure()
        x_values, y_values = zip(*points)
        plt.plot(x_values, y_values)
        self.summary_writer.add_figure(title, fig, self.epoch)
        plt.close(fig)
        self.summary_writer.flush()

    def log_image_infered(self, image, **masks):
        self.summary_writer.add_image(
            "Infered Images",
            image,
            self.epoch
        )
        for key, value in masks.items():
            self.summary_writer.add_image(
                f"Infered {key}",
                value,
                self.epoch
            )
        self.summary_writer.flush()

    def cleanup(self):
        self.summary_writer.flush()
        self.summary_writer.close()

    def log_scalar(self, data, title):
        self.summary_writer.add_scalar(title, data, self.epoch)
        self.summary_writer.flush()


def isOnline():
    try:
        s = socket.create_connection(("www.google.ca", 80))
        if s is not None:
            s.close()
        return True
    except OSError:
        pass
    return False


class WandBLogger(Logger):

    def __init__(self, output_dir: str, current_epoch=0, dataset_name=None, config=None) -> None:
        super().__init__(output_dir, current_epoch)
        resume = False
        if os.path.exists(f"{output_dir}/.wandb_id.txt"):
            with open(f"{output_dir}/.wandb_id.txt", "r") as f:
                resume = True
                wandb_id = str(f.read()).strip()
        else:
            wandb_id = str(uuid.uuid4())
            with open(f"{output_dir}/.wandb_id.txt", "w") as f:
                f.write(wandb_id)

        name = output_dir.split("/")[-1]
        try:
            wandb.require("core")
        except:
            ...
            
        wandb.login(
            key=WANDB_API_KEY
        )
        wandb.init(
            project=dataset_name,
            dir=f"{output_dir}",
            name=name,
            id=wandb_id,
            resume="must" if resume else None,
            config=config,
            mode="online" if isOnline() else "offline",
            entity=WANDB_ENTITY
        )
        self.has_logged_net = False

    def epoch_end(self, train_loss: float, val_loss: float, learning_rate: float, duration: float) -> None:
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "epoch_duration": duration,
            "epoch": self.epoch
        }, step=self.epoch)
        super().epoch_end(train_loss, val_loss, learning_rate, duration)

    def plot_confusion_matrix(self, predictions: List, labels: List, class_names, set_name: str = "val"):
        wandb.log({
            f"confusion_matrix_{set_name}": wandb.plot.confusion_matrix(
                y_true=labels,
                preds=predictions,
                class_names=class_names
            )
        }, step=self.epoch)

    def log_augmented_image(self, image: Any, mask: Any = None, name="augmented_image"):
        data = {
            name: wandb.Image(
                image,
                masks=None if mask is None else {"augmented_mask": {"mask_data": mask}},
            ),
        }
        wandb.log(data, step=self.epoch)

    def log_histogram(self, data:dict, title):
        wandb.log({
            title: wandb.Histogram(sequence=data)
        }, step=self.epoch)

    def log_net_structure(self, net, *inputs):
        if not self.has_logged_net:
            self.has_logged_net = True
            wandb.watch(net, log="all", log_freq=1000, log_graph=True)
        else:
            logging.info("Network structure logging skipped. already logged.")

    def log_parameters(self, total, trainable):
        wandb.log({
            "total_params": total,
            "trainable_params": trainable
        }, step=self.epoch)

    def log_graph(self, points: List[Tuple[float, float]], x="x", y="y", title="2D Graph"):
        table = wandb.Table(data=points, columns=[x, y])
        wandb.log({
            title: wandb.plot.line(table, x=x, y=y, title=title)
        }, step=self.epoch)

    def log_image_infered(self, image, **masks):
        wandb.log({
            "infered_image": wandb.Image(image, masks={k:{"mask_data":v} for k,v in masks.items()})
        }, step=self.epoch)


    def log_image(self, title, image, **masks):
        wandb.log({
            title: wandb.Image(image, masks={k:{"mask_data":v} for k,v in masks.items()})
        }, step=self.epoch)

    def cleanup(self):
        wandb.finish()

    def log_scalar(self, data, title):
        wandb.log({
            title: data
        }, step=self.epoch)
