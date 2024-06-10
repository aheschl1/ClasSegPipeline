import click
import torch
import torch.nn as nn

from classeg.dataloading.datapoint import Datapoint
from classeg.inference.inferer import Inferer
from classeg.utils.constants import PREPROCESSED_ROOT
from classeg.utils.utils import read_json, write_json
import torchvision.transforms as transforms


class ClassificationInferer(Inferer):
    def __init__(self,
                 dataset_name: str,
                 fold: int,
                 name: str,
                 weights: str,
                 input_root: str):
        """
        Inferer for pipeline.
        :param dataset_name: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_name, fold, name, weights, input_root)
        assert input_root is not None, "Provide input root argument for classification inference"
        self.results = {}
        self.id_to_label = read_json(f"{PREPROCESSED_ROOT}/{self.dataset_name}/id_to_label.json")

    def get_augmentations(self):
        mean_data = read_json(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{self.fold}/mean_std.json")
        return transforms.Compose([
            transforms.Resize(self.config.get('target_size', [512, 512]), antialias=True),
            transforms.Normalize(mean=mean_data["mean"], std=mean_data["std"])
        ])

    def infer_single_sample(self, image: torch.Tensor, datapoint: Datapoint) -> None:
        image = image.to(self.device)
        predictions = nn.Softmax(dim=1)(self.model(image))[0]
        predicted_class = torch.argmax(predictions).detach().item()
        self.results[datapoint.im_path] = self.id_to_label[str(predicted_class)]

    def post_infer(self):
        write_json(self.results, f"{self.save_path}/results.json")


@click.command()
@click.option('-dataset_id', '-d', required=True)
@click.option('-fold', '-f', required=True, type=int)
@click.option('-result_folder', '-r', required=True)
@click.option('-data_path', '-data', required=True)
@click.option('-weights', '-w', default='best')
def main(dataset_id: str, fold: int, result_folder: str, data_path: str, weights: str) -> None:
    inferer = Inferer(dataset_id, fold, result_folder, data_path, weights)
    inferer.infer()


if __name__ == "__main__":
    main()
