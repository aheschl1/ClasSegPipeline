import torch
import torchvision.transforms as transforms

from classeg.dataloading.datapoint import Datapoint
from classeg.inference.inferer import Inferer
from classeg.utils.constants import PREPROCESSED_ROOT
from classeg.utils.utils import read_json


class SegmentationInferer(Inferer):
    def __init__(self,
                 dataset_id: str,
                 fold: int,
                 name: str,
                 weights: str,
                 input_root: str):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_id, fold, name, weights, input_root)

    def get_augmentations(self):
        mean_data = read_json(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{self.fold}/mean_std.json")
        return transforms.Compose([
            transforms.Resize(self.config["input_shape"]),
            transforms.Normalize(mean=mean_data["mean"], std=mean_data["std"])
        ])

    def infer_single_sample(self, image: torch.Tensor, datapoint: Datapoint) -> None:
        image = image.to(self.device)
        prediction = self.model(image)[0]
        segmentation = torch.argmax(prediction, dim=0).detach().cpu()
        torch.save(segmentation, f"{self.save_path}/{datapoint.im_path.split('/')[-1].split('.')[0]}.pth")
