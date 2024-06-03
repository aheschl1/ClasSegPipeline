import importlib
from typing import List

import click

from classeg.inference.default_inferers.classification_inferer import ClassificationInferer
from classeg.inference.default_inferers.segmentation_inferer import SegmentationInferer
from classeg.inference.default_inferers.self_supervised_inferer import SelfSupervisedInferer
from classeg.utils.constants import CLASSIFICATION, SEGMENTATION, SELF_SUPERVISED
from classeg.utils.utils import get_dataset_name_from_id, get_dataset_mode_from_name, import_from_recursive


@click.command()
@click.option("-dataset_id", "-d", required=True)  # 10
@click.option("-fold", "-f", required=True, type=int)  # 0
@click.option("-name", "-n", required=True)
@click.option("-input_folder", "-i", required=True, help="Path to the input data folder")
@click.option("-weights", "-w", default="best")
@click.option("-extension", "-ext", help="Name of the extension to load for inference")
@click.option("-dataset_desc", "-dd", required=False,
              help="Description of dataset. Useful if you have overlapping ids.")  # 10
@click.argument('extra_args', nargs=-1)
def main(dataset_id: str, fold: int, name: str, input_folder: str, weights: str, extension: str,
         dataset_desc: str, extra_args: List[str]) -> None:

    kwargs = {}
    for arg in extra_args:
        if "=" not in arg:
            raise ValueError(
                "For inference, all positional arguments must contain '='. They are used for passing arguments to "
                "extension inferer.")
        key, value = arg.split('=')
        kwargs[key] = value

    dataset_name = get_dataset_name_from_id(dataset_id, name=dataset_desc)
    mode = get_dataset_mode_from_name(dataset_name)

    if extension is not None:
        module = importlib.import_module(f"classeg.extensions.{extension}")
        inferer_name = getattr(module, "INFERER_CLASS_NAME")
        inferer_class = import_from_recursive(f"pipe.extensions.{extension}.inference", inferer_name)
    else:
        inferer_class = {
            CLASSIFICATION: ClassificationInferer,
            SEGMENTATION: SegmentationInferer,
            SELF_SUPERVISED: SelfSupervisedInferer
        }[mode]

    inferer = inferer_class(dataset_name, fold, name, weights, input_folder, **kwargs)
    inferer.infer()
    print("Completed inference!")


if __name__ == "__main__":
    main()
