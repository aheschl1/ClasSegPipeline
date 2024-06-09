from typing import List

import click

from classeg.utils.constants import *
from classeg.utils.utils import get_preprocessor_from_extension, get_dataset_name_from_id


@click.command()
@click.option("-folds", "-f", help="How many folds should be generated.", type=int, required=True)
@click.option("-processes", "-p", help="How many processes can be used.", type=int, default=DEFAULT_PROCESSES)
@click.option("--normalize", "--n", help="Should we compute and save normalized data.", type=bool, is_flag=True, )
@click.option("-dataset_id", "-d", help="The dataset id to work on.", type=str, required=True)
@click.option("-dataset_description", "-dd", help="Short description/dataset name", type=str, default=None)
@click.option("-extension", "-ext", help="Name of the extension you want to use. Default behavior is available")
@click.argument('extra_args', nargs=-1)
def main(folds: int, processes: int, normalize: bool, dataset_id: str, extension: str, dataset_description: str, extra_args: List[str]):
    kwargs = {}
    for arg in extra_args:
        if "=" not in arg:
            raise ValueError("For preprocessing, all positional arguments must contain '='. They are used for passing arguments to extension preprocessors.")
        key, value = arg.split('=')
        kwargs[key] = value

    preprocessor = get_preprocessor_from_extension(extension, get_dataset_name_from_id(dataset_id, dataset_description))
    preprocessor = preprocessor(
        dataset_id=dataset_id, normalize=normalize, folds=folds, processes=processes, dataset_desc=dataset_description, **kwargs
    )
    preprocessor.process()
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
