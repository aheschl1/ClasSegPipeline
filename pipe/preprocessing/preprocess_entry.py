import warnings
from typing import Type, List

import click

from pipe.preprocessing.preprocessor import Preprocessor
from pipe.utils.constants import *
from pipe.utils.utils import import_from


def get_preprocessor_from_name(name: str) -> Type[Preprocessor]:
    if name is None:
        return Preprocessor
    try:
        return import_from("pipe.extensions.custom_preprocessors", name)
    except ImportError as e:
        print(e)
        print(f"Ensure you create the {name} class inside the custom_preprocessors package.")
        raise SystemExit


@click.command()
@click.option("-folds", "-f", help="How many folds should be generated.", type=int)
@click.option("-processes", "-p", help="How many processes can be used.", type=int, default=DEFAULT_PROCESSES)
@click.option("--normalize", "--n", help="Should we compute and save normalized data.", type=bool, is_flag=True, )
@click.option("-dataset_id", "-d", help="The dataset id to work on.", type=str)
@click.option("-dataset_desc", "-dd", help="The dataset name", type=str, default=None)
@click.option("-preprocessor", help="Identifier of the preprocessor you want to use. Default preprocessor is available")
@click.argument('extra_args', nargs=-1)
def main(folds: int, processes: int, normalize: bool, dataset_id: str, preprocessor: str, dataset_desc: str, extra_args: List[str]):
    kwargs = {}
    for arg in extra_args:
        if "=" not in arg:
            raise ValueError("For preprocessing, all positional arguments must contain '='. They are used for passing arguments to extension preprocessors.")
        key, value = arg.split('=')
        kwargs[key] = value

    preprocessor = get_preprocessor_from_name(preprocessor)
    preprocessor = preprocessor(
        dataset_id=dataset_id, normalize=normalize, folds=folds, processes=processes, dataset_desc=dataset_desc, **kwargs
    )
    preprocessor.process()
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
