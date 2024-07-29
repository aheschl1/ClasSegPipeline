from classeg.utils.constants import *
from glob import glob


def get_raw_datasets() -> set:
    return {x.split('/')[-1] for x in glob(f"{RAW_ROOT}/*") if os.path.isdir(x)}


def get_preprocessed_datasets() -> set:
    return {x.split('/')[-1] for x in glob(f"{PREPROCESSED_ROOT}/*") if os.path.isdir(x)}


def get_available_datasets() -> list:
    return list(get_raw_datasets() | get_preprocessed_datasets())
