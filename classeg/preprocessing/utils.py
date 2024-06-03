import glob
import os
import shutil
from typing import List

from classeg.utils.constants import PREPROCESSED_ROOT, RAW_ROOT


def maybe_make_preprocessed(dataset_name: str, query_overwrite: bool = True) -> None:
    """
    Checks if the preprocessed folder should be made for a dataset. Makes it if needed.
    :param dataset_name: Dataset name
    :param query_overwrite: Whether to ask before overwriting.
    :return: None
    """
    target_folder = f"{PREPROCESSED_ROOT}/{dataset_name}"
    if os.path.exists(target_folder):
        remove = input(f"Warning! You are about to overwrite the existing path {target_folder}. Continue? (y/n): ") \
                 == "y" if query_overwrite else True
        if not remove:
            print("Killing program.")
            raise SystemExit
        print("Clearing the preprocessed folder. This may take a while depending on how large the dataset is...")
        shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder, exist_ok=True)
