import glob
import importlib
import os
import pkgutil
from typing import Any

from classeg.utils.constants import RAW_ROOT, PREPROCESSED_ROOT, SELF_SUPERVISED, SEGMENTATION, CLASSIFICATION


def import_from_recursive(from_package: str, class_name: str) -> Any:
    module = importlib.import_module(from_package)
    # Iterate through all modules in the package
    for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
        # Import module
        submodule = importlib.import_module(f"{from_package}.{name}")
        # Check if class_name exists in the module
        if hasattr(submodule, class_name):
            return getattr(submodule, class_name)

    # If class is not found in any submodule, raise ImportError
    raise ImportError(f"Class '{class_name}' not found in package '{from_package}'")


def get_dataset_mode_from_name(dataset_name: str):
    """
    Based on raw or preprocessed structure, can determine the dataset mode.
    :param dataset_name:
    :return:
    """
    print(dataset_name)
    raw_root = f"{RAW_ROOT}/{dataset_name}"
    preprocessed_root = f"{PREPROCESSED_ROOT}/{dataset_name}"
    if os.path.exists(raw_root):
        first_level = glob.glob(f"{raw_root}/*")
        if not os.path.isdir(first_level[0]):
            mode = SELF_SUPERVISED
        elif len(first_level) == 2 and "imagesTr" in [first_level[i].split("/")[-1] for i in [0, 1]]:
            mode = SEGMENTATION
        else:
            mode = CLASSIFICATION
    else:
        first_level = glob.glob(f"{preprocessed_root}/*")
        if "id_to_label.json" in [x.split('/')[-1] for x in first_level]:
            mode = CLASSIFICATION
        else:
            second_level = glob.glob(f"{preprocessed_root}/fold_0/train/*")
            if os.path.isdir(second_level[0]):
                mode = SEGMENTATION
            else:
                mode = SELF_SUPERVISED
    return mode
