import glob
import json
import logging
import os
import pkgutil
from contextlib import contextmanager
from typing import Dict, List, Union, Tuple, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classeg.dataloading.datapoint import Datapoint
from classeg.dataloading.dataset import PipelineDataset
from classeg.utils.constants import PREPROCESSED_ROOT, RAW_ROOT, SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED
import importlib


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


def get_trainer_from_extension(extension: Union[str, None], dataset_name: Union[str, None] = None) -> Any:
    """
    Given an extension, returns the trainer class.
    :param extension: The extension to fetch.
    :param dataset_name: The name of the dataset.
    :return: The trainer class.
    """
    if extension is None:
        if dataset_name is None:
            raise ValueError("You must provide either an extension or a dataset name.")
        extension = {
            CLASSIFICATION: "default_class",
            SEGMENTATION: "default_seg",
            SELF_SUPERVISED: "default_ssl"
        }[get_dataset_mode_from_name(dataset_name)]
    module = importlib.import_module(f"classeg.extensions.{extension}")
    trainer_name = getattr(module, "TRAINER_CLASS_NAME")
    trainer_class = import_from_recursive(f"classeg.extensions.{extension}.training", trainer_name)
    return trainer_class


def get_preprocessor_from_extension(extension: Union[str, None], dataset_name: Union[str, None] = None) -> Any:
    """
    Given an extension, returns the preprocessor class.
    :param extension: The extension to fetch.
    :param dataset_name: The name of the dataset.
    :return: The preprocessor class.
    """
    if extension is None:
        if dataset_name is None:
            raise ValueError("You must provide either an extension or a dataset name.")
        extension = {
            CLASSIFICATION: "default_class",
            SEGMENTATION: "default_seg",
            SELF_SUPERVISED: "default_ssl"
        }[get_dataset_mode_from_name(dataset_name)]
    module = importlib.import_module(f"classeg.extensions.{extension}")
    preprocessor_name = getattr(module, "PREPROCESSOR_CLASS_NAME")
    preprocessor_class = import_from_recursive(f"classeg.extensions.{extension}.preprocessing", preprocessor_name)
    return preprocessor_class


def get_inferer_from_extension(extension: Union[str, None], dataset_name: Union[str, None] = None) -> Any:
    """
    Given an extension, returns the inferer class.
    :param extension: The extension to fetch.
    :param dataset_name: The name of the dataset.
    :return: The inferer class.
    """
    if extension is None:
        if dataset_name is None:
            raise ValueError("You must provide either an extension or a dataset name.")
        extension = {
            CLASSIFICATION: "default_class",
            SEGMENTATION: "default_seg",
            SELF_SUPERVISED: "default_ssl"
        }[get_dataset_mode_from_name(dataset_name)]
    module = importlib.import_module(f"classeg.extensions.{extension}")
    inferer_name = getattr(module, "INFERER_CLASS_NAME")
    inferer_class = import_from_recursive(f"classeg.extensions.{extension}.inference", inferer_name)
    return inferer_class


def write_json(data: Union[Dict, List], path: str, create_folder: bool = False) -> None:
    """
    Write helper for json.
    :param data: Dictionary data to be written.
    :param path: The path to write.
    :param create_folder: If the path doesn't exist, should we create folders?
    :return: None
    """
    if not os.path.exists('/'.join(path.split('/')[0:-1])):
        assert create_folder, 'Path does not exist, and you did not indicate create_folder.'
        os.makedirs(path)

    with open(path, 'w') as file:
        file.write(json.dumps(data, indent=4))


def read_json(path: str) -> Dict:
    """
    Read json file.
    :param path:
    :return: Dictionary data of json file.
    """
    with open(path, 'r') as file:
        return json.load(file)


def get_dataset_name_from_id(id: Union[str, int], name: str = None) -> str:
    """
    Given a dataset if that could be xxx or xx or x. Formats dataset id into dataset name.
    :param name:
    :param id: The dataset id
    :return: The dataset name.
    """
    id = str(id)
    id = '0' * (3 - len(id)) + id
    dataset_name = f"Dataset_{id}"
    if name is None:
        preprocessed_folders = [x.split("/")[-1] for x in glob.glob(f"{PREPROCESSED_ROOT}/*") if
                                f"_{id}" in x.split("/")[-1]]
        raw_folders = [x.split("/")[-1] for x in glob.glob(f"{RAW_ROOT}/*") if f"_{id}" in x.split("/")[-1]]
        if len(preprocessed_folders) > 1 or len(raw_folders) > 1:
            raise EnvironmentError(f"Found more than one dataset with id {id}.")
        if len(preprocessed_folders) == 1 and len(raw_folders) == 1:
            if preprocessed_folders[0] != raw_folders[0]:
                raise EnvironmentError(f"Found more than one dataset with id {id}.")
        if len(preprocessed_folders) == 1:
            dataset_name = preprocessed_folders[0]
        elif len(raw_folders) == 1:
            dataset_name = raw_folders[0]
    else:
        dataset_name = dataset_name.replace("_", f"_{name}_")
    return dataset_name


if __name__ == "__main__":
    print(get_dataset_name_from_id(420))


def check_raw_exists(dataset_name: str) -> bool:
    """
    Checks if the raw folder for a given dataset exists.
    :param dataset_name: The name of the dataset to check.
    :return: True if the raw folder exists, False otherwise.
    """
    assert "Dataset_" in dataset_name, f"You passed {dataset_name} to utils/check_raw_exists. Expected a dataset " \
                                       f"folder name."
    if os.path.exists(f"{RAW_ROOT}/{dataset_name}"):
        return False
    os.makedirs(f"{RAW_ROOT}/{dataset_name}")
    return True


def verify_case_name(case_name: str) -> None:
    """
    Verifies that a case is named appropriately.
    If the case is named wrong, crashes the program.
    :param case_name: The name to check.
    :return: None
    """
    assert 'case_' in case_name, f"Invalid case name {case_name} in one of your folders. Case name " \
                                 "should be format case_xxxxx."
    assert len(case_name.split('_')[-1]) >= 5, f"Invalid case name {case_name} in one of your folders. Case name " \
                                               "should be format case_xxxxx, with >= 5 x's"


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


def get_raw_datapoints(dataset_name: str) -> List[Datapoint]:
    """
    Given the name of a dataset, gets a list of datapoint objects.
    :param dataset_name: The name of the dataset.
    :return: List of datapoints in the dataset.
    """

    dataset_root = f"{RAW_ROOT}/{dataset_name}"
    label_to_id_mapping = None
    datapoints = []
    logging.info("Reading dataset paths.")
    mode = get_dataset_mode_from_name(dataset_name)
    if mode == SEGMENTATION:
        sample_paths = glob.glob(f"{dataset_root}/imagesTr/*")
    elif mode == CLASSIFICATION:
        label_to_id_mapping = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/label_to_id.json")
        sample_paths = glob.glob(f"{dataset_root}/*/*", recursive=True)
    else:
        sample_paths = glob.glob(f"{dataset_root}/*")
    logging.info("Paths reading has completed.")
    for path in tqdm(sample_paths, desc="Preparing datapoints"):
        case_name = path.split('/')[-1].split('.')[0]
        verify_case_name(case_name)
        if mode == SEGMENTATION:
            label = path.replace("imagesTr", "labelsTr")
        elif mode == CLASSIFICATION:
            label = label_to_id_mapping[path.split("/")[-2]]
        else:
            label = None

        datapoints.append(Datapoint(path, label, case_name=case_name, dataset_name=dataset_name))

    return datapoints


def get_label_case_mapping_from_dataset(dataset_name: str) -> Dict:
    """
    Given a dataset name looks for a case_label_mapping file.
    :param dataset_name: The name of the dataset.
    :return: case_label_mapping dictionary
    """
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/case_label_mapping.json"
    return read_json(path)


def get_labels_from_raw(dataset_name: str) -> List[str]:
    """
    Given a dataset name checks what labels are in the dataset.
    :param dataset_name: Name of the dataset.
    :return: List of labels.
    """
    path = f"{RAW_ROOT}/{dataset_name}"
    folders = glob.glob(f"{path}/*")
    return [f.split('/')[-1] for f in folders if os.path.isdir(f)]


@contextmanager
def dummy_context():
    class Dummy:
        def __enter__(self):
            ...

        def __exit__(self, exc_type, exc_val, exc_tb):
            ...

        def update(self):
            ...

    yield Dummy()


def get_preprocessed_datapoints(dataset_name: str, fold: int,
                                cache: bool = False, verbose=True) -> Tuple[List[Datapoint], List[Datapoint]]:
    """
    Returns the datapoints of preprocessed cases.
    :param dataset_name:
    :param fold:
    :return: Train points, Val points.
    """

    case_label_mapping = None
    train_root = f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/train"
    val_root = f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/val"
    if verbose:
        logging.info("Reading dataset paths.")
    mode = get_dataset_mode_from_name(dataset_name)
    if mode == SEGMENTATION:
        val_paths = glob.glob(f"{val_root}/imagesTr/*")
        train_paths = glob.glob(f"{train_root}/imagesTr/*")
    else:
        if mode == CLASSIFICATION:
            case_label_mapping = get_label_case_mapping_from_dataset(dataset_name)
        val_paths = glob.glob(f"{val_root}/*")
        train_paths = glob.glob(f"{train_root}/*")

    if verbose:
        logging.info("Paths reading has completed.")
    sample_paths = val_paths + train_paths
    train_datapoints, val_datapoints = [], []
    context = tqdm(total=len(sample_paths), desc=f"Preparing datapoints with{'' if cache else 'out'} cache") if verbose \
        else dummy_context()
    with context as pbar:
        for path in train_paths:
            name = path.split('/')[-1].split('.')[0]
            # -12 is sentinel value for segmentation labels to ensure intention.
            verify_case_name(name)
            if mode == SEGMENTATION:
                label = path.replace("imagesTr", "labelsTr")
            elif mode == CLASSIFICATION:
                label = case_label_mapping[name]
            else:
                label = None
            train_datapoints.append(
                Datapoint(path,
                          label,
                          case_name=name,
                          dataset_name=dataset_name,
                          cache=cache)
            )
            pbar.update()
        for path in val_paths:
            name = path.split('/')[-1].split('.')[0]
            # -12 is sentinel value for segmentation labels to ensure intention.
            verify_case_name(name)
            if mode == SEGMENTATION:
                label = path.replace("imagesTr", "labelsTr")
            elif mode == CLASSIFICATION:
                label = case_label_mapping[name]
            val_datapoints.append(
                Datapoint(path, label, case_name=name, dataset_name=dataset_name, cache=cache)
            )
            pbar.update()

    return train_datapoints, val_datapoints


def get_raw_datapoints_folded(dataset_name: str, fold: int) -> Tuple[List[Datapoint], List[Datapoint]]:
    """
    Given a dataset name, returns the train and val points given a fold.
    :param dataset_name: The name of the dataset.
    :param fold: The fold to fetch.
    :return: Train and val points.
    """
    fold = get_folds_from_dataset(dataset_name)[str(fold)]
    datapoints = get_raw_datapoints(dataset_name)
    train_points, val_points = [], []
    # Now we populate the train and val lists
    for point in datapoints:
        if point.case_name in fold['train']:
            train_points.append(point)
        elif point.case_name in fold['val']:
            val_points.append(point)
        else:
            raise SystemError(f'{point.case_name} was not found in either the train or val fold! Maybe rerun '
                              f'preprocessing.')
    return train_points, val_points


def get_folds_from_dataset(dataset_name: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Fetches and loads the fold json from a dataset.
    :param dataset_name: The name of the dataset.
    :return: The fold dictionary.
    """
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/folds.json"
    return read_json(path)


def get_config_from_dataset(dataset_name: str, config_name: str = 'config', output_dir=None) -> Dict:
    """
    Given a dataset name looks for a config file.
    :param config_name: Name of the config file to load
    :param dataset_name: The name of the dataset.
    :return: Config dictionary.
    """
    if output_dir is not None:
        return read_json(f"{output_dir}/config.json")
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/{config_name}.json"
    return read_json(path)


def batch_collate_fn(batch: List[Tuple[torch.Tensor, Datapoint]]):
    """
    Combines data fetched by dataloader into proper format.
    :param batch: List of data points from loader.
    :return: Batched tensor data, labels, and list of datapoints.
    """
    images, labels = [], []
    points = []

    for images_data, labels_data, point in batch:
        images.append(images_data)
        labels.append(labels_data)
        points.append(point)

    images = torch.stack(images)
    if labels[0] is not None:
        labels = torch.stack(labels)

    return images, labels, points


def get_dataloaders_from_fold(dataset_name: str,
                              fold: int,
                              train_transforms=None,
                              val_transforms=None,
                              preprocessed_data: bool = True,
                              store_metadata: bool = False,
                              config_name="config",
                              cache=False,
                              **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the train and val dataloaders for a specific dataset fold.
    :param config_name:
    :param dataset_name: The name of the dataset.
    :param fold: The fold to grab.
    :param train_transforms: The transforms to apply to training data.
    :param val_transforms: The transforms to apply to val data.
    :param preprocessed_data: If true, grabs the preprocessed data,if false grabs the raw data.
    :param store_metadata: If true, will tell the datapoints reader/writer to save metadata on read.
    :param kwargs: Can overwrite some settings.
    :return: Train and val dataloaders.
    """

    config = get_config_from_dataset(dataset_name, config_name)

    train_points, val_points = get_preprocessed_datapoints(dataset_name, fold, cache=cache) if preprocessed_data \
        else get_raw_datapoints_folded(dataset_name, fold)

    train_dataset = PipelineDataset(train_points, dataset_name, train_transforms,
                                    store_metadata=store_metadata)
    val_dataset = PipelineDataset(val_points, dataset_name, val_transforms,
                                  store_metadata=store_metadata)
    train_sampler, val_sampler = None, None
    if 'sampler' in kwargs and kwargs['sampler'] is not None:
        assert 'rank' in kwargs and 'world_size' in kwargs, \
            "If supplying 'sampler' you must also supply 'world_size' and 'rank'"
        train_sampler = kwargs['sampler'](train_dataset, rank=kwargs['rank'],
                                          num_replicas=kwargs['world_size'], shuffle=True)
        val_sampler = kwargs['sampler'](val_dataset, rank=kwargs['rank'],
                                        num_replicas=kwargs['world_size'], shuffle=False)

    batch_size = kwargs.get('batch_size', config['batch_size'])
    if 'world_size' in kwargs:
        batch_size //= kwargs['world_size']

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['processes'],
        shuffle=train_sampler is None,
        pin_memory=not cache,
        collate_fn=batch_collate_fn,
        sampler=train_sampler,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config['processes'],
        shuffle=False,
        pin_memory=not cache,
        collate_fn=batch_collate_fn,
        sampler=val_sampler,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader


def get_case_name_from_number(c: int) -> str:
    """
    Given a case number, returns the string name.
    :param c: The case number.
    :return: The case name in form case_xxxxx
    """
    c = str(c)
    zeros = '0' * (5 - len(c))
    return f"case_{zeros}{c}"
