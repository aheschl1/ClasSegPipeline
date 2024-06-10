import importlib
from typing import Union, Any, Tuple, Type

from classeg.dataloading.datapoint import Datapoint
from classeg.dataloading.dataset import PipelineDataset
from classeg.utils.constants import CLASSIFICATION, SEGMENTATION, SELF_SUPERVISED
from classeg.utils.import_utils import get_dataset_mode_from_name, import_from_recursive


class State:
    """
    The State class is a singleton that holds the current state of the application.
    It provides access to the current Datapoint, Dataset, Trainer, Preprocessor, and Inferer classes.
    These classes are determined based on the provided extension and dataset name.
    """
    _datapoint, _dataset = None, None
    _trainer, _preprocessor, _inferer = None, None, None
    _dataset_name, _extension = None, None

    def __init__(self, extension: Union[str, None] = None, dataset_name: str = None):
        """
        Initializes the State with the provided extension and dataset name.

        :param extension: The extension to use.
        :param dataset_name: The name of the dataset.
        """
        State._extension = extension
        State._dataset_name = dataset_name

        State._trainer = get_trainer_from_extension(extension, dataset_name)
        State._preprocessor = get_preprocessor_from_extension(extension, dataset_name)
        State._inferer = get_inferer_from_extension(extension, dataset_name)
        State._datapoint, State._dataset = get_datapoint_and_dataset_from_extension(extension)

    @classmethod
    def getDatapointClass(cls) -> Type[Datapoint]:
        if cls._datapoint is None:
            raise ValueError("You must initialize the state before accessing the Datapoint class.")
        return State._datapoint

    @classmethod
    def getDatasetClass(cls) -> Type[PipelineDataset]:
        if cls._dataset is None:
            raise ValueError("You must initialize the state before accessing the Dataset class.")
        return State._dataset

    @classmethod
    def getTrainerClass(cls):
        if cls._trainer is None:
            raise ValueError("You must initialize the state before accessing the Trainer class.")
        return State._trainer

    @classmethod
    def getPreprocessorClass(cls):
        if cls._preprocessor is None:
            raise ValueError("You must initialize the state before accessing the Preprocessor class.")
        return State._preprocessor

    @classmethod
    def getInfererClass(cls):
        if cls._inferer is None:
            raise ValueError("You must initialize the state before accessing the Inferer class.")
        return State._inferer


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


def get_datapoint_and_dataset_from_extension(
        extension: Union[str, None]
) -> Tuple[Any, Any]:
    """
    Given an extension, returns the inferer class.
    :param extension: The extension to fetch.
    :return: The inferer class.
    """

    from classeg.dataloading.datapoint import Datapoint
    from classeg.dataloading.dataset import PipelineDataset

    if extension is None:
        return Datapoint, PipelineDataset
    try:
        datapoint_class = import_from_recursive(f"classeg.extensions.{extension}.dataloading", "Datapoint")
    except ImportError:
        datapoint_class = Datapoint
    try:
        dataset_class = import_from_recursive(f"classeg.extensions.{extension}.dataloading", "PipelineDataset")
    except ImportError:
        dataset_class = PipelineDataset
    return datapoint_class, dataset_class
