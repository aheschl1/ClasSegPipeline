from typing import Dict

import numpy as np
from PIL import ImageFile
from overrides import override

from classeg.preprocessing.preprocessor import Preprocessor
from classeg.preprocessing.splitting import Splitter
from classeg.utils.constants import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Extensions require to keep class name the same for proper loading
"""


class ExtensionPreprocessor(Preprocessor):
    """
    A class used to represent an ExtensionPreprocessor

    ...

    Attributes
    ----------
    dataset_id : str
        a formatted string to print out the id of the dataset
    folds : int
        an integer to represent the number of folds to generate
    processes : int
        an integer to represent the number of processes to be used
    normalize : bool
        a boolean to represent whether normalized data should be saved
    dataset_desc : str
        a formatted string to print out the description of the dataset

    Methods
    -------
    get_config()
        Returns the configuration of the preprocessor
    normalize_function(data: np.array)
        Performs normalization on the data
    post_preprocessing()
        Called at the end of preprocessing
    pre_preprocessing()
        Called before standard preprocessing flow
    process()
        Triggers the preprocessing process
    get_folds(k: int)
        Gets random fold at 80/20 split
    process_fold(fold: int)
        Preprocesses a fold
    """

    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, dataset_desc: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        dataset_id : str
            The id of the dataset
        folds : int
            The number of folds to generate
        processes : int
            The number of processes to be used
        normalize : bool
            Whether normalized data should be saved
        dataset_desc : str, optional
            The description of the dataset
        """
        super().__init__(dataset_id, folds, processes, normalize, dataset_desc, **kwargs)

    def get_config(self) -> Dict:
        """
        Returns
        -------
        dict
            The configuration of the preprocessor
        """
        return {
            "batch_size": 32,
            "processes": DEFAULT_PROCESSES,
            "lr": 0.001,
            "epochs": 50,
            "momentum": 0,
            "weight_decay": 0.0001,
            "target_size": [224, 224]
        }

    def normalize_function(self, data: np.array) -> np.array:
        """
        Perform normalization. z-score normalization will still always occur for classification and segmentation

        Parameters
        ----------
        data : np.array
            The data to be normalized

        Returns
        -------
        np.array
            The normalized data
        """
        return data

    def post_preprocessing(self):
        """
        Called at the end of preprocessing
        """
        ...

    @override
    def pre_preprocessing(self):
        """
        Called before standard preprocessing flow
        """
        ...

    def process(self) -> None:
        """
        Triggers the preprocessing process
        """
        super().process()

    def get_folds(self, k: int) -> Dict[int, Dict[str, list]]:
        """
        Gets random fold at 80/20 split. Returns in a map.

        Parameters
        ----------
        k : int
            The number of folds for kfold cross validation

        Returns
        -------
        dict
            The folds map
        """
        splitter = Splitter(self.datapoints, k)
        return splitter.get_split_map()

    def process_fold(self, fold: int) -> None:
        """
        Preprocesses a fold. This method indirectly triggers saving of metadata if necessary,
        writes data to proper folder, and will perform any other future preprocessing.

        Parameters
        ----------
        fold : int
            The fold that we are currently preprocessing
        """
        super().process_fold(fold)
