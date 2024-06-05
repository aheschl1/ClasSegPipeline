import copy
import os.path
import warnings
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple, Union

import numpy as np
import torch
from classeg.utils.constants import SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED
from classeg.utils.normalizer import get_normalizer_from_extension
from classeg.utils.reader_writer import get_reader_writer, get_reader_writer_from_extension


class Datapoint:
    def __init__(self,
                 im_path: str,
                 label: Union[None, str, torch.Tensor],
                 dataset_name: str = None,
                 case_name: str = None,
                 writer: str = None,
                 cache: bool = False) -> None:
        """
        Initialize a Datapoint object. This object supports caching and determines its own read/write and normalizer.

        Parameters:
        im_path (str): The path to the image.
        label (Union[None, str, torch.Tensor]): Either a class label, a path to a segmentation, or None.
        dataset_name (str, optional): The name of the dataset. Example: Dataset_000. Defaults to None.
        case_name (str, optional): The name of the case. Example: case_00000. Defaults to None.
        writer (str, optional): Overwrite writer class. Defaults to None.
        cache (bool, optional): If enabled, stores data in shared memory. Defaults to False.
        """
        self._shared_mem = None
        self.im_path = im_path
        self.label = label
        if label is None:
            self.mode = SELF_SUPERVISED
        elif isinstance(label, str) and os.path.exists(label) and not os.path.isdir(label):
            self.mode = SEGMENTATION
        else:
            self.mode = CLASSIFICATION
        self.cache = cache
        if cache:
            warnings.filterwarnings("ignore", category=UserWarning)

        self.num_classes = None
        self._shared_mem: Union[SharedMemory, None] = None

        self._shape = None
        self._dtype = None
        # reader
        if writer is not None:
            self.reader_writer = get_reader_writer(writer)
        else:
            self.reader_writer = get_reader_writer_from_extension(self.extension)
        self.normalizer = get_normalizer_from_extension(self.extension)
        self.reader_writer = self.reader_writer(case_name=case_name, dataset_name=dataset_name)
        self._case_name = case_name

        if cache:
            self._shared_mem = self._cache(self._standardize(self.reader_writer.read(self.im_path)))

    def _standardize(self, data: np.array) -> np.array:
        """
        Standardize the data. If the data shape is missing a channel, it is added. If the data shape does not match
        the reader/writer's image dimensions, a ValueError is raised.

        Parameters:
        data (np.array): The data to be standardized.

        Returns:
        np.array: The standardized data.
        """
        if len(data.shape) == self.reader_writer.image_dimensions:
            # Add channels in - it is missing
            data = data[..., np.newaxis]
        if len(data.shape) - 1 != self.reader_writer.image_dimensions:
            raise ValueError(f"There is a shape mismatch. The reader/writer indicates "
                             f"{self.reader_writer.image_dimensions} spacial dimensions. "
                             f"With channels, your data shape is {data.shape}.")
        return data.astype(np.float32)

    def get_data(self, **kwargs) -> Tuple[np.array, np.array]:
        """
        Returns the datapoint data. If caching is enabled, it checks the cache first.

        Parameters:
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Tuple[np.array, np.array]: The image and label data as numpy arrays.
        """
        if self.cache:
            image = self._get_cached()
        else:
            image = self.reader_writer.read(self.im_path, **kwargs).astype(np.float32)
            # Enforce [H, W, C] or [H, W, D, C]
            image = self._standardize(image)
        label = None
        if self.mode == SEGMENTATION:
            label = self.reader_writer.read(self.label, **kwargs)
            label = self._standardize(label)
        elif self.mode == CLASSIFICATION:
            label = np.array(int(self.label))

        return image, label

    def set_num_classes(self, n: int) -> None:
        """
        Set the number of classes.

        Parameters:
        n (int): The number of classes.
        """
        self.num_classes = n

    def _cache(self, data: np.array) -> SharedMemory:
        """
        Cache the data.

        Parameters:
        data (np.array): The data to be cached.
        """
        self._shape = data.shape
        self._dtype = data.dtype
        shared_mem = SharedMemory(size=data.nbytes, name=self.case_name, create=True)
        temp_buf = np.ndarray(data.shape, dtype=data.dtype, buffer=shared_mem.buf)
        temp_buf[:] = data[:]
        return shared_mem

    def _get_cached(self) -> np.array:
        """
        Fetch the cached data.

        Returns:
        np.array: The cached data.
        """
        return np.asarray(np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared_mem.buf))

    @property
    def case_name(self) -> str:
        """
        Get the case name. If the case name is not set, a NameError is raised.

        Returns:
        str: The case name.
        """
        if self._case_name is not None:
            return self._case_name
        raise NameError('You are trying to access case name when you never set it. set case_name when constructing '
                        'object.')

    @property
    def extension(self) -> str:
        """
         Get the extension of the image file.

         Returns:
         str: The extension of the image file.
         """
        name = self.im_path.split('/')[-1]
        return '.'.join(name.split('.')[1:])

    def __del__(self):
        if self._shared_mem is not None:
            self._shared_mem.close()
