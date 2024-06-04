import os.path
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple, Union

import numpy as np
import torch

from classeg.utils.constants import SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED
from classeg.utils.reader_writer import get_reader_writer, get_reader_writer_from_extension, NaturalReaderWriter, \
    SimpleITKReaderWriter
from classeg.utils.normalizer import get_normalizer, get_normalizer_from_extension


class Datapoint:
    def __init__(self,
                 im_path: str,
                 label: Union[None, str, torch.Tensor],
                 dataset_name: str = None,
                 case_name: str = None,
                 writer: str = None,
                 cache: bool = False) -> None:
        """
        Datapoint object. Supports caching, and determines own read/write and normalizer.
        :param dataset_name: Name of the dataset. ex: Dataset_000
        :param label: Either a class label, or a path to a segmentation, or None
        :param case_name: The name of the case. ex: case_00000
        :param writer: Overwrite writer class
        :param cache: If enabled, stores data in shared memory
        """

        self.im_path = im_path
        self.label = label
        if label is None:
            self.mode = SELF_SUPERVISED
        elif isinstance(label, str) and os.path.exists(label) and not os.path.isdir(label):
            self.mode = SEGMENTATION
        else:
            self.mode = CLASSIFICATION
        self.cache = cache
        self.num_classes = None
        self._shared_mem = None
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

    def get_data(self, **kwargs) -> Tuple[np.array, np.array]:
        """
        Returns datapoint data. Checks cache if enabled.
        :param kwargs:
        :return: Data as np.array. DO NOT EDIT DIRECTLY IF CACHE ENABLED!!!
        """
        image = self.reader_writer.read(self.im_path, **kwargs).astype(np.float32)
        # Enforce [H, W, C] or [H, W, D, C]
        if len(image.shape) == self.reader_writer.image_dimensions:
            # Add channels in - it is missing
            image = image[..., np.newaxis]
        if len(image.shape)-1 != self.reader_writer.image_dimensions:
            raise ValueError(f"There is a shape mismatch. The reader/writer indicates "
                             f"{self.reader_writer.image_dimensions} spacial dimensions. "
                             f"With channels, your IMAGE shape is {image.shape}.")
        if self.mode == SEGMENTATION:
            label = self.reader_writer.read(self.label, **kwargs)
            if len(label.shape) == self.reader_writer.image_dimensions:
                # Add channels in - it is missing
                label = label[..., np.newaxis]
            if len(label.shape) - 1 != self.reader_writer.image_dimensions:
                raise ValueError(f"There is a shape mismatch. The reader/writer indicates "
                                 f"{self.reader_writer.image_dimensions} spacial dimensions. "
                                 f"With channels, your LABEL shape is {label.shape}.")
            return image, label
        elif self.mode == CLASSIFICATION:
            return image, np.array(int(self.label))
        else:
            return image, None

    def set_num_classes(self, n: int) -> None:
        self.num_classes = n

    def _cache(self, data: np.array) -> None:
        """
        Cache the data
        :return: None
        """
        try:
            self._shared_mem = SharedMemory(size=data.nbytes, create=True, name=self._case_name)
            temp_buf = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shared_mem.buf)
            temp_buf[:] = data
        except FileExistsError:
            ...  # Already exists

    def _get_cached(self) -> np.array:
        """
        Fetch the cached data
        :return: The data
        """
        return np.asarray(np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared_mem.buf))

    @property
    def case_name(self) -> str:
        if self._case_name is not None:
            return self._case_name
        raise NameError('You are trying to access case name when you never set it. set case_name when constructing '
                        'object.')

    @property
    def extension(self) -> str:
        name = self.im_path.split('/')[-1]
        return '.'.join(name.split('.')[1:])

    def __del__(self):
        """
        Unlink shared memory
        :return:
        """
        if self._shared_mem is not None:
            self._shared_mem.unlink()
