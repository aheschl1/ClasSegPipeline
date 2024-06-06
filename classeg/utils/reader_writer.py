import pickle
from typing import Type, Union

import SimpleITK as sitk
import numpy as np
import torch

from classeg.utils.constants import *
from PIL import Image


class BaseReaderWriter:
    """
    Base class for reading and writing data. This class should not be used directly.
    Instead, use a subclass that overrides the necessary methods.
    """

    def __init__(self, case_name: str, dataset_name: str = None):
        """
        Initialize a BaseReaderWriter object.

        Parameters:
        case_name (str): The name of the case.
        dataset_name (str, optional): The name of the dataset. Defaults to None.
        """
        super().__init__()
        self.direction = None
        self.spacing = None
        self.origin = None
        self.has_read = False
        self.case_name = case_name
        self.dataset_name = dataset_name

    def _verify_extension(self, extension: str) -> None:
        """
        Verify the file extension. This method should be overridden by subclasses.

        Parameters:
        extension (str): The file extension.
        """
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def read(self, path: str, **kwargs) -> np.array:
        """
        Read data from a file. This method should be overridden by subclasses.

        Parameters:
        path (str): The path to the file.

        Returns:
        np.array: The data read from the file.
        """
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        """
        Write data to a file. This method should be overridden by subclasses.

        Parameters:
        data (Union[Type[np.array], Type[torch.Tensor]]): The data to write.
        path (str): The path to the file.
        """
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")

    def _store_metadata(self) -> None:
        """
        Store metadata. This method should be overridden by subclasses.
        """
        ...

    @property
    def image_dimensions(self) -> int:
        """
        Get the number of spatial dimensions that this reader/writer manages, ignoring channels.
        This method should be overridden by subclasses.

        Returns:
        int: The number of spatial dimensions.
        """
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")


class SimpleITKReaderWriter(BaseReaderWriter):

    def _verify_extension(self, extension: str) -> None:
        assert extension in ['nii.gz', 'nrrd'], f'Invalid extension {extension} for reader SimpleITKReader.'

    def _store_metadata(self) -> None:
        assert self.dataset_name is not None, "Can not store metadata from SimpleITK reader/writer without knowing " \
                                              "dataset name."
        expected_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}/metadata"
        expected_file = f"{expected_folder}/{self.case_name}.pkl"
        # Assumption being made here is that images and masks will have the same metadata within a case.
        data = {
            'spacing': self.spacing,
            'direction': self.direction,
            'origin': self.origin
        }
        if os.path.exists(expected_file):
            os.remove(expected_file)
        os.makedirs(expected_folder, exist_ok=True)

        with open(expected_file, 'wb') as file:
            pickle.dump(data, file)

    def read(self, path: str, store_metadata: bool = False, **kwargs) -> np.array:
        self.has_read = True
        self._verify_extension('.'.join(path.split('.')[1:]))
        image = sitk.ReadImage(path)
        self.spacing = image.GetSpacing()
        self.direction = image.GetDirection()
        self.origin = image.GetOrigin()
        if store_metadata:
            self._store_metadata()
        return sitk.GetArrayFromImage(image)

    def check_for_metadata_folder(self) -> Union[dict, None]:
        expected_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}/metadata"
        expected_file = f"{expected_folder}/{self.case_name}.pkl"
        if os.path.exists(expected_file):
            with open(expected_file, 'rb') as file:
                return pickle.load(file)
        return None

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        if not self.has_read:
            meta = self.check_for_metadata_folder()
            if meta is None:
                raise ValueError(f'SimpleITK reader writer can not find metadata for this image {self.case_name}. If '
                                 f'you read first we can save.')
            try:
                self.spacing = meta['spacing']
                self.direction = meta['direction']
                self.origin = meta['origin']
            except KeyError:
                raise ValueError(f'Invalid metadata found for {self.case_name} in SimpleITKReaderWriter.')
        self._verify_extension('.'.join(path.split('.')[1:]))
        if isinstance(data, torch.Tensor):
            data = np.array(data.detach().cpu())
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        sitk.WriteImage(image, path)

    @property
    def image_dimensions(self) -> int:
        return 3


class NaturalReaderWriter(BaseReaderWriter):

    def _verify_extension(self, extension: str) -> None:
        assert extension in ['png', 'jpg', 'npy', 'jpeg', 'JPEG'], (f'Invalid extension {extension} for reader '
                                                                    f'NaturalReaderWriter.')

    def read(self, path: str, store_metadata: bool = False, **kwargs) -> np.array:
        name = path.split('/')[-1]
        extension = '.'.join(name.split('.')[1:])
        self._verify_extension(extension)
        if extension == 'npy':
            image = np.load(path)
        else:
            image = np.array(Image.open(path))
        return image

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        name = path.split('/')[-1]
        self._verify_extension('.'.join(name.split('.')[1:]))
        if isinstance(data, torch.Tensor):
            data = np.array(data.detach().cpu())
        np.save(path, data)

    @property
    def image_dimensions(self) -> int:
        return 2


def get_reader_writer(io: str) -> Type[BaseReaderWriter]:
    assert io in [SIMPLE_ITK, NATURAL], f'Unrecognized reader/writer {io}.'
    reader_writer_mapping = {
        SIMPLE_ITK: SimpleITKReaderWriter,
        NATURAL: NaturalReaderWriter
    }
    return reader_writer_mapping[io]


def get_reader_writer_from_extension(extension: str) -> Type[BaseReaderWriter]:
    mapping = {
        'nii.gz': SimpleITKReaderWriter,
        'nrrd': SimpleITKReaderWriter,
        'png': NaturalReaderWriter,
        'jpg': NaturalReaderWriter,
        'jpeg': NaturalReaderWriter,
        'npy': NaturalReaderWriter
    }
    assert extension in mapping.keys(), f"Currently unsupported extension {extension}"
    return mapping[extension]
