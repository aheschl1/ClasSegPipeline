import gc
from typing import Tuple, Type, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Normalize
from classeg.utils.constants import CT, NATURAL


class Normalizer:
    def __init__(self, dataloader: DataLoader, active: bool = True, calculate_early: bool = True) -> None:
        """
        Given a dataloader, will wrap it and perform a normalization operation specific to the type of data.
        :param dataloader: Iterator to wrap
        :param active: If normalization should be applied to the data.
        :param calculate_early: If calculations should be performed. If false, you must sync with another.
        """
        self.active = active
        self.calculate_early = calculate_early
        self.mean, self.std = None, None
        self.length = len(dataloader)
        self._init(dataloader)
        self.dataloader = iter(dataloader)

    def _init(self, dataloader: DataLoader) -> None:
        """
        Performs calculations on the data.
        :param dataloader: The wrapper dataloader.
        :return: Nothing
        """
        raise NotImplemented('Do not use the base class as an iterator.')

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Next step in the iterator. If not active, just mirror the original dataloader.
        :return:
        """
        if not self.active:
            return next(self.dataloader)
        return self._normalize(*next(self.dataloader))

    def sync(self, other):
        """
        Syncs one normalizer with another. This is to share calculations across similar sets.
        :param other: The other normalizer to sync with.
        :return:
        """
        assert self.__class__ == other.__class__, \
            f"You tried syncing type {self.__class__} with type {other.__class__}."

    def _normalize(self,
                   data: torch.Tensor,
                   label: torch.Tensor,
                   point: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]: ...


class NaturalImageNormalizer(Normalizer):
    def _init(self, dataloader: DataLoader):
        """
        Computes three channel mean and std for natural image normalization.
        :param dataloader:
        :return:
        """
        if not self.active or not self.calculate_early:
            return
        means = None
        total = 0
        shape_len = None
        for data, _, _ in tqdm(dataloader, desc="Calculating mean"):
            assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
            if shape_len is None:
                shape_len = len(data.shape)
            data = torch.moveaxis(data, 1, -1)
            assert shape_len == len(data.shape), "Some images are grayscale, some are RGB!!"
            if means is None:
                means = torch.mean(data.float(), dim=[i for i in range(shape_len)])
            else:
                means += torch.mean(data.float(), dim=[i for i in range(shape_len)])
            total += 1

        means /= total
        mu_rgb = means
        variances = None
        total = 0
        for data, _, _ in tqdm(dataloader, desc="Calculating std"):
            assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
            data = torch.moveaxis(data, 1, -1)
            var = torch.mean((data - mu_rgb) ** 2, dim=[i for i in range(len(data.shape))])
            if variances is None:
                variances = var
            else:
                variances += var
            total += 1
        variances /= total
        std_rgb = torch.sqrt(variances)
        self.mean = mu_rgb
        self.std = std_rgb

    def sync(self, other):
        super().sync(other)
        self.mean = other.mean
        self.std = other.std

    def _normalize(self,
                   data: torch.Tensor,
                   label: torch.Tensor,
                   point: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Method where data normalization is computed.
        :param data: The data to normalize.
        :param label:
        :param point:
        :return: Normalized data and other two points
        """
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        # data = data.float().permute(0, 3, 1, 2)
        return Normalize(mean=self.mean, std=self.std)(data.float()), label, point


class CTNormalizer(Normalizer):
    def _init(self, dataloader: DataLoader) -> None:
        """
        Computes mean and std across entire image for CT normalization.
        :param dataloader:
        :return:
        """
        if not self.active or not self.calculate_early:
            return
        psum = torch.tensor([0.0])
        psum_sq = torch.tensor([0.0])
        pixel_count = 0

        # loop through images
        for data, _, _ in tqdm(dataloader, desc="Calculating mean and std"):
            assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
            psum += data.sum()
            psum_sq += (data ** 2).sum()

            pixels = 1.
            for i in data.shape:
                pixels *= i
            pixel_count += pixels

        # mean and std
        total_mean = psum / pixel_count
        total_var = (psum_sq / pixel_count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)

        # output
        self.mean = total_mean
        self.std = total_std

    def sync(self, other):
        super().sync(other)
        self.mean = other.mean
        self.std = other.std

    def _normalize(self,
                   data: torch.Tensor,
                   label: torch.Tensor,
                   point: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Method where data normalization is computed.
        :param data: The data to normalize.
        :param label:
        :param point:
        :return: Normalized data and other two points
        """
        return Normalize(mean=self.mean, std=self.std)(data.float()), label, point


def get_normalizer(norm: str) -> Type[Normalizer]:
    """
    Given a name of a normalizer, returns the class.
    :param norm: The name of the desired normalizer. Use constants.
    :return: Normalizer subclass.
    """
    assert norm in [NATURAL, CT], f'Unrecognized normalizer type {norm}.'
    norm_mapping = {
        CT: CTNormalizer,
        NATURAL: NaturalImageNormalizer
    }
    return norm_mapping[norm]


def get_normalizer_from_extension(extension: str) -> Type[Normalizer]:
    """
    Given a file extension, determines which normalizer would be appropriate for the data. Returns it.
    :param extension: The extension of the data.
    :return: Normalizer subclass
    """
    mapping = {
        'nii.gz': CTNormalizer,
        'png': NaturalImageNormalizer,
        'jpg': NaturalImageNormalizer,
        'jpeg': NaturalImageNormalizer,
        'JPEG': NaturalImageNormalizer,
        'npy': NaturalImageNormalizer
    }
    assert extension in mapping.keys(), f"Currently unsupported extension {extension}"
    return mapping[extension]
