import glob
import os
import shutil

from tqdm import tqdm

from classeg.cli.preprocess_entry import Preprocessor
from classeg.utils.constants import RAW_ROOT
from classeg.utils.utils import get_case_name_from_number


class ExtensionPreprocessor(Preprocessor):
    def __init__(
            self, dataset_id: str, folds: int, processes: int, normalize: bool, data_path: str, **kwargs
    ):
        super().__init__(dataset_id, folds, processes, normalize, **kwargs)
        if data_path is None:
            raise ValueError("Pass data_path='path to dataset' in preprocessing arguments.")
        self.data_path = f"{data_path}/1"
        assert os.path.exists(self.data_path)
        self._prepare_raw_folder()

    def _prepare_raw_folder(self):
        cases = glob.glob(f"{self.data_path}/*")
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/labelsTr", exist_ok=True)
        os.makedirs(f"{RAW_ROOT}/{self.dataset_name}/imagesTr", exist_ok=True)
        for case in tqdm(cases, desc="Moving data"):
            case_name = get_case_name_from_number(int(case.split("/")[-1]))
            shutil.copy(f"{case}/Mask.png", f"{RAW_ROOT}/{self.dataset_name}/labelsTr/{case_name}.jpg")
            shutil.copy(f"{case}/Image.jpg", f"{RAW_ROOT}/{self.dataset_name}/imagesTr/{case_name}.jpg")
