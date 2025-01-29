import pytest
import os
import shutil
from classeg.utils.utils import get_extensions, write_yaml, write_json, get_dataset_name_from_id, check_raw_exists
from classeg.utils.constants import PREPROCESSED_ROOT, RAW_ROOT, RESULTS_ROOT
from classeg.utils import constants

def test_load_all_extensions():
    extensions = get_extensions()
    assert all([x in extensions for x in ["default_class", "default_seg", "default_ssl"]])

def test_write_yaml():
    out_dir = os.path.join(".", "test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.yaml")
    write_yaml({"test": "test"}, out_path)
    assert os.path.exists(out_path)
    shutil.rmtree(out_dir)

def test_write_json():
    out_dir = os.path.join(".", "test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.json")
    write_json({"test": "test"}, out_path)
    assert os.path.exists(out_path)
    shutil.rmtree(out_dir)

def test_get_datasetname_from_id():
    out_dir = os.path.join(".", "test")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "p"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "p", "Dataset_009"), exist_ok=True)
    constants.PREPROCESSED_ROOT = os.path.join(out_dir, "p")
    name = get_dataset_name_from_id(9)
    assert name == "Dataset_009"
    shutil.rmtree(out_dir)
