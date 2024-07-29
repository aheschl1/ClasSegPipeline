import os
import json
import platform

if platform.system() == 'Windows':
    raise SystemExit('Windows is not supported (you should fix this and make a pull request, it is just paths)')

SIMPLE_ITK = 'SimpleITK'
NATURAL = 'NATURAL'
CT = 'CT'
TENSORBOARD = 'tensorboard'
WANDB = 'wandb'

if not os.path.exists(os.path.expanduser("~/.classegrc")):
    default_config = {
        "best_epoch_celebration": "That is a new best epoch, saving the state!",
        "default_processes": os.cpu_count(),
        "model_bucket_directory": f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/model_definitions",
        "logger": TENSORBOARD
    }

    with open(os.path.expanduser("~/.classegrc"), "w+") as config_file:
        json.dump(default_config, config_file, indent=4)

with open(os.path.expanduser("~/.classegrc"), "r") as config_file:
    current_config = json.load(config_file)

if "wandb_entity" in current_config and "WANDB_ENTITY" not in os.environ:
    os.environ["WANDB_ENTITY"] = current_config["wandb_entity"]

LOGGER_TYPE = current_config.get("logger", None)

RAW_ROOT = os.getenv('RAW_ROOT', current_config.get("RAW_ROOT", None))
if RAW_ROOT is None or not os.path.exists(RAW_ROOT):
    raise NotADirectoryError('You must define $RAW_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), or in ~/.classegrc, and make sure that the path exists.')

PREPROCESSED_ROOT = os.getenv('PREPROCESSED_ROOT', current_config.get("PREPROCESSED_ROOT", None))
if PREPROCESSED_ROOT is None or not os.path.exists(PREPROCESSED_ROOT):
    raise NotADirectoryError('You must define $PREPROCESSED_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), or in ~/.classegrc, and make sure that the path exists.')

RESULTS_ROOT = os.getenv('RESULTS_ROOT', current_config.get("RESULTS_ROOT", None))
if RESULTS_ROOT is None or not os.path.exists(RESULTS_ROOT):
    raise NotADirectoryError('You must define $RESULTS_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), or in ~/.classegrc, and make sure that the path exists.')
                             
AUTOENCODER = os.getenv('AUTOENCODER')
# if AUTOENCODER is None or not os.path.exists(AUTOENCODER):
#     raise NotADirectoryError('You must define $AUTOENCODER in your environment variables '
#                              '(in ~/.bashrc or ~/.profile), and make sure that the path exists.')


BEST_EPOCH_CELEBRATION = current_config.get("best_epoch_celebration", "That is a new best epoch, saving the state!")
DEFAULT_PROCESSES = current_config.get("default_processes", os.cpu_count())
MODEL_BUCKET_DIRECTORY = current_config.get(
    "model_bucket_directory",
    f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/model_definitions"
)

ENB6 = 'enb6'
ENB4 = 'enb4'
ENV2 = 'env2'
ENB0 = 'enb0'
ENB1 = "enb1"

ENB6_P = 'enb6p'
ENB4_P = 'enb4p'
ENV2_P = 'env2p'
ENB0_P = 'enb0p'
ENB1_P = "enb1p"

CONCAT = 'concat'
ADD = 'add'
TWO_D = '2d'
THREE_D = '3d'
INSTANCE = 'instance'
BATCH = "batch"
BASE = 'base'
MNIST = 'mnist'
WHEAT = 'wheat'
ECHO = 'echo'
IMAGENET = 'imagenet'
NUTRIENT = 'nutrient'
PATIENT_PATH = "patient_path"
FILE_NAME = "file_name"
INSTANCE_NUMBER = "InstanceNumber"
LABEL = "label"
ENDOCARDIAL_QUALITY = "endocardial_quality"
AXIS_QUALITY = "axis_quality"
STRUCTURE_QUALITY = "structure_quality"
QUALITY_SUM = "quality_sum"

SEGMENTATION = "segmentation"
CLASSIFICATION = "classification"
SELF_SUPERVISED = "self_supervised"
EXTENSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'extensions')
