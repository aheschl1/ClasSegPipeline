from pipe.training.trainer import Trainer
import glob
import os.path
import click
import multiprocessing_logging
import shutil
from pipe.utils.constants import *
from pipe.utils.utils import get_dataset_name_from_id
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import datetime


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Prepares the ddp on a specific process.
    :param rank: The device we are initializing.
    :param world_size: The total number of devices.
    :return: None
    """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12345"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def ddp_training(rank, world_size: int, dataset_id: int,
                 fold: int, model: str,
                 session_id: str, continue_training: bool, config: str, preload: bool) -> None:
    """
    Launches training on a single process using pytorch ddp.
    :param preload:
    :param config: The name of the config to load.
    :param session_id: Session id to be used for folder name on output.
    :param rank: The rank we are starting.
    :param world_size: The total number of devices
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param model: The path to the model json definition
    :param continue_training: Continue training from latest epoch
    :return: Nothing
    """
    setup_ddp(rank, world_size)
    dataset_name = get_dataset_name_from_id(dataset_id)
    trainer = None
    try:
        trainer = Trainer(
            dataset_name,
            fold,
            model,
            rank,
            session_id,
            config,
            continue_training=continue_training,
            preload=preload, world_size=world_size)
        trainer.train()
    except Exception as e:
        if trainer is not None and trainer.output_dir is not None:
            out_files = glob.glob(f"{trainer.output_dir}/*")
            if len(out_files) < 4:
                shutil.rmtree(trainer.output_dir, ignore_errors=True)
        raise e
    destroy_process_group()


@click.command()
@click.option("-fold", "-f", help="Which fold to train.", type=int, required=True)
@click.option("-dataset_id", "-d", help="The dataset id to train.", type=str, required=True)
@click.option("-model", "-m", help="Path to model json definition or model name.", type=str, default="PIMPNet")
@click.option("--gpus", "-g", help="How many gpus for ddp", type=int, default=1)
@click.option("--continue_training", "--c", help="Continue training from best", type=bool, is_flag=True)
@click.option("-config", "-c", help="Name of the config file to utilize.", type=str, default="config")
@click.option("--preload", "--p", help="Should the datasets preload.", is_flag=True, type=bool)
@click.option("-name", "-n", help="Output folder name.", type=str, default=None)
def main(
        fold: int,
        dataset_id: str,
        model: str,
        gpus: int,
        continue_training: bool,
        config: str,
        preload: bool,
        name: str
) -> None:
    """
    Initializes training on multiple processes, and initializes logger.
    :param preload: Should datasets preload
    :param config: The name oof the config file to load.
    :param gpus: How many gpus to train with
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param model: The path to the model json definition
    :param continue_training: The weights to load, or None
    :param name: The name of the output folder. Will timestamp by default.
    :return:
    """
    multiprocessing_logging.install_mp_handler()
    assert "json" not in model or os.path.exists(
        model
    ), "The model path you specified doesn't exist."
    # This sets the behavior of some modules in json models utils.
    session_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%f") if name is None else name
    if gpus > 1:
        mp.spawn(
            ddp_training,
            args=(
                gpus,
                dataset_id,
                fold,
                model,
                session_id,
                continue_training,
                config,
                preload
            ),
            nprocs=gpus,
            join=True,
        )
    elif gpus == 1:
        dataset_name = get_dataset_name_from_id(dataset_id)
        trainer = None
        try:
            trainer = Trainer(
                dataset_name,
                fold,
                model,
                0,
                session_id,
                config,
                continue_training=continue_training,
                preload=preload,
                world_size=1
            )
            trainer.train()
        except Exception as e:
            if trainer is not None and trainer.output_dir is not None:
                out_files = glob.glob(f"{trainer.output_dir}/*")
                if len(out_files) < 4:
                    shutil.rmtree(trainer.output_dir, ignore_errors=True)
            raise e
    else:
        raise NotImplementedError("You must set gpus to >= 1")
