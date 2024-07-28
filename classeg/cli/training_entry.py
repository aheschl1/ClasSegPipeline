import datetime
import glob
import logging
import os.path
import shutil
import socketserver
import warnings
from multiprocessing.shared_memory import SharedMemory
from typing import Type, List

import click
import multiprocessing_logging
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from classeg.training.trainer import Trainer
from classeg.utils.constants import *
from classeg.utils.utils import get_dataset_name_from_id, get_dataset_mode_from_name, \
    get_preprocessed_datapoints, get_trainer_from_extension


def cleanup(dataset_name, fold, cache):
    if not cache:
        return
    print("Cleaning up the shared memory...")
    train_points, val_points = get_preprocessed_datapoints(dataset_name, fold, cache=False, verbose=False)
    for point in train_points + val_points:
        try:
            SharedMemory(point.case_name).unlink()
        except FileNotFoundError:
            ...


def _get_free_port() -> str:
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    return str(free_port)


def setup_ddp(rank: int, world_size: int, port: str) -> None:
    """
    Prepares the ddp on a specific process.
    :param rank: The device we are initializing.
    :param world_size: The total number of devices.
    :param port: The port to use for communication.
    :return: None
    """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = port
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def ddp_training(rank, world_size: int, dataset_id: int,
                 fold: int, model: str,
                 session_id: str, resume: bool,
                 config: str, trainer_class: Type[Trainer], dataset_desc: str,
                 mem_cache: bool, port: str, kwargs: dict) -> None:
    """
    Launches training on a single process using pytorch ddp.
    :param config: The name of the config to load.
    :param session_id: Session id to be used for folder name on output.
    :param rank: The rank we are starting.
    :param world_size: The total number of devices
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param model: The path to the model json definition
    :param resume: Continue training from latest epoch
    :param trainer_class: Trainer class to use
    :param dataset_desc: Trainer class to useZ,
    :param mem_cache: Cache the data in memory
    :param port: The port to use for communication
    :param kwargs: Extra arguments to pass to the trainer
    :return: Nothing
    """
    setup_ddp(rank, world_size, port)
    dataset_name = get_dataset_name_from_id(dataset_id, dataset_desc)
    try:
        trainer = trainer_class(
            dataset_name,
            fold,
            model,
            rank,
            session_id,
            config,
            cache=mem_cache,
            resume=resume,
            world_size=world_size,
            **kwargs
        )
        trainer.train()
    except Exception as e:
        raise e
    finally:
        destroy_process_group()
        cleanup(dataset_name, fold, mem_cache)


@click.command()
@click.option("-fold", "-f", help="Which fold to train.", type=int, required=True)
@click.option("-dataset_id", "-d", help="The dataset id to train.", type=str, required=True)
@click.option("-model", "-m", help="Path to model json definition, or name of the model class.",
              type=str, required=False)
@click.option("-gpus", "-g", help="How many gpus for ddp", type=int, default=1)
@click.option("--resume", "--r", help="Resume training from latest", type=bool, is_flag=True)
@click.option("-config", "-c", help="Name of the config file to utilize.", type=str, default="config")
@click.option("-name", "-n", help="Output folder name.", type=str, default=None)
@click.option("-extension", "-ext", help="Name of the extension that you want to use.", type=str, default=None)
@click.option("-dataset_desc", "-dd", required=False, default=None,
              help="Description of dataset. Useful if you have overlapping ids.")  # 10
@click.option("--mem_cache", help="Cache the data in memory.", type=bool, is_flag=True)
@click.option("--force_override", "--fo",
              help="Ignore that the experiment name is the same as one existing, even though you did not specify to resume.",
              is_flag=True, type=bool)
@click.argument('extra_args', nargs=-1)
def main(
        fold: int,
        dataset_id: str,
        model: str,
        gpus: int,
        resume: bool,
        config: str,
        name: str,
        extension: str,
        dataset_desc: str,
        mem_cache: bool,
        force_override: bool,
        extra_args: List[str]
) -> None:
    """
    Initializes training on multiple processes, and initializes logger.
    :param config: The name oof the config file to load.
    :param gpus: How many gpus to train with
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param model: The path to the model json definition
    :param resume: The weights to load, or None
    :param name: The name of the output folder. Will timestamp by default.
    :param extension: The name of the trainer class to use
    :param dataset_desc: Dataset description
    :param mem_cache: Cache the data in memory
    :param force_override:
    :param extra_args: Extra arguments to pass to the extension.
    :return:
    """
    multiprocessing_logging.install_mp_handler()
    kwargs = {}
    for arg in extra_args:
        if "=" not in arg:
            raise ValueError(
                "For preprocessing, all positional arguments must contain '='. They are used for passing arguments to extension preprocessors.")
        key, value = arg.split('=')
        kwargs[key] = value

    dataset_name = get_dataset_name_from_id(dataset_id, dataset_desc)
    output_dir = f"{RESULTS_ROOT}/{dataset_name}/fold_{fold}/{name}"
    if os.path.exists(output_dir) and not resume:
        if force_override:
            shutil.rmtree(output_dir)
        else:
            raise ValueError(
                f"An experiment with name {name} already exists. Do you want to resume it? Then use --r. Otherwise, pick a new name, or run with --force_override"
            )
    if resume and name is None:
        raise ValueError("You must provide a name for the session if you want to resume training.")

    if model is None:
        warnings.warn("No model provided. "
                      "Make sure you use an extension that does not need an explicit model argument.")
    if model is not None and not os.path.exists(model) and "json" in model:
        # try to find it in the default model bucket
        available_models = [x for x in glob.glob(f"{MODEL_BUCKET_DIRECTORY}/**/*", recursive=True) if "json" in x]
        for model_path in available_models:
            if model_path.split('/')[-1] == model:
                print(model_path.split('/')[-1].split('.')[0])
                model = model_path
                break

    mode = get_dataset_mode_from_name(get_dataset_name_from_id(dataset_id, dataset_desc))
    trainer_class = get_trainer_from_extension(extension, dataset_name)
    logging.info(f"Training detected mode {mode}")
    # This sets the behavior of some modules in json models utils.
    session_id = datetime.datetime.now().strftime("%d_%H_%M_%f") if name is None else name
    if gpus > 1:
        port = _get_free_port()
        mp.spawn(
            ddp_training,
            args=(gpus, dataset_id, fold, model, session_id, resume,
                  config, trainer_class, dataset_desc, mem_cache, port, kwargs),
            nprocs=gpus,
            join=True
        )
    elif gpus == 1:
        try:
            trainer = trainer_class(dataset_name, fold, model, 0, session_id, config,
                                    resume=resume, cache=mem_cache, world_size=1, **kwargs)
            trainer.train()
        except Exception as e:
            raise e
        finally:
            cleanup(dataset_name, fold, mem_cache)
    else:
        raise NotImplementedError("You must set gpus to >= 1")


if __name__ == "__main__":
    main()
