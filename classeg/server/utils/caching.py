import random
import shutil
import os
import time

import numpy as np
from PIL import Image
from uuid import uuid4
import cv2
from multiprocessing import Process

if not os.path.exists(os.path.expanduser("~/.classeg_cache")):
    os.makedirs(os.path.expanduser("~/.classeg_cache"))


def _clear_cache():
    shutil.rmtree(os.path.expanduser("~/.classeg_cache"), ignore_errors=True)
    os.makedirs(os.path.expanduser("~/.classeg_cache"))


def clear_cache():
    process = Process(target=_clear_cache)
    process.start()


def cache_array_as_image(array):
    """
    Cache an array as an image. This is useful for caching preprocessed images.

    Parameters:
    array (np.array): The array to cache.
    path (str): The path to the cache file.
    """
    array = np.moveaxis(array, 0, -1)
    path = os.path.expanduser(f"~/.classeg_cache/{str(uuid4())}.png")
    cv2.imwrite(path, cv2.cvtColor((array - np.min(array)) / (np.max(array)) * 255, cv2.COLOR_BGR2RGB))
    return path
