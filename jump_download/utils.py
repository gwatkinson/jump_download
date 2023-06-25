"""Utilities for download module."""

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu


def get_otsu_threshold(img_arr, nbins=256):
    threshold = threshold_otsu(image=img_arr, nbins=nbins)
    foreground_area = (img_arr > threshold).sum() / img_arr.size

    return threshold, foreground_area


def robust_convert_to_8bit(img, percentile=1.0):
    """Convert a array to a 8-bit image by min percentile normalisation and clipping."""
    img = img.astype(np.float32)
    img = (img - np.percentile(img, percentile)) / (
        np.percentile(img, 100 - percentile) - np.percentile(img, percentile) + np.finfo(float).eps
    )
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def simple_convert_to_8bit(img):
    img = (img // 256).astype(np.uint8)
    return img


def get_subdictionary(dictionary, keys):
    return {key: dictionary[key] for key in keys if key in dictionary}


def apply_dtypes_with_large_dict(df: pd.DataFrame, dtypes: dict):
    cols = df.columns
    sub_dict = get_subdictionary(dtypes, cols)
    df = df.astype(sub_dict)
    return df


def crop_min_resolution(img, min_resolution_x=970, min_resolution_y=970):
    """Crop the image to the minimum resolution."""
    if min_resolution_x <= 0 or min_resolution_x > img.shape[0]:
        min_resolution_x = img.shape[0]

    if min_resolution_y <= 0 or min_resolution_y > img.shape[1]:
        min_resolution_y = img.shape[1]

    x_start = img.shape[0] // 2 - min_resolution_x // 2
    x_end = x_start + min_resolution_x
    y_start = img.shape[1] // 2 - min_resolution_y // 2
    y_end = y_start + min_resolution_y

    return img[x_start:x_end, y_start:y_end]


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {x}"
        num /= 1024.0


def readable_to_num(x):
    if x.endswith("TB"):
        return float(x[:-3]) * 1024
    elif x.endswith("GB"):
        return float(x[:-3])
    elif x.endswith("MB"):
        return float(x[:-3]) / 1024
    else:
        return float(x[:-3]) / 1024**2


def get_size(path):
    return os.path.getsize(path)


def get_size_of_folder(path):
    return sum(f.stat().st_size for f in Path(path).glob("**/*") if f.is_file())


def get_resolution_tif(path):
    with Image.open(path) as img:
        return img.size


def get_resolution_png(path):
    return get_resolution_tif(path)


def get_resolution_np(path):
    image = np.load(path)
    return image["images"].shape[1:]


def get_resolution_np_dict(path):
    image = np.load(path)
    return image["DNA"].shape


def get_resolution_h5(path):
    with h5py.File(path, "r") as f:
        return f["images"].shape


def get_datetime_str():
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    return now


def hr_time(start, end):
    diff_time = end - start
    return time.strftime("%M:%S", time.gmtime(diff_time))


def time_decorator(func):
    def wrapper(*args, **kwargs) -> Tuple[Any, str]:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        exec_time = hr_time(start, end)

        print(f"Time elapsed: {exec_time}")
        return result, exec_time

    return wrapper


class LogToFileAndTerminal:
    """Log to file and terminal.

    Example:
        sys.stdout = LogToFileAndTerminal("log.txt")
    """

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")  # noqa

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ErrorToFileAndTerminal(LogToFileAndTerminal):
    """Log errors to file and terminal.

    Example:
        sys.stderr = ErrorToFileAndTerminal("log.txt")
    """

    def __init__(self, log_file):
        super().__init__(log_file)
        self.terminal = sys.stderr
