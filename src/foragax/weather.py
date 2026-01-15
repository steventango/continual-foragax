"""Helper functions for loading temperature data for the ForagerWeather env."""

import csv
import os
from glob import glob

import jax.numpy as jnp
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ECA_non-blended_custom")
FILE_PATHS = sorted(glob(f"{DATA_PATH}/TG_*.txt"))


def load_data(file_path: str):
    """Load and normalize temperature data from a file"""
    with open(file_path, "r") as f:
        # Skip header lines
        for _ in range(21):
            next(f)
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() for fn in reader.fieldnames]
        tg = []
        for row in reader:
            if not row or row["Q_TG"].strip() != "0":
                continue
            tg.append(float(row["TG"]))
        tg = np.array(tg)
        mean_temperature = tg / 10
        min_temp = mean_temperature.min()
        max_temp = mean_temperature.max()
        normalized = (mean_temperature - min_temp) / (max_temp - min_temp) * 2 - 1
        return jnp.array(normalized)


def get_temperature(rewards: jnp.ndarray, clock: int, repeat: int) -> float:
    """Get the temperature for a given clock time."""
    return rewards[clock // repeat % len(rewards)]
