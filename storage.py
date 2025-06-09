import json
from datetime import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import h5py

from configs import CONFIGS


def get_foldername(config_name, time):
    return "data/" + config_name + time.strftime("-%Y%m%d-%H%M%S%f")


def get_config_filename(config_name, time):
    return get_foldername(config_name, time) + "/config.json"


def get_config_fields():
    return ("name", "time", "seed", "drive", "emissions", "logging", "saving", "snapshot_period")


def save_config(config):
    file_name = get_config_filename(config["name"], config["time"])
    config_info = {k: config[k].strftime("%Y%m%d-%H%M%S%f") if k == "time" else config[k] 
                   for k in get_config_fields()}
    file = Path(file_name)
    file.parent.mkdir(exist_ok=True, parents=True)
    with file.open("w") as f:
        json.dump(config_info, f, indent=2)


def load_config(config_name, time):
    file_name = get_config_filename(config_name, time)
    with open(file_name, "r") as f:
        config_info = json.load(f)

    config_info["time"] = datetime.strptime(config_info["time"], "%Y%m%d-%H%M%S%f")
    config = CONFIGS[config_info["name"]]
    config |= config_info
    return config


def get_system_fields():
    return ("step", "R", "P", "net_field")


def extract_stored_data(data):
    return {key: getattr(data, key) for key in get_system_fields()}


def get_hdf5_filename(config_name, time):
    return get_foldername(config_name, time) + "/data.hdf5"


def initialize_hdf5(data, config_name, time):
    file_name = get_hdf5_filename(config_name, time)
    reps = extract_stored_data(data)
    with h5py.File(file_name, "w") as f:
        for key, val in reps.items():
            array = np.asarray(val)
            f.create_dataset(key, shape=(0, *array.shape), maxshape=(None, *array.shape), 
                             chunks=True, dtype=array.dtype)


def save_state(config_name, time, data):
    file_name = get_hdf5_filename(config_name, time)
    data_dict = extract_stored_data(data)
    with h5py.File(file_name, "a") as f:
        for key, val in data_dict.items():
            dataset = f[key]
            i = dataset.shape[0]
            dataset.resize((i + 1, *dataset.shape[1:]))
            dataset[i] = np.asarray(val)


def load_states(config_name, time):
    file_name = get_hdf5_filename(config_name, time)
    with h5py.File(file_name, "r") as f:
        states = namedtuple("States", get_system_fields())(
            **{field: f[field][:] for field in get_system_fields()})

    return states


def string_to_datetime(timestamp):
    return datetime.strptime(timestamp, "%Y%m%d-%H%M%S%f")