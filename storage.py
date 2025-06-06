import json

import numpy as np
import jax
import jax.numpy as jnp
import h5py

from configs import CONFIGS


def get_foldername(config_name, time):
    return config_name + time.strftime("-%Y%m%d-%H%M%S%f")


def get_config_filename(config_name, time):
    return generate_foldername(config_name, time) + "/config.json"


def save_config(config):
    file_name = get_config_filename(config["name"], config["time"])
    config_info = {k: config[k] for k in ("name", "time", "seed")}
    with open(file_name, "w") as f:
        json.dump(config_info, f, indent=2)


def load_config(config_name, time):
    file_name = get_config_filename(config_name, time)
    with open(file_name, "r") as f:
        config_info = json.load(f)

    config = CONFIGS[config_info["name"]]
    config |= config_info
    return config


def get_fields():
    return ("step", "R", "P", "net_field")


def extract_stored_data(data):
    return {key: data[key] for key in get_fields()}


def get_hdf5_filename(config_name, time):
    return generate_foldername(config_name, time) + "/data.hdf5"


def initialize_hdf5(data, config_name, time):
    file_name = get_hdf5_filename(config_name, time)
    reps = extract_stored_data(data)
    with h5py.File(file_name, "w") as f:
        for key, val in reps.items():
            array = np.asarray(val)
            f.create_dataset(key, shape=(0, *array.shape), maxshape=(None, *array.shape), 
                             chunks=True, dtype=array.dtype)


def save_state(data, config_name, time):
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
        states = tuple(f[field] for field in get_fields())

    return states