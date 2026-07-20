import warnings
import argparse
from contextlib import nullcontext

import jax
import jax.numpy as jnp

from system import ParticleSystem as System
#from system_test import run, ParticleSystem as System
from log import setup_logging, jax_log_info


warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_traceback_in_locations_limit", -1)


def get_config():
    parser = argparse.ArgumentParser(description="Run a driven particle model.")
    parser.add_argument("--config")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    config = CONFIGS[args.config.lower()]
    return config, args.profile, args.cpu


CONFIGS = {
    "toy_tiny": {
        "n": 10,
        "k": 10,
        "t": 2,
        "N": jnp.array((8, 2)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "time_unit": 1.0,
        "speed_limit": 3,
        "boundstate_speed_limit": 2,

        "mu": 0.1,

        "rho": 2,
        "point_func": lambda r: jnp.minimum(0.0, jnp.log2(1.5*r + 1) - 2.0),
        "energy_lower_bound": -20.0,
        "factor_limit": 4,

        "bond_energy": -2.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": -1,
        "charge_pad_value": -2,
        "particle_limit": 4,
        "boundstate_limit": 10,
        "boundstate_nbhr_limit": 10,

        "kappa": 50,
        "proposal_samples": 5,
        "field_preloads": 2,
    },
    "toy_small": {
        "n": 40,
        "k": 100,
        "t": 2,
        "N": jnp.array((80, 20)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "time_unit": 1.0,
        "speed_limit": 3,
        "boundstate_speed_limit": 2,

        "mu": 0.1,

        "rho": 2,
        "point_func": lambda r: jnp.minimum(0.0, jnp.log2(1.5*r + 1) - 2.0),
        "energy_lower_bound": -20.0,
        "factor_limit": 12,

        "bond_energy": -2.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": -1,
        "charge_pad_value": -2,
        "particle_limit": 20,
        "boundstate_limit": 20,
        "boundstate_nbhr_limit": 24,

        "kappa": 50,
        "proposal_samples": 10,
        "field_preloads": 2,
    },
    "toy_medium": {
        "n": 120,
        "k": 400,
        "t": 2,
        "N": jnp.array((320, 80)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "time_unit": 1.0,
        "speed_limit": 3,
        "boundstate_speed_limit": 2,

        "mu": 0.1,

        "rho": 2,
        "point_func": lambda r: jnp.minimum(0.0, jnp.log2(1.5*r + 1) - 2.0),
        "energy_lower_bound": -20.0,
        "factor_limit": 24,

        "bond_energy": -2.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": -1,
        "charge_pad_value": -2,
        "particle_limit": 50,
        "boundstate_limit": 50,
        "boundstate_nbhr_limit": 48,

        "kappa": 50,
        "proposal_samples": 10,
        "field_preloads": 2,
    },
    "toy_large": {
        "n": 250,
        "k": 2000,
        "t": 2,
        "N": jnp.array((1600, 400)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "time_unit": 1.0,
        "speed_limit": 8,
        "boundstate_speed_limit": 4,

        "mu": 0.1,

        "rho": 2,
        "point_func": lambda r: jnp.minimum(0.0, jnp.log2(1.5*r + 1) - 2.0),
        "energy_lower_bound": -20.0,
        "factor_limit": 24,

        "bond_energy": -2.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": -1,
        "charge_pad_value": -2,
        "particle_limit": 200,
        "boundstate_limit": 200,
        "boundstate_nbhr_limit": 196,

        "kappa": 50,
        "proposal_samples": 24,
        "field_preloads": 2,
    },
    "toy_massive": {
        "n": 750,
        "k": 20000,
        "t": 2,
        "N": jnp.array((16000, 4000)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "time_unit": 1.0,
        "speed_limit": 8,
        "boundstate_speed_limit": 4,

        "mu": 0.1,

        "rho": 2,
        "point_func": lambda r: jnp.minimum(0.0, jnp.log2(1.5*r + 1) - 2.0),
        "energy_lower_bound": -20.0,
        "factor_limit": 24,

        "bond_energy": -2.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": -1,
        "charge_pad_value": -2,
        "particle_limit": 1000,
        "boundstate_limit": 1000,
        "boundstate_nbhr_limit": 964,

        "kappa": 50,
        "proposal_samples": 48,
        "field_preloads": 2,
    }
}

toy_small2 = CONFIGS["toy_small"]
toy_small2["n"] = 80
CONFIGS["toy_small2"] = toy_small2


if __name__ == '__main__':
    config, profiling, use_cpu = get_config()
    if use_cpu:
        device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", device)
    else:
        device = jax.devices()[0]

    setup_logging()
    jax_log_info("Using device: " + device.__repr__())

    key = jax.random.key(12)
    particle_system = System(**config)

    if profiling:
        jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)

    data, internal_data, key = particle_system.run(key, 4)
    jax.block_until_ready((particle_system, data, internal_data, key))

    if profiling:
        jax.profiler.stop_trace()