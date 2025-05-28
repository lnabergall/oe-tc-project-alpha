import warnings
import argparse
from contextlib import nullcontext

import jax
import jax.numpy as jnp

from system import ParticleSystem as System
from log import setup_logging, jax_log_info


warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_traceback_in_locations_limit", -1)


def get_config():
    parser = argparse.ArgumentParser(description="Run a driven particle model.")
    parser.add_argument("--config")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nojit", action="store_true")
    args = parser.parse_args()
    config = CONFIGS[args.config.lower()]
    return config, args.profile, args.cpu, args.nojit


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

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 20,
        "charge_pad_value": -2,
        "emission_streams": 3,
        "boundstate_streams": 3,
        "particle_limit": 4,
        "boundstate_limit": 10,

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

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 200,
        "charge_pad_value": -2,
        "emission_streams": 5,
        "boundstate_streams": 5,
        "particle_limit": 10,
        "boundstate_limit": 20,

        "field_preloads": 2,
    },
    "toy_medium": {
        "n": 100,
        "k": 500,
        "t": 2,
        "N": jnp.array((400, 100)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 1000,
        "charge_pad_value": -2,
        "emission_streams": 20,
        "boundstate_streams": 20,
        "particle_limit": 50,
        "boundstate_limit": 100,

        "field_preloads": 2,
    },
}


if __name__ == '__main__':
    config, profiling, use_cpu, no_jit = get_config()
    if use_cpu:
        device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", device)
    else:
        device = jax.devices()[0]

    if no_jit:
        jax.config.update("jax_disable_jit", True)

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