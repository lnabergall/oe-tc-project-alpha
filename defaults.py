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
    parser.add_argument("--steps")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nojit", action="store_true")
    args = parser.parse_args()
    config = CONFIGS[args.config.lower()]
    return config, args.steps, args.profile, args.cpu, args.nojit


CONFIGS = {
    "toy_tiny": {
        "n": 12,    # multiple of 4
        "k": 12,
        "t": 2,
        "N": jnp.array((9, 3)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 24,
        "emission_streams": 3,
        "boundstate_streams": 3,
        "particle_limit": 6,
        "boundstate_limit": 6,

        "field_preloads": 10,
    },
    "toy_small": {
        "n": 40,    # multiple of 4
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
        "emission_streams": 5,
        "boundstate_streams": 5,
        "particle_limit": 30,
        "boundstate_limit": 40,

        "field_preloads": 10,
    },
    "toy_medium": {
        "n": 100,    # multiple of 4
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
        "emission_streams": 20,
        "boundstate_streams": 20,
        "particle_limit": 120,
        "boundstate_limit": 160,

        "field_preloads": 10,
    },
    "toy_large": {
        "n": 248,    # multiple of 4
        "k": 2500,
        "t": 2,
        "N": jnp.array((2000, 500)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 5000,
        "emission_streams": 100,
        "boundstate_streams": 100,
        "particle_limit": 500,
        "boundstate_limit": 800,

        "field_preloads": 10,
    },
    "toy_huge": {
        "n": 500,    # multiple of 4
        "k": 10000,
        "t": 2,
        "N": jnp.array((8000, 2000)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 20000,
        "emission_streams": 300,
        "boundstate_streams": 300,
        "particle_limit": 1800,
        "boundstate_limit": 3000,

        "field_preloads": 10,
    },
    "toy_massive": {
        "n": 800,    # multiple of 4
        "k": 25000,
        "t": 2,
        "N": jnp.array((20000, 5000)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 50000,
        "emission_streams": 600,
        "boundstate_streams": 600,
        "particle_limit": 4000,
        "boundstate_limit": 6000,

        "field_preloads": 10,
    },
}


if __name__ == '__main__':
    config, steps, profiling, use_cpu, no_jit = get_config()
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

    data, internal_data, key = particle_system.run(key, jnp.int32(steps))
    jax.block_until_ready((particle_system, data, internal_data, key))

    if profiling:
        jax.profiler.stop_trace()