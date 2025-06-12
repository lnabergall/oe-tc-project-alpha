import warnings
import argparse
from contextlib import nullcontext
from datetime import datetime, UTC

import jax
import jax.numpy as jnp

from system import ParticleSystem as System
from log import setup_logging, jax_log_info
from visualize import create_movie
from storage import *


warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_traceback_in_locations_limit", -1)


def get_config():
    parser = argparse.ArgumentParser(description="Run a driven particle model.")
    parser.add_argument("--config")
    parser.add_argument("--steps")
    parser.add_argument("--no-emissions", action="store_true")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--saving", action="store_true")
    parser.add_argument("--snapshot_period")
    parser.add_argument("--make-movie", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--nojit", action="store_true")
    args = parser.parse_args()

    config = CONFIGS[args.config.lower()]
    config["emissions"] = not args.no_emissions
    config["drive"] = not args.no_drive

    return (config, args.steps, args.saving, args.snapshot_period, args.make_movie,
            args.logging, args.cpu, args.profile, args.nojit)


if __name__ == '__main__':
    config, steps, saving, snapshot_period, make_movie, logging, use_cpu, profiling, no_jit = get_config()
    if use_cpu:
        device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", device)
    else:
        device = jax.devices()[0]

    if no_jit:
        jax.config.update("jax_disable_jit", True)

    setup_logging()
    jax_log_info("Using device: " + device.__repr__())

    config["time"] = datetime.now(UTC)
    config["seed"] = 12
    config["logging"] = logging
    config["saving"] = saving
    config["snapshot_period"] = 0 if snapshot_period is None else int(snapshot_period)
    key = jax.random.key(config["seed"])

    particle_system = System(**config)

    if saving:
        save_config(config)

    if profiling:
        jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)

    data, internal_data, key = particle_system.run(key, jnp.int32(steps))
    jax.block_until_ready((particle_system, data, internal_data, key))

    if profiling:
        jax.profiler.stop_trace()

    if make_movie:
        jax_log_info("Making movie...")
        create_movie(particle_system.name, particle_system.time)
        jax_log_info("Movie created.")