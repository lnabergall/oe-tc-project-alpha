import warnings
import argparse
from contextlib import nullcontext
from datetime import datetime, UTC
from pprint import pformat

import jax
import jax.numpy as jnp

from system import ParticleSystem as System
from log import setup_logging, jax_log_info
from visualize import produce_graphics
from storage import *
from configs import *


warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_traceback_in_locations_limit", -1)


def get_config():
    parser = argparse.ArgumentParser(description="Run a driven particle model.")

    for key, val in DEFAULTS.items():
        parser.add_argument("--" + key, default=val)
    parser.add_argument("--steps")
    parser.add_argument("--no-emissions", action="store_true")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--saving", action="store_true")
    parser.add_argument("--snapshot_period")
    parser.add_argument("--make-graphics", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--nojit", action="store_true")
    args = parser.parse_args()

    param_indices = {key: int(val) for key, val in vars(args).items() if key in PARAMS}
    config = build_config(param_indices)
    config["emissions"] = not args.no_emissions
    config["drive"] = not args.no_drive

    return (config, args.steps, args.saving, args.snapshot_period, args.make_graphics,
            args.logging, args.cpu, args.profile, args.nojit)


if __name__ == '__main__':
    config, steps, saving, snapshot_period, make_graphics, logging, use_cpu, profiling, no_jit = get_config()
    if use_cpu:
        device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", device)
    else:
        device = jax.devices()[0]

    if no_jit:
        jax.config.update("jax_disable_jit", True)

    setup_logging()
    jax_log_info("Using device: " + device.__repr__())
    jax_log_info("Configuration: \n" + pformat(config))

    config["time"] = datetime.now(UTC)
    config["seed"] = 12     # eventually make variable
    config["logging"] = logging
    config["saving"] = saving
    config["snapshot_period"] = 1 if snapshot_period is None else int(snapshot_period)
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

    if make_graphics:
        jax_log_info("Generating graphics...")
        produce_graphics(particle_system.name, particle_system.time)
        jax_log_info("Graphics created.")