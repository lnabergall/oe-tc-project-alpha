import warnings
import argparse
from contextlib import nullcontext

import jax
import jax.numpy as jnp

from log import setup_logging, jax_log_info


warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_traceback_in_locations_limit", -1)