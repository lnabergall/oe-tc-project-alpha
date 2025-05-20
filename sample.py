from datetime import date
from functools import partial

import jax
import jax.numpy as jnp

from utils import *


key = jax.random.key(int(date.today().strftime("%Y%m%d")))


@partial(jax.jit, static_argnums=[1, 2, 3])
def sample_lattice_points(key, n, k, replace=False):
    """Returns particle positions uniformly randomly sampled without replacement."""
    samples_1D = jax.random.choice(key, n**2, (k,), replace=replace)
    samples = jnp.stack((samples_1D // n, samples_1D % n), axis=1)
    return samples


def sample(key, probabilities, n):
    """Sample from 'n' distinguishable elements according to 'probabilities'."""
    i = jax.random.choice(key, n, p=probabilities)
    return i


def gibbs_sample(key, probabilities):
    k = probabilities.shape[0]
    keys = jax.random.split(key, num=k+1)
    key, keys_sample = keys[0], keys[1:]
    next_indices = jax.vmap(sample)(keys_sample, probabilities, 5)
    return next_indices, key