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


def binomial_sampling(key, log_p_accept, x, x_new):
    """Sample from {x, x_new} with log probability of acceptance log_p_accept."""
    p_accept = jnp.clip(jnp.exp(log_p_accept), max=1)
    accept = jax.random.bernoulli(key, p_accept)
    x_sampled = jax.lax.cond(accept, lambda _: x_new, lambda _: x, operand=None)
    return x_sampled, p_accept, accept


def compute_acceptance_factor(log_p, scale_factor=1):
    """Scale log probability log_p by scale_factor."""
    return jnp.log(scale_factor) + log_p


def sample(key, logdensity, x, x_proposed, scale_factor):
    key, key_accept = jax.random.split(key)
    log_p_accept = compute_acceptance_factor(logdensity, scale_factor)
    x_new, p_accept, accept = binomial_sampling(key_accept, log_p_accept, x, x_proposed)
    return x_new, (p_accept, accept)


uniform_boundstate_proposal_generator = partial(uniform_proposal_generator, bound_state=True)