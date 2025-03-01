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


def uniform_position_proposal_generator(key, position, range_, num_samples):
    """
    Generates a new position uniformly sampled from all points 
    within 'range_' distance of the particle. Does not exclude any points, 
    e.g. points of infinite potential for the particle.
    """
    key, subkey = jax.random.split(key)
    sample_indices = jax.random.randint(subkey, (num_samples, 2), -range_, range_ + 1)
    shifted_proposals = discrete_ball_map(sample_indices)
    proposals = position + shifted_proposals
    return proposals


def uniform_orientation_proposal_generator(key, angular_range, num_samples):
    """
    Generates a new orientation uniformly sampled from all orientations
    within 'angular_range' distance of the current orientation. Does not exclude any orientations,
    e.g. orientations of infinite potential for the bound state. 
    """
    key, subkey = jax.random.split(key)
    max_quarter_spins = angular_range // 90
    proposals = jax.random.randint(subkey, (num_samples,), -max_quarter_spins, max_quarter_spins + 1)
    proposals *= 90
    return proposals


@partial(jax.jit, static_argnums=[3, 5])
def uniform_proposal_generator(key, position, range_, num_samples, angular_range=None, 
                               bound_state=False):
    proposed_position = uniform_position_proposal_generator(key, position, range_, num_samples)
    if bound_state:
        proposed_orientation = uniform_orientation_proposal_generator(key, angular_range, num_samples)
        return proposed_position, proposed_orientation
    else:
        return proposed_position


uniform_boundstate_proposal_generator = partial(uniform_proposal_generator, bound_state=True)