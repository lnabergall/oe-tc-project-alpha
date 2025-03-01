import warnings

import jax
import jax.numpy as jnp

from system import ParticleSystem as System
from log import setup_logging

warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

toy_tiny = {
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
    "particle_limit": 10,
    "boundstate_limit": 10,
    "boundstate_nbhr_limit": 10,

    "key": jax.random.key(12),
    "kappa": 50,
    "proposal_samples": 5,
    "field_preloads": 2,
}

particle_system = System(**toy_tiny)

if __name__ == '__main__':
    setup_logging()
    data, internal_data, key = particle_system.run(2)