import warnings

import jax
import jax.numpy as jnp

from system import ParticleSystem as System
from log import setup_logging

warnings.filterwarnings('error')  # turn warnings into errors

jnp.set_printoptions(precision=3, suppress=True)

n = 10
k = 10
t = 2
N = jnp.array((8, 2))
T_M = jnp.array((1, 2))
T_Q = jnp.array((-1, 1))

beta = 1.0
gamma = 1.0

time_unit = 1.0
speed_limit = 3
boundstate_speed_limit = 2

mu = 0.1

rho = 2
point_func = lambda r: jnp.minimum(0.0, jnp.log2(1.5*r + 1) - 2.0)
energy_lower_bound = -20.0
factor_limit = 4

bond_energy = -2.0

alpha = 1.0

epsilon = 1.0
delta = 2

pad_value = -1
charge_pad_value = -2
particle_limit = 10
boundstate_limit = 10
boundstate_nbhr_limit = 10

key = jax.random.key(12)
kappa = 50
proposal_samples = 5
field_preloads = 2

particle_system = System(n, k, t, N, T_M, T_Q, beta, gamma, time_unit, speed_limit, boundstate_speed_limit, 
						 mu, rho, point_func, energy_lower_bound, factor_limit, bond_energy, alpha, epsilon, 
						 delta, pad_value, charge_pad_value, particle_limit, boundstate_limit, 
						 boundstate_nbhr_limit, key, kappa, proposal_samples, field_preloads)

if __name__ == '__main__':
	setup_logging()
	data, internal_data, key = particle_system.run(2)