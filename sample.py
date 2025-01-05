from typing import NamedTuple
from datetime import date
from functools import partial
from dataclasses import dataclass, field, asdict

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from physics import *


key = jax.random.key(int(date.today().strftime("%Y%m%d")))


@register_pytree_node_class
@dataclass
class ParticleSystem:
	# system
	n: int 									# number of lattice points along one dimension
	k: int 									# number of particles
	t: int  								# number of particle types
	N: jax.Array							# number particles of each type, should sum to k. 1D, t.
	T_M: jax.Array							# particle type masses. 1D, t.
	T_Q: jax.Array							# particle type charges. 1D, t.

	# viscous heat bath
	beta: float 							# inverse temperature, 1/T
	gamma: float							# collision coefficient

	# kinetics
	time_unit: float						# unit converting between velocity and distance in a timestep

	# potential energy 
	rho: float 								# range of the potential
	point_func: callable 					# point function of the potential
	energy_lower_bound: float				# lower bound for the potential energy of a particle

	# bonding 
	bond_energy: float						# maximum energy defining a bond

	# external drive
	alpha: float							# scale factor in Wien approximation to Planck's law

	# radiation emission
	delta: float							# range of radiation emission

	# stochasticity
	key: jax.Array							# PRNG key for stochastic processes

	### --- data fields with defaults ---

	# particles
	pauli_exclusion: bool = True 			# Pauli exclusion indicator
	T: jax.Array = field(default=None)		# particle types, to be initialized. 1D, k.
	M: jax.Array = field(default=None)		# particle masses, to be initialized. 1D, k.
	Q: jax.Array = field(default=None)		# particle charges, to be initialized. 1D, k.

	def __post_init__(self):
		if self.T is None:
			self._assign_properties()

	def _assign_properties(self):
		self.T, self.M, self.Q = assign_properties(self.t, self.k, self.N, self.T_M, self.T_Q)

	def tree_flatten(self):
		fields = asdict(self)
		static_fields = {attr: val for attr, val in fields.items() if not isinstance(val, jax.Array)}
		other_fields = tuple(val for val in fields.values() if isinstance(val, jax.Array))
		return (other_fields, static_fields)

	@classmethod
	def tree_unflatten(cls, aux_data, children):
		return cls(**aux_data, *children)

	def initialize(self):
		"""Returns particle positions uniformly randomly sampled without replacement."""
		return sample_lattice_points(self.n, self.k, replace=False)

	def generate_brownian_field(self, steps=1):
		"""Generates field values of Brownian fluctuations and the index of the starting set."""
		num_samples = self.k * steps
		samples = brownian_noise(self.key, self.beta, self.gamma, num_samples)
		brownian_field = jnp.reshape(samples, (steps, self.k))
		return brownian_field, 0

	def generate_external_drive(self, steps=1):
		"""Generates external field values at the top of the lattice and the index of the starting set."""
		num_samples = self.n * steps
		unmasked_samples = wien_approximation(self.key, self.beta, self.alpha, num_samples)
		external_field = jnp.reshape(unmasked_samples, (steps, self.n))
		return external_field, 0

	def system_update_data(self, system_data):
		"""Particle-independent data needed for the update process."""
		R, brownian_field, bf_idx, external_field, ef_idx = system_data

		potential = potential_everywhere(R, self.Q, lattice_positions(self.n), self.point_func)
		interaction_field = generate_interaction_field(potential)

		brownian_field = brownian_field[bf_idx]
		external_field = generate_drive_field(self.n, R, external_field[ef_idx], masked)
		
		return potential, interaction_field, brownian_field, external_field

	def particle_update_data(self, system_data, state):
		"""Data needed for multiple particle-dependent functions of the update process."""
		interaction_field, brownian_field, external_field = system_data
		i, position = state
		mass, charge = self.M[i], self.Q[i]

		velocity_bound = compute_velocity_bound(position, interaction_field, brownian_field, 
			external_field, mass, charge, self.gamma, self.time_unit)

		return velocity_bound

	def step(self):
		pass

	def particle_logdensity_function(self, system_data, state, proposed_position, velocity_bound):
		R, interaction_field, brownian_field, external_field = system_data
		i, position = state
		mass, charge = self.M[i], self.Q[i]

		# compute particle path
		path = canonical_shortest_path(position, proposed_position, self.n)

		# assemble potential function
		potential_func = partial(potential_at, R, self.Q, self.point_func, 
								 pauli_exclusion=self.pauli_exclusion)

		# energy barrier
		B_path = energy_barrier(i, path, potential_func, charge, self.energy_lower_bound)

		# energy difference
		delta_E = energy_difference(i, charge, position, proposed_position, potential_func)

		# work
		start_end = jnp.array((position, proposed_position), dtype=int)
		brownian_work, total_brownian_work = calculate_work(start_end, brownian_field, jnp.sqrt(mass))
		drive_work, total_drive_work = calculate_work(path, external_field, mass)
		work = brownian_work + drive_work
		total_work = calculate_total_work(total_brownian_work + total_drive_work, brownian_field, 
										  external_field, mass, position, proposed_position, velocity_bound)

		logdensity = beta * (work - delta_E - B_path)
		# return both, for excess work and work appled even if proposal rejected
		return logdensity, total_work, work 	

	def bound_state_logdensity_function(self, system_state, state, proposed_position):
		pass

	def particle_proposal_generator(self, state, range_):
		return uniform_proposal_generator(self.key, state, self.R, range_)

	def boundstate_proposal_generator(self, state, range_, angular_range):
		return uniform_boundstate_proposal_generator(self.key, state, self.R, range_, angular_range)


def assign_properties(t, k, N, T_M, T_Q):
	T = jnp.repeat(jnp.arange(t), N, total_repeat_length=self.k) 	# particle types
	M = jnp.fromfunction(lambda i: T_M[T[i]], (k,), dtype=int)  	# particle masses
	Q = jnp.fromfunction(lambda i: T_Q[T[i]], (k,), dtype=int)  	# particle charges
	return T, M, Q


@partial(jax.jit, static_argnums=[2])
def sample_lattice_points(n, k, replace=False):
	samples_1D = jax.random.choice(n**2, (k,), replace=replace)
	samples = jnp.stack((samples_1D // n, samples_1D % n), axis=1)
	return samples


class FSInfo(NamedTuple):
	"""
	Data involved in a single step of factorized sampling.

	accept_probability
		The acceptance probability of the transition.
	accepted
		Whether the proposed position was accepted or rejected.
	position
		The position proposed during this step.
	"""
	accept_probability: float
	accepted: bool
	position: jax.Array


def select_with_fills(padded_indices, X, pad_value):
	"""
	Select rows of 'X' using the 2D Array padded_indices containing rows of row indices of 'X'
	padded with 'pad_value' to account for variable length slices. 
	"""
	valid_indices = jnp.where(padded_indices == pad_value, 0, X)
	selected = X[valid_indices]
	valid_mask = padded_indices != pad_value
	return selected, index_mask


def binomial_sampling(key, log_p_accept, x, x_new):
	"""Sample from {x, x_new} with log probability of acceptance log_p_accept."""
	p_accept = jnp.clip(jnp.exp(log_p_accept), max=1)
	accept = jax.random.bernoulli(key, p_accept)
	x_sampled = jax.lax.cond(accept, lambda _: x_new, lambda _: x, operand=None)
	return x_sampled, p_accept, accept


def compute_acceptance_factor(log_p, scale_factor=1):
	"""Scale log probability log_p by scale_factor."""
	return jnp.log(scale_factor) + log_p


def factorized_step(key, state, logdensity_fn, proposal_generator, scale_factor):
	"""
	Perform one step of factorized sampling, where we generate a proposal for the next position
	and apply an acceptance criteria based on the log density function. 
	
	key
		PRNG key.
	state
		A tuple (i, position), where 'i' is the index of a particle 
		and 'position' is a 1D Array containing the position of a single particle. 
	logdensity_fn
		A function that computes P(R'|R) given a particle state and proposal for the next position. 
	proposal_generator
		A function that samples a proposal given a PRNG key and a particle state. 
	scale_factor
		A scalar determining how to scale the density when computing the acceptance probability. 
	"""
	key_proposal, key_accept = jax.random.split(key, 2)
	proposed_position = proposal_generator(key_proposal, state)
	log_p = logdensity_fn(state, proposed_position)
	log_p_accept = compute_acceptance_factor(log_p, scale_factor)
	new_position, p_accept, accept = binomial_sampling(
		key_accept, log_p_accept, state[1], proposed_position)
	return new_position, (p_accept, accept, proposed_position)


def jaxify_step_fn(step_fn, pad_value, *args):
	"""Construct a step function with inputs restricted to only PyTrees."""

	def jax_step_fn(key, state):
		placeholder = jnp.zeros((2), dtype=int)
		placeholder_func = lambda *_: (placeholder, (0, 0, placeholder))
		valid_func = lambda key, state: step_fn(key, state, *args)
		return jax.lax.cond(state[0] == pad_value, placeholder_func, valid_func, key, state)

	return jax_step_fn


def gibbs_kernel(key, positions, partition, logdensity_fn, 
				 proposal_generator, step_fn, pad_value, parameters):
	"""
	Gibbs kernel. 
	Updates each component of 'positions' conditioned on all the others 
	using a sampling algorithm defined by 'step_fn'. 

	key
		PRNG key.
	positions
		Array of particle positions. 2D, Kx2.  
	partition
		Array specifying a partition of the particles into 'independent' sets; 
		each part of a partition is a row of 'partition' filled with particle indices 
		and padded with 'pad_value'. 2D. 
	logdensity_fn
		A function that computes P(R'|R) for a single partition transition, 
		given an input particle state (index and position), proposal for the next position, 
		and array of all particle positions.
	proposal_generator
		A function that samples a proposal given a PRNG key, a particle state, 
		and an array of all particle positions.
	step_fn
		A function that samples from the distribution P(R'|R) for a single particle update.
		Accepts a PRNG key, particle state, logdensity_fn, proposal_generator as inputs, 
		as well as any additional parameters provided by 'parameters'. 
	pad_value
		The scalar value used for padding, in particular in 'partition'. 
	parameters
		A tuple of additional parameters to be input into step_fn.
	"""
	keys = jax.random.split(key, num=len(partition)) 

	def gibbs_step(positions, part_data):
		key, indices_padded = part_data
		keys = jax.random.split(key, num=len(indices_padded))
		
		def proposal_generator(key, state): 
			return proposal_generator(key, state, positions)

		def logdensity_fn(state, proposed_position):
			return logdensity_fn(state, proposed_position, positions)

		step_fn = jax.vmap(jaxify_step_fn(
			step_fn, pad_value, logdensity_fn, proposal_generator, *parameters))
		independent_set = positions.at[indices_padded].get(mode="clip")
		new_positions, infos = step_fn(keys, (indices_padded, independent_set))
		positions = positions.at[indices_padded].set(new_positions, mode="drop")

		return positions, infos

	positions, _ = jax.lax.scan(gibbs_step, positions, (keys, partition))
	return positions


### need to make logdensity_fn ((i, position), proposed_position, positions) -> log_probability
### and proposal_generator (key, (i, position), positions) -> proposed_position


def discrete_ball_map(indices):
	map_matrix = jnp.array([[1, -1], [1, 1]])
	return (map_matrix @ indices) / 2	# divide by 2 after for possible efficiency


def uniform_position_proposal_generator(key, position, range_):
	"""
	Generates a new position uniformly sampled from all points 
	within 'range_' distance of the particle. Does not exclude any points, 
	e.g. points of infinite potential for the particle.
	"""
	key, subkey = jax.random.split(key)
	sample_indices = jax.random.randint(subkey, (2,), -range_, range_ + 1)
	shifted_proposal = discrete_ball_map(sample_indices)
	proposal = position + shifted_proposal
	return proposal


def uniform_orientation_proposal_generator(key, angular_range):
	"""
	Generates a new orientation uniformly sampled from all orientations
	within 'angular_range' distance of the current orientation. Does not exclude any orientations,
	e.g. orientations of infinite potential for the bound state. 
	"""
	key, subkey = jax.random.split(key)
	max_quarter_spins = angular_range // 90
	proposal = jax.random.randint(subkey, (1,), -max_quarter_spins, max_quarter_spins + 1)
	proposal *= 90
	return proposal


@partial(jax.jit, static_argnums=[5])
def uniform_proposal_generator(key, state, positions, range_, angular_range=None, bound_state=False):
	position_proposal = uniform_position_proposal_generator(key, state[1], range_)
	if bound_state:
		orientation_proposal = uniform_orientation_proposal_generator(key, angular_range)
		return proposed_position, orientation_proposal
	else:
		return proposed_position


uniform_boundstate_proposal_generator = partial(uniform_proposal_generator, bound_state=True)


def particle_logdensity_function(state, proposed_position, positions, charges, masses, beta):
	i, position = state
	pass

### only for first phase, generalize to second as well---just an extra sample for the rotation