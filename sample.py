from typing import NamedTuple
from datetime import date
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from physics import *
from geometry import *
from utils import *


key = jax.random.key(int(date.today().strftime("%Y%m%d")))


class SystemData(NamedTuple):
	# system
	R: jax.Array							# particle positions. 2D, kx2.
	L: jax.Array 							# labeled occupation lattice. 3D, nxnx3.
	LQ: jax.Array 							# charge-labeled occupation lattice. 3D, nxnx3.
	L_test: jax.Array 						# labeled occupation lattice for test particles. 3D, nxnx3.
	LQ_test: jax.Array 						# charge-labeled occupation lattice for test particles. 3D, nxnx3.

	# fields
	potential: jax.Array 					# potential field. 2D, nxn. 
	brownian_fields: jax.Array				# fields produced by Brownian fluctuations. 4D, axnxnx2.
	bf_idx: int 							# index of the Brownian field for the current timestep 
	external_fields: jax.Array				# top fields produced by external drive. 3D, bxnx2.
	ef_idx: int 							# index of the external field for the current timestep
	interaction_field: jax.Array			# field produced by the particle potential. 3D, nxnx2.
	brownian_field: jax.Array				# field produced by Brownian fluctations. 3D, nxnx2. 
	external_field: jax.Array				# field produced by external drive. 3D, nxnx2. 
	energy_field: jax.Array					# energy field produced by excitations. 3D, kx2. 
	transfer_field: jax.Array				# field transferred to particles by other influences. 2D, kx2.
	net_field: jax.Array 					# net non-transferred field experienced by each particle. 2D, kx2. 

	# bound states
	bound_states: jax.Array 				# bound state index of each particle. 1D, k.
	masses: jax.Array						# mass of each bound state. 1D, k, 0-padded.
	coms: jax.Array							# center of mass of each bound state. 2D, kx2, 0-padded.
	MOIs: jax.Array							# moment of inertia of each bound state. 1D, k, 0-padded.


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

	# energy
	mu: float								# discernability threshold for energy

	# potential energy 
	rho: int 								# range of the potential
	point_func: callable 					# point function of the potential
	energy_lower_bound: float				# lower bound for the potential energy of a particle
	factor_limit: int						# upper bound for the number of particles with range of a point

	# bonding 
	bond_energy: float						# maximum energy defining a bond

	# external drive
	alpha: float							# scale factor in Wien approximation to Planck's law

	# radiation emission
	epsilon: float 							# threshold constant of radiation emission
	delta: float							# range of radiation emission

	# bookkeeping
	pad_value: int 							# scalar used for padding most arrays
	particle_limit: int						# upper bound on the size of an independent set of particles
	boundstate_limit: int 					# upper bound on the size of an independent set of molecules

	# sampling 
	kappa: float							# scale factor used to pseudo-normalize densities

	### --- data fields with defaults ---

	# system
	lattice: field(default=None)   			# lattice indices/positions, to be initialized. 3D, nxnx2.

	# particles
	pauli_exclusion: bool = True 			# Pauli exclusion indicator
	I: jax.Array = field(default=None)      # particle indices, to be initialized. 1D, k.
	T: jax.Array = field(default=None)		# particle types, to be initialized. 1D, k.
	M: jax.Array = field(default=None)		# particle masses, to be initialized. 1D, k.
	Q: jax.Array = field(default=None)		# particle charges, to be initialized. 1D, k.

	def __post_init__(self):
		if self.T is None:
			self._assign_properties()

	def _assign_properties(self):
		self.I, self.T, self.M, self.Q = assign_properties(self.t, self.k, self.N, self.T_M, self.T_Q)
		self.lattice = lattice_positions(self.n)

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

	def particle_gibbs_step(self, data, I, I_mask, key):
		state = (I, data.R[I])

		potential, interaction_field, L_test, LQ_test = self.gibbs_update_data(self, data, I)
		data = data._replace(potential=potential, interaction_field=interaction_field,
							 L_test=L_test, LQ_test=LQ_test)

		update_data_fn = jax.vmap(self.particle_update_data, in_axes=(None, (0, 0)))
		velocity_bounds = update_data_fn(data, state)

		# generate proposal
		key, *keys_proposal = jax.random.split(key, num=I.shape[0]+1)
		proposal_generator_fn = jax.vmap(self.particle_proposal_generator, in_axes=(None, (0, 0), 0, 0))
		proposed_positions = proposal_generator_fn(data, state, keys_proposal, velocity_bounds)

		# logdensity
		logdensity_fn = jax.vmap(self.particle_logdensity_function, in_axes=(None, (0, 0), 0, 0))
		logdensities, works, total_works, energies, proposal_energies = logdensity_fn(
			data, state, proposed_positions, velocity_bounds)
		excess_works = total_works - works

		# sample next positions
		key, *keys_sample = jax.random.split(key, num=I.shape[0]+1)
		sample_fn = jax.vmap(sample, in_axes=(0, 0, 0, 0, None))
		accepted_positions, sample_info = sample_fn(
			keys_sample, logdensities, state[1], proposed_positions, self.kappa)
		R = data.R.at[I_mask].set(accepted_positions)
		data = data._replace(R=R)

		# compute particle potential energies
		potential = potential_everywhere(data.R, self.Q, self.lattice, self.point_func)
		potential_energies = potential_energy_all(data.R, self.Q, potential)

		# get bound states and transfer excess work
		L, LQ, bound_states, masses, coms, MOIs = self.determine_bound_states(data, I)
		data = data._replace(L=L, LQ=LQ, bound_states=bound_states, masses=masses, coms=coms, MOIs=MOIs)
		state_extra = (state[0], I_mask, state[1], accepted_positions)
		transfer_field, net_field, energy_field = self.generate_transfer_field(
			data, state_extra, excess_works, potential_energies)
		data = data._replace(potential=potential, transfer_field=transfer_field, 
							 net_field=net_field, energy_field=energy_field)

		# generate new excitations and update data
		energy_field = self.generate_excitations(data, I, excess_works, potential_energies)
		data = data._replace(energy_field=energy_field)

		return (data, accepted_positions, logdensities, works, 
				total_works, energies, proposal_energies, sample_info)

	def boundstate_gibbs_step(self, data, I, key):
		pass

	def generate_brownian_fields(self, key, steps=1):
		"""Generates field values of Brownian fluctuations and the index of the starting set."""
		num_samples = self.k * steps
		samples = brownian_noise(key, self.beta, self.gamma, num_samples)
		brownian_fields = jnp.reshape(samples, (steps, self.k))
		return brownian_fields, 0

	def generate_external_drives(self, key, steps=1):
		"""Generates external field values at the top of the lattice and the index of the starting set."""
		num_samples = self.n * steps
		unmasked_samples = wien_approximation(key, self.beta, self.alpha, num_samples)
		external_fields = jnp.reshape(unmasked_samples, (steps, self.n))
		return external_fields, 0

	def system_update_data(self, data):
		"""Particle-independent data needed for the update process."""
		brownian_field = data.brownian_fields[data.bf_idx]
		external_field = generate_drive_field(self.n, data.R, data.external_fields[data.ef_idx], masked)
		
		return brownian_field, external_field

	def gibbs_update_data(self, data, state):
		"""Multi-particle data needed for a single Gibbs update step."""
		I = state

		potential = potential_everywhere(data.R, self.Q, self.lattice, self.point_func)
		interaction_field = generate_interaction_field(potential)

		# labeled occupation lattices without particles in I
		L_test = remove_from_lattice(data.L, I, data.R, self.pad_value)
		LQ_test = generate_property_lattice(L_test, self.Q, self.pad_value)

		return potential, interaction_field, L_test, LQ_test

	def particle_update_data(self, data, state):
		"""Data needed for multiple particle-dependent functions of the update process."""
		i, position = state
		mass, charge = self.M[i], self.Q[i]
		molecule_mass = data.masses[data.bound_states[i]]

		velocity_bound = compute_velocity_bound(i, position, data.interaction_field, data.brownian_field, 
			data.external_field, data.energy_field, data.transfer_field, mass, charge, 
			molecule_mass, self.gamma, self.time_unit)

		return velocity_bound

	def boundstate_gibbs_update_data(self, data, state, proposed_coms, proposed_orientations):
		"""Multi-molecule data needed for a single bound state Gibbs update step."""
		I = state
		coms = data.coms[data.bound_states]
		proposed_coms = proposed_coms[data.bound_states]
		proposed_orientations = proposed_orientations[data.bound_states]

		R_moved = move(data.R, coms, proposed_coms)
		R_proposed = rotate(R, proposed_coms, proposed_orientations)

		I_particles = get_particles(I, data.bound_states, self.pad_value)
		L_test, LQ_test = self.gibbs_update_data(data, I_particles)

		return I_particles, R_moved, R_proposed, L_test, LQ_test

	def determine_bound_states(self, data, I):
		"""Determines all bounds states."""
		R = data.R
		potential_energy_factors_fn = jax.vmap(
			potential_energy_factors, in_axes=(0, 0, None, None, None, None, None, None))
		bonds_fn = jax.vmap(compute_bonds, in_axes=(0, 0, 0, None, None))
		L = add_to_lattice(self.L_test, I, R, self.pad_value)
		LQ = generate_property_lattice(L, self.Q, self.pad_value)

		all_energy_factors, all_factor_indices = potential_energy_factors_fn(
			R, self.Q, L, LQ, self.rho, self.point_func, self.factor_limit, self.pad_value)
		all_bonds = bonds_fn(
			self.I, all_energy_factors, all_factor_indices, self.bond_energy, self.pad_value)

		bound_states = compute_bound_states(all_bonds, self.pad_value)
		coms, masses = centers_of_mass(R, self.M, bound_states, self.n)
		MOIs = moments_of_inertia(R, self.M, coms, bound_states, self.n)

		return L, LQ, bound_states, masses, coms, MOIs

	def generate_transfer_field(self, data, state, excess_works, potential_energies):
		"""Generate new transfer forces, which are added to the transfer field."""
		I, I_mask, positions, accepted_positions = state

		end_field_fn = jax.vmap(calculate_end_field, in_axes=(None, None, None, None, None, 0, 0, 0, None))
		end_field = end_field_fn(data.brownian_field, data.external_field, data.energy_field, 
			data.transfer_field, I, positions, accepted_positions, self.M[I], self.gamma)
		net_field = data.net_field.at[I_mask].set(end_field)
		new_transfer_field = data.transfer_field.at[I_mask].set(0.0)
		energy_field = data.energy_field.at[I_mask].set(0.0)

		transfer_field = generate_full_transferred_field(
			I, net_field, data.R, data.bound_states, data.masses, data.coms, data.MOIs, self.pad_value)
		net_field = net_field.at[I_mask].set(0.0)
		transfer_field = transfer_field.at[:, :].multiply(
			transfer_mask(excess_works, potential_energies, self.epsilon))
		net_field = net_field.at[:, :].add(transfer_field)
		transfer_field = transfer_field.at[:, :].add(new_transfer_field) 

		return transfer_field, net_field, energy_field

	def generate_excitations(self, data, I, excess_works, potential_energies):
		"""Generate new excitations, which are added to the energy field."""
		L = generate_unlabeled_lattice(data.R, self.n)
		excitations_fn = jax.vmap(calculate_excitations, in_axes=(0, None, None, 0, None))

		# generate excitations for *all* particles, consider changing to loop if vast majority don't excite
		excitation_arrays, position_arrays = excitations_fn(I, data.R, L, excess_works, self.delta)

		# filter for those that generate excitations (and are not padding)
		excitation_arrays = excitation_arrays.at[:, :, :].multiply(
			(I != self.pad_value) * excitation_mask(excess_works, potential_energies[I], self.epsilon))

		# merge into an energy field
		new_energy_field = merge_excitations(data.R, L, excitation_arrays, position_arrays)
		new_energy_field = new_energy_field[data.R[:, 0], data.R[:, 1]]
		energy_field = data.energy_field.at[:, :].add(new_energy_field)

		return energy_field

	def particle_logdensity_function(self, data, state, proposed_position, velocity_bound):
		i, position = state
		mass, charge = self.M[i], self.Q[i]

		# compute particle path
		path = canonical_shortest_path(position, proposed_position, self.n, self.pad_value)

		# assemble potential function
		potential_energy_func = make_potential_energy_func(
			data.L_test, data.LQ_test, self.rho, self.point_func, self.pad_value)

		# energy difference
		energy = potential_energy_func(position, charge)
		proposal_energy = potential_energy_func(proposed_position, charge)
		deltaE = proposal_energy - energy

		# energy barrier
		max_boundary_energy = jnp.max(jnp.array((energy, proposal_energy)))
		B_path = energy_barrier(path, potential_energy_func, max_boundary_energy, 
								charge, self.energy_lower_bound)

		# work
		transfer_work, total_transfer_work = calculate_work_step(
			position, proposed_position, data.transfer_field[i], mass)
		brownian_work, total_brownian_work = calculate_work_step(
			position, proposed_position, data.brownian_field[position], jnp.sqrt(mass))
		drive_work, total_drive_work = calculate_work(path, data.external_field, mass)
		excitation_work, total_excitation_work = calculate_excitation_work(
			i, position, proposed_position, data.energy_field)
		work = brownian_work + transfer_work + drive_work + excitation_work
		total_work = calculate_total_work(
			total_brownian_work + total_transfer_work + total_drive_work + total_excitation_work, 
			data.brownian_field, data.external_field, data.transfer_field, 
			i, mass, position, proposed_position, velocity_bound)

		logdensity = self.beta * (work - deltaE - B_path)
		# return both, for excess work and work applied even if proposal rejected
		return logdensity, work, total_work, energy, proposal_energy

	def bulk_boundstate_logdensity_function(
			self, data, state, R_moved, R_proposed, proposed_coms, proposed_orientations, 
			linear_velocity_bounds, angular_velocity_bounds):
		### check proposed_coms shape, should be Ix2
		I = state
		I_particles = data.I_particles
		particle_mask = I_particles == self.pad_value
		molecule_indices = bound_state_indices(I, data.bound_states, pad_value)
		masses, coms, MOIs = data.masses[I], data.coms[I], data.MOIs[I]

		# compute molecule path, first translation then add rotation at the end
		com_paths = jax.vmap(canonical_shortest_path, in_axes=(0, 0, None, None))(
			coms, proposed_coms, self.n, self.pad_value)
		com_paths_byparticle = com_paths[molecule_indices]
		particle_paths_nospin_nofilter = jax.vmap(shift_path)(com_paths_byparticle, data.R)
		particle_paths_nofilter = jax.vmap(append_on_path)(particle_paths_nospin_nofilter, R_proposed)

		# assemble potential function
		potential_energy_func = make_potential_energy_func(
			data.L_test, data.LQ_test, self.rho, self.point_func, self.pad_value)
		potential_energy_func = jax.vmap(potential_energy_func)

		# energy difference, all particles 
		energies = potential_energy_func(data.R, self.Q)
		proposal_energies = potential_energy_func(R_proposed, self.Q)
		deltaEs_by_particle = particle_mask * (proposal_energies - energies) 	# mask probably not needed
		deltaEs = jnp.zeros(I.shape).at[molecule_indices].add(deltaEs_by_particle)

		# energy barrier, all particle paths
		energy_barrier_fn = jax.vmap(energy_barrier, in_axes=(0, None, 0, 0, None))
		max_boundary_energies = jnp.max(jnp.stack((energies, proposal_energies), axis=-1), axis=-1)
		B_paths_by_particle = energy_barrier_fn(
			particle_paths_nofilter, potential_energy_func, 
			max_boundary_energies, self.Q, self.energy_lower_bound)
		B_paths_by_particle = B_paths_by_particle.at[:].multiply(particle_mask)
		B_paths = jnp.zeros(I.shape).at[molecule_indices].add(B_paths_by_particle)

		# work, com + orientation only
		work_fn = jax.vmap(calculate_boundstate_work, in_axes=(0, None, None, None, 0, 0, 0))
		works, total_forces, total_torques = work_fn(I, data.bound_states, data.net_field, data.R, 
													 coms, proposed_coms, proposed_orientations)

		logdensities = self.beta * (works - deltaEs - B_paths)
		return logdensities, works, energies, proposal_energies, total_forces, total_torques 

	def particle_proposal_sampler(self, data, state, key, range_, num_samples=1):
		"""
		Samples 'num_samples' particle position proposals uniformly at random within range 
		and selects the first proposal that does not violate Pauli exclusion.  
		"""
		i, position = state
		q = self.Q[i]
		proposed_positions = uniform_proposal_generator(key, position, range_, num_samples)
		invalid_fn = lambda r: violates_pauli_exclusion(data.LQ_test[r], q)
		invalids = jax.vmap(invalid_fn)(proposed_positions)
		proposed_position = proposed_positions[jnp.argmin(invalids)]
		return proposed_position

	def particle_proposal_generator(self, data, state, key, range_):
		"""
		Samples a particle position proposal uniformly at random from the positions within range
		that do not violate Pauli exclusion.  
		"""
		key, subkey = jax.random.split(key)
		i, position = state
		q = self.Q[i]
		Lr_square, Lr_square_positions = centered_square(data.L_test, position, range_)
		Qr_square, _ = centered_square(data.LQ_test, position, range_)
		distances = lattice_distances_2D(r, Lr_square_positions, self.n)

		# determine possible positions
		invalid_fn = jax.vmap(jax.vmap(violates_pauli_exclusion, (0, None)), (1, None))
		valid_mask = (distances <= range_) & ~invalid_fn(Qr_square, q)
		valid_count = jnp.sum(valid_mask)

		# sample proposal index
		proposal_idx = jax.random.randint(subkey, (), 0, valid_count)

		# convert index to position
		flat_valid_mask = jnp.ravel(valid_mask)
		proposal_idx = ith_nonzero_index(flat_valid_mask, proposal_idx)
		proposal_indices = unravel_2Dindex(proposal_idx, Lr_square_positions.shape[1])
		proposed_position = Lr_square_positions[proposal_indices]

		return proposed_position

	def boundstate_proposal_sampler(self, data, state, key, range_, angular_range, num_samples=1):
		"""
		Samples 'num_samples' bound state centers-of-mass and orientation proposals uniformly at random 
		within range and selects the first proposal that does not violate strong Pauli exclusion."""
		i = state
		com = data.coms[i]
		Rs_proposed = jnp.broadcast_to(data.R, (num_samples,) + data.R.shape)
		curr_coms = jnp.broadcast_to(data.coms[data.bound_states], (num_samples,) + data.bound_states.shape)

		# generate proposals
		proposed_coms, proposed_orientations = uniform_boundstate_proposal_generator(
			key, com, range_, angular_range, num_samples)

		# move particles by proposed coms
		new_coms = jnp.broadcast_to(data.coms, (num_samples,) + data.coms.shape).at[:, i, :].set(proposed_coms)
		new_coms = new_coms[:, data.bound_states, :]
		Rs_proposed = move(Rs_proposed, curr_coms, new_coms)

		# rotate particles by proposed orientations
		new_orientations = jnp.zeros((num_samples, self.k))
		new_orientations = new_orientations.at[:, i].set(proposed_orientations)
		new_orientations = new_orientations[:, data.bound_states]
		Rs_proposed = jax.vmap(rotate)(Rs_proposed, new_coms, standardize_angle(new_orientations))

		# get strong Pauli valid or first
		invalids = jax.vmap(violates_strong_pauli_exclusion)(Rs_proposed)
		proposed_idx = jnp.argmin(invalids)

		return proposed_coms[proposed_idx], proposed_orientations[proposed_idx]

### make sure work (excess work) / fields are being tracked / combined / reset correctly
### filter out small work?

### generate initial lattice L and any other needed data

### need to cutoff velocity bound?
### following that, cuttoff paths and consider vmap version

### use sparse arrays to speed up bound state discovery, among other things
### vmap over particles (all? independent/dense set?), finding each molecule, then merge


def assign_properties(t, k, N, T_M, T_Q):
	I = jnp.arange(k)
	T = jnp.repeat(jnp.arange(t), N, total_repeat_length=self.k) 	# particle types
	M = jnp.fromfunction(lambda i: T_M[T[i]], (k,), dtype=int)  	# particle masses
	Q = jnp.fromfunction(lambda i: T_Q[T[i]], (k,), dtype=int)  	# particle charges
	return I, T, M, Q


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


def sample(key, logdensity, x, x_proposed, scale_factor):
	key, key_accept = jax.random.split(key)
	log_p_accept = compute_acceptance_factor(logdensity, scale_factor)
	x_new, p_accept, accept = binomial_sampling(key_accept, log_p_accept, x, x_proposed)
	return x_new, (p_accept, accept)


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


def discrete_ball_map(indices):
	map_matrix = jnp.array([[1, -1], [1, 1]])
	return (indices @ map_matrix.T ) / 2	# divide by 2 after for possible efficiency


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


@partial(jax.jit, static_argnums=[5, 6])
def uniform_proposal_generator(key, position, range_, angular_range=None, 
							   bound_state=False, num_samples):
	position_proposal = uniform_position_proposal_generator(key, position, range_, num_samples)
	if bound_state:
		proposed_orientation = uniform_orientation_proposal_generator(key, angular_range, num_samples)
		return proposed_position, proposed_orientation
	else:
		return proposed_position


uniform_boundstate_proposal_generator = partial(uniform_proposal_generator, bound_state=True)