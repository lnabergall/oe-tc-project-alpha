"""
Note: we use matrix indexing, so row indexing is the first or x dimension and column indexing
is the second or y dimension. 
"""

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from utils import *
from geometry import *


def violates_strong_pauli_exclusion(R):
    """Checks for three particles at the same point."""
    _, counts = jnp.unique(R, axis=0, return_counts=True, size=R.shape[0], fill_value=-1)
    return jnp.any(counts >= 3)


def violates_pauli_exclusion(q_point, q, pad_value):
    """Checks if placing a test particle at 'r' would place three particles at the same point."""
    too_many_particles = jnp.count_nonzero(q_point != pad_value) >= 2
    like_charges = jnp.any(q_point == q)
    return too_many_particles | like_charges


def local_potential(i, r, L, LQ, rho, point_func, pad_value):
    """Potential at r excluding interactions from particle i."""
    n = L.shape[0]
    Lr_square, Lr_square_positions = centered_square(L, r, rho)
    Qr_square, _ = centered_square(LQ, r, rho)
    r_ = tuple(r)
    self_mask = jnp.isin(Lr_square, i)
    Lr_square = jnp.where(self_mask, pad_value, Lr_square)
    Qr_square = jnp.where(self_mask, pad_value, Qr_square)
    distances = lattice_distances_2D(r, Lr_square_positions, n)
    V_expanded = Qr_square * jnp.expand_dims(point_func(distances), -1)
    V_expanded = jnp.where(Lr_square == pad_value, 0.0, V_expanded)
    V = jnp.sum(V_expanded)
    return V


def potential_all(I, R, L, LQ, rho, point_func, pad_value):
    """Potential of each particle in I, excluding self-interaction."""
    n = L.shape[0]
    V = jax.vmap(local_potential, in_axes=(0, 0, None, None, None, None, None))(
        I, R, L, LQ, rho, point_func, pad_value)
    return V


def potential_energy_all(Q, V):
    """
    Q
        Array of particle charges. 1D, k.
    V
        Array of potentials at all positions in space. 1D, k.
    """
    return Q * V


def test_potential_factors(r, L, LQ, rho, point_func, pad_value):
    """
    Computes the individual contribution to the potential at 'r' produced by each particle. 
    Returns a square lattice of source potential values around 'r' of radius 'rho'.

    r
        Position we are computing the potential at. 1D, 2. 
    L
        Array of lattice points, with three sites per point filled with either a particle index
        or pad_value. 3D, nxnx3, pad_value-padded.
    LQ
        Array of lattice points, with three sites per point filled with either a particle charge
        or 0. 3D, nxnx3, 0-padded.
    rho
        Integer giving the range of the potential.
    point_func
        Function specifying the potential generated by a particle at a given distance. 
    """
    n = L.shape[0]
    Lr_square, Lr_square_positions = centered_square(L, r, rho)
    Qr_square, _ = centered_square(LQ, r, rho)
    distances = lattice_distances_2D(r, Lr_square_positions, n)
    V_expanded = Qr_square * jnp.expand_dims(point_func(distances), -1)
    V_expanded = jnp.where(Lr_square == pad_value, 0.0, V_expanded)
    return V_expanded, Lr_square


@partial(jax.jit, static_argnums=[3, 4, 5, 6])
def test_potential(r, L, LQ, rho, point_func, pad_value, pauli_exclusion=False):
    """
    The potential at 'r' generated by particles in the occupation lattices L and LQ, 
    with the shape of the potential generated by a point source determined by point_func. 
    When determining the potential energy of a particle i using this function the particle 
    itself should not be included in these lattices, only (at minimum) all other particles 
    that can interact via the potential with i. As such it cannot properly and fully factor in
    Pauli exclusion; for efficiency we set the option off by default, assuming that 
    Pauli exclusion is obeyed everywhere. 

    r
        Position we are computing the potential at. 1D, 2. 
    L
        Array of lattice points, with three sites per point filled with either a particle index
        or pad_value. 3D, nxnx3, pad_value-padded.
    LQ
        Array of lattice points, with three sites per point filled with either a particle charge
        or 0. 3D, nxnx3, 0-padded.
    rho
        Integer giving the range of the potential.
    point_func
        Function specifying the potential generated by a particle at a given distance. 
    pad_value
        Scalar.
    pauli_exclusion
        Boolean indicating whether to apply Pauli exclusion based on charge polarity or not.
    """
    def compute():
        V_expanded, _ = test_potential_factors(r, L, LQ, rho, point_func, pad_value)
        V = jnp.sum(V_expanded)
        return V

    if pauli_exclusion:
        r = tuple(r)
        r_sites, r_qsites = L[r], LQ[r]
        three_particles = jnp.all(r_sites != pad_value)
        one_particle = jnp.all(r_sites[1:] == pad_value)
        pauli_valid = jnp.all(jnp.unique_counts(r_qsites).counts == 1)
        finite = ~(three_particles) & (one_particle | pauli_valid)
        V = jax.lax.cond(finite, compute, lambda: jnp.inf)
    else:
        V = compute()

    return V


def potential_for(V_test, q_point, q, pad_value):
    """
    Calculates the potential of a test particle from the test potential, 
    incoporating Pauli exclusion of like charges.
    """
    violates_pe = violates_pauli_exclusion(q_point, q, pad_value)
    return jnp.where(violates_pe, jnp.inf, V_test)


def potential_energy(V, q):
    return jnp.where(V == jnp.inf, jnp.inf, q * V)


def make_potential_energy_func(L, LQ, rho, point_func, L_pad_value, LQ_pad_value):
    
    def func(r, q):
        V_test = test_potential(r, L, LQ, rho, point_func, L_pad_value)
        V = potential_for(V_test, LQ[tuple(r)], q, LQ_pad_value)
        U = potential_energy(V, q)
        return U

    return func


#@partial(jax.jit, static_argnums=[4, 5, 6, 7])
def potential_energy_factors(i, r, q, L, LQ, rho, point_func, factor_limit, pad_value=-1):
    """
    Computes the individual contribution to the potential energy of a particle i at 'r'
    of charge 'q' produced by each other particle. Assumes all potential factors are finite; 
    i.e. Pauli exclusion is never violated in 'L'. 

    i
        Scalar, index of the particle.
    r
        Position we are computing the potential at. 1D, 2. 
    q
        Scalar, charge of the particle. 
    L
        Array of lattice points, with three sites per point filled with either a particle index
        or pad_value. 3D, nxnx3, pad_value-padded.
    LQ
        Array of lattice points, with three sites per point filled with either a particle charge
        or 0. 3D, nxnx3, 0-padded.
    rho
        Integer giving the range of the potential.
    point_func
        Function specifying the potential generated by a particle at a given distance. 
    factor_limit
        Integer scalar giving the maximum number of particles within potential range of i. 
    pad_value
        Integer scalar used for padding to remove most zero energy factors. Defaults to -1. 
    """
    potential_factors, Lr_square = test_potential_factors(r, L, LQ, rho, point_func, pad_value)

    # remove self potential
    self_interact_mask = Lr_square == i
    Lr_square = jnp.where(self_interact_mask, pad_value, Lr_square)
    potential_factors = jnp.where(self_interact_mask, 0.0, potential_factors)

    energy_factors = potential_energy(potential_factors, q)

    # reduce to nonzero factors 
    pad_index = 2 * Lr_square.size
    nonzero_lattice_indices = jnp.nonzero(energy_factors, size=factor_limit, fill_value=pad_index)
    indices = Lr_square.at[nonzero_lattice_indices].get(mode="fill", fill_value=pad_value)
    energy_factors = energy_factors.at[nonzero_lattice_indices].get(mode="fill", fill_value=0.0)
    return energy_factors, indices


def bonded(U, max_energy):
    return U <= max_energy


def compute_bonds(i, energy_factors, factor_indices, max_energy, pad_value):
    """
    Calculates the bonds of particle i, with bonds defined by a maximum energy of i due to the bond.
    Assumes that particle i is bonded and max_energy < 0. Returns a padded 1D Array 
    containing the indices of particles bonded to i, sorted by increasing bond energy.  

    energy_factors:
        Array containing portions of the potential energy of particle i 
        generated by the interaction of other particles. 1D. 
    """
    sorted_indices = jnp.argsort(energy_factors)
    energy_factors = energy_factors[sorted_indices]
    factor_indices = factor_indices[sorted_indices]

    def find_bond(args):
        curr_index, _, remaining_energies = args
        energy_sums = jnp.cumsum(remaining_energies)
        index = jnp.argmax(energy_sums < max_energy)
        remaining_energies = zero_prefix(remaining_energies, index+1)
        return index, curr_index, remaining_energies

    def cond_fn(args):
        curr_index, last_index, remaining_energies = args
        return (curr_index != 0) | (remaining_energies[0] != 0)

    index = jax.lax.while_loop(cond_fn, find_bond, (0, 0, energy_factors))[1]
    factor_indices = fill_suffix(factor_indices, index, pad_value)
    return factor_indices


def compute_bound_state(i, all_bonds, pad_value):
    """
    i
        Index of the particle whose bound state we are computing. 
    all_bonds
        Array of indices, pad_value-padded, which indicates the particles bonded 
        to each particle of the system. 2D, kxm.
     """
    state = bfs(all_bonds, i, pad_value)
    return state


def compute_bound_states(all_bonds, pad_value):
    """
    Calculates all of the bound states, or "molecules", specified by the array all_bounds. 
    Returns a 1D Array of size k giving the label integer of the bound state associated to each particle. 
    """
    k = all_bonds.shape[0]
    curr_particle = jnp.int32(0)
    visited = jnp.zeros(k, dtype=bool)
    bound_states = jnp.zeros(k, dtype=int)
    count = jnp.int32(-1)

    def cond_fn(args):
        visited = args[1]
        return ~jnp.all(visited)

    def get_molecule(curr_particle, visited, bound_states, count):
        count += 1
        molecule = compute_bound_state(curr_particle, all_bonds, pad_value)
        bound_states = jnp.where(molecule, count, bound_states)
        visited = jnp.where(molecule, True, visited)
        curr_particle += 1
        return curr_particle, visited, bound_states, count

    def body_fn(args):
        curr_particle, visited, _, _ = args
        seen = visited[curr_particle]
        next_fn = lambda j, *_args: (j+1,) + _args
        return jax.lax.cond(seen, next_fn, get_molecule, *args)

    curr_particle, visited, bound_states, count = jax.lax.while_loop(
        cond_fn, body_fn, (curr_particle, visited, bound_states, count))
    return bound_states


def get_particles(I, bound_states, pad_value):
    
    def collection_fn(i):
        molecule_id = bound_states[i]
        return jnp.where(jnp.isin(molecule_id, I), i, pad_value)

    I_particles = jnp.fromfunction(collection_fn, bound_states.shape, dtype=int)
    return I_particles


def get_boundstates(bound_states, pad_value):
    I = jnp.arange(bound_states.shape[0])
    max_state = jnp.max(bound_states)
    I = jnp.where(I <= max_state, I, pad_value)
    return I


def bound_state_indices(I, bound_states, pad_value):
    
    def collection_fn(i):
        idx = jnp.argmax(I == i)
        return jnp.where(I[idx] == i, idx, pad_value)

    indices = jax.vmap(collection_fn)(bound_states)
    return indices


def centers_of_mass(R, M, bound_states):
    """
    Returns the center of mass and mass of each bound state 
    in a 0-padded 'k' length 2D Array and 1D Array, respectively.
    """
    k = R.shape[0]
    position_sums = jnp.zeros((k, 2), dtype=int)
    mass_sums = jnp.zeros((k,), dtype=int)
    position_sums = position_sums.at[bound_states].add(jnp.expand_dims(M, axis=-1) * R)
    mass_sums = mass_sums.at[bound_states].add(M)
    # use a mask to handle padding
    mass_sums_expanded = jnp.expand_dims(mass_sums, axis=-1)
    centers = jnp.where(mass_sums_expanded > 0, position_sums / mass_sums_expanded, 0)    
    return centers, mass_sums


def moments_of_inertia(R, M, coms, bound_states, n):
    """Returns the moment of inertia of each bound state in a 0-padded 'k' length 1D Array."""
    k = R.shape[0]
    MOIs = jnp.zeros((k,))
    arm_lengths = lattice_distances(R, coms, n) ** 2
    particle_MOIs = M * arm_lengths
    MOIs = MOIs.at[bound_states].add(particle_MOIs)
    return MOIs


def move(R, curr_coms, new_coms):
    """Move each particle using reference points. Should be a valid lattice translation."""
    return R + (new_coms - curr_coms).astype(int)


def standardize_angle(theta):
    """Standardizes an angle to between 0 and 360 degrees."""
    return theta % 360


def rotate(R, coms, thetas):
    """
    Rotates each particle about a given center of mass by a multiple 'theta' of 90 degrees,
    upper bounded by 270 degrees. Then shifts each particle to the nearest lattice site. 

    R
        Array of particle positions. 2D, kx2.
    coms
        Array of centers of mass. 2D, kx2.
    thetas
        Array of angles. 1D, k. 
    """
    # if at all slow could convert to direct array creation
    rotate_matrix90 = jnp.array([[0, -1], [1, 0]])
    rotate_matrix180 = jnp.array([[-1, 0], [0, -1]])
    rotate_matrix270 = jnp.array([[0, 1], [-1, 0]])

    @partial(jax.jit, static_argnums=[1])
    def origin_rotate(r, theta=90):
        if theta == 0:
            r_new = r
        elif theta == 90:
            r_new = rotate_matrix90 @ r 
        elif theta == 180:
            r_new = rotate_matrix180 @ r
        elif theta == 270:
            r_new = rotate_matrix270 @ r
        return r_new

    rotation_branches = tuple(partial(origin_rotate, theta=i) for i in (0, 90, 180, 270))
    
    def rotation_fn(r, com, theta):
        theta_idx = (theta // 90).astype(int)
        r_centered = r - com
        r_centered_rotated = jax.lax.switch(theta_idx, rotation_branches, r_centered)
        r_rotated = r_centered_rotated + com
        return r_rotated

    R_rotated = jax.vmap(rotation_fn)(R, coms, thetas)
    R_rotated = jnp.rint(R_rotated).astype(int)     # should be a uniform shift
    return R_rotated


def generate_interaction_field(I, R, L, LQ, rho, point_func, pad_value):
    n = L.shape[0]

    # generate neighborhood of each particle
    shifts = jnp.expand_dims(jnp.array(([0, 0], [1, 0], [-1, 0], [0, 1], [0, -1])), axis=0)
    R_nbhd = jnp.expand_dims(R, axis=1) + shifts
    R_nbhd = cylindrical_coordinates(R_nbhd, n)

    # generate potential; not necessary to correct for non-periodic boundary here
    potential_fn = jax.vmap(local_potential, in_axes=(None, 0, None, None, None, None, None))
    potential_fn = jax.vmap(potential_fn, in_axes=(0, 0, None, None, None, None, None))
    potentials = potential_fn(I, R_nbhd, L, LQ, rho, point_func, pad_value)

    # calculate the field, handling the boundary
    def gradient(r, V_nbhd):
        delta_y = jax.lax.select(r[1] == n-1, 0.0, V_nbhd[3] - V_nbhd[0])
        nabla_y = jax.lax.select(r[1] == 0, 0.0, V_nbhd[0] - V_nbhd[4])
        y_comp = (delta_y + nabla_y) / 2.0
        x_comp = (V_nbhd[1] - V_nbhd[2]) / 2.0
        return jnp.array((x_comp, y_comp))

    interaction_field = jax.vmap(gradient)(R, potentials)
    return interaction_field


@jax.jit
def planck_logdensity_fn(F, beta, alpha, delta):
    return jnp.log(alpha) + (jnp.log(F) * 3) - jnp.log(jnp.exp(beta * delta * F) - 1)


@jax.jit
def gamma_logdensity_fn(x, a, theta):
    return jnp.log(x)*(a - 1) - (x / theta)


@partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6])
def planck(key, beta, alpha, delta, theta, M, num_samples):
    keys = jax.random.split(key, num=num_samples)

    @jax.jit
    def sample_gamma_fn(data):
        key, _ = data
        key, subkey = jax.random.split(key)
        F = jnp.exp(jax.random.loggamma(subkey, 4)) * theta
        return key, F

    @jax.jit
    def sample_fn(key):

        @jax.jit
        def reject_fn(data):
            key, F = data
            key, subkey = jax.random.split(key)
            planck_logdensity = planck_logdensity_fn(F, beta, alpha, delta)
            gamma_logdensity = gamma_logdensity_fn(F, 4, theta)
            accept_ratio = jnp.exp(planck_logdensity - jnp.log(M) - gamma_logdensity)
            u = jax.random.uniform(subkey)
            return u >= accept_ratio

        key, F = sample_gamma_fn((key, 0))
        _, F = jax.lax.while_loop(reject_fn, sample_gamma_fn, (key, F))
        return F

    samples = jax.vmap(sample_fn)(keys)
    return samples


#@partial(jax.jit, static_argnums=[1, 2, 3])
def wien_approximation(key, beta, alpha, num_samples):
    keys = jax.random.split(key, num=num_samples)

    #@jax.jit
    def sample_fn(key):
        return jnp.exp(jax.random.loggamma(key, 4) - jnp.log(alpha) - jnp.log(beta))

    samples = jax.vmap(sample_fn)(keys)
    return samples


def occupation_mask(n, R):
    """
    Returns a particle occupation indicator array for an nxn space, 
    based on the particle positions contained in R.

    n
        Number of lattice sites in each dimension of the space.
    R
        Array of particle positions. 2D, kx2. 
    """
    mask = jnp.zeros((n, n), dtype=int).at[R[:, 0], R[:, 1]].set(1)
    return mask


def masked(x, y, occupation_mask):
    """
    Implements drive masking that treats particles as perfect barriers. 
    Returns 0 if there is a particle in {x} x [0,y), 1 otherwise. 
    Drive comes in from the side, making x-dimension correspond to rows, per C indexing.   
    """
    row = occupation_mask[x]
    i = jnp.argmax(row)
    v = jnp.max(row)
    return (v == 0) | (y <= i)


@partial(jax.jit, static_argnums=[0, 3])
def generate_drive_field(n, R, top_field, mask_func):
    occupied = occupation_mask(n, R)
    field = jnp.broadcast_to(top_field, (n, n))
    mask_fn = lambda x, y: mask_func(x, y, occupied)
    particle_mask = jnp.fromfunction(mask_fn, (n, n), dtype=int)
    return jnp.stack((jnp.zeros((n, n)), particle_mask * field), axis=-1)


#@partial(jax.jit, static_argnums=[1, 2, 3])
def brownian_noise(key, beta, gamma, num_samples):
    keys = jax.random.split(key, num=num_samples)

    #@jax.jit
    def sample_fn(key):
        return jnp.sqrt(2*gamma / beta) * jax.random.normal(key)

    samples = jax.vmap(sample_fn)(keys)
    return samples


def energy_to_velocity(field, m):
    norm = jnp.linalg.vector_norm(field, axis=-1, keepdims=True)
    speed = jnp.sqrt(2 * norm / m)
    return speed * jnp.where(norm == 0.0, field, field / norm)


def compute_particle_velocity_bound(i, r, interaction_field, brownian_field, external_field, energy_field, 
                                       transfer_field, mass, charge, molecule_mass, gamma, time_unit):
    r = tuple(r)
    nontransfer_force = ((charge * interaction_field[i]) + (mass * external_field[r]) 
                         + (jnp.sqrt(mass) * brownian_field[i]))
    transfer_force = mass * transfer_field[i]
    nontransfer_velocity = nontransfer_force / (gamma * mass)
    transfer_velocity = transfer_force / (gamma * molecule_mass)
    energy_velocity = energy_to_velocity(energy_field[i], mass)
    speed = jnp.linalg.vector_norm(nontransfer_velocity + transfer_velocity + energy_velocity)
    distance = speed * time_unit
    return distance


def compute_boundstate_velocity_bounds(I, R, bound_states, interaction_field, 
                                       net_field, masses, coms, M, Q, gamma, time_unit):
    M_expanded = jnp.expand_dims(M, axis=-1)
    Q_expanded = jnp.expand_dims(Q, axis=-1)
    masses_by_particle = jnp.expand_dims(masses[bound_states], axis=-1)
    coms_by_particle = coms[bound_states]
    interact_velocities_by_particle = Q_expanded * interaction_field / (gamma * masses_by_particle)
    other_velocities_by_particle = M_expanded * net_field / (gamma * masses_by_particle)
    net_velocities = interact_velocities_by_particle + other_velocities_by_particle
    net_angular_velocities = calculate_torque(R, coms_by_particle, net_velocities)

    def sum_velocities(i):
        particle_mask = bound_states == i 
        velocity = jnp.sum(jnp.expand_dims(particle_mask, axis=-1) * net_velocities, axis=0)
        angular_velocity = jnp.sum(particle_mask * net_angular_velocities, axis=0)
        return velocity, angular_velocity

    velocities, angular_velocities = jax.vmap(sum_velocities)(I)
    distances = jnp.linalg.vector_norm(velocities, axis=-1) * time_unit
    angular_distances = angular_velocities * time_unit
    return distances, angular_distances


def energy_barrier(path, potential_energy_func, max_boundary_energy, coupling_constant, lower_bound):
    """
    Calculates the energy barrier encountered by a particle moving along a path
    from a potential and coupling constant. 

    path 
        Array of adjacent positions, (-1)-padded. 2D, nx2. Assuming the path is length at most n-1. 
    potential_func
        Function that takes a particle index and a position 
        and returns the potential at that position relative to the particle.
    max_boundary_energy
        Scalar.
    coupling_constant
        Scalar. 
    lower_bound
        Scalar giving a lower bound for the potential energy. 
    """
    end = jnp.argmin(path, axis=0)[0] - 1

    def step_fn(i, max_energy):
        energy = potential_energy_func(path[i], coupling_constant)
        return jnp.max(jnp.array((energy, max_energy)))

    max_energy = jax.lax.fori_loop(1, end, step_fn, lower_bound)
    max_energy = jnp.max(jnp.array((max_energy, max_boundary_energy)))
    return max_energy - max_boundary_energy


def calculate_work_step(r_start, r_end, field, coupling_constant):
    delta = r_end - r_start
    work = jnp.dot(field, delta)
    total_work = jnp.linalg.vector_norm(field) * jnp.linalg.vector_norm(delta)
    return work, total_work


def calculate_work(path, field, coupling_constant):
    """
    Calculates the work done by a field on a particle moving along a path, 
    with the strength of the interaction determined by the appropriate 
    coupling constant of the particle (e.g. charge or mass). Also calculates
    the total work done by the field on the system, returning both values. 

    path
        Array of pairwise adjacent positions, (-1)-padded. 2D, lx2. 
        Assuming the path is length at most l-1. 
    field
        Array of field values at each position in space. 2D, nxnx2.
    coupling_constant
        Scalar. 
    """
    steps = jnp.diff(path, axis=0)
    end = jnp.argmin(path, axis=0)[0] - 1

    def add_work_step(i, vals):
        work, total_work = vals
        r = tuple(path[i])
        delta = steps[i]
        field_val = field[r]
        work_step = jnp.dot(field_val, delta)
        total_work_step = jnp.linalg.vector_norm(field_val)
        return work + work_step, total_work + total_work_step

    work, total_work = jax.lax.fori_loop(0, end, add_work_step, (0, 0))
    # multiply coupling at the end for efficiency
    return coupling_constant * work, coupling_constant * total_work


def calculate_work_v2(path, field, coupling_constant):
    """
    Calculates the work done by a field on a particle moving along a path, 
    with the strength of the interaction determined by the appropriate 
    coupling constant of the particle (e.g. charge or mass).

    path
        Array of adjacent positions, (-1)-padded. 2D, nx2. Assuming the path is length at most n-1. 
    field
        Array of field values at each position in space. 2D, nxnx2.
    coupling_constant
        Scalar. 
    """
    steps = jnp.diff(path, axis=0)
    end = jnp.argmin(path, axis=0)[0] - 1

    def get_work_step(r, delta):
        return jnp.dot(field[tuple(r)], delta)

    work_steps = jax.vmap(get_work_step)(path, steps)
    work_steps[end] = 0        # to remove contribution from last fake delta
    work = jnp.sum(work_steps)

    return coupling_constant * work     # multiply coupling at the end for efficiency


def calculate_total_work(total_path_work, brownian_field, external_field, 
                         transfer_field, i, mass, r_start, r_end, d):
    diff_norm = jnp.linalg.norm(r_end - r_start)
    excess_force_magnitude = mass * (jnp.linalg.norm(external_field[tuple(r_end)]) 
                                      + jnp.linalg.norm(transfer_field[i]))
    excess_force_magnitude += jnp.sqrt(mass) * jnp.linalg.norm(brownian_field[i])
    total_work = total_path_work + (excess_force_magnitude * (d - diff_norm))
    return total_work


def calculate_boundstate_work(i, bound_states, net_field, R, M, start_com, end_com, end_theta):
    particle_mask = jnp.expand_dims(bound_states == i, axis=-1)
    M = jnp.expand_dims(M, axis=-1)

    # total force
    forces = M * particle_mask * net_field
    total_force = jnp.sum(forces, axis=0)

    # translation work
    delta = end_com - start_com
    translate_work = jnp.dot(total_force, delta)

    # total torque
    torques = calculate_torque(R, start_com, forces)
    total_torque = jnp.sum(torques, axis=0)

    # rotation work
    rotate_work = jnp.dot(total_torque, jnp.deg2rad(end_theta))

    work = translate_work + rotate_work
    return work, total_force, total_torque


def calculate_end_field(brownian_field, external_field, energy_field, 
                        transfer_field, i, r_start, r_end, mass, gamma):
    extra_field = energy_to_velocity(energy_field[i], mass) * gamma
    return ((brownian_field[i] / jnp.sqrt(mass)) + external_field[tuple(r_end)] 
            + extra_field + transfer_field[i])


def calculate_torque(r, r_axis, f_vector):
    position_vec = r - r_axis
    return jnp.cross(position_vec, f_vector)


def torque_force(torque, r, r_axis):
    position_vec = r - r_axis
    return cross_1D(torque, position_vec) / jnp.linalg.vector_norm(position_vec) ** 2


def cross_1D(y_1D, x_2D):
    y_3D = jnp.concatenate((jnp.zeros(2), jnp.expand_dims(y_1D, axis=0)))
    x_3D = jnp.concatenate((x_2D, jnp.zeros(1)))
    return jnp.cross(y_3D, x_3D)[:2]


def transferred_field(r_source, force_source, r_dest, m_block, I_block, torque):
    angular_component = jnp.cross(torque, r_dest - r_source) / I_block
    linear_component = force_source / m_block
    return linear_component + angular_component


def generate_full_transferred_field(I, net_field, R, bound_states, masses, coms, MOIs, I_all, pad_value):
    """
    I
        Array of particle indices specifying a subset of the particles. 1D, l, pad_value-padded.
    net_field
        Array giving the field experienced by each particle in its current state. 2D, kx2. 
    R
        Array of particle positions. 2D, kx2. 
    bound_states
        Array giving the bound state index for each particle. 1D, k.
    masses
        Array giving the mass of each bound state, 0-padded. 1D, k.
    coms
        Array giving the center of mass of each bound state, 0-padded. 1D, kx2.
    MOIs 
        Array giving the moment of inertia of each bound state, 0-padded. 1D, k.
    I_all
        Array containing the indices of all particles. 1D, k.
    pad_value
        Scalar.
    """
    def source_factors(i):
        r, molecule_id = R[i], bound_states[i]
        mass, com, MOI = masses[molecule_id], coms[molecule_id], MOIs[molecule_id]
        field_i = net_field[i]
        weight = jnp.where(MOI == 0.0, 0.0, mass / MOI)
        weighted_torque_i = weight * calculate_torque(r, com, field_i)
        torque_field_part = cross_1D(weighted_torque_i, -r)
        return field_i, weighted_torque_i, torque_field_part

    field_parts, weighted_torques, torque_field_parts = jax.vmap(source_factors)(I)
    molecules_I = bound_states[I]
    pad_mask = I != pad_value

    def collect_by_molecule(i):
        particle_mask = pad_mask & (molecules_I == i)
        particle_mask_expanded = jnp.expand_dims(particle_mask, axis=-1)
        field_part = jnp.sum(particle_mask_expanded * field_parts, axis=0)
        weighted_torque = jnp.sum(particle_mask * weighted_torques, axis=0)
        torque_field_part = jnp.sum(particle_mask_expanded * torque_field_parts, axis=0)
        return field_part, weighted_torque, torque_field_part

    molecule_indices = jnp.arange(masses.shape[0])
    field_parts, weighted_torques, torque_field_parts = jax.vmap(collect_by_molecule)(molecule_indices)

    def calculate_transfer(i):
        r, molecule_id = R[i], bound_states[i]
        field_part = field_parts[molecule_id]
        torque_field_part1 = torque_field_parts[molecule_id]
        weighted_torque = weighted_torques[molecule_id]
        torque_field_part2 = cross_1D(weighted_torque, r)
        return field_part + torque_field_part1 + torque_field_part2

    transfer_field = jax.vmap(calculate_transfer)(I_all)
    return transfer_field


def transfer_mask(excess_works, potential_energies, epsilon):
    return excess_works < epsilon * potential_energies


def calculate_excitations(i, R, L, work, delta):
    """
    Calculates the energy received by each particle in an energy emission event.
    Uses a simple dispersal method where each particle within 'delta' distance 
    receives an energy amount proportional to its distance from the source particle i. 
    No screening from other particles.

    i
        Index of the particle generating the spontaneous emission. 
    R
        Array of particle positions. 2D, kx2.
    L
        Array of lattice sites, with each element indicating whether the site is empty 
        or filled by 1 or 2 particles. Assumes at most two particles may occupy each site. 
        2D, nxn. 
    work
        Scalar, the amount of work released during the emission. 
    delta
        Range of radiation emission. 

    Returns:
        An Array representing a delta radius square around i with each element 
        giving the amount of energy received at that position. 
    """
    r = R[i]
    n = L.shape[0]
    r_square, r_square_positions = centered_square(L, r, delta)
    distances = lattice_distances_2D(r, r_square_positions, n)

    excited_mask = (distances <= delta) & (distances != 0)    # exclude r itself from excitation
    excited_particle_mask = (r_square != 0) & excited_mask

    inv_distances = 1 / distances
    inv_distances = jnp.where(inv_distances == jnp.inf, 0.0, inv_distances)
    # factor of 2 to account for two particle sites at each posiiton
    normalizer = 1 / jnp.sum(2 * inv_distances * excited_mask)    

    excitations = normalizer * work * inv_distances * excited_particle_mask
    return excitations, r_square_positions


def excitation_mask(excess_works, potential_energies, epsilon):
    return excess_works >= epsilon * potential_energies


def merge_excitations(R, L, excitation_arrays, position_arrays):
    """
    Returns an energy field, a 3D array of shape nxnx2 giving an 'energy vector' at each lattice site. 
    R 
        Array of particle positions. 2D, kx2. 
    L
        Array of lattice sites, with each element indicating whether the site is empty 
        or filled by 1 or 2 particles. Assumes at most two particles may occupy each site. 
        2D, nxn. 
    excitation_arrays
        Array where each row is a 2D array of energies. 3D, kxdxd.
    position_arrays
        Array where each row is a 3D array giving the positions corresponding to the energies
        in excitation_arrays. 4D, kxdxdx2. 
    """
    R_expanded = jnp.expand_dims(jnp.expand_dims(R, axis=1), axis=1)
    excitation_vectors = normalize(position_arrays - R_expanded)
    excitation_arrays = jnp.expand_dims(excitation_arrays, axis=-1) * excitation_vectors
    energy_field = jnp.zeros(L.shape + (2,), dtype=float)
    positions = position_arrays.reshape(-1, 2)
    excitations = excitation_arrays.reshape(-1, 2)
    energy_field = energy_field.at[positions[:, 0], positions[:, 1], :].add(excitations)
    return energy_field


def calculate_excitation_work(i, r_start, r_end, energy_field):
    """
    Calculates the absorbed work and total work produced by an energy field on a particle 
    travelling from 'r_start' to 'r_end'. 
    """
    update_vec = r_end - r_start
    excitation = energy_field[i]
    energy = jnp.linalg.vector_norm(excitation)
    excitation_work = jnp.max(jnp.array((jnp.dot(excitation, update_vec), energy)))
    total_excitation_work = jnp.linalg.vector_norm(excitation)
    return excitation_work, total_excitation_work


def add_boundstate_energy_transfers(I, I_bound_states, excess_works, total_forces, total_torques, 
                                    coms, R, curr_bound_states, prev_bound_states, energy_field, 
                                    curr_energy_factors, curr_factor_indices, prev_energy_factors, 
                                    prev_factor_indices, pad_value):
    """
    Assumes each particle has at most one bound state of I within interaction range. 
    Assumes prev_bound_states is a refinement of curr_bound_states. 

    I
        Array of all particle indices. 1D, k.
    I_bound_states
        Array of bound state indices. 1D, l, pad_value-padded.
    excess_works
        Array of excess work values. 1D, l, 0-padded.
    total_forces
        Array of total force vectors. 2D, lx2, 0-padded.
    total_torques
        Array of total torque vectors. 1D, l, 0-padded.
    coms
        Array of centers of mass. 1D, kx2, 0-padded.
    R
        Array of particle positions. 2D, kx2.
    curr_bound_states
        Array giving the bound state index for each particle. 1D, k.
    prev_bound_states
        Array giving the previous bound state index for each particle. 1D, k.
    energy_field
        Array of energy field values. 2D, kx2. 
    curr_energy_factors
        Array of energy factors for each particle. 2D, kxd, pad_value-padded. 
    curr_factor_indices
        Array containing indices of the particles generating each of the energy factors
        in curr_energy_factors. 2D, kxd, pad_value-padded. 
    prev_energy_factors
        Array of previous energy factors for each particle. 2D, kxd, pad_value-padded. 
    prev_factor_indices
        Array containing indices of the particles generating each of the energy factors
        in prev_energy_factors. 2D, kxd, pad_value-padded. 
    pad_value
        Scalar.
    """
    k = curr_bound_states.shape[0]
    l = excess_works.shape[0]
    I_bound_states = bound_state_indices(I_bound_states, prev_bound_states, pad_value)

    def calculate_factor(EF, FI, i_B, is_receiving, weighted=True):
        FI = replace(FI, pad_value, 2*k)
        FI_curr_molecules = curr_bound_states.at[FI].get(mode="fill", fill_value=0)
        FI_prev_molecules = prev_bound_states.at[FI].get(mode="fill", fill_value=0)
        A_mask = jnp.logical_xor(FI_curr_molecules, FI_prev_molecules)
        i_A = prev_bound_states[FI_prev_molecules[jnp.unravel_index(jnp.argmax(A_mask), FI.shape)]]
        B_mask = FI_curr_molecules == i_B
        A_factor = jnp.sum(is_receiving * A_mask * B_mask * EF)
        return A_factor, i_A

    def aggregate_factors(i, EF_curr, FI_curr, EF_prev, FI_prev): 
        i_B = curr_bound_states[i]
        is_receiving = prev_bound_states[i] == i_B

        A_factor_curr, i_A = calculate_factor(EF_curr, FI_curr, i_B, is_receiving)
        A_factor_prev, _ = calculate_factor(EF_prev, FI_prev, i_B, is_receiving)
        return A_factor_prev - A_factor_curr, i_A

    molecule_delta_factors, transfer_bound_states = jax.vmap(aggregate_factors)(
        I, curr_energy_factors, curr_factor_indices, prev_energy_factors, prev_factor_indices)     

    normalizers = compute_group_sums(molecule_delta_factors, transfer_bound_states, l)
    normalizers = jnp.expand_dims(normalizers, axis=-1)

    # assemble excess works and energy vectors
    excess_works = jnp.expand_dims(excess_works[I_bound_states[transfer_bound_states]], axis=-1)
    total_forces = total_forces[I_bound_states[transfer_bound_states]]
    total_torques = total_torques[I_bound_states[transfer_bound_states]]
    coms = coms[transfer_bound_states]
    f_taus = jax.vmap(torque_force)(total_torques, R, coms)
    unit_energy_vectors = normalize(total_forces + f_taus)

    molecule_delta_factors = jnp.expand_dims(molecule_delta_factors, axis=-1)

    energy_field = energy_field.at[:].add(
        molecule_delta_factors * excess_works * unit_energy_vectors / normalizers)
    return energy_field