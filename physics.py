"""
Note: we use matrix indexing, so row indexing is the first or x dimension and column indexing
is the second or y dimension. 
"""

from functools import partial

import jax
import jax.numpy as jnp

from utils import *
from geometry import *


def occupied_neighbors(nbhd_I, R, n, pad_value):
    return neighborhood_mask(nbhd_I, R, n, pad_value).any(1)


def test_potential_energies(R_I, Q_I, R_test, Q, n, pad_value):
    nbhd = jax.vmap(generate_open_neighborhood, in_axes=(0, None, None))(R_I, n, pad_value)
    nbhd_mask = neighborhood_mask(nbhd, R_test, n, pad_value).any(-1)
    V = jnp.sum(nbhd_mask * Q[None, :], axis=-1)
    U = V * Q_I
    return U


@partial(jax.jit, static_argnums=[4, 5])
def neighborhood_potential_energies(I, nbhd_I, Q, R, n, pad_value):
    Q_I = Q[I]
    R_test = R.at[I].set(pad_value)

    # for pauli exclusion
    nbhd_occupation = occupied_neighbors(nbhd_I, R_test, n, pad_value)

    U_nbhd = jax.vmap(test_potential_energies, in_axes=(1, None, None, None, None, None))(
        nbhd_I, Q_I, R_test, Q, n, pad_value).T
    U_nbhd += jnp.nan_to_num(nbhd_occupation * jnp.inf, posinf=jnp.inf)
    return U_nbhd


@partial(jax.jit, static_argnums=[5, 6, 7])
def neighborhood_potential_energies_dynamic(I, nbhd_I, Q, R, indices, n, buffer_size, pad_value):
    remaining = I != pad_value
    slice_ = jnp.extract(remaining, indices, size=buffer_size, fill_value=pad_value)
    U_nbhd = jnp.zeros(nbhd_I.shape[:2])

    def cond_fn(args):
        remaining = args[2]
        return jnp.any(remaining)

    def potential_fn(args):
        U_nbhd, slice_, remaining = args
        I_slice = I[slice_]
        nbhd_I_slice = nbhd_I[slice_]
        Q_Is = Q[I_slice]
        R_test = R.at[I_slice].set(pad_value)

        # for pauli exclusion
        nbhd_occupation = occupied_neighbors(nbhd_I_slice, R_test, n, pad_value)

        U_nbhd_slice = jax.vmap(test_potential_energies, in_axes=(1, None, None, None, None, None))(
            nbhd_I_slice, Q_Is, R_test, Q, n, pad_value).T

        # update the carry
        U_nbhd.at[slice_].add(U_nbhd_slice + jnp.nan_to_num(nbhd_occupation * jnp.inf, posinf=jnp.inf))
        remaining = remaining.at[slice_].set(False)
        slice_ = jnp.extract(remaining, I, size=buffer_size, fill_value=pad_value)
        return U_nbhd, slice_, remaining

    U_nbhd = jax.lax.while_loop(cond_fn, potential_fn, (U_nbhd, slice_, remaining))[0]
    return U_nbhd


def bound(E):
    return E < 0


def compute_bond_data(i, K, R, L, Q, pad_value):
    n, k = L.shape[0], R.shape[0]
    r, q = R[i], Q[i]

    r_nbhd = generate_open_neighborhood(r, n, pad_value)
    nbhd = L.at[r_nbhd[:, 0], r_nbhd[:, 1]].get(mode="fill", fill_value=pad_value)
    nbhd_charges = Q.at[nbhd].get(mode="fill", fill_value=0)

    energy_factors = q * nbhd_charges
    is_bound = (jnp.sum(energy_factors) + K) < 0
    return nbhd, energy_factors, is_bound


@partial(jax.jit, static_argnums=[3])
def determine_bonds(energy_factors, nbhd, is_bound, pad_value):
    nbhd_bound = is_bound[nbhd]
    bonded_nbhd = (energy_factors < 0) & nbhd_bound
    return jnp.extract(bonded_nbhd, nbhd, size=4, fill_value=pad_value)


@partial(jax.jit, static_argnums=[2])
def compute_bound_state(i, bonds, pad_value):
    """
    i 
        Index of the particle whose bound state we are computing.
    bonds
        Array of indices with index-safe padding which indicates the particles bonded to each particle. 
        Should contain all the particles in a subset of the bound states. 2D, mx4.
    """
    state = bfs(bonds, i, pad_value)
    return state


@partial(jax.jit, static_argnums=[4, 5])
def compute_bound_states(I, bonds, bound_states, visited, l, pad_value):
    """
    Calculates the bound states, or "molecules", of particles in 'I' specified by the array 'bonds' 
    starting from the partially discovered 'bound_states' inhibitated by particles that are 'visited'.

    Returns a 1D Array of size k giving the label integer of the bound state 
    associated to each particle.
    """
    I_slice = I[:l]

    def cond_fn(args):
        visited = args[2]
        return ~jnp.all(visited)

    def find_fn(args):
        I, I_slice, visited, bound_states = args

        new_states = jax.vmap(compute_bound_state, in_axes=(0, None, None))(I_slice, bonds, pad_value)
        labels = jnp.argmax(new_states, axis=-1, keepdims=True)
        repeats = jnp.sum(new_states, axis=0)

        new_states = jnp.sum(new_states * labels, axis=0) // repeats    # 0 becomes -1 from floor divide
        visited_new = repeats.astype(bool)
        bound_states = jnp.where(visited_new, new_states, bound_states)
        visited = jnp.logical_or(visited_new, visited)

        I_slice = jnp.extract(~visited, I, size=l, fill_value=pad_value)

        return I, I_slice, visited, bound_states

    bound_states = jax.lax.while_loop(cond_fn, find_fn, (I, I_slice, visited, bound_states))[3]
    return bound_states

### unused ideas
### - could consider vmapping by sorting I to separate particles likely 
### in different bound states to reduce redundant computation
### - could consider early termination of the bfs
### - more difficult: stitching bound states together, allowing for reduced computation


def centers_of_mass(R, M, bound_states):
    """
    Returns the center of mass and mass of each bound state 
    in a 0-padded 'k' length 2D Array and 1D Array, respectively.
    """
    mass_sums = compute_group_sums(M, bound_states)
    position_sums = compute_group_sums(jnp.expand_dims(M, axis=-1) * R, bound_states)
    # use a mask to handle padding
    mass_sums_expanded = jnp.expand_dims(mass_sums, axis=-1)
    centers = jnp.where(mass_sums_expanded > 0, position_sums / mass_sums_expanded, 0)
    return centers, mass_sums


def move(R, coms_curr, coms_new, n):
    """Move each particle using reference points. Should be a valid lattice translation."""
    return cylindrical_coordinates(R + (coms_new - coms_curr).astype(int), n)


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


@partial(jax.jit, static_argnums=[1, 2, 3])
def wien_approximation(key, beta, alpha, num_samples):
    keys = jax.random.split(key, num=num_samples)

    @jax.jit
    def sample_fn(key):
        return jnp.exp(jax.random.loggamma(key, 4) - jnp.log(alpha) - jnp.log(beta))

    samples = jax.vmap(sample_fn)(keys)
    return samples


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


@partial(jax.jit, static_argnums=[2, 3])
def generate_drive_field(R, top_field, mask_func, n):
    occupied = generate_unlabeled_lattice(R, n)
    field = jnp.broadcast_to(top_field, (n, n))
    mask_fn = lambda x, y: mask_func(x, y, occupied)
    particle_mask = jnp.fromfunction(mask_fn, (n, n), dtype=int)
    return jnp.stack((jnp.zeros((n, n)), particle_mask * field), axis=-1)


@partial(jax.jit, static_argnums=[3, 4])
def calculate_momentum_vectors(p, field, q, mu, gamma):
    p_v = (1 - gamma) * p
    impulse = mu * q * field
    p_nv = p + impulse
    p_ne = p_v + impulse
    return p_v, p_nv, p_ne


def lattice_norm_squared(p):
    return jnp.linalg.vector_norm(p, axis=-1, ord=1) ** 2


def kinetic_energy(p, m):
    return lattice_norm_squared(p) / (2 * m)


def calculate_partial_kinetic_factors(p_nv, p_ne, double_m):
    K_ne = lattice_norm_squared(p_ne) / double_m
    K_nv = lattice_norm_squared(p_nv) / double_m
    Q_delta_momentum = K_nv - K_ne
    return Q_delta_momentum, K_nv, K_ne


def calculate_kinetic_factors(p, p_nv, p_ne, m):
    double_m = 2 * m
    K = lattice_norm_squared(p) / double_m
    Q_delta_momentum, K_nv, K_ne = calculate_partial_kinetic_factors(p_nv, p_ne, double_m)
    E_emit = K_nv - K
    return Q_delta_momentum, E_emit, K_ne


@partial(jax.jit, static_argnums=[5])
def calculate_heat_released(p_ne, e, Q_delta_mom, U, U_e, mu):
    e = e.astype(p_ne.dtype)
    Q_delta_pot = U_e - U
    Q_delta = Q_delta_mom + (mu * jnp.dot(p_ne, e)) - Q_delta_pot
    return Q_delta


@partial(jax.jit, static_argnums=[5, 6])
def calculate_probabilities(P_ne, Q_delta_mom, U, U_e, boundary_mask, mu, beta):
    shifts = get_shifts()
    heat_fn = jax.vmap(jax.vmap(calculate_heat_released, 
        in_axes=(None, 0, None, None, 0, None)), in_axes=(0, None, 0, 0, 0, None))
    Q_deltas = heat_fn(P_ne, shifts, Q_delta_mom, U, U_e, mu)
    logdensities = beta * Q_deltas
    densities = boundary_mask * jnp.exp(logdensities)
    Z = jnp.sum(densities, axis=-1)
    probabilities = densities / jnp.expand_dims(Z, axis=-1)
    return probabilities, logdensities


@partial(jax.jit, static_argnums=[4])
def determine_emissions(U, is_bound, E_emit, K_ne, epsilon):
    energy_to_emit = E_emit > 0
    high_energy = K_ne >= (-epsilon * U)
    return jnp.logical_and(jnp.logical_and(energy_to_emit, is_bound), high_energy)


@partial(jax.jit, static_argnums=[6, 7, 8])
def calculate_emissions(r, L, M, Q, P, E_emit, delta, mu, pad_value):
    """
    Calculates the field update vector for each particle in an energy emission event at 'r'.
    Uses a simple dispersal method where each particle within 'delta' distance receives
    an energy amount proportional to its distance from the source particle at 'r'.
    No screening from other particles.

    r
        Position of the source particle, matching L. 2D, 2.
    L
        Array of lattice sites, with each filled site indicated by the index of the particle.
        Otherwise filled with 'pad_value'. 2D, nxn.
    M
        Array of particle masses. 1D, k.
    Q 
        Array of particle charges. 1D, k.
    P
        Array of particle momenta. 2D, kx2. 
    E_emit
        Energy emitted by the particle at 'r'. Assumed to be positive.
    delta
        Range of energy emission.
    mu
        Scalar.
    """
    n = L.shape[0]
    Lr_square, Lr_square_positions = centered_square(L, r, delta)
    Mr_square, Qr_square, Pr_square = M[Lr_square], Q[Lr_square], P[Lr_square]
    distances = lattice_distances_2D(r, Lr_square_positions, n)

    excited_mask = (distances <= delta) & (distances != 0)  # exclude r itself from excitation
    excited_particle_mask = (Lr_square != pad_value) & excited_mask

    inv_distances = 1 / distances
    inv_distances = jnp.where(inv_distances == jnp.inf, 0.0, inv_distances)
    normalizer = 1 / jnp.sum(inv_distances * excited_mask)

    directions = Lr_square_positions - jnp.expand_dims(r, axis=0)
    Pr_norm_square = jnp.linalg.vector_norm(Pr_square, axis=-1, ord=1)

    field = 2 * Mr_square * normalizer * E_emit
    field = (Pr_norm_square ** 2) + (field * inv_distances)
    field = jnp.expand_dims(jnp.sqrt(field) - Pr_norm_square, axis=-1)
    field = directions * jnp.expand_dims(inv_distances, axis=-1) * field
    field = field / (mu * jnp.expand_dims(Qr_square, axis=-1))
    field = jnp.expand_dims(excited_particle_mask, axis=-1) * field 

    return field, Lr_square_positions


@partial(jax.jit, static_argnums=[2])
def merge_excitations(field_arrays, position_arrays, n):
    field = jnp.zeros((n, n, 2), dtype=float)
    positions = position_arrays.reshape(-1, 2)
    excitations = field_arrays.reshape(-1, 2)
    field = field.at[positions[:, 0], positions[:, 1], :].add(excitations)
    return field