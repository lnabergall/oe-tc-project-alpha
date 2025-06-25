from functools import partial

import jax
import jax.numpy as jnp

from utils import *


def lattice_indices(A, n):
    return (A[..., 0] * n) + A[..., 1]


@partial(jax.jit, static_argnums=[1])
def cylindrical_coordinates(A, n):
    """
    Convert an array A of 2D vectors to cylindrical coordinates, 
    with a periodic first or 'x' dimension.
    """
    return A.at[..., 0].set(A[..., 0] % n)


@partial(jax.jit, static_argnums=[1])
def periodic_norm(x_diff, n):
    return jnp.min(jnp.stack((x_diff, n - x_diff)))


@partial(jax.jit, static_argnums=[2])
def lattice_distance(r, s, n):
    """Compute lattice distance between points r and s."""
    diff = jnp.abs(r - s)
    y_norm = diff[1]
    x_diff = diff[0]
    x_norm = periodic_norm(x_diff, n)
    norm = x_norm + y_norm
    return norm


lattice_distances = jax.vmap(lattice_distance, (0, 0, None))
lattice_distances_1D = jax.vmap(lattice_distance, (None, 0, None))
lattice_distances_2D = jax.vmap(lattice_distances_1D, (None, 0, None))


@jax.jit
def get_shifts():
    return jnp.array(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)), dtype=int)


@partial(jax.jit, static_argnums=[1])
def generate_neighborhood(x, n, pad_value):
    """
    Generate the neighborhood of a point 'x' in a lattice of size n, 
    with periodic first dimension.
    """
    nbhd = cylindrical_coordinates(x[None, ...] + get_shifts(), n)
    nbhd = jnp.where((nbhd >= n) | (nbhd < 0), pad_value, nbhd)
    return nbhd


@partial(jax.jit, static_argnums=[1])
def generate_open_neighborhood(x, n, pad_value):
    """
    Generate the neighborhood of a point 'x' in a lattice of size n, with periodic first dimension, 
    strict boundaries on the second dimension, and excluding 'x' itself.
    """
    steps = jnp.array(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=int)
    nbhd = cylindrical_coordinates(x.reshape((1, 2)) + steps, n)
    nbhd = jnp.where((nbhd >= n) | (nbhd < 0), pad_value, nbhd)
    return nbhd


def neighborhood_mask(nbhd, R, n, pad_value):
    """
    Generates a 2D mask indicating where elements of 'nbhd' and 'R' coincide. 
    The first dimension indexes 'nbhd' while the second indexes 'R'.
    """
    nbhd_indices = lattice_indices(nbhd, n)     # open nbhd
    R_indices = lattice_indices(R, n)
    nbhd_mask = nbhd_indices[:, None, :] == R_indices[:, None][None, :, :]
    return nbhd_mask


@partial(jax.jit, static_argnums=[1,2])
def square_indices(r, delta, n):
    x, y = r
    diameter = (2 * delta) + 1
    row_indices = (x + jnp.arange(diameter) - delta) % n
    col_start = jnp.clip(y - delta, 0, n - diameter)
    col_indices = jnp.arange(diameter) + col_start
    return row_indices, col_indices


@partial(jax.jit, static_argnums=[2])
def centered_square(L, r, delta):
    """
    Generate the subset of the lattice 'L' consisting of a square of radius 'delta' centered at 'r',
    along with the positions in that subset. Adjusts for periodicity in the x-dimension and 
    boundary limits in the y-dimension.
    """
    n = L.shape[0]
    x_indices, y_indices = square_indices(r, delta, n)
    square = L[x_indices[:, None], y_indices[None, :]]
    positions = jnp.stack(jnp.meshgrid(x_indices, y_indices, indexing="ij"), axis=-1)
    return square, positions


@partial(jax.jit, static_argnums=[1, 2])
def generate_lattice(R, n, pad_value):
    L = jnp.full((n, n), pad_value)
    L = L.at[R[:, 0], R[:, 1]].set(jnp.arange(R.shape[0]))
    return L


@partial(jax.jit, static_argnums=[3])
def add_to_lattice(L, I, R, pad_value):
    """Add particles in I to lattice L. Assumes no particles are co-located."""
    R_I = R.at[I].get(mode="fill", fill_value=pad_value)
    return L.at[R_I[:, 0], R_I[:, 1]].set(I, mode="drop")


@partial(jax.jit, static_argnums=[2])
def remove_from_lattice(L, I, pad_value):
    """Remove particles in I from lattice L."""
    removal_mask = jnp.isin(L, jnp.where(I == pad_value, pad_value + 1, I))
    return jnp.where(removal_mask, pad_value, L)


@partial(jax.jit, static_argnums=[1])
def generate_unlabeled_lattice(R, n):
    """
    Generate the unlabeled occupation lattice corresponding to R. The lattice is an nxn Array; 
    occupation is indicated by 0 or 1, indicating whether a site is filled by a particle or not. 
    """
    L = jnp.zeros((n, n), dtype=int)
    L = L.at[R[:, 0], R[:, 1]].set(1)
    return L


@partial(jax.jit, static_argnums=[2])
def generate_property_lattice(L, C, pad_value):
    """Gather property values from a 1D Array C using indices in an Array L while ignoring padding."""
    return C.at[L].get(mode="fill", fill_value=pad_value)


@partial(jax.jit, static_argnums=[2, 3])
def four_partition(R, L, m, pad_value):
    """
    Partitions the lattice 'L' into eight isomorphic sets whose elements are all distance four apart.
    The number of elements in each independent set should be at most 'm'. 

    R
        Array of particle positions. 2D, kx2.
    L
        Labeled occupation array. 2D, nxn.
    pad_value
        Scalar.

    Returns:
        8xm Array whose rows contain the eight independent sets, pad_value padded. 
    """
    n = L.shape[0]

    def collect_part(i):
        row_shift = i >= 4
        col_shift = i % 4
        half_grid = jnp.indices(((n + 3) // 4, (n + 3) // 4)).at[:].multiply(4)
        half_grid = half_grid.at[0].add(row_shift)
        half_grid = half_grid.at[1].add(col_shift)
        missing_col = half_grid[:, :, 0].at[0].add(2).at[1].add(-2)
        missing_col = jnp.where(col_shift >= 2, missing_col, 2*n)
        half_grid = half_grid.reshape((half_grid.shape[0], -1))
        indices = jnp.concat((half_grid, missing_col, half_grid + 2), axis=-1)
        return L.at[indices[0], indices[1]].get(mode="fill", fill_value=pad_value)

    P = jax.vmap(collect_part)(jnp.arange(8))
    P = rearrange_padding(P, pad_value)[:, :m] 

    return P


@partial(jax.jit, static_argnums=[3])
def four_group_partition(bound_states, L, key, part_size, pad_value):
    k, m, n = bound_states.shape[0], jnp.max(bound_states), L.shape[0]
    L_bs = jnp.where(L == pad_value, pad_value, bound_states[L])
    label_indicator = jnp.zeros((k,), dtype=bool).at[bound_states].set(True)
    coloring = jnp.full((k,), pad_value)

    def any_true_per_id(L_mask):
        flat_mask = L_mask.ravel().astype(int)
        flat_ids = L_bs.ravel()
        mask = jnp.zeros((k,), dtype=int)
        return mask.at[flat_ids].max(flat_mask, mode="drop")

    def cond_fn(args):
        coloring = args[1]
        return jnp.any((coloring == pad_value) & label_indicator)

    def color_fn(args):
        step, coloring, key = args
        key, subkey = jax.random.split(key)

        # random 32-bit priorities for mixing
        priorities = jax.random.randint(subkey, (k,), 1, 2**30, dtype=int)
        priorities = jnp.where((coloring == pad_value) & label_indicator, priorities, 0)
        L_p = priorities.at[L_bs].get(mode="fill", fill_value=0)

        # max priority in every 7x7 square
        wrapped = jnp.concatenate((L_p[-3:], L_p, L_p[:3]), axis=0)
        nbhd_max = jax.lax.reduce_window(wrapped, 0, jax.lax.max, (7, 7), (1, 1), "SAME")[3:n+3]
        L_loses = (nbhd_max != L_p) & (L_bs != pad_value)
        lose_indicator = any_true_per_id(L_loses)

        # winners are uncoloured that never lost
        winners_mask = (lose_indicator == 0) & (coloring == pad_value) & label_indicator
        winner_priorities = priorities * winners_mask.astype(int)

        # choose at most part_size winners
        top_vals, top_ids = jax.lax.top_k(winner_priorities, part_size)
        valid = top_vals > 0
        selected_ids = jnp.where(valid, top_ids, pad_value)
        coloring = coloring.at[selected_ids].set(step, mode="drop")

        return step + 1, coloring, key

    _, coloring, _ = jax.lax.while_loop(cond_fn, color_fn, (0, coloring, key))
    return coloring