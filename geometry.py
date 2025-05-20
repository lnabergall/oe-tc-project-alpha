from functools import partial

import jax
import jax.numpy as jnp

from utils import *


def cylindrical_coordinates(A, n):
    """
    Convert an array A of 2D vectors to cylindrical coordinates, 
    with a periodic first or 'x' dimension.
    """
    return A.at[..., 0].set(A[..., 0] % n)


def periodic_norm(x_diff, n):
    return jnp.min(jnp.stack((x_diff, n - x_diff)))


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


def get_shifts():
    return jnp.array(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)), dtype=int)


def generate_neighorhood(x, n):
    """
    Generate the neighborhood of a point 'x' in a lattice of size n, 
    with periodic first dimension.
    """
    return cylindrical_coordinates(x.reshape((1, 2)) + shifts(), n)


def generate_open_neighorhood(x, n):
    """
    Generate the neighborhood of a point 'x' in a lattice of size n, 
    with periodic first dimension and excluding 'x' itself.
    """
    steps = jnp.array(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=int)
    return cylindrical_coordinates(x.reshape((1, 2)) + steps, n)


def square_indices(r, delta, n):
    x, y = r
    diameter = (2 * delta) + 1
    row_indices = (x + jnp.arange(diameter) - delta) % n
    col_start = jnp.clip(y - delta, 0, n - diameter)
    col_indices = jnp.arange(diameter) + col_start
    return row_indices, col_indices


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


def add_to_lattice(L, I, R, pad_value):
    """Add particles in I to lattice L. Assumes no particles are co-located."""
    R_I = R[I]
    return L.at[R_I[:, 0], R_I[:, 1]].set(I, mode="drop")


def remove_from_lattice(L, I, pad_value):
    """Remove particles in I from lattice L."""
    removal_mask = jnp.isin(L, jnp.where(I == pad_value, pad_value - 1, I))
    return jnp.where(removal_mask, pad_value, L)


def generate_unlabeled_lattice(R, n):
    """
    Generate the unlabeled occupation lattice corresponding to R. The lattice is an nxn Array; 
    occupation is indicated by 0 or 1, indicating whether a site is filled by a particle or not. 
    """
    L = jnp.zeros((n, n), dtype=int)
    L = L.at[R[:, 0], R[:, 1]].set(1)
    return L


def generate_property_lattice(L, C, L_pad_value, C_pad_value):
    """Gather property values from a 1D Array C using indices in an Array L while ignoring padding."""
    return jnp.where(L != L_pad_value, C[L], C_pad_value)


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
    k, n = R.shape[0], L.shape[0]
    half_grid = jnp.indices(((n + 3) // 4, (n + 3) // 4)).at[:].multiply(4)

    def collect_part(i):
        row_shift = i >= 4
        col_shift = i % 4
        half_grid = half_grid.at[0].add(row_shift)
        half_grid = half_grid.at[1].add(col_shift)
        half_grid = half_grid.reshape((half_grid.shape[0], -1))
        indices = jnp.concat((half_grid, half_grid + 2), axis=-1)
        return L.at[indices[0], indices[1]].get(mode="fill", fill_value=pad_value)

    P = jax.vmap(collect_part)(jnp.arange(8))
    P = rearrange_padding(P, pad_value)[:, :m] 

    return P


def four_group_partition(bound_states, L, key, pad_value):
    k, m = bound_states.shape[0], jnp.max(bound_states)
    L_bs = jnp.where(L == pad_value, 2*k, bound_states[L])
    coloring = jnp.full((k,), pad_value)
    padding_mask = bound_states != pad_value

    def any_true_per_id(L_mask):
        flat_mask = L_mask.ravel().astype(int)
        flat_ids = L_bs.ravel()
        mask = jnp.zeros((k,), dtype=int)
        return mask.at[flat_ids].max(flat_mask, mode="drop")

    def cond_fn(args):
        coloring = args[1]
        return jnp.any(color == pad_value)

    def color_fn(args):
        step, coloring, key = args
        key, subkey = jax.random.split(key)

        # random 64-bit priorities for mixing
        priorities = jax.random.randint(subkey, (k,), 1, 2**63, dtype=jnp.uint64)
        priorities = jnp.where(coloring == pad_value, priorities, 0)
        L_p = priorities.at[L_bs].get(mode="fill", fill_value=0)

        # max priority in every 7x7 square
        nbhd_max = jax.lax.reduce_window(L_p, 0, jax.lax.max, (7, 7), (1, 1), "SAME")
        L_loses = (nbhd_max != L_p) & (L_bs != pad_value)
        lose_indicator = any_true_per_id(L_loses)

        # choose uncoloured that never lost
        chosen = (lose_indicator == 0) & (coloring == pad_value) & padding_mask
        coloring = coloring.at[chosen].set(step)

        return step + 1, coloring, key

    coloring, _ = jax.lax.while_loop(cond_fn, color_fn, (0, coloring, key))
    return coloring