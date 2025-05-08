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


def get_shifts():
    return jnp.array(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)), dtype=int)


def generate_neighorhood(x, n):
    """
    Generate the neighborhood of a point 'x' in a lattice of size n, 
    with periodic first dimension.
    """
    return cylindrical_coordinates(x.reshape((1, 2)) + shifts(), n)


def remove_from_lattice(L, I, pad_value):
    """Remove particles in I from lattice L."""
    removal_mask = jnp.isin(L, jnp.where(I == pad_value, pad_value - 1, I))
    return jnp.where(removal_mask, pad_value, L)


def generate_property_lattice(L, C, L_pad_value, C_pad_value):
    """Gather property values from a 1D Array C using indices in an Array L while ignoring padding."""
    return jnp.where(L != L_pad_value, C[L], C_pad_value)


def four_partition(R, L, pad_value):
    """
    Partitions the lattice L into eight isomorphic sets whose elements are all distance four apart
    and collect the contained particles.

    R
        Array of particle positions. 2D, kx2.
    L
        Labeled occupation array. 2D, nxn.
    pad_value
        Scalar.

    Returns:
        8xk Array whose rows contain the eight independent sets, pad_value padded. 
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
    P = rearrange_padding(P, pad_value)[:, :k]

    return P
