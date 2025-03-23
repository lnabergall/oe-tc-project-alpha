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


def using_periodicity(x_diff, n):
    return (n - x_diff) <= x_diff


@partial(jax.jit, static_argnums=[2, 3])
def canonical_shortest_path(r_start, r_end, max_length, n, pad_value):
    """
    Returns a canonical shortest path between r_start and r_end, where we take 
    a staircase path until we reach a straight line to r_end. 
    """
    # first change r_start labeling to account for periodic x-dim
    x_diff = jnp.abs(r_end - r_start)[0]
    x_start = r_start[0]
    x_new = jnp.where(x_start > n // 2, x_start - n, x_start + n)
    x_new = jnp.where(using_periodicity(x_diff, n), x_new, x_start)
    r_start = r_start.at[0].set(x_new)

    r_diff = r_end - r_start
    r_diff_sign = jnp.sign(r_diff)
    r_diff_abs = jnp.abs(r_diff)
    length = jnp.sum(r_diff_abs)

    i_square_half = jnp.min(r_diff_abs)
    i_square = 2 * i_square_half 
    element_square = r_start + (r_diff_sign * jnp.array((i_square_half, i_square_half)))
    step_outside_square = r_diff_sign * jnp.where(
        r_diff_abs[0] >= r_diff_abs[1], jnp.array([1, 0]), jnp.array([0, 1]))
    padding = jnp.full(2, pad_value, dtype=int)

    def path_creator(i):
        half, parity = i // 2, i % 2
        i_diff = i - i_square
        element_within_square = r_start + (r_diff_sign * jnp.array((half + parity, half)))
        element_outside_square = element_square + (i_diff * step_outside_square)
        # mod n here to account for periodicity
        element_on_path = jnp.where(
            i <= i_square, element_within_square, element_outside_square) % n     
        element = jnp.where(i <= length, element_on_path, padding)
        return element

    path = jnp.fromfunction(path_creator, (max_length,), dtype=r_start.dtype)
    return path


def shift_path(path, r): 
    # result should be path of lattice points
    return (path + r - path[0]).astype(int)


def append_on_path(path, r):
    insert_id = jnp.argmin(path, axis=0)[0]
    return jnp.insert(path, insert_id, r, axis=0)


def lattice_positions(n):
    return jnp.stack(jnp.indices((n, n)), axis=-1)


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


def square_indices_nowrap(r, delta, n):
    x, y = r
    diameter = (2 * delta) + 1
    col_start = jnp.clip(x - delta, 0)
    col_indices = jnp.arange(diameter) + col_start
    row_start = jnp.clip(y - delta, 0)
    row_indices = jnp.arange(diameter) + row_start
    return row_indices, col_indices


def centered_square_nowrap(L, r, delta):
    """
    Generate the subset of the lattice 'L' consisting of a square of radius 'delta' centered at 'r',
    along with the positions in that subset.  
    """
    n = L.shape[0]
    x_indices, y_indices = square_indices_nowrap(r, delta, n)
    square = L[x_indices[:, None], y_indices[None, :]]
    positions = jnp.stack(jnp.meshgrid(x_indices, y_indices, indexing="ij"), axis=-1)
    return square, positions


@partial(jax.jit, static_argnums=[1, 2, 3])
def generate_lattice(R, n, pad_value, sites=2):
    """
    Generate the occupation lattice corresponding to R. The lattice is nxnxsites, 
    with 'sites' slots per position of the lattice, to accomodate test positions; 
    occupation is indicated by a particle index in one of the slots. 
    """
    k = R.shape[0]
    L = jnp.full((n, n, sites), pad_value, dtype=int)
    counts = jnp.zeros((n, n), dtype=int)

    def particle_fn(L, counts, i, x, y):
        idx = counts[x, y]
        L = L.at[x, y, idx].set(i)
        counts = counts.at[x, y].add(1)
        return (L, counts)

    def padding_fn(L, counts, *args):
        return L, counts

    def map_point(i, args):
        L, counts = args
        x, y = R[i]
        return jax.lax.cond(x == pad_value, padding_fn, particle_fn, L, counts, i, x, y)

    L, counts = jax.lax.fori_loop(0, k, map_point, (L, counts))
    return L, counts


def add_to_lattice(L, I, R, pad_value):
    """
    Add particles in I to lattice L. Assumes 2-particle co-location limit is obeyed
    and no particles in I are co-located.
    """
    n = L.shape[0]
    R_I = R[I]
    empty_sites = jnp.argmin(L[R_I[:, 0], R_I[:, 1]], axis=-1)
    empty_sites = jnp.where(I == pad_value, 2*n, empty_sites)
    return L.at[R_I[:, 0], R_I[:, 1], empty_sites].set(I, mode="drop")


def remove_from_lattice(L, I, R, pad_value):
    """Remove particles in I from lattice L. Assumes that R matches L."""
    removal_mask = jnp.isin(L, jnp.where(I == pad_value, pad_value - 1, I))
    return jnp.where(removal_mask, pad_value, L)


def unlabel_lattice(L, pad_value):
    return jnp.sum(L != pad_value, axis=-1)


def generate_unlabeled_lattice(R, n):
    """
    Generate the unlabeled occupation lattice corresponding to R. The lattice is an nxn Array; 
    occupation is indicated by 0, 1, or 2, indicating the number of particles in the site. 
    """
    L = jnp.zeros((n, n), dtype=int)
    L = L.at[R[:, 0], R[:, 1]].add(1)
    return L


def generate_property_lattice(L, C, L_pad_value, C_pad_value):
    """Gather property values from a 1D Array C using indices in an Array L while ignoring padding."""
    return jnp.where(L != L_pad_value, C[L], C_pad_value)


def partition_size(min_sep):
    return 2 * (2 * (min_sep // 2) + 1) ** 2


def independent_partition(R, L, part_size, min_sep, pad_value):
    """
    Partitions L into square cells of diameter min_sep 
    and assigns each site in a cell to a different independent set. 
    min_sep must be even. 

    R
        Array of particle positions. 2D, kx2. Not used currently.
    L 
        Labeled occupation array. 3D, nxnx2.
    part_size
        Scalar. Not used currently.
    min_sep
        Scalar, minimum separation required between particles in an independent set. Must be even.
    pad_value
        Scalar.
    """
    k, n = R.shape[0], L.shape[0]
    sep_radius = min_sep // 2
    cell_dim = ceil_div(n, min_sep)
    center_indices = jnp.arange(cell_dim ** 2)
    
    def scan_cell(i):
        # get cell center
        i, j = i // cell_dim, i % cell_dim
        i = (i * min_sep) + sep_radius
        j = (j * min_sep) + sep_radius
        r = jnp.array((i, j))

        # get cell
        cell, cell_positions = centered_square_nowrap(L, r, sep_radius)
        cell = jnp.where(cell_positions >= n, pad_value, cell)

        # partition cell
        return cell.ravel()

    P = jax.vmap(scan_cell)(center_indices)
    P = P.transpose()
    P = rearrange_padding(P, pad_value)[:, :part_size]

    return P


def independent_group_partition(I, groups, R, L, part_size, min_sep, max_nbhrs, pad_value):
    k, n = R.shape[0], L.shape[0]
    sep_radius = min_sep // 2

    L_alt = jnp.where(L == pad_value, 2*k, L)
    L_groups = groups.at[L_alt].get(mode="fill", fill_value=pad_value)

    cell_dim = ceil_div(n, min_sep)
    center_indices = lattice_positions(cell_dim)
    
    def scan_cell(r):
        # get cell center
        r = (r * min_sep) + sep_radius

        # get cell
        cell, cell_positions = centered_square_nowrap(L_groups, r, sep_radius)
        cell = jnp.where(cell_positions >= n, pad_value, cell)

        # collect groups in the cell
        cell_groups = jnp.unique(cell, size=min_sep**2, fill_value=pad_value)

        return cell_groups

    groups_by_cell = jax.vmap(jax.vmap(scan_cell, in_axes=0), in_axes=1)(center_indices)

    shifts = jnp.array([[0, 1], [1, 0], [1, 1], [1, -1]])
    shifts = jnp.concatenate((jnp.zeros((1, 2), dtype=int), shifts, -shifts))

    def find_group(i):
        # cells containing group i
        cells = jnp.any(groups_by_cell == i, axis=-1)
        cell_indices = jnp.where(jnp.expand_dims(cells, axis=-1), center_indices, -2)
        cell_indices = jnp.reshape(cell_indices, (cell_dim**2, 2))

        # cells containing group i or adjacent to such a group
        with_neighbors = cell_indices[:, None, :] + shifts[None, :, :]
        with_neighbors = jnp.reshape(with_neighbors, (9 * cell_indices.shape[0], 2))
        with_neighbors = jnp.where(with_neighbors < 0, pad_value, with_neighbors)
        with_neighbors = jnp.unique(with_neighbors, axis=0, 
                                    size=cell_indices.shape[0], fill_value=2*k)

        # get all groups in these cells
        adjacent_groups = groups_by_cell.at[
            with_neighbors[:, 0], with_neighbors[:, 1]].get(mode="fill", fill_value=pad_value)
        adjacent_groups = jnp.unique(adjacent_groups, size=max_nbhrs, fill_value=pad_value)

        return adjacent_groups

    group_graph = jax.vmap(find_group)(I)

    partition = greedy_graph_coloring(group_graph, pad_value)
    partition = expand_partition(partition, part_size, pad_value)

    return partition