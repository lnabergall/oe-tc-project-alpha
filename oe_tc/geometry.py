"""Vectorized geometry kernels for the cylindrical simulation lattice.

Coordinates are ordered ``(x, y)``.  The horizontal ``x`` coordinate is
periodic, while ``y`` is closed with the irradiated top surface at ``y = 0``.
The occupation lattice stores particle ids and uses ``N`` as its empty value.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


NUM_DIRECTIONS = 4
POS_X = 0
NEG_X = 1
POS_Y = 2
NEG_Y = 3

DIRECTIONS = jnp.asarray(
    ((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=jnp.int32
)
OPPOSITE_DIRECTIONS = jnp.asarray((NEG_X, POS_X, NEG_Y, POS_Y), dtype=jnp.int32)


class NeighborGeometry(NamedTuple):
    """Safe neighboring coordinates and closed-boundary validity mask."""

    coordinates: jax.Array
    valid: jax.Array


class ColumnSurface(NamedTuple):
    """Topmost particle id and occupancy flag for every column."""

    particle: jax.Array
    occupied: jax.Array


class MatchingEdges(NamedTuple):
    """One fixed-size lattice-edge class, padded by ``valid=False`` entries."""

    source: jax.Array
    target: jax.Array
    direction: jax.Array
    valid: jax.Array


def wrap_x(coordinates: jax.Array, n: int) -> jax.Array:
    """Wrap only the periodic horizontal coordinate into ``[0, n)``."""

    coordinates = jnp.asarray(coordinates, dtype=jnp.int32)
    return coordinates.at[..., 0].set(jnp.mod(coordinates[..., 0], n))


def site_is_valid(coordinates: jax.Array, n: int) -> jax.Array:
    """Return whether coordinates are canonical sites of the cylinder."""

    coordinates = jnp.asarray(coordinates)
    return (
        (coordinates[..., 0] >= 0)
        & (coordinates[..., 0] < n)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 1] < n)
    )


def lattice_indices(coordinates: jax.Array, n: int) -> jax.Array:
    """Flatten canonical ``(x, y)`` coordinates in row-major order."""

    coordinates = jnp.asarray(coordinates, dtype=jnp.int32)
    return coordinates[..., 0] * n + coordinates[..., 1]


def lattice_coordinates(indices: jax.Array, n: int) -> jax.Array:
    """Invert :func:`lattice_indices` for fixed-size arrays of indices."""

    indices = jnp.asarray(indices, dtype=jnp.int32)
    return jnp.stack((indices // n, indices % n), axis=-1)


def build_lattice(
    positions: jax.Array, n: int, empty: int | None = None
) -> jax.Array:
    """Build ``L[x, y]`` from unique, valid particle positions."""

    positions = wrap_x(positions, n)
    num_particles = positions.shape[0]
    if empty is None:
        empty = num_particles
    particle_ids = jnp.arange(num_particles, dtype=jnp.int32)
    lattice = jnp.full((n, n), empty, dtype=jnp.int32)
    return lattice.at[positions[:, 0], positions[:, 1]].set(particle_ids)


# Compatibility with the topology module and natural call-site terminology.
build_occupancy = build_lattice
occupancy_from_positions = build_lattice


def neighbor_coordinates(positions: jax.Array, n: int) -> NeighborGeometry:
    """Return the four neighbors of every position with shape ``(..., 4, 2)``.

    Invalid vertical neighbors are clipped to a safe gather location and are
    identified by ``valid=False``.  Horizontal neighbors always wrap.
    """

    positions = jnp.asarray(positions, dtype=jnp.int32)
    raw = positions[..., None, :] + DIRECTIONS
    x = jnp.mod(raw[..., 0], n)
    valid = (raw[..., 1] >= 0) & (raw[..., 1] < n)
    y = jnp.clip(raw[..., 1], 0, n - 1)
    return NeighborGeometry(jnp.stack((x, y), axis=-1), valid)


def neighbor_indices(
    positions: jax.Array, lattice: jax.Array, empty: int | None = None
) -> jax.Array:
    """Gather neighboring particle ids, padding walls and vacancies by ``N``."""

    if empty is None:
        empty = positions.shape[0]
    geometry = neighbor_coordinates(positions, lattice.shape[0])
    gathered = lattice[
        geometry.coordinates[..., 0], geometry.coordinates[..., 1]
    ]
    return jnp.where(
        geometry.valid, gathered, jnp.asarray(empty, dtype=gathered.dtype)
    )


neighbor_particles = neighbor_indices
gather_lattice_neighbors = neighbor_indices


def exposure_fraction(
    positions: jax.Array, lattice: jax.Array, empty: int | None = None
) -> jax.Array:
    """Fraction of four directions leading to valid, unoccupied sites.

    A closed-boundary wall is not an unoccupied lattice site.  Consequently an
    isolated particle at ``y=0`` or ``y=n-1`` has exposure ``3/4``.
    """

    if empty is None:
        empty = positions.shape[0]
    geometry = neighbor_coordinates(positions, lattice.shape[0])
    gathered = lattice[
        geometry.coordinates[..., 0], geometry.coordinates[..., 1]
    ]
    unoccupied = geometry.valid & (gathered == empty)
    return jnp.mean(unoccupied.astype(jnp.float32), axis=-1)


def topmost_particles(
    lattice: jax.Array, empty: int | None = None
) -> ColumnSurface:
    """Return the occupied site of smallest ``y`` in each column.

    Empty columns contain the supplied sentinel in ``particle``.  The separate
    Boolean mask lets callers scatter without ever gathering at that sentinel.
    """

    if empty is None:
        # This fallback is useful for unlabeled/full lattices, but simulation
        # callers normally pass N because N cannot be inferred from L alone.
        empty = lattice.size
    occupied_sites = lattice != empty
    occupied_columns = jnp.any(occupied_sites, axis=1)
    top_y = jnp.argmax(occupied_sites, axis=1)
    x = jnp.arange(lattice.shape[0], dtype=jnp.int32)
    particle = lattice[x, top_y]
    particle = jnp.where(occupied_columns, particle, empty)
    return ColumnSurface(particle, occupied_columns)


def matching_edges(n: int, class_id: int | jax.Array) -> MatchingEdges:
    """Enumerate one of four fixed edge matchings on an even cylinder.

    Classes 0--1 contain ``+x`` edges and classes 2--3 contain ``+y``
    edges.  Anchor checkerboard parity separates all incident edges and also
    separates parallel edges one perpendicular lattice step apart.  The
    ``n*n`` output shape is independent of the class; missing bottom-boundary
    vertical edges are masked by ``valid``.
    """

    sites = lattice_coordinates(jnp.arange(n * n, dtype=jnp.int32), n)
    class_id = jnp.asarray(class_id, dtype=jnp.int32)
    horizontal = class_id < 2
    direction = jnp.where(horizontal, POS_X, POS_Y).astype(jnp.int32)
    offset = DIRECTIONS[direction]
    target = sites + offset
    target = target.at[:, 0].set(jnp.mod(target[:, 0], n))
    parity = jnp.bitwise_and(sites[:, 0] + sites[:, 1], 1)
    correct_parity = parity == jnp.bitwise_and(class_id, 1)
    vertical_valid = target[:, 1] < n
    valid = correct_parity & (horizontal | vertical_valid)
    safe_target = target.at[:, 1].set(jnp.clip(target[:, 1], 0, n - 1))
    return MatchingEdges(
        source=sites,
        target=safe_target,
        direction=jnp.broadcast_to(direction, (n * n,)),
        valid=valid,
    )


def occupancy_is_consistent(
    positions: jax.Array, lattice: jax.Array, empty: int | None = None
) -> jax.Array:
    """Check the position/lattice bijection and sentinel range as a JAX scalar."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    num_particles = positions.shape[0]
    if empty is None:
        empty = num_particles
    n = lattice.shape[0]
    valid = jnp.all(site_is_valid(positions, n))
    safe = jnp.clip(positions, 0, n - 1)
    expected = jnp.arange(num_particles, dtype=lattice.dtype)
    ids_match = jnp.all(lattice[safe[:, 0], safe[:, 1]] == expected)
    entries_valid = jnp.all((lattice >= 0) & (lattice <= empty))
    count_matches = jnp.sum(lattice != empty) == num_particles
    return valid & ids_match & entries_valid & count_matches


__all__ = [
    "NUM_DIRECTIONS",
    "POS_X",
    "NEG_X",
    "POS_Y",
    "NEG_Y",
    "DIRECTIONS",
    "OPPOSITE_DIRECTIONS",
    "NeighborGeometry",
    "ColumnSurface",
    "MatchingEdges",
    "wrap_x",
    "site_is_valid",
    "lattice_indices",
    "lattice_coordinates",
    "build_lattice",
    "build_occupancy",
    "occupancy_from_positions",
    "neighbor_coordinates",
    "neighbor_indices",
    "neighbor_particles",
    "gather_lattice_neighbors",
    "exposure_fraction",
    "topmost_particles",
    "matching_edges",
    "occupancy_is_consistent",
]
