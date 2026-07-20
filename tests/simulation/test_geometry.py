from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.geometry import (
    build_lattice,
    exposure_fraction,
    lattice_indices,
    matching_edges,
    neighbor_coordinates,
    neighbor_indices,
    occupancy_is_consistent,
    site_is_valid,
    topmost_particles,
    wrap_x,
)


def test_cylindrical_wrapping_and_closed_y_boundaries() -> None:
    coordinates = jnp.asarray(((-1, 0), (4, 3), (2, -1), (2, 4)))
    wrapped = wrap_x(coordinates, 4)
    np.testing.assert_array_equal(wrapped[:, 0], np.asarray((3, 0, 2, 2)))
    np.testing.assert_array_equal(
        site_is_valid(wrapped, 4), np.asarray((True, True, False, False))
    )

    neighbors = neighbor_coordinates(jnp.asarray(((0, 0),)), 4)
    np.testing.assert_array_equal(
        neighbors.coordinates[0], np.asarray(((1, 0), (3, 0), (0, 1), (0, 0)))
    )
    np.testing.assert_array_equal(neighbors.valid[0], (True, True, True, False))


def test_lattice_neighbors_use_n_as_wall_and_vacancy_sentinel() -> None:
    positions = jnp.asarray(((0, 0), (3, 0), (0, 1), (2, 3)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    assert bool(occupancy_is_consistent(positions, lattice))
    expected = np.asarray(
        (
            (4, 1, 2, 4),
            (0, 4, 4, 4),
            (4, 4, 4, 0),
            (4, 4, 4, 4),
        ),
        dtype=np.int32,
    )
    np.testing.assert_array_equal(neighbor_indices(positions, lattice), expected)

    compiled = jax.jit(neighbor_indices)(positions, lattice)
    np.testing.assert_array_equal(compiled, expected)


def test_exposure_counts_only_valid_empty_lattice_sites() -> None:
    interior = jnp.asarray(((1, 1),), dtype=jnp.int32)
    top = jnp.asarray(((1, 0),), dtype=jnp.int32)
    np.testing.assert_allclose(exposure_fraction(interior, build_lattice(interior, 4)), 1.0)
    np.testing.assert_allclose(exposure_fraction(top, build_lattice(top, 4)), 0.75)


def test_topmost_particle_is_minimum_y_and_empty_columns_are_padded() -> None:
    positions = jnp.asarray(((0, 2), (0, 0), (2, 3)), dtype=jnp.int32)
    surface = topmost_particles(build_lattice(positions, 4), empty=3)
    np.testing.assert_array_equal(surface.particle, np.asarray((1, 3, 2, 3)))
    np.testing.assert_array_equal(surface.occupied, (True, False, True, False))


def test_four_classes_cover_every_edge_once_and_each_is_a_matching() -> None:
    n = 4
    all_edges: list[tuple[int, int]] = []
    for class_id in range(4):
        edge_class = matching_edges(n, class_id)
        source = np.asarray(lattice_indices(edge_class.source, n))
        target = np.asarray(lattice_indices(edge_class.target, n))
        valid = np.asarray(edge_class.valid)
        endpoints = np.concatenate((source[valid], target[valid]))
        assert len(endpoints) == len(np.unique(endpoints))
        all_edges.extend(
            tuple(sorted((int(i), int(j))))
            for i, j in zip(source[valid], target[valid], strict=True)
        )

    # n*n periodic horizontal edges plus n*(n-1) closed vertical edges.
    assert len(all_edges) == n * n + n * (n - 1)
    assert len(set(all_edges)) == len(all_edges)

    compiled = jax.jit(matching_edges, static_argnums=0)(n, jnp.asarray(3))
    np.testing.assert_array_equal(compiled.valid, matching_edges(n, 3).valid)
