from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from oe_tc import topology
from tests.simulation.reference import components as reference_components
from tests.simulation.reference import occupancy as reference_occupancy


def _random_bonds(rng: np.random.Generator, R: np.ndarray, n: int) -> np.ndarray:
    """Generate a valid random bond graph over occupied nearest neighbors."""

    N = len(R)
    L = reference_occupancy(R, n)
    bonds = np.zeros(N, dtype=np.uint8)
    for particle, (x, y) in enumerate(R):
        for direction, (dx, dy), opposite in ((0, (1, 0), 1), (2, (0, 1), 3)):
            yy = y + dy
            if yy >= n:
                continue
            other = int(L[(x + dx) % n, yy])
            if other < N and rng.random() < 0.4:
                bonds[particle] |= np.uint8(1 << direction)
                bonds[other] |= np.uint8(1 << opposite)
    return bonds


def test_bit_helpers_and_quarter_turn_permutation() -> None:
    assert [int(topology.opposite_direction(d)) for d in range(4)] == [1, 0, 3, 2]
    assert [int(topology.rotate_direction(d, 1)) for d in range(4)] == [2, 3, 1, 0]

    all_masks = jnp.arange(16, dtype=jnp.uint8)
    rotated = topology.rotate_bond_mask(all_masks, 1)
    assert np.array_equal(
        np.asarray(topology.rotate_bond_mask(rotated, -1)), np.arange(16)
    )
    assert np.array_equal(
        np.asarray(topology.rotate_bond_mask(all_masks, 4)), np.arange(16)
    )

    mask = jnp.uint8(0)
    mask = topology.set_bond_bit(mask, topology.POS_X)
    assert bool(topology.has_bond(mask, topology.POS_X))
    mask = topology.toggle_bond_bit(mask, topology.POS_X)
    assert int(mask) == 0


def test_occupancy_and_neighbor_gather_respect_cylinder_boundaries() -> None:
    R = jnp.asarray(((0, 0), (3, 0), (0, 1), (0, 3)), dtype=jnp.int32)
    L = topology.build_occupancy(R, 4)
    expected = np.full((4, 4), 4, dtype=np.int32)
    expected[0, 0], expected[3, 0], expected[0, 1], expected[0, 3] = range(4)
    assert np.array_equal(np.asarray(L), expected)

    neighbors = np.asarray(topology.gather_lattice_neighbors(L, R))
    assert np.array_equal(neighbors[0], (4, 1, 2, 4))
    assert neighbors[3, topology.POS_Y] == 4


def test_bonded_neighbor_gather_requires_mirrored_endpoint_bits() -> None:
    R = jnp.asarray(((0, 0), (1, 0), (0, 1)), dtype=jnp.int32)
    L = topology.build_occupancy(R, 4)
    bonds = jnp.asarray((1 << topology.POS_X, 1 << topology.NEG_X, 0), dtype=jnp.uint8)
    gathered = np.asarray(topology.gather_bonded_neighbors(L, R, bonds))
    assert gathered[0, topology.POS_X] == 1
    assert gathered[1, topology.NEG_X] == 0
    assert np.count_nonzero(gathered < 3) == 2

    asymmetric = bonds.at[1].set(0)
    assert np.all(np.asarray(topology.gather_bonded_neighbors(L, R, asymmetric)) >= 3)
    assert not bool(topology.topology_is_valid(L, R, asymmetric))


def test_parallel_components_match_host_bfs_on_random_graphs() -> None:
    n = 8
    grid = np.asarray([(x, y) for x in range(n) for y in range(n)], dtype=np.int32)
    rng = np.random.default_rng(20260719)
    R = grid[rng.permutation(len(grid))]
    L = reference_occupancy(R, n)

    compiled_components = jax.jit(
        lambda lattice, positions, masks: topology.connected_components(
            lattice, positions, masks, max_iters=64
        )
    )
    for _ in range(24):
        bonds = _random_bonds(rng, R, n)
        expected = reference_components(R, bonds, n)
        actual = np.asarray(
            compiled_components(jnp.asarray(L), jnp.asarray(R), jnp.asarray(bonds))
        )
        assert np.array_equal(actual, expected)


def test_components_handle_isolates_bridge_splits_and_periodic_edge() -> None:
    R = np.asarray(((0, 0), (1, 0), (2, 0), (3, 0), (0, 2)), dtype=np.int32)
    L = topology.build_occupancy(jnp.asarray(R), 4)
    bonds = np.asarray((1, 3, 3, 2, 0), dtype=np.uint8)
    root = topology.connected_components(L, jnp.asarray(R), jnp.asarray(bonds), 16)
    assert np.array_equal(np.asarray(root), (0, 0, 0, 0, 4))

    bonds[1] &= np.uint8(~(1 << topology.POS_X) & 0xFF)
    bonds[2] &= np.uint8(~(1 << topology.NEG_X) & 0xFF)
    root = topology.connected_components(L, jnp.asarray(R), jnp.asarray(bonds), 16)
    assert np.array_equal(np.asarray(root), (0, 0, 2, 2, 4))

    bonds[3] |= np.uint8(1 << topology.POS_X)
    bonds[0] |= np.uint8(1 << topology.NEG_X)
    root = topology.connected_components(L, jnp.asarray(R), jnp.asarray(bonds), 16)
    assert np.array_equal(np.asarray(root), (0, 0, 0, 0, 4))


def test_component_iteration_metric_reports_executed_rounds() -> None:
    R = jnp.asarray(((0, 0), (1, 0), (2, 0), (3, 0)), dtype=jnp.int32)
    L = topology.build_occupancy(R, 4)
    bonds = jnp.asarray((1, 3, 3, 2), dtype=jnp.uint8)
    root, iterations, converged = topology.connected_components_with_status(
        L, R, bonds, max_iters=64
    )
    assert np.array_equal(np.asarray(root), (0, 0, 0, 0))
    assert 0 < int(iterations) < 64
    assert bool(converged)

    pair_root, pair_iterations = topology.connected_components_with_iterations(
        L, R, bonds, max_iters=64
    )
    assert np.array_equal(np.asarray(pair_root), np.asarray(root))
    assert int(pair_iterations) == int(iterations)
    assert np.array_equal(
        np.asarray(topology.connected_components(L, R, bonds, max_iters=64)),
        np.asarray(root),
    )

    isolates, isolate_iterations, isolates_converged = (
        topology.connected_components_with_status(
            L, R, jnp.zeros_like(bonds), max_iters=64
        )
    )
    assert np.array_equal(np.asarray(isolates), np.arange(4))
    assert int(isolate_iterations) == 1
    assert bool(isolates_converged)


def test_component_cap_reports_unconverged_partial_labeling() -> None:
    # Particle ids follow the geometric path 0--2--1.  In the first parallel
    # hooking round root 2 attaches to root 0, but root 1 is a competing local
    # minimum and cannot see root 0 until the next round.
    R = jnp.asarray(((0, 0), (2, 0), (1, 0)), dtype=jnp.int32)
    L = topology.build_occupancy(R, 4)
    bonds = jnp.asarray((1, 2, 3), dtype=jnp.uint8)

    partial, iterations, converged = topology.connected_components_with_status(
        L, R, bonds, max_iters=1
    )
    assert int(iterations) == 1
    assert not bool(converged)
    assert np.array_equal(np.asarray(partial), (0, 1, 0))
    assert not np.array_equal(np.asarray(partial), (0, 0, 0))

    complete, iterations, converged = topology.connected_components_with_status(
        L, R, bonds, max_iters=8
    )
    assert bool(converged)
    assert int(iterations) == 3
    assert np.array_equal(np.asarray(complete), (0, 0, 0))


def test_topology_kernel_is_jittable() -> None:
    R = jnp.asarray(((0, 0), (1, 0)), dtype=jnp.int32)
    L = topology.build_occupancy(R, 4)
    bonds = jnp.asarray((1, 2), dtype=jnp.uint8)
    root = jax.jit(lambda x: topology.connected_components(L, R, x, 8))(bonds)
    assert np.array_equal(np.asarray(root), (0, 0))
