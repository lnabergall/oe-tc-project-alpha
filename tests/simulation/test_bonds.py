from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from oe_tc.bonds import (
    bond_phase,
    bond_proposal_terms,
    candidate_edges_for_class,
    catalysis_count,
    edge_class_masks,
    update_bond_class,
)
from oe_tc.config import StaticConfig, default_params
from oe_tc.state import State
from oe_tc.topology import (
    NEG_X,
    NEG_Y,
    POS_X,
    POS_Y,
    build_occupancy,
    topology_is_valid,
)
from tests.simulation.reference import configurational_energy


def _bond(
    bonds: np.ndarray, source: int, direction: int, target: int, opposite: int
) -> None:
    bonds[source] |= np.uint8(1 << direction)
    bonds[target] |= np.uint8(1 << opposite)


def test_four_edge_classes_partition_canonical_edges_into_matchings() -> None:
    n = 6
    R = jnp.asarray([(x, y) for x in range(n) for y in range(n)], dtype=jnp.int32)
    L = build_occupancy(R, n)
    masks = np.asarray(edge_class_masks(R, n))
    assert masks.shape == (4, n * n)

    seen: set[tuple[int, int]] = set()
    for class_index in range(4):
        edges = candidate_edges_for_class(L, R, class_index)
        source = np.asarray(edges.source)
        target = np.asarray(edges.target)
        valid = np.asarray(edges.valid)
        endpoints: list[int] = []
        for i, j in zip(source[valid], target[valid], strict=True):
            edge = tuple(sorted((int(i), int(j))))
            assert edge not in seen
            seen.add(edge)
            endpoints.extend(edge)
        assert len(endpoints) == len(set(endpoints))

    assert len(seen) == n * n + n * (n - 1)


def test_same_class_edges_do_not_change_each_others_catalytic_stencil() -> None:
    n = 6
    R = np.asarray([(x, y) for x in range(n) for y in range(n)], dtype=np.int32)
    L = build_occupancy(jnp.asarray(R), n)
    for class_index in range(4):
        edges = candidate_edges_for_class(L, jnp.asarray(R), class_index)
        direction = int(edges.direction)
        source_ids = np.asarray(edges.source)[np.asarray(edges.valid)]
        coordinates = R[source_ids]
        for a, coordinate in zip(source_ids, coordinates, strict=True):
            if direction == POS_X:
                catalyst_coordinates = {
                    (coordinate[0] % n, coordinate[1] - 1),
                    (coordinate[0] % n, coordinate[1] + 1),
                }
            else:
                catalyst_coordinates = {
                    ((coordinate[0] - 1) % n, coordinate[1]),
                    ((coordinate[0] + 1) % n, coordinate[1]),
                }
            for b, other_coordinate in zip(source_ids, coordinates, strict=True):
                if a != b:
                    assert tuple(other_coordinate) not in catalyst_coordinates


def test_catalysis_uses_exactly_two_parallel_neighbor_edges() -> None:
    n = 6
    R = jnp.asarray(
        ((1, 2), (2, 2), (1, 3), (2, 3), (1, 1), (2, 1), (1, 4), (2, 4)),
        dtype=jnp.int32,
    )
    L = build_occupancy(R, n)
    bonds = np.zeros(8, dtype=np.uint8)
    _bond(bonds, 2, POS_X, 3, NEG_X)
    _bond(bonds, 4, POS_X, 5, NEG_X)
    _bond(bonds, 6, POS_X, 7, NEG_X)

    count = np.asarray(catalysis_count(L, R, jnp.asarray(bonds), POS_X))
    assert count[0] == 2

    _bond(bonds, 0, POS_X, 1, NEG_X)
    assert int(catalysis_count(L, R, jnp.asarray(bonds), POS_X)[0]) == 2

    lower = int(np.flatnonzero(np.all(np.asarray(R) == (1, 1), axis=1))[0])
    assert int(catalysis_count(L, R, jnp.asarray(bonds), POS_X)[lower]) == 1


def test_vertical_catalysis_wraps_across_periodic_x_boundary() -> None:
    n = 6
    R = jnp.asarray(
        ((0, 1), (0, 2), (1, 1), (1, 2), (5, 1), (5, 2)), dtype=jnp.int32
    )
    L = build_occupancy(R, n)
    bonds = np.zeros(6, dtype=np.uint8)
    _bond(bonds, 2, POS_Y, 3, NEG_Y)
    _bond(bonds, 4, POS_Y, 5, NEG_Y)
    assert int(catalysis_count(L, R, jnp.asarray(bonds), POS_Y)[0]) == 2


def test_channel_affinities_and_internal_legality() -> None:
    params = default_params()._replace(bond_frequency=1.0)
    endpoint_energy = jnp.asarray(((2.0, 4.0), (2.0, 4.0)), dtype=jnp.float32)
    is_bonded = jnp.asarray((False, False))
    bath_channel = jnp.asarray((True, False))
    terms = bond_proposal_terms(
        endpoint_energy, is_bonded, jnp.asarray((0, 0)), bath_channel, params
    )

    delta_u = -(1.0 - params.eta)
    assert np.allclose(np.asarray(terms.delta_u), delta_u)
    assert np.allclose(np.asarray(terms.endpoint_energy[0]), (2.0, 4.0))
    assert np.allclose(
        np.asarray(terms.endpoint_energy[1]),
        np.asarray(endpoint_energy[1]) - delta_u / 2.0,
    )
    assert np.isclose(float(terms.affinity[0]), -delta_u / params.bath_temperature)
    expected_entropy = params.heat_capacity * np.log(
        ((2.0 - delta_u / 2.0) * (4.0 - delta_u / 2.0)) / (2.0 * 4.0)
    )
    assert np.isclose(float(terms.affinity[1]), expected_entropy)

    cold_break = bond_proposal_terms(
        jnp.asarray(((0.2, 0.2),), dtype=jnp.float32),
        jnp.asarray((True,)),
        jnp.asarray((0,)),
        jnp.asarray((False,)),
        params,
    )
    assert not bool(cold_break.internal_legal[0])
    assert np.isneginf(float(cold_break.log_acceptance[0]))


def test_log_space_kinetics_caps_catalysis_without_overflow() -> None:
    params = default_params()._replace(
        bond_frequency=1.0e30,
        catalysis_strength=1.0e30,
        catalysis_cap=4.0,
    )
    terms = bond_proposal_terms(
        jnp.asarray(((1.0e30, 1.0e30),), dtype=jnp.float32),
        jnp.asarray((False,)),
        jnp.asarray((2,)),
        jnp.asarray((True,)),
        params,
    )
    assert np.isfinite(float(terms.log_acceptance[0]))
    assert float(terms.log_acceptance[0]) <= 0.0


def test_internal_class_update_conserves_configurational_plus_internal_energy() -> None:
    static = StaticConfig(n=4, num_particles=2)
    params = default_params()._replace(
        bath_channel_probability=0.0,
        bond_frequency=1.0e6,
        catalysis_strength=0.0,
    )
    R = jnp.asarray(((0, 0), (1, 0)), dtype=jnp.int32)
    L = build_occupancy(R, static.n)
    E = jnp.asarray((10.0, 12.0), dtype=jnp.float32)
    masks = jnp.zeros((2,), dtype=jnp.uint8)
    before = float(jnp.sum(E)) + configurational_energy(
        np.asarray(R), np.asarray(masks), static.n, params.eta
    )

    E2, masks2, metrics = update_bond_class(
        jax.random.key(1), R, E, L, masks, 0, params, static.empty
    )
    after = float(jnp.sum(E2)) + configurational_energy(
        np.asarray(R), np.asarray(masks2), static.n, params.eta
    )
    assert int(metrics.accepted_flips) == 1
    assert np.isclose(float(metrics.bath_energy), 0.0)
    assert np.isclose(before, after, atol=2e-6)
    assert np.allclose(np.asarray(E2 - E), (0.45, 0.45))
    assert np.array_equal(np.asarray(masks2), (1 << POS_X, 1 << NEG_X))


def test_bath_class_update_changes_configurational_energy_not_endpoints() -> None:
    static = StaticConfig(n=4, num_particles=2)
    params = default_params()._replace(
        bath_channel_probability=1.0,
        bond_frequency=1.0e6,
        catalysis_strength=0.0,
    )
    R = jnp.asarray(((0, 0), (1, 0)), dtype=jnp.int32)
    L = build_occupancy(R, static.n)
    E = jnp.asarray((10.0, 12.0), dtype=jnp.float32)
    masks = jnp.zeros((2,), dtype=jnp.uint8)
    E2, masks2, metrics = update_bond_class(
        jax.random.key(2), R, E, L, masks, 0, params, static.empty
    )
    assert int(metrics.accepted_flips) == 1
    assert np.array_equal(np.asarray(E2), np.asarray(E))
    assert np.isclose(float(metrics.bath_energy), -(1.0 - params.eta))
    assert np.array_equal(np.asarray(masks2), (1 << POS_X, 1 << NEG_X))


def test_full_phase_randomizes_classes_and_recomputes_components_once_at_end() -> None:
    static = StaticConfig(n=4, num_particles=3)
    params = default_params()._replace(
        bath_channel_probability=0.0,
        bond_frequency=1.0e6,
        catalysis_strength=0.0,
    )
    R = jnp.asarray(((0, 0), (1, 0), (2, 0)), dtype=jnp.int32)
    L = build_occupancy(R, static.n)
    state = State(
        R=R,
        E=jnp.asarray((10.0, 10.0, 10.0), dtype=jnp.float32),
        L=L,
        bonds=jnp.zeros((3,), dtype=jnp.uint8),
        root=jnp.asarray((0, 1, 2), dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.int32),
    )
    next_state, metrics = bond_phase(jax.random.key(3), state, params, static)
    assert int(metrics.accepted_flips) == 2
    assert bool(metrics.components_converged)
    assert sorted(np.asarray(metrics.class_order).tolist()) == [0, 1, 2, 3]
    assert np.array_equal(np.asarray(next_state.root), (0, 0, 0))
    assert bool(topology_is_valid(next_state.L, next_state.R, next_state.bonds))
    assert np.isclose(
        float(jnp.sum(next_state.E - state.E)),
        -float(metrics.configurational_delta),
        atol=2e-6,
    )

    orders = {
        tuple(np.asarray(bond_phase(jax.random.key(seed), state, params, static)[1].class_order))
        for seed in range(8)
    }
    assert len(orders) > 1


def test_bond_phase_propagates_component_convergence_status() -> None:
    params = default_params()._replace(bond_frequency=0.0)
    R = jnp.asarray(((0, 0), (2, 0), (1, 0)), dtype=jnp.int32)
    state = State(
        R=R,
        E=jnp.asarray((2.0, 2.0, 2.0), dtype=jnp.float32),
        L=build_occupancy(R, 4),
        bonds=jnp.asarray((1, 2, 3), dtype=jnp.uint8),
        root=jnp.arange(3, dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.int32),
    )

    short = StaticConfig(n=4, num_particles=3, component_max_iters=1)
    partial_state, partial_metrics = bond_phase(
        jax.random.key(20), state, params, short
    )
    assert not bool(partial_metrics.components_converged)
    assert int(partial_metrics.component_iterations) == 1
    assert np.array_equal(np.asarray(partial_state.root), (0, 1, 0))

    sufficient = StaticConfig(n=4, num_particles=3, component_max_iters=8)
    complete_state, complete_metrics = bond_phase(
        jax.random.key(20), state, params, sufficient
    )
    assert bool(complete_metrics.components_converged)
    assert int(complete_metrics.component_iterations) == 3
    assert np.array_equal(np.asarray(complete_state.root), (0, 0, 0))


def test_bond_phase_is_jittable_with_static_shape_config() -> None:
    static = StaticConfig(n=4, num_particles=2)
    params = default_params()._replace(bond_frequency=0.0)
    R = jnp.asarray(((0, 0), (1, 0)), dtype=jnp.int32)
    state = State(
        R,
        jnp.asarray((1.0, 1.0), dtype=jnp.float32),
        build_occupancy(R, static.n),
        jnp.zeros((2,), dtype=jnp.uint8),
        jnp.arange(2, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    compiled = jax.jit(lambda key, current, p: bond_phase(key, current, p, static))
    next_state, metrics = compiled(jax.random.key(9), state, params)
    assert int(metrics.accepted_flips) == 0
    assert bool(metrics.components_converged)
    assert np.array_equal(np.asarray(next_state.bonds), (0, 0))
