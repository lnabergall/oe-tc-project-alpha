from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.conduction import (
    conduction_class_from_draws,
    conduction_sweep,
    pair_exchange_proposal,
)
from oe_tc.geometry import build_lattice


QUANTUM = 0.25
FLOOR = 0.05
CAPACITY = 2.5


def test_pair_proposal_selects_kinetics_and_conserves_energy() -> None:
    source = jnp.asarray((1.5, 1.5))
    target = jnp.asarray((0.5, 0.5))
    proposal = pair_exchange_proposal(
        source,
        target,
        jnp.asarray((False, True)),
        jnp.ones(2),
        QUANTUM,
        FLOOR,
        CAPACITY,
        0.2,
        0.8,
    )

    np.testing.assert_allclose(proposal.source_energy, (1.25, 1.25))
    np.testing.assert_allclose(proposal.target_energy, (0.75, 0.75))
    np.testing.assert_allclose(
        proposal.source_energy + proposal.target_energy, source + target
    )
    np.testing.assert_allclose(proposal.coefficient, (0.2, 0.8))
    expected_affinity = CAPACITY * np.log((1.25 * 0.75) / (1.5 * 0.5))
    np.testing.assert_allclose(proposal.affinity, expected_affinity)
    np.testing.assert_allclose(
        proposal.log_acceptance,
        np.log((0.2, 0.8)),
    )


def test_forward_reverse_probability_ratio_is_entropy_ratio() -> None:
    forward = pair_exchange_proposal(
        jnp.asarray(1.5),
        jnp.asarray(0.5),
        jnp.asarray(False),
        jnp.asarray(1.0),
        QUANTUM,
        FLOOR,
        CAPACITY,
        0.4,
        0.8,
    )
    reverse = pair_exchange_proposal(
        forward.source_energy,
        forward.target_energy,
        jnp.asarray(False),
        jnp.asarray(-1.0),
        QUANTUM,
        FLOOR,
        CAPACITY,
        0.4,
        0.8,
    )

    np.testing.assert_allclose(reverse.affinity, -forward.affinity, rtol=1e-6)
    np.testing.assert_allclose(
        forward.log_acceptance - reverse.log_acceptance,
        forward.affinity,
        rtol=1e-6,
    )


def test_floor_invalid_pair_exchange_is_rejected() -> None:
    proposal = pair_exchange_proposal(
        jnp.asarray(0.2),
        jnp.asarray(1.0),
        jnp.asarray(False),
        jnp.asarray(1.0),
        QUANTUM,
        FLOOR,
        CAPACITY,
        1.0,
        1.0,
    )
    assert not bool(proposal.floor_valid)
    assert np.isneginf(float(proposal.log_acceptance))


def test_periodic_contact_and_bonded_exchange_from_fixed_draws() -> None:
    positions = jnp.asarray(((0, 1), (3, 1)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((1.0, 3.0))
    signs = jnp.ones(2)
    uniforms = jnp.full(2, 0.5)

    contact = conduction_class_from_draws(
        energy,
        positions,
        lattice,
        jnp.zeros(2, dtype=jnp.uint8),
        0.0,
        1.0,
        QUANTUM,
        FLOOR,
        CAPACITY,
        0,
        signs,
        uniforms,
    )
    np.testing.assert_allclose(contact.energy, energy)
    assert not bool(jnp.any(contact.accepted))

    # Particle 1 has +x and particle 0 has the reciprocal -x bit.
    bonds = jnp.asarray((2, 1), dtype=jnp.uint8)
    bonded = conduction_class_from_draws(
        energy,
        positions,
        lattice,
        bonds,
        0.0,
        1.0,
        QUANTUM,
        FLOOR,
        CAPACITY,
        0,
        signs,
        uniforms,
    )
    np.testing.assert_allclose(bonded.energy, (1.25, 2.75))
    assert int(jnp.sum(bonded.accepted)) == 1
    np.testing.assert_allclose(bonded.throughput, QUANTUM)
    np.testing.assert_allclose(bonded.heat, 0.0)


def test_closed_vertical_boundary_does_not_conduct_across_top_and_bottom() -> None:
    positions = jnp.asarray(((1, 0), (1, 3)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((1.0, 4.0))
    result = conduction_sweep(
        jax.random.key(0),
        energy,
        positions,
        lattice,
        jnp.zeros(2, dtype=jnp.uint8),
        1.0,
        1.0,
        QUANTUM,
        FLOOR,
        CAPACITY,
    )
    np.testing.assert_allclose(result.energy, energy)
    assert not bool(jnp.any(result.active))


def test_each_full_lattice_class_has_disjoint_endpoints() -> None:
    n = 4
    positions = jnp.stack(
        jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    lattice = build_lattice(positions, n)
    energy = jnp.arange(1, n * n + 1, dtype=jnp.float32)
    bonds = jnp.zeros(n * n, dtype=jnp.uint8)
    signs = jnp.ones_like(energy)
    uniforms = jnp.full_like(energy, 0.5)

    expected_counts = (8, 8, 6, 6)
    for class_id, expected_count in enumerate(expected_counts):
        result = conduction_class_from_draws(
            energy,
            positions,
            lattice,
            bonds,
            0.2,
            0.8,
            QUANTUM,
            FLOOR,
            CAPACITY,
            class_id,
            signs,
            uniforms,
        )
        active = np.flatnonzero(np.asarray(result.active))
        target = np.asarray(result.edge_target)[active]
        endpoints = np.concatenate((active, target))
        assert len(active) == expected_count
        assert len(endpoints) == len(np.unique(endpoints))


def test_sweep_is_reproducible_conservative_and_respects_floor() -> None:
    positions = jnp.asarray(((0, 0), (0, 1), (0, 2)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((4.0, 2.0, 0.5))
    order = jnp.asarray((2, 3, 0, 1), dtype=jnp.int32)
    key = jax.random.key(4)
    arguments = (
        key,
        energy,
        positions,
        lattice,
        jnp.zeros(3, dtype=jnp.uint8),
        1.0,
        1.0,
        QUANTUM,
        FLOOR,
        CAPACITY,
        order,
    )
    result = conduction_sweep(*arguments)

    np.testing.assert_allclose(jnp.sum(result.energy), jnp.sum(energy), atol=1e-6)
    np.testing.assert_allclose(result.heat, 0.0, atol=1e-6)
    assert float(jnp.min(result.energy)) >= FLOOR
    np.testing.assert_allclose(
        result.throughput,
        QUANTUM * jnp.sum(result.accepted),
        atol=1e-6,
    )

    compiled = jax.jit(conduction_sweep)(*arguments)
    for field in result._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(compiled, field)),
            np.asarray(getattr(result, field)),
        )
