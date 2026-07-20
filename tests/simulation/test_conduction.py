from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.conduction import conduction_class, conduction_sweep, pair_energy_flux
from oe_tc.geometry import build_lattice


def test_pair_flux_selects_contact_or_bond_coefficient() -> None:
    source = jnp.asarray((3.0, 3.0))
    target = jnp.asarray((1.0, 1.0))
    flux = pair_energy_flux(source, target, jnp.asarray((False, True)), 0.2, 0.8)
    np.testing.assert_allclose(flux, (0.2, 0.8))


def test_periodic_contact_and_bonded_conduction_are_exact() -> None:
    positions = jnp.asarray(((0, 1), (3, 1)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((1.0, 3.0))
    contact = conduction_sweep(
        energy, positions, lattice, jnp.zeros(2, dtype=jnp.uint8), 0.5, 1.0
    )
    np.testing.assert_allclose(contact.energy, (1.5, 2.5))

    # Particle 1 has +x and particle 0 has the reciprocal -x bit.
    bonds = jnp.asarray((2, 1), dtype=jnp.uint8)
    bonded = conduction_sweep(energy, positions, lattice, bonds, 0.5, 1.0)
    np.testing.assert_allclose(bonded.energy, (2.0, 2.0))


def test_closed_vertical_boundary_does_not_conduct_across_top_and_bottom() -> None:
    positions = jnp.asarray(((1, 0), (1, 3)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((1.0, 4.0))
    result = conduction_sweep(
        energy, positions, lattice, jnp.zeros(2, dtype=jnp.uint8), 1.0, 1.0
    )
    np.testing.assert_allclose(result.energy, energy)
    assert not bool(jnp.any(result.active))


def test_each_full_lattice_class_has_disjoint_endpoints() -> None:
    n = 4
    positions = jnp.stack(jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij"), axis=-1).reshape((-1, 2))
    lattice = build_lattice(positions, n)
    energy = jnp.arange(1, n * n + 1, dtype=jnp.float32)
    bonds = jnp.zeros(n * n, dtype=jnp.uint8)

    expected_counts = (8, 8, 6, 6)
    for class_id, expected_count in enumerate(expected_counts):
        result = conduction_class(energy, positions, lattice, bonds, 0.2, 0.8, class_id)
        active = np.flatnonzero(np.asarray(result.active))
        target = np.asarray(result.edge_target)[active]
        endpoints = np.concatenate((active, target))
        assert len(active) == expected_count
        assert len(endpoints) == len(np.unique(endpoints))


def test_classes_are_sequential_and_the_sweep_conserves_energy_and_floor() -> None:
    positions = jnp.asarray(((0, 0), (0, 1), (0, 2)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((4.0, 2.0, 0.5))
    order = jnp.asarray((2, 3, 0, 1), dtype=jnp.int32)
    result = conduction_sweep(
        energy,
        positions,
        lattice,
        jnp.zeros(3, dtype=jnp.uint8),
        1.0,
        1.0,
        order,
    )
    np.testing.assert_allclose(result.energy, (3.0, 1.75, 1.75))
    np.testing.assert_allclose(jnp.sum(result.energy), jnp.sum(energy))
    np.testing.assert_allclose(result.heat, 0.0)
    assert float(jnp.min(result.energy)) >= 0.5

    compiled = jax.jit(conduction_sweep)(
        energy,
        positions,
        lattice,
        jnp.zeros(3, dtype=jnp.uint8),
        1.0,
        1.0,
        order,
    )
    np.testing.assert_allclose(compiled.energy, result.energy)
