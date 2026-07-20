from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.bath import (
    bath_proposal,
    collision_coefficient,
    direct_bath_exchange,
    exposure_properties,
    heat_exchange_from_draws,
    molecule_translation_drag,
    sample_heat_exchange,
    thermal_coupling,
)
from oe_tc.geometry import build_lattice


def test_exposure_coefficients_and_molecule_drag_are_linear() -> None:
    exposure = jnp.asarray((0.0, 0.5, 1.0))
    np.testing.assert_allclose(collision_coefficient(exposure, 1.0, 2.0), (1.0, 2.0, 3.0))
    np.testing.assert_allclose(thermal_coupling(exposure, 0.1, 0.4), (0.1, 0.3, 0.5))
    drag = molecule_translation_drag(jnp.asarray((1.0, 2.0, 4.0)), jnp.asarray((0, 0, 2)))
    np.testing.assert_allclose(drag, (3.0, 0.0, 4.0))


def test_exposure_properties_use_closed_boundary_geometry() -> None:
    positions = jnp.asarray(((1, 0), (2, 2)), dtype=jnp.int32)
    properties = exposure_properties(
        positions, build_lattice(positions, 4), 1.0, 2.0, 0.1, 0.4
    )
    np.testing.assert_allclose(properties.fraction, (0.75, 1.0))
    np.testing.assert_allclose(properties.gamma, (2.5, 3.0))
    np.testing.assert_allclose(properties.kappa, (0.4, 0.5))


def test_deterministic_draws_respect_floor_and_report_heat() -> None:
    energy = jnp.asarray((1.0, 0.2, 2.0))
    result = heat_exchange_from_draws(
        energy=energy,
        exposure=jnp.asarray((0.0, 1.0, 0.5)),
        signs=jnp.asarray((1.0, -1.0, -1.0)),
        uniforms=jnp.asarray((0.0, 0.0, 0.0)),
        energy_quantum=0.2,
        energy_floor=0.1,
        heat_capacity=1.0,
        bath_temperature=1.0,
        kappa_base=1.0,
        kappa_exposure=0.0,
    )
    np.testing.assert_array_equal(result.accepted, (True, False, True))
    expected_energy = np.asarray((1.2, 0.2, 1.8), dtype=np.float32)
    np.testing.assert_allclose(result.energy, expected_energy)
    realized_heat = expected_energy - np.asarray(energy)
    np.testing.assert_array_equal(result.particle_heat, realized_heat)
    np.testing.assert_allclose(result.heat, np.sum(realized_heat), atol=1e-7)
    np.testing.assert_array_equal(np.asarray(result.energy - energy), result.particle_heat)
    assert np.isneginf(float(result.log_acceptance[1]))


def test_forward_reverse_bath_acceptance_has_affinity_ratio() -> None:
    exposure = jnp.asarray((0.3,))
    forward = bath_proposal(
        jnp.asarray((1.5,)), exposure, jnp.asarray((1.0,)), 0.2, 0.1, 2.0, 1.3, 0.2, 0.4
    )
    reverse = bath_proposal(
        jnp.asarray((1.7,)), exposure, jnp.asarray((-1.0,)), 0.2, 0.1, 2.0, 1.3, 0.2, 0.4
    )
    np.testing.assert_allclose(reverse.affinity, -forward.affinity, atol=1e-6)
    np.testing.assert_allclose(
        forward.log_acceptance - reverse.log_acceptance, forward.affinity, atol=1e-6
    )


def test_zero_coupling_rejects_and_sampled_exchange_is_reproducible_and_jittable() -> None:
    energy = jnp.asarray((1.0, 1.5))
    exposure = jnp.asarray((0.5, 1.0))
    rejected = heat_exchange_from_draws(
        energy, exposure, jnp.ones(2), jnp.zeros(2), 0.1, 0.1, 2.0, 1.0, 0.0, 0.0
    )
    assert not bool(jnp.any(rejected.accepted))
    np.testing.assert_allclose(rejected.energy, energy)

    key = jax.random.key(11)
    eager = sample_heat_exchange(key, energy, exposure, 0.1, 0.1, 2.0, 1.0, 0.2, 0.4)
    repeated = sample_heat_exchange(key, energy, exposure, 0.1, 0.1, 2.0, 1.0, 0.2, 0.4)
    np.testing.assert_array_equal(eager.energy, repeated.energy)

    positions = jnp.asarray(((0, 0), (2, 2)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    compiled = jax.jit(direct_bath_exchange)(
        key, energy, positions, lattice, 0.1, 0.1, 2.0, 1.0, 0.2, 0.4
    )
    direct = direct_bath_exchange(
        key, energy, positions, lattice, 0.1, 0.1, 2.0, 1.0, 0.2, 0.4
    )
    np.testing.assert_array_equal(compiled.energy, direct.energy)
    np.testing.assert_allclose(
        jnp.sum(compiled.energy - energy), compiled.heat, atol=5e-8
    )
