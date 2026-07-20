from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.geometry import build_lattice
from oe_tc.source import (
    absorb_packets,
    irradiate,
    planck_mode_probabilities,
    planck_tail_probability_bound,
    sample_planck_packets,
)


def test_bounded_planck_mixture_weights_and_error_bound() -> None:
    probabilities = planck_mode_probabilities(64)
    np.testing.assert_allclose(np.sum(np.asarray(probabilities)), 1.0, atol=1e-7)
    assert np.all(np.diff(np.asarray(probabilities)) < 0.0)
    assert planck_tail_probability_bound(64) < 1.2e-6


def test_planck_sampler_is_positive_reproducible_and_has_correct_mean() -> None:
    key = jax.random.key(7)
    beta = 2.0
    samples = sample_planck_packets(key, beta, 80_000, terms=64)
    repeated = sample_planck_packets(key, beta, 80_000, terms=64)
    np.testing.assert_array_equal(samples, repeated)
    assert bool(jnp.all(samples > 0.0))

    expected_mean = 3.832229496 / beta
    np.testing.assert_allclose(np.mean(np.asarray(samples)), expected_mean, rtol=0.015)


def test_packets_hit_only_the_topmost_particle_and_empty_columns_escape() -> None:
    positions = jnp.asarray(((0, 2), (0, 0), (2, 3)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((1.0, 2.0, 3.0))
    packets = jnp.asarray((0.5, 1.0, 1.5, 2.0))
    result = absorb_packets(energy, lattice, packets)

    np.testing.assert_allclose(result.particle_heat, (0.0, 0.5, 1.5))
    np.testing.assert_allclose(result.energy, (1.0, 2.5, 4.5))
    np.testing.assert_allclose(result.heat, 2.0)
    np.testing.assert_allclose(result.escaped_energy, 3.0)
    np.testing.assert_allclose(result.incident_energy, 5.0)
    np.testing.assert_allclose(result.heat + result.escaped_energy, result.incident_energy)
    np.testing.assert_array_equal(result.targets, (1, 3, 2, 3))


def test_composed_irradiation_is_jittable_and_energy_auditable() -> None:
    positions = jnp.asarray(((0, 1), (3, 0)), dtype=jnp.int32)
    lattice = build_lattice(positions, 4)
    energy = jnp.asarray((1.0, 1.5))
    key = jax.random.key(3)
    eager = irradiate(key, energy, lattice, 4.0, terms=32)
    compiled = jax.jit(lambda k, e, l: irradiate(k, e, l, 4.0, terms=32))(
        key, energy, lattice
    )
    np.testing.assert_allclose(compiled.energy, eager.energy)
    np.testing.assert_allclose(compiled.heat, eager.heat)
    np.testing.assert_allclose(jnp.sum(compiled.energy - energy), compiled.heat)
