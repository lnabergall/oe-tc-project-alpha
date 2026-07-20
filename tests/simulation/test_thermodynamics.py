from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.thermodynamics import (
    bath_channel_affinity,
    entropy,
    event_affinity,
    event_thermodynamics,
    internal_channel_affinity,
    kinetic_metropolis_log_probability,
    metropolis_probability,
    temperature,
)


def test_temperature_and_entropy_definition() -> None:
    energy = jnp.asarray((0.5, 1.0, 2.0))
    np.testing.assert_allclose(temperature(energy, 2.0), (0.25, 0.5, 1.0))
    np.testing.assert_allclose(entropy(energy, 3.0, 0.5)[0], 0.0)
    np.testing.assert_allclose(entropy(energy, 3.0, 0.5)[2], 3.0 * np.log(4.0))


def test_general_event_reports_explicit_energy_and_entropy_fluxes() -> None:
    before = jnp.asarray((2.0, 3.0))
    after = jnp.asarray((1.8, 2.7))
    result = event_thermodynamics(0.8, before, after, 2.0, 0.1, 1.5)
    expected_delta_e = -0.5
    expected_heat = 0.3
    expected_delta_s = 2.0 * np.sum(np.log(np.asarray(after) / np.asarray(before)))
    np.testing.assert_allclose(result.internal_energy_change, expected_delta_e, atol=1e-6)
    np.testing.assert_allclose(result.heat, expected_heat, atol=1e-6)
    np.testing.assert_allclose(result.entropy_change, expected_delta_s, atol=1e-6)
    np.testing.assert_allclose(
        result.affinity, expected_delta_s - expected_heat / 1.5, atol=1e-6
    )
    assert bool(result.floor_valid)


def test_channel_affinities_reduce_to_the_model_formulas() -> None:
    np.testing.assert_allclose(bath_channel_affinity(0.6, 1.5), -0.4)
    before = jnp.asarray((2.0, 3.0))
    after = jnp.asarray((2.4, 2.6))
    expected = 4.0 * np.sum(np.log(np.asarray(after) / np.asarray(before)))
    np.testing.assert_allclose(
        internal_channel_affinity(before, after, 4.0, 0.1), expected, atol=1e-7
    )


def test_metropolis_probability_obeys_forward_reverse_ratio() -> None:
    affinity = jnp.asarray((-3.0, -0.2, 0.2, 3.0))
    probability = metropolis_probability(affinity)
    reverse = metropolis_probability(-affinity)
    np.testing.assert_allclose(probability / reverse, np.exp(np.asarray(affinity)))
    assert np.all((np.asarray(probability) >= 0.0) & (np.asarray(probability) <= 1.0))

    log_probability = kinetic_metropolis_log_probability(affinity, 0.25)
    np.testing.assert_allclose(np.exp(np.asarray(log_probability)), 0.25 * probability)
    assert np.isneginf(float(kinetic_metropolis_log_probability(1.0, 0.0)))


def test_event_affinity_is_jittable() -> None:
    before = jnp.asarray((1.0, 2.0))
    after = jnp.asarray((1.2, 1.7))
    eager = event_affinity(0.4, before, after, 3.0, 0.1, 1.25)
    compiled = jax.jit(event_affinity)(0.4, before, after, 3.0, 0.1, 1.25)
    np.testing.assert_allclose(compiled, eager)
