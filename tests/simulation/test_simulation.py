from __future__ import annotations

import jax
import numpy as np

from oe_tc.config import StaticConfig, default_params
from oe_tc.initialization import initialize_state
from oe_tc.simulation import run_chunk, sweep
from oe_tc.validation import validate_state


def _internal_only_params():
    return default_params()._replace(
        source_enabled=False,
        kappa_base=0.0,
        kappa_exposure=0.0,
        bath_channel_probability=0.0,
        translation_frequency=0.5,
        rotation_frequency=0.5,
        bond_frequency=0.5,
    )


def test_jitted_sweep_preserves_invariants_and_energy_ledger() -> None:
    config = StaticConfig(n=6, num_particles=12)
    params = _internal_only_params()
    state = initialize_state(jax.random.key(1), params, config)
    base_key = jax.random.key(2)

    step = jax.jit(lambda current: sweep(current, base_key, params, config))
    state, metrics = step(state)
    state, metrics = jax.device_get((state, metrics))

    validation = validate_state(state, params, config)
    assert bool(validation.valid)
    assert int(state.sweep) == 1
    assert int(metrics.accepted_conduction_exchanges) >= 0
    assert np.isclose(
        float(metrics.conduction_energy_throughput),
        params.conduction_energy_quantum
        * int(metrics.accepted_conduction_exchanges),
        atol=2e-6,
    )
    assert abs(float(metrics.energy_residual)) < 2e-5


def test_chunk_partition_does_not_change_trajectory() -> None:
    config = StaticConfig(n=6, num_particles=8)
    params = default_params()._replace(source_enabled=False)
    initial = initialize_state(jax.random.key(3), params, config)
    base_key = jax.random.key(4)

    whole, whole_metrics = run_chunk(initial, base_key, params, config, 3)
    split, first_metrics = run_chunk(initial, base_key, params, config, 1)
    split, second_metrics = run_chunk(split, base_key, params, config, 2)

    for expected, actual in zip(jax.device_get(whole), jax.device_get(split)):
        np.testing.assert_array_equal(expected, actual)
    for field in whole_metrics._fields:
        combined = np.concatenate(
            (
                np.asarray(getattr(first_metrics, field)),
                np.asarray(getattr(second_metrics, field)),
            )
        )
        np.testing.assert_array_equal(np.asarray(getattr(whole_metrics, field)), combined)


def test_source_and_bath_fluxes_close_total_energy_balance() -> None:
    config = StaticConfig(n=4, num_particles=6)
    params = default_params()
    state = initialize_state(jax.random.key(5), params, config)
    state, metrics = sweep(state, jax.random.key(6), params, config)

    assert float(metrics.source_incident_energy) + 1e-6 >= float(
        metrics.source_energy
    )
    assert np.isclose(
        float(metrics.source_incident_energy),
        float(metrics.source_energy + metrics.source_escaped_energy),
        atol=2e-6,
    )
    assert abs(float(metrics.energy_residual)) < 2e-5
    assert int(state.sweep) == 1


def test_compiled_chunk_has_leading_metric_axis() -> None:
    config = StaticConfig(n=4, num_particles=4)
    params = default_params()._replace(source_enabled=False)
    state = initialize_state(jax.random.key(7), params, config)
    compiled = jax.jit(
        lambda current: run_chunk(current, jax.random.key(8), params, config, 2)
    )

    state, metrics = compiled(state)
    jax.block_until_ready((state, metrics))
    assert int(state.sweep) == 2
    assert {getattr(metrics, field).shape for field in metrics._fields} == {(2,)}
