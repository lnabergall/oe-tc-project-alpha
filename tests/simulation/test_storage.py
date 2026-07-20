import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oe_tc.config import StaticConfig, default_params
from oe_tc.state import State, StepMetrics
from oe_tc.storage import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    load_checkpoint,
    manifest,
    save_checkpoint,
    save_metrics,
)


def test_manifest_is_versioned_and_complete():
    config = StaticConfig(n=4, num_particles=2)
    result = manifest(config, default_params(), seed=17)

    assert result["model_version"] == MODEL_VERSION
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["static"]["n"] == 4
    assert "bath_temperature" in result["params"]


def test_checkpoint_round_trip(tmp_path):
    state = State(
        R=jnp.array([[0, 0], [1, 1]], dtype=jnp.int32),
        E=jnp.array([1.0, 2.0]),
        L=jnp.array([[0, 2], [2, 1]], dtype=jnp.int32),
        bonds=jnp.zeros(2, dtype=jnp.uint8),
        root=jnp.arange(2, dtype=jnp.int32),
        sweep=jnp.asarray(3, dtype=jnp.uint32),
    )
    key = jax.random.key(5)
    path = tmp_path / "checkpoint.h5"

    save_checkpoint(path, state, key)
    restored, restored_key = load_checkpoint(path)

    for expected, actual in zip(state, restored):
        np.testing.assert_array_equal(np.asarray(expected), actual)
    np.testing.assert_array_equal(
        jax.random.key_data(key), jax.random.key_data(restored_key)
    )


def _metric_row(value: float) -> StepMetrics:
    return StepMetrics(**{name: np.asarray(value) for name in StepMetrics._fields})


def test_metrics_append_and_reject_schema_mismatch(tmp_path):
    path = tmp_path / "metrics.h5"
    save_metrics(path, [_metric_row(1.0), _metric_row(2.0)])
    save_metrics(path, [_metric_row(3.0)])
    with h5py.File(path, "r+") as file:
        assert set(file) == set(StepMetrics._fields)
        assert {dataset.shape for dataset in file.values()} == {(3,)}
        file.attrs["schema_version"] = SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="schema"):
        save_metrics(path, [_metric_row(4.0)])
