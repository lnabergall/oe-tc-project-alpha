from __future__ import annotations

import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oe_tc import runner
from oe_tc.state import State, StepMetrics
from oe_tc.storage import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    load_checkpoint,
    manifest,
    save_checkpoint,
    save_metrics,
)


def _fake_initialize(key, params, config):
    del key
    positions = jnp.stack(
        (jnp.arange(config.num_particles) % config.n, jnp.arange(config.num_particles) // config.n),
        axis=1,
    ).astype(jnp.int32)
    lattice = jnp.full((config.n, config.n), config.empty, dtype=jnp.int32)
    lattice = lattice.at[positions[:, 0], positions[:, 1]].set(
        jnp.arange(config.num_particles, dtype=jnp.int32)
    )
    return State(
        R=positions,
        E=jnp.full(
            (config.num_particles,),
            params.heat_capacity * params.bath_temperature,
            dtype=jnp.float32,
        ),
        L=lattice,
        bonds=jnp.zeros((config.num_particles,), dtype=jnp.uint8),
        root=jnp.arange(config.num_particles, dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )


def _fake_run_chunk(state, base_key, params, config, num_steps):
    del base_key, params, config
    start = state.sweep.astype(jnp.float32)
    values = start + jnp.arange(1, num_steps + 1, dtype=jnp.float32)
    metrics = StepMetrics(**{name: values for name in StepMetrics._fields})
    return state._replace(sweep=state.sweep + jnp.uint32(num_steps)), metrics


@pytest.fixture
def fake_runtime():
    return runner.Runtime(
        jax=jax,
        initialize_state=_fake_initialize,
        run_chunk=_fake_run_chunk,
        manifest=manifest,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
        save_metrics=save_metrics,
        model_version=MODEL_VERSION,
        schema_version=SCHEMA_VERSION,
    )


def _args(*values: str):
    return runner.build_parser().parse_args(values)


def _create_zero_run(output, fake_runtime, *, particles=2):
    spec = runner.resolve_run_spec(
        _args(
            "--n",
            "4",
            "--particles",
            str(particles),
            "--steps",
            "0",
            "--output",
            str(output),
        ),
        fake_runtime,
    )
    runner.run(spec, fake_runtime, platform="cpu", quiet=True)


def _resume_spec(output, fake_runtime, *, steps=0):
    return runner.resolve_run_spec(
        _args("--steps", str(steps), "--resume", str(output)),
        fake_runtime,
    )


def test_inline_parameter_overrides_and_source_switch(tmp_path, fake_runtime):
    args = _args(
        "--n",
        "4",
        "--particles",
        "3",
        "--steps",
        "0",
        "--output",
        str(tmp_path / "run"),
        "--params-json",
        '{"eta": 0.2, "second_conduction": true}',
        "--no-source",
    )
    spec = runner.resolve_run_spec(args, fake_runtime)

    assert spec.config.n == 4
    assert spec.config.num_particles == 3
    assert spec.params.eta == pytest.approx(0.2)
    assert spec.params.second_conduction is True
    assert spec.params.source_enabled is False


def test_parameter_override_file_and_unknown_name(tmp_path, fake_runtime):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"bath_temperature": 1.5}), encoding="utf-8")
    args = _args("--steps", "0", "--output", str(tmp_path / "run"), "--params-json", str(path))
    assert runner.resolve_run_spec(args, fake_runtime).params.bath_temperature == pytest.approx(1.5)

    bad = _args(
        "--steps",
        "0",
        "--output",
        str(tmp_path / "other"),
        "--params-json",
        '{"not_a_parameter": 1}',
    )
    with pytest.raises(ValueError, match="unknown model parameters"):
        runner.resolve_run_spec(bad, fake_runtime)


def test_chunked_run_and_resume_are_deterministic(tmp_path, fake_runtime):
    resumed_dir = tmp_path / "resumed"
    first = runner.resolve_run_spec(
        _args(
            "--n",
            "4",
            "--particles",
            "2",
            "--steps",
            "3",
            "--chunk-size",
            "2",
            "--seed",
            "17",
            "--output",
            str(resumed_dir),
        ),
        fake_runtime,
    )
    first_state = runner.run(first, fake_runtime, platform="cpu", quiet=True)
    assert int(first_state.sweep) == 3

    resumed = runner.resolve_run_spec(
        _args("--steps", "5", "--chunk-size", "2", "--resume", str(resumed_dir)),
        fake_runtime,
    )
    resumed_state = runner.run(resumed, fake_runtime, platform="cpu", quiet=True)

    straight_dir = tmp_path / "straight"
    straight = runner.resolve_run_spec(
        _args(
            "--n",
            "4",
            "--particles",
            "2",
            "--steps",
            "5",
            "--chunk-size",
            "4",
            "--seed",
            "17",
            "--output",
            str(straight_dir),
        ),
        fake_runtime,
    )
    straight_state = runner.run(straight, fake_runtime, platform="cpu", quiet=True)

    assert int(resumed_state.sweep) == int(straight_state.sweep) == 5
    np.testing.assert_array_equal(resumed_state.R, straight_state.R)
    with h5py.File(resumed_dir / runner.METRICS_NAME, "r") as file:
        assert set(file) == set(StepMetrics._fields)
        assert {dataset.shape[0] for dataset in file.values()} == {5}
    saved_state, resumed_key = load_checkpoint(resumed_dir / runner.CHECKPOINT_NAME)
    _, straight_key = load_checkpoint(straight_dir / runner.CHECKPOINT_NAME)
    assert int(saved_state.sweep) == 5
    np.testing.assert_array_equal(jax.random.key_data(resumed_key), jax.random.key_data(straight_key))


def test_resume_rejects_physics_changes(tmp_path, fake_runtime):
    output = tmp_path / "run"
    _create_zero_run(output, fake_runtime)
    changed = _args(
        "--steps",
        "1",
        "--resume",
        str(output),
        "--params-json",
        '{"eta": 0.2}',
    )
    with pytest.raises(ValueError, match="parameters cannot change"):
        runner.resolve_run_spec(changed, fake_runtime)


def test_resume_rejects_checkpoint_shape_corruption(tmp_path, fake_runtime):
    output = tmp_path / "run"
    _create_zero_run(output, fake_runtime)
    with h5py.File(output / runner.CHECKPOINT_NAME, "a") as file:
        del file["R"]
        file.create_dataset("R", data=np.zeros((1, 2), dtype=np.int32))

    with pytest.raises(ValueError, match="field 'R' has shape"):
        runner.run(_resume_spec(output, fake_runtime), fake_runtime, platform="cpu", quiet=True)


def test_resume_rejects_checkpoint_dtype_corruption(tmp_path, fake_runtime):
    output = tmp_path / "run"
    _create_zero_run(output, fake_runtime)
    with h5py.File(output / runner.CHECKPOINT_NAME, "a") as file:
        energy = file["E"][()].astype(np.float64)
        del file["E"]
        file.create_dataset("E", data=energy)

    with pytest.raises(ValueError, match="field 'E' has dtype"):
        runner.run(_resume_spec(output, fake_runtime), fake_runtime, platform="cpu", quiet=True)


def test_resume_rejects_invalid_checkpoint_energy(tmp_path, fake_runtime):
    output = tmp_path / "run"
    _create_zero_run(output, fake_runtime)
    with h5py.File(output / runner.CHECKPOINT_NAME, "a") as file:
        file["E"][0] = np.float32(0.0)

    with pytest.raises(ValueError, match="energy_floor"):
        runner.run(_resume_spec(output, fake_runtime), fake_runtime, platform="cpu", quiet=True)


def test_resume_rejects_invalid_checkpoint_topology(tmp_path, fake_runtime):
    output = tmp_path / "run"
    _create_zero_run(output, fake_runtime)
    with h5py.File(output / runner.CHECKPOINT_NAME, "a") as file:
        file["bonds"][0] = np.uint8(1)

    with pytest.raises(ValueError, match="bonds"):
        runner.run(_resume_spec(output, fake_runtime), fake_runtime, platform="cpu", quiet=True)


@pytest.mark.parametrize("bad_seed", [None, -1, 2**32, 1.5, True])
def test_resume_rejects_missing_or_invalid_manifest_seed(tmp_path, fake_runtime, bad_seed):
    output = tmp_path / f"run-{bad_seed}"
    _create_zero_run(output, fake_runtime)
    path = output / runner.MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if bad_seed is None:
        del value["seed"]
    else:
        value["seed"] = bad_seed
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="unsigned 32-bit seed"):
        _resume_spec(output, fake_runtime)


def test_reconcile_metrics_truncates_only_ahead_rows(tmp_path, fake_runtime):
    path = tmp_path / "metrics.h5"
    with h5py.File(path, "w") as file:
        file.attrs["model_version"] = fake_runtime.model_version
        file.attrs["schema_version"] = fake_runtime.schema_version
        file.create_dataset("a", data=np.arange(4), maxshape=(None,))
        file.create_dataset("b", data=np.arange(4), maxshape=(None,))

    runner._reconcile_metrics(path, 3, fake_runtime)
    assert runner._metric_length(
        path,
        model_version=fake_runtime.model_version,
        schema_version=fake_runtime.schema_version,
    ) == 3
    with pytest.raises(ValueError, match="checkpoint is ahead"):
        runner._reconcile_metrics(path, 4, fake_runtime)


def test_resume_rejects_metrics_schema_mismatch(tmp_path, fake_runtime):
    output = tmp_path / "run"
    spec = runner.resolve_run_spec(
        _args("--n", "4", "--steps", "1", "--output", str(output)),
        fake_runtime,
    )
    runner.run(spec, fake_runtime, platform="cpu", quiet=True)
    with h5py.File(output / runner.METRICS_NAME, "a") as file:
        file.attrs["schema_version"] = fake_runtime.schema_version + 1

    with pytest.raises(ValueError, match="metrics schema"):
        runner.run(_resume_spec(output, fake_runtime, steps=1), fake_runtime, platform="cpu", quiet=True)


def test_atomic_replace_retries_transient_permission_error(tmp_path, monkeypatch):
    source = tmp_path / ".checkpoint.tmp.h5"
    destination = tmp_path / "checkpoint.h5"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    real_replace = runner.os.replace
    attempts = []
    delays = []

    def flaky_replace(current_source, current_destination):
        attempts.append((current_source, current_destination))
        if len(attempts) < 3:
            raise PermissionError("transient sharing violation")
        real_replace(current_source, current_destination)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", delays.append)

    runner._replace_with_retry(source, destination)

    assert destination.read_bytes() == b"new"
    assert not source.exists()
    assert len(attempts) == 3
    assert delays == [0.025, 0.05]


def test_environment_configuration(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    runner.configure_environment("gpu", False)
    assert runner.os.environ["JAX_PLATFORMS"] == "cuda"
    runner.configure_environment("cpu", True)
    assert runner.os.environ["JAX_PLATFORMS"] == "cpu"
    assert runner.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
