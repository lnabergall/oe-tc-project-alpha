from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

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


def _initialize(key, params, config):
    del key
    positions = jnp.asarray(((0, 0), (1, 0)), dtype=jnp.int32)
    lattice = jnp.full((config.n, config.n), config.empty, dtype=jnp.int32)
    lattice = lattice.at[positions[:, 0], positions[:, 1]].set(jnp.arange(2, dtype=jnp.int32))
    return State(
        R=positions,
        E=jnp.full((2,), params.heat_capacity * params.bath_temperature, dtype=jnp.float32),
        L=lattice,
        bonds=jnp.zeros((2,), dtype=jnp.uint8),
        root=jnp.arange(2, dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )


def _metrics(num_steps, *, component_iterations=0):
    values = {}
    for name in StepMetrics._fields:
        if name == "components_converged":
            values[name] = jnp.ones((num_steps,), dtype=bool)
        elif name == "component_iterations":
            values[name] = jnp.full((num_steps,), component_iterations, dtype=jnp.int32)
        else:
            values[name] = jnp.zeros((num_steps,), dtype=jnp.float32)
    return StepMetrics(**values)


def _run_chunk(state, base_key, params, config, num_steps):
    del base_key, params, config
    return state._replace(sweep=state.sweep + jnp.uint32(num_steps)), _metrics(num_steps)


def _runtime(run_chunk=_run_chunk):
    return runner.Runtime(
        jax=jax,
        initialize_state=_initialize,
        run_chunk=run_chunk,
        manifest=manifest,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
        save_metrics=save_metrics,
        model_version=MODEL_VERSION,
        schema_version=SCHEMA_VERSION,
    )


def _args(*values):
    return runner.build_parser().parse_args(tuple(map(str, values)))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshots_align_chunks_and_resume_without_overwriting(tmp_path):
    runtime = _runtime()
    output = tmp_path / "run"
    initial = runner.resolve_run_spec(
        _args(
            "--n",
            4,
            "--particles",
            2,
            "--steps",
            4,
            "--chunk-size",
            5,
            "--snapshot-every",
            2,
            "--output",
            output,
        ),
        runtime,
    )
    runner.run(initial, runtime, platform="cpu", quiet=True)

    snapshot_0 = output / runner.SNAPSHOT_DIRECTORY / "sweep_0000000000.h5"
    snapshot_2 = output / runner.SNAPSHOT_DIRECTORY / "sweep_0000000002.h5"
    snapshot_4 = output / runner.SNAPSHOT_DIRECTORY / "sweep_0000000004.h5"
    assert snapshot_0.is_file() and snapshot_2.is_file() and snapshot_4.is_file()
    assert int(load_checkpoint(snapshot_0)[0].sweep) == 0
    assert int(load_checkpoint(snapshot_2)[0].sweep) == 2
    assert int(load_checkpoint(snapshot_4)[0].sweep) == 4
    original_digests = (_digest(snapshot_2), _digest(snapshot_4))
    saved_manifest = json.loads((output / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved_manifest["runtime"]["snapshot_every"] == 2

    resumed = runner.resolve_run_spec(
        _args("--steps", 6, "--chunk-size", 5, "--resume", output),
        runtime,
    )
    assert resumed.snapshot_every == 2
    runner.run(resumed, runtime, platform="cpu", quiet=True)

    assert (_digest(snapshot_2), _digest(snapshot_4)) == original_digests
    snapshot_6 = output / runner.SNAPSHOT_DIRECTORY / "sweep_0000000006.h5"
    assert int(load_checkpoint(snapshot_6)[0].sweep) == 6


def test_initial_snapshot_is_written_without_periodic_snapshots(tmp_path):
    runtime = _runtime()
    output = tmp_path / "run"
    spec = runner.resolve_run_spec(
        _args("--n", 4, "--particles", 2, "--steps", 1, "--output", output),
        runtime,
    )

    runner.run(spec, runtime, platform="cpu", quiet=True)

    initial = output / runner.SNAPSHOT_DIRECTORY / "sweep_0000000000.h5"
    assert int(load_checkpoint(initial)[0].sweep) == 0


def test_snapshot_interval_is_positive_and_immutable(tmp_path):
    runtime = _runtime()
    invalid = _args("--steps", 0, "--snapshot-every", 0, "--output", tmp_path / "bad")
    with pytest.raises(ValueError, match="positive unsigned 32-bit"):
        runner.resolve_run_spec(invalid, runtime)

    output = tmp_path / "run"
    spec = runner.resolve_run_spec(
        _args(
            "--n",
            4,
            "--particles",
            2,
            "--steps",
            0,
            "--snapshot-every",
            2,
            "--output",
            output,
        ),
        runtime,
    )
    runner.run(spec, runtime, platform="cpu", quiet=True)
    changed = _args("--steps", 1, "--snapshot-every", 3, "--resume", output)
    with pytest.raises(ValueError, match="cannot change"):
        runner.resolve_run_spec(changed, runtime)


def test_component_iteration_cap_does_not_persist_chunk(tmp_path):
    def saturated_chunk(state, base_key, params, config, num_steps):
        del base_key, params
        state = state._replace(sweep=state.sweep + jnp.uint32(num_steps))
        metrics = _metrics(num_steps, component_iterations=config.component_max_iters)
        if hasattr(metrics, "components_converged"):
            metrics = metrics._replace(
                components_converged=jnp.zeros((num_steps,), dtype=bool)
            )
        return state, metrics

    runtime = _runtime(saturated_chunk)
    output = tmp_path / "run"
    spec = runner.resolve_run_spec(
        _args("--n", 4, "--particles", 2, "--steps", 1, "--output", output),
        runtime,
    )
    with pytest.raises(RuntimeError, match="did not converge|iteration cap"):
        runner.run(spec, runtime, platform="cpu", quiet=True)

    assert not (output / runner.METRICS_NAME).exists()
    assert int(load_checkpoint(output / runner.CHECKPOINT_NAME)[0].sweep) == 0


def test_explicit_component_convergence_failure_is_guarded():
    class Metrics(NamedTuple):
        components_converged: np.ndarray
        component_iterations: np.ndarray

    metrics = Metrics(np.asarray((True, False)), np.asarray((1, 1)))
    with pytest.raises(RuntimeError, match="did not converge"):
        runner._guard_component_convergence(metrics, runner.StaticConfig(n=4, num_particles=2))
