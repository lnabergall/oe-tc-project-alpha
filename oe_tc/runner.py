"""Production-oriented, chunked command-line execution for OE-TC model.

Simulation work remains on device for a complete fixed-shape chunk. Compact
metrics and a restartable state cross to the host only at chunk boundaries.
Random events derive from an immutable base key and the checkpointed sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np

from oe_tc.config import Params, StaticConfig, default_params, validate_params


DEFAULT_N = 256
DEFAULT_CHUNK_SIZE = 128
MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.h5"
CHECKPOINT_NAME = "checkpoint.h5"
SNAPSHOT_DIRECTORY = "snapshots"


@dataclass(frozen=True)
class RunPaths:
    """Files belonging to one append-only logical run."""

    directory: Path
    manifest: Path
    metrics: Path
    checkpoint: Path
    snapshots: Path


@dataclass(frozen=True)
class RunSpec:
    """Resolved simulation inputs after defaults or resume validation."""

    config: StaticConfig
    params: Params
    seed: int
    target_sweep: int
    chunk_size: int
    snapshot_every: int | None
    paths: RunPaths
    resumed: bool


class Runtime(NamedTuple):
    """Lazily imported runtime dependencies, exposed for isolated tests."""

    jax: Any
    initialize_state: Callable[..., Any]
    run_chunk: Callable[..., Any]
    manifest: Callable[..., dict[str, Any]]
    save_checkpoint: Callable[..., None]
    load_checkpoint: Callable[..., tuple[Any, Any]]
    save_metrics: Callable[..., None]
    model_version: int
    schema_version: int


def build_parser() -> argparse.ArgumentParser:
    """Build the simulation command-line parser without importing JAX."""

    parser = argparse.ArgumentParser(
        description=(
            "Run OE-TC model in fixed-shape JAX chunks. On resume, --steps "
            "is the total target sweep rather than an additional step count."
        )
    )
    parser.add_argument("--n", type=int, help=f"even lattice width (default: {DEFAULT_N})")
    parser.add_argument(
        "--particles",
        type=int,
        help="particle count (default: one quarter of lattice sites)",
    )
    parser.add_argument("--steps", type=int, required=True, help="total target sweep")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"host synchronization interval (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument("--seed", type=int, help="unsigned 32-bit seed (default: 0)")
    parser.add_argument(
        "--output",
        type=Path,
        help="new run directory; inferred from --resume when resuming",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="checkpoint file, or a run directory containing checkpoint.h5",
    )
    parser.add_argument(
        "--params-json",
        help="JSON object or path to a JSON object overriding model parameters",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        dest="source_enabled",
        action="store_true",
        help="enable irradiation",
    )
    source.add_argument(
        "--no-source",
        dest="source_enabled",
        action="store_false",
        help="disable irradiation",
    )
    parser.set_defaults(source_enabled=None)
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
        help="JAX platform selected before runtime initialization",
    )
    parser.add_argument(
        "--no-preallocate",
        action="store_true",
        help="disable XLA GPU memory preallocation",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        help="save an immutable state snapshot every K sweeps",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress chunk progress")
    return parser


def configure_environment(platform: str, no_preallocate: bool) -> None:
    """Set JAX environment options before any runtime import."""

    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform
    if no_preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _load_runtime() -> Runtime:
    """Import JAX and implementations after accelerator configuration."""

    import jax

    from oe_tc.initialization import initialize_state
    from oe_tc.simulation import run_chunk
    from oe_tc.storage import (
        MODEL_VERSION,
        SCHEMA_VERSION,
        load_checkpoint,
        manifest,
        save_checkpoint,
        save_metrics,
    )

    return Runtime(
        jax=jax,
        initialize_state=initialize_state,
        run_chunk=run_chunk,
        manifest=manifest,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
        save_metrics=save_metrics,
        model_version=MODEL_VERSION,
        schema_version=SCHEMA_VERSION,
    )


def _read_json_object(spec: str | None) -> dict[str, Any]:
    """Read a JSON object from inline text or a filesystem path."""

    if spec is None:
        return {}
    candidate = Path(spec)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    text = candidate.read_text(encoding="utf-8") if is_file else spec
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid parameter JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("parameter JSON must contain one object")
    return value


def _apply_param_overrides(
    params: Params,
    overrides: dict[str, Any],
    source_enabled: bool | None,
) -> Params:
    unknown = sorted(set(overrides) - set(params._fields))
    if unknown:
        raise ValueError(f"unknown model parameters: {', '.join(unknown)}")
    if source_enabled is not None:
        if "source_enabled" in overrides and bool(overrides["source_enabled"]) != source_enabled:
            raise ValueError("--source/--no-source conflicts with parameter JSON")
        overrides = {**overrides, "source_enabled": source_enabled}
    result = params._replace(**overrides)
    validate_params(result)
    return result


def _checkpoint_from_argument(path: Path) -> Path:
    return path / CHECKPOINT_NAME if path.is_dir() else path


def _paths_for_new(output: Path) -> RunPaths:
    return RunPaths(
        directory=output,
        manifest=output / MANIFEST_NAME,
        metrics=output / METRICS_NAME,
        checkpoint=output / CHECKPOINT_NAME,
        snapshots=output / SNAPSHOT_DIRECTORY,
    )


def _paths_for_resume(resume: Path, output: Path | None) -> RunPaths:
    checkpoint = _checkpoint_from_argument(resume)
    directory = checkpoint.parent
    if output is not None and output.resolve() != directory.resolve():
        raise ValueError("a resumed run must use the checkpoint's run directory as --output")
    return RunPaths(
        directory=directory,
        manifest=directory / MANIFEST_NAME,
        metrics=directory / METRICS_NAME,
        checkpoint=checkpoint,
        snapshots=directory / SNAPSHOT_DIRECTORY,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"resume manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid resume manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("resume manifest must contain one object")
    return value


def _config_from_manifest(value: dict[str, Any]) -> StaticConfig:
    try:
        return StaticConfig(**value["static"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest contains an invalid static configuration") from error


def _params_from_manifest(value: dict[str, Any]) -> Params:
    try:
        params = Params(**value["params"])
        validate_params(params)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest contains invalid model parameters") from error
    return params


def _seed_from_manifest(value: dict[str, Any]) -> int:
    """Read a strictly typed unsigned 32-bit seed from a manifest."""

    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("manifest contains an invalid unsigned 32-bit seed")
    return seed


def _snapshot_interval_from_manifest(value: dict[str, Any]) -> int | None:
    """Read the immutable snapshot cadence from a run manifest."""

    runtime = value.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("manifest contains invalid runtime metadata")
    interval = runtime.get("snapshot_every")
    if interval is None:
        return None
    if isinstance(interval, bool) or not isinstance(interval, int) or not 0 < interval < 2**32:
        raise ValueError("manifest contains an invalid snapshot interval")
    return interval


def resolve_run_spec(args: argparse.Namespace, runtime: Runtime) -> RunSpec:
    """Resolve defaults or validate immutable inputs of a resumed run."""

    if not 0 <= args.steps < 2**32:
        raise ValueError("--steps must be an unsigned 32-bit sweep target")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if args.snapshot_every is not None and not 0 < args.snapshot_every < 2**32:
        raise ValueError("--snapshot-every must be a positive unsigned 32-bit interval")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be an unsigned 32-bit integer")

    overrides = _read_json_object(args.params_json)
    if args.resume is None:
        if args.output is None:
            raise ValueError("--output is required for a new run")
        n = DEFAULT_N if args.n is None else args.n
        particles = max(1, n * n // 4) if args.particles is None else args.particles
        seed = 0 if args.seed is None else args.seed
        config = StaticConfig(n=n, num_particles=particles)
        params = _apply_param_overrides(default_params(), overrides, args.source_enabled)
        paths = _paths_for_new(args.output)
        occupied = [
            path
            for path in (paths.manifest, paths.metrics, paths.checkpoint, paths.snapshots)
            if path.exists()
        ]
        if occupied:
            names = ", ".join(str(path) for path in occupied)
            raise ValueError(f"run output already exists: {names}")
        return RunSpec(
            config=config,
            params=params,
            seed=seed,
            target_sweep=args.steps,
            chunk_size=args.chunk_size,
            snapshot_every=args.snapshot_every,
            paths=paths,
            resumed=False,
        )

    paths = _paths_for_resume(args.resume, args.output)
    if not paths.checkpoint.is_file():
        raise ValueError(f"resume checkpoint not found: {paths.checkpoint}")
    saved = _load_manifest(paths.manifest)
    if saved.get("model_version") != runtime.model_version:
        raise ValueError("manifest belongs to a different model version")
    if saved.get("schema_version") != runtime.schema_version:
        raise ValueError("unsupported manifest schema")

    config = _config_from_manifest(saved)
    params = _params_from_manifest(saved)
    seed = _seed_from_manifest(saved)
    snapshot_every = _snapshot_interval_from_manifest(saved)
    requested_params = _apply_param_overrides(params, overrides, args.source_enabled)

    if args.n is not None and args.n != config.n:
        raise ValueError("--n cannot change when resuming")
    if args.particles is not None and args.particles != config.num_particles:
        raise ValueError("--particles cannot change when resuming")
    if args.seed is not None and args.seed != seed:
        raise ValueError("--seed cannot change when resuming")
    if args.snapshot_every is not None and args.snapshot_every != snapshot_every:
        raise ValueError("--snapshot-every cannot change when resuming")
    if requested_params != params:
        raise ValueError("model parameters cannot change when resuming")
    return RunSpec(
        config=config,
        params=params,
        seed=seed,
        target_sweep=args.steps,
        chunk_size=args.chunk_size,
        snapshot_every=snapshot_every,
        paths=paths,
        resumed=True,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(runtime: Runtime, path: Path, state: Any, base_key: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    runtime.save_checkpoint(temporary, state, base_key)
    os.replace(temporary, path)


def _snapshot_path(paths: RunPaths, sweep: int) -> Path:
    return paths.snapshots / f"sweep_{sweep:010d}.h5"


def _metric_length(
    path: Path,
    *,
    model_version: int | None = None,
    schema_version: int | None = None,
) -> int | None:
    """Return a consistent metric length and optionally validate its schema."""

    if not path.exists():
        return None
    import h5py

    with h5py.File(path, "r") as file:
        if model_version is not None and file.attrs.get("model_version") != model_version:
            raise ValueError("metrics file belongs to a different model version")
        if schema_version is not None and file.attrs.get("schema_version") != schema_version:
            raise ValueError("unsupported metrics schema")
        lengths = {dataset.shape[0] for dataset in file.values()}
    if not lengths:
        return 0
    if len(lengths) != 1:
        raise ValueError("metrics datasets have inconsistent lengths")
    return lengths.pop()


def _reconcile_metrics(path: Path, completed_sweeps: int, runtime: Runtime) -> None:
    """Validate metrics and repair only the safe crash window."""

    length = _metric_length(
        path,
        model_version=runtime.model_version,
        schema_version=runtime.schema_version,
    )
    if length is None:
        if completed_sweeps:
            raise ValueError("resume metrics are missing before a nonzero checkpoint")
        return
    if length < completed_sweeps:
        raise ValueError("checkpoint is ahead of recorded metrics")
    if length == completed_sweeps:
        return

    import h5py

    with h5py.File(path, "a") as file:
        for dataset in file.values():
            dataset.resize((completed_sweeps,) + dataset.shape[1:])


def _batched_metric_rows(metrics: Any, expected: int) -> list[Any]:
    """Convert an arbitrary metrics NamedTuple batch to host rows generically."""

    metric_type = type(metrics)
    arrays = {name: np.asarray(value) for name, value in metrics._asdict().items()}
    lengths = {value.shape[0] for value in arrays.values()}
    if lengths != {expected}:
        raise RuntimeError(
            f"simulation returned metric batch lengths {sorted(lengths)}; expected {expected}"
        )
    return [metric_type(**{name: value[index] for name, value in arrays.items()}) for index in range(expected)]


def _validate_checkpoint_state(
    state: Any,
    params: Params,
    config: StaticConfig,
    target_sweep: int,
    runtime: Runtime,
) -> int:
    """Validate checkpoint representation and physical invariants on the host."""

    expected = {
        "R": ((config.num_particles, 2), np.dtype(np.int32)),
        "E": ((config.num_particles,), np.dtype(np.float32)),
        "L": ((config.n, config.n), np.dtype(np.int32)),
        "bonds": ((config.num_particles,), np.dtype(np.uint8)),
        "root": ((config.num_particles,), np.dtype(np.int32)),
        "sweep": ((), np.dtype(np.uint32)),
    }
    for name, (shape, dtype) in expected.items():
        if not hasattr(state, name):
            raise ValueError(f"checkpoint state is missing field {name!r}")
        value = np.asarray(getattr(state, name))
        if value.shape != shape:
            raise ValueError(
                f"checkpoint field {name!r} has shape {value.shape}; expected {shape}"
            )
        if value.dtype != dtype:
            raise ValueError(
                f"checkpoint field {name!r} has dtype {value.dtype}; expected {dtype}"
            )

    sweep = int(np.asarray(state.sweep))
    if not 0 <= sweep <= target_sweep:
        raise ValueError(
            f"checkpoint sweep {sweep} lies outside the requested range [0, {target_sweep}]"
        )

    from oe_tc.validation import validate_state

    try:
        result = runtime.jax.device_get(validate_state(state, params, config))
    except Exception as error:
        raise ValueError(f"checkpoint state invariant validation failed: {error}") from error
    failures = [
        name
        for name in ("positions_and_lattice", "energy_floor", "bonds", "components")
        if not bool(np.asarray(getattr(result, name)))
    ]
    if failures:
        raise ValueError(f"checkpoint state invariants failed: {', '.join(failures)}")
    return sweep


def _ensure_snapshot(
    spec: RunSpec,
    runtime: Runtime,
    state: Any,
    base_key: Any,
    sweep: int,
) -> None:
    """Create one immutable snapshot or validate an existing identical file."""

    path = _snapshot_path(spec.paths, sweep)
    if not path.exists():
        _atomic_checkpoint(runtime, path, state, base_key)
        return

    try:
        saved_state, saved_key = runtime.load_checkpoint(path)
    except Exception as error:
        raise ValueError(f"unable to load existing snapshot {path}: {error}") from error
    saved_sweep = _validate_checkpoint_state(
        saved_state,
        spec.params,
        spec.config,
        sweep,
        runtime,
    )
    if saved_sweep != sweep:
        raise ValueError(f"existing snapshot {path} records sweep {saved_sweep}")
    mismatches = [
        name
        for name, value in state._asdict().items()
        if not np.array_equal(np.asarray(value), np.asarray(getattr(saved_state, name)))
    ]
    key_matches = np.array_equal(
        np.asarray(runtime.jax.random.key_data(base_key)),
        np.asarray(runtime.jax.random.key_data(saved_key)),
    )
    if mismatches or not key_matches:
        detail = ", ".join(mismatches) or "base_key"
        raise ValueError(f"existing snapshot {path} conflicts with checkpoint fields: {detail}")


def _compile_chunk(
    runtime: Runtime,
    params: Params,
    config: StaticConfig,
    num_steps: int,
) -> Callable[[Any, Any], tuple[Any, Any]]:
    """Compile one fixed scan length and donate the evolving state buffer."""

    def chunk(state: Any, base_key: Any) -> tuple[Any, Any]:
        return runtime.run_chunk(state, base_key, params, config, num_steps)

    return runtime.jax.jit(chunk, donate_argnums=(0,))


def _new_manifest(runtime: Runtime, spec: RunSpec, platform: str) -> dict[str, Any]:
    value = runtime.manifest(spec.config, spec.params, spec.seed)
    value["runtime"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "initial_target_sweep": spec.target_sweep,
        "initial_chunk_size": spec.chunk_size,
        "snapshot_every": spec.snapshot_every,
        "platform_request": platform,
    }
    return value


def _chunk_length(spec: RunSpec, completed: int) -> int:
    """Choose a fixed scan length without crossing a snapshot boundary."""

    count = min(spec.chunk_size, spec.target_sweep - completed)
    if spec.snapshot_every is not None:
        boundary = (completed // spec.snapshot_every + 1) * spec.snapshot_every
        count = min(count, boundary - completed)
    return count


def _guard_component_convergence(metrics: Any, config: StaticConfig) -> None:
    """Refuse to persist a chunk whose component solver may be truncated."""

    if hasattr(metrics, "components_converged"):
        converged = np.asarray(metrics.components_converged, dtype=bool)
        if np.any(~converged):
            raise RuntimeError(
                "connected-component solver did not converge; chunk was not persisted"
            )
    else:
        iterations = np.asarray(metrics.component_iterations)
        if np.any(iterations >= config.component_max_iters):
            maximum = int(np.max(iterations))
            raise RuntimeError(
                "connected-component solver reached its iteration cap "
                f"({maximum} >= {config.component_max_iters}); chunk was not persisted"
            )


def run(spec: RunSpec, runtime: Runtime, *, platform: str, quiet: bool) -> Any:
    """Execute or resume one run and return its final host-backed state."""

    jax = runtime.jax
    if spec.resumed:
        try:
            state, base_key = runtime.load_checkpoint(spec.paths.checkpoint)
        except Exception as error:
            raise ValueError(f"unable to load checkpoint: {error}") from error
        completed = _validate_checkpoint_state(
            state,
            spec.params,
            spec.config,
            spec.target_sweep,
            runtime,
        )
        _reconcile_metrics(spec.paths.metrics, completed, runtime)
        if spec.snapshot_every is not None and completed and completed % spec.snapshot_every == 0:
            _ensure_snapshot(spec, runtime, state, base_key, completed)
        state = jax.device_put(state)
        base_key = jax.device_put(base_key)
    else:
        spec.paths.directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(spec.paths.manifest, _new_manifest(runtime, spec, platform))
        seed_key = jax.random.key(spec.seed)
        init_key = jax.random.fold_in(seed_key, 0)
        base_key = jax.random.fold_in(seed_key, 1)
        state = runtime.initialize_state(init_key, spec.params, spec.config)
        host_state = jax.device_get(state)
        completed = int(np.asarray(host_state.sweep))
        _atomic_checkpoint(runtime, spec.paths.checkpoint, host_state, base_key)

    if not quiet:
        device = jax.tree.leaves(state)[0].device
        action = "Resuming" if spec.resumed else "Starting"
        print(
            f"{action} OE-TC model at sweep {completed}/{spec.target_sweep} "
            f"on {device}; output={spec.paths.directory}"
        )

    compiled: dict[int, Callable[[Any, Any], tuple[Any, Any]]] = {}
    while completed < spec.target_sweep:
        count = _chunk_length(spec, completed)
        chunk = compiled.get(count)
        if chunk is None:
            chunk = _compile_chunk(runtime, spec.params, spec.config, count)
            compiled[count] = chunk

        state, batched_metrics = chunk(state, base_key)
        host_state, host_metrics = jax.device_get((state, batched_metrics))
        next_sweep = int(np.asarray(host_state.sweep))
        if next_sweep != completed + count:
            raise RuntimeError(
                f"simulation advanced from {completed} to {next_sweep}; expected {completed + count}"
            )

        _guard_component_convergence(host_metrics, spec.config)
        rows = _batched_metric_rows(host_metrics, count)
        runtime.save_metrics(spec.paths.metrics, rows)
        _atomic_checkpoint(runtime, spec.paths.checkpoint, host_state, base_key)
        completed = next_sweep
        if spec.snapshot_every is not None and completed % spec.snapshot_every == 0:
            _ensure_snapshot(spec, runtime, host_state, base_key, completed)
        if not quiet:
            print(f"Completed sweep {completed}/{spec.target_sweep}")

    return jax.device_get(state)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Return a process status for embedding and tests."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_environment(args.platform, args.no_preallocate)
    try:
        runtime = _load_runtime()
        spec = resolve_run_spec(args, runtime)
        run(spec, runtime, platform=args.platform, quiet=args.quiet)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
