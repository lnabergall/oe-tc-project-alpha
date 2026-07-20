"""Deterministic host-side analysis of a completed or in-progress simulation run.

The analyzer consumes immutable snapshots in sweep order and then the latest
checkpoint.  It performs no simulation work and never modifies the run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from oe_tc.config import Params, StaticConfig, validate_params
from oe_tc.evaluation import NoveltyRecord, NoveltyTracker, StateSummary, summarize_state
from oe_tc.state import State
from oe_tc.storage import MODEL_VERSION, SCHEMA_VERSION, load_checkpoint


MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "checkpoint.h5"
SNAPSHOT_DIRECTORY = "snapshots"
_SNAPSHOT_PATTERN = re.compile(r"sweep_(\d+)\.h5\Z")


def build_parser() -> argparse.ArgumentParser:
    """Build the analysis command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarize molecular structure, energy gradients, novelty, and "
            "turnover across a simulation run."
        )
    )
    parser.add_argument("run_directory", type=Path, help="simulation run directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the exact JSON report to this file",
    )
    return parser


def _read_manifest(path: Path) -> tuple[dict[str, Any], StaticConfig, Params]:
    """Load and strictly validate the immutable run manifest."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"run manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("run manifest must contain one object")

    model_version = value.get("model_version")
    schema_version = value.get("schema_version")
    if type(model_version) is not int or model_version != MODEL_VERSION:
        raise ValueError("manifest belongs to a different model version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")

    static = value.get("static")
    expected_static = {field.name for field in fields(StaticConfig)}
    if not isinstance(static, dict) or set(static) != expected_static:
        raise ValueError("manifest contains an invalid static configuration")
    if any(type(static[name]) is not int for name in expected_static):
        raise ValueError("manifest static configuration values must be integers")
    try:
        config = StaticConfig(**static)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest contains an invalid static configuration") from error

    parameters = value.get("params")
    if not isinstance(parameters, dict) or set(parameters) != set(Params._fields):
        raise ValueError("manifest contains invalid model parameters")
    try:
        params = Params(**parameters)
        validate_params(params)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest contains invalid model parameters") from error

    seed = value.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("manifest contains an invalid unsigned 32-bit seed")
    return value, config, params


def _load_state(path: Path, config: StaticConfig) -> State:
    """Load one versioned checkpoint and validate its fixed host representation."""

    try:
        state, _ = load_checkpoint(path)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"unable to load state file {path}: {error}") from error

    expected = {
        "R": ((config.num_particles, 2), np.dtype(np.int32)),
        "E": ((config.num_particles,), np.dtype(np.float32)),
        "L": ((config.n, config.n), np.dtype(np.int32)),
        "bonds": ((config.num_particles,), np.dtype(np.uint8)),
        "root": ((config.num_particles,), np.dtype(np.int32)),
        "sweep": ((), np.dtype(np.uint32)),
    }
    for name, (shape, dtype) in expected.items():
        array = np.asarray(getattr(state, name))
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"state file {path} has invalid {name!r} shape or dtype"
            )
    return state


def _states_equal(first: State, second: State) -> bool:
    return all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(first, second)
    )


def _summary_json(summary: StateSummary) -> dict[str, Any]:
    """Serialize compact summary values without duplicating molecular graphs."""

    return {
        "molecule_count": summary.molecule_count,
        "molecule_type_count": summary.molecule_type_count,
        "molecule_size_mean": summary.molecule_size_mean,
        "molecule_size_max": summary.molecule_size_max,
        "molecule_size_entropy": summary.molecule_size_entropy,
        "type_shannon_entropy": summary.type_shannon_entropy,
        "effective_type_diversity": summary.effective_type_diversity,
        "internal_energy_mean": summary.internal_energy_mean,
        "internal_energy_std": summary.internal_energy_std,
        "vertical_energy_gradient": summary.vertical_energy_gradient,
        "top_bottom_energy_difference": summary.top_bottom_energy_difference,
        "molecule_size_counts": [
            {"size": size, "count": count} for size, count in summary.size_counts
        ],
    }


def _novelty_json(record: NoveltyRecord) -> dict[str, Any]:
    return {
        "new_type_count": record.new_type_count,
        "cumulative_type_count": record.cumulative_type_count,
        "type_turnover": record.type_turnover,
        "abundance_turnover": record.abundance_turnover,
    }


def _observation(
    state: State,
    source: str,
    config: StaticConfig,
    tracker: NoveltyTracker,
) -> dict[str, Any]:
    summary = summarize_state(state, n=config.n)
    # Reuse the signatures already computed by summarize_state instead of
    # canonicalizing every molecule a second time for novelty tracking.
    novelty = tracker.observe(
        signature
        for signature, count in summary.type_counts
        for _ in range(count)
    )
    return {
        "sweep": int(np.asarray(state.sweep)),
        "source": source,
        "summary": _summary_json(summary),
        "novelty": _novelty_json(novelty),
    }


def analyze_run(run_directory: str | Path) -> dict[str, Any]:
    """Analyze snapshots and the final checkpoint of one simulation run."""

    directory = Path(run_directory)
    if not directory.is_dir():
        raise ValueError(f"run directory not found: {directory}")
    manifest_value, config, params = _read_manifest(directory / MANIFEST_NAME)

    snapshot_directory = directory / SNAPSHOT_DIRECTORY
    snapshot_paths = sorted(snapshot_directory.glob("sweep_*.h5"))
    tracker = NoveltyTracker()
    observations: list[dict[str, Any]] = []
    last_state: State | None = None
    last_sweep = -1

    for path in snapshot_paths:
        match = _SNAPSHOT_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid snapshot filename: {path.name}")
        filename_sweep = int(match.group(1))
        state = _load_state(path, config)
        sweep = int(np.asarray(state.sweep))
        if sweep != filename_sweep:
            raise ValueError(
                f"snapshot {path.name} records sweep {sweep}, not {filename_sweep}"
            )
        if sweep <= last_sweep:
            raise ValueError("snapshot sweeps must be unique and strictly increasing")
        observations.append(
            _observation(
                state,
                path.relative_to(directory).as_posix(),
                config,
                tracker,
            )
        )
        last_state = state
        last_sweep = sweep

    checkpoint_path = directory / CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        raise ValueError(f"final checkpoint not found: {checkpoint_path}")
    checkpoint = _load_state(checkpoint_path, config)
    checkpoint_sweep = int(np.asarray(checkpoint.sweep))
    if checkpoint_sweep < last_sweep:
        raise ValueError("final checkpoint precedes the latest immutable snapshot")
    if checkpoint_sweep == last_sweep:
        if last_state is None or not _states_equal(last_state, checkpoint):
            raise ValueError(
                "final checkpoint disagrees with the immutable snapshot at the same sweep"
            )
    else:
        observations.append(
            _observation(checkpoint, CHECKPOINT_NAME, config, tracker)
        )

    return {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "seed": manifest_value["seed"],
            "static": asdict(config),
            "params": dict(params._asdict()),
        },
        "observation_count": len(observations),
        "first_sweep": observations[0]["sweep"] if observations else None,
        "last_sweep": observations[-1]["sweep"] if observations else None,
        "cumulative_molecule_type_count": tracker.cumulative_type_count,
        "observations": observations,
    }


def report_json(report: dict[str, Any]) -> str:
    """Return the canonical human-readable representation of one report."""

    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        text = report_json(analyze_run(args.run_directory))
        if args.output is not None:
            _atomic_write(args.output, text)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
