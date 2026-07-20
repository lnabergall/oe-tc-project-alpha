"""Versioned host-side storage for simulation checkpoints and metrics."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import h5py
import jax
import numpy as np

from oe_tc.config import Params, StaticConfig
from oe_tc.state import State, StepMetrics
from oe_tc.version import MODEL_VERSION, SCHEMA_VERSION


def _json_value(value):
    if isinstance(value, (bool, float, int, str)):
        return value
    return np.asarray(value).item()


def manifest(config: StaticConfig, params: Params, seed: int) -> dict:
    """Return the complete JSON-serializable run manifest."""

    return {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "static": asdict(config),
        "params": {name: _json_value(value) for name, value in params._asdict().items()},
    }


def save_manifest(path: str | Path, config: StaticConfig, params: Params, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(config, params, seed), indent=2), encoding="utf-8")


def save_checkpoint(path: str | Path, state: State, base_key: jax.Array) -> None:
    """Save one restartable checkpoint after transferring arrays to the host."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = jax.device_get(state._asdict())
    key_data = jax.device_get(jax.random.key_data(base_key))

    with h5py.File(path, "w") as file:
        file.attrs["model_version"] = MODEL_VERSION
        file.attrs["schema_version"] = SCHEMA_VERSION
        for name, value in arrays.items():
            array = np.asarray(value)
            filters = {} if array.ndim == 0 else {"compression": "gzip", "shuffle": True}
            file.create_dataset(name, data=array, **filters)
        file.create_dataset("base_key", data=np.asarray(key_data))


def load_checkpoint(path: str | Path) -> tuple[State, jax.Array]:
    """Load a checkpoint as NumPy-backed state and a typed JAX PRNG key."""

    with h5py.File(path, "r") as file:
        if int(file.attrs["model_version"]) != MODEL_VERSION:
            raise ValueError("checkpoint belongs to a different model version")
        if int(file.attrs["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        missing = set(State._fields) - set(file)
        if missing:
            raise ValueError(f"checkpoint is missing state fields: {sorted(missing)}")
        state = State(**{name: file[name][()] for name in State._fields})
        try:
            base_key = jax.random.wrap_key_data(file["base_key"][()])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint contains an invalid base PRNG key") from error
    return state, base_key


def save_metrics(path: str | Path, metrics: Iterable[StepMetrics]) -> None:
    """Append one chunk of compact per-sweep metrics with schema checks."""

    rows = list(metrics)
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        name: np.asarray([np.asarray(row[i]) for row in rows])
        for i, name in enumerate(StepMetrics._fields)
    }

    with h5py.File(path, "a") as file:
        existing = bool(file.keys())
        if existing:
            if int(file.attrs.get("model_version", -1)) != MODEL_VERSION:
                raise ValueError("metrics belong to a different model version")
            if int(file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError("unsupported metrics schema")
            fields = set(file.keys())
            if fields != set(StepMetrics._fields):
                raise ValueError("metrics datasets do not match the current schema")
            lengths = {dataset.shape[0] for dataset in file.values()}
            if len(lengths) != 1:
                raise ValueError("metrics datasets have inconsistent lengths")

        file.attrs["model_version"] = MODEL_VERSION
        file.attrs["schema_version"] = SCHEMA_VERSION
        for name, chunk in values.items():
            if name not in file:
                file.create_dataset(
                    name,
                    data=chunk,
                    maxshape=(None,) + chunk.shape[1:],
                    chunks=True,
                    compression="gzip",
                    shuffle=True,
                )
                continue
            dataset = file[name]
            if dataset.shape[1:] != chunk.shape[1:]:
                raise ValueError(f"metric shape changed for {name}")
            start = dataset.shape[0]
            dataset.resize((start + chunk.shape[0],) + dataset.shape[1:])
            dataset[start:] = chunk
