"""Frozen-geometry controls for conduction-assisted bath dissipation.

Positions and bonds are held fixed, irradiation and structural transitions are
disabled, and only conduction followed by direct bath exchange is applied.
This isolates whether spreading excess energy across particles increases the
loss enabled by one fixed-quantum bath proposal per particle and sweep.

JAX is imported only after command-line platform configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from oe_tc.backend import configure_environment


METRIC_NAMES = (
    "total_energy",
    "bath_heat",
    "conduction_throughput",
    "excess_participation",
    "energetic_particles",
    "exposed_bath_heat",
    "shielded_bath_heat",
    "maximum_energy",
)


@dataclass(frozen=True)
class ConductionSetting:
    name: str
    contact: float
    bond: float


@dataclass(frozen=True)
class BathSetting:
    name: str
    base: float
    exposure: float


@dataclass(frozen=True)
class Condition:
    distribution: str
    conduction: str
    bath: str


CONDUCTION_SETTINGS = (
    ConductionSetting("default", 0.02, 0.10),
    ConductionSetting("no_bond_advantage", 0.02, 0.02),
    ConductionSetting("off", 0.0, 0.0),
)
BATH_SETTINGS = (
    BathSetting("default", 0.10, 0.40),
    BathSetting("transport_limited", 0.01, 0.49),
    BathSetting("exposed_only", 0.0, 0.50),
)
DISTRIBUTION_NAMES = (
    "actual",
    "uniform",
    "concentrated_shielded",
    "concentrated_exposed",
    "canonical_background",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen-geometry thermal transport controls."
    )
    parser.add_argument("run_directory", type=Path, help="source simulation run")
    parser.add_argument(
        "--state-file",
        type=Path,
        help="checkpoint or snapshot (relative paths are resolved inside the run)",
    )
    parser.add_argument(
        "--quench-total",
        type=float,
        help="rescale the source energy distribution to this common total",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/thermal_quench"),
    )
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--platform", choices=("auto", "cpu", "gpu", "tpu"), default="auto"
    )
    parser.add_argument("--no-preallocate", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _bootstrap(arguments: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--platform", choices=("auto", "cpu", "gpu", "tpu"), default="auto"
    )
    parser.add_argument("--no-preallocate", action="store_true")
    args, _ = parser.parse_known_args(arguments)
    configure_environment(args.platform, args.no_preallocate)


def _preserve_total(values: np.ndarray, total: float, index: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    values[index] += np.float32(total - float(np.sum(values, dtype=np.float64)))
    return values


def canonical_bath_background(
    size: int,
    heat_capacity: float,
    bath_temperature: float,
    energy_floor: float,
    seed: int,
) -> np.ndarray:
    """Sample the canonical internal-energy law conditioned on the floor.

    With ``S(E) = C log(E/E0)`` and ``k_B = 1``, the canonical density is a
    Gamma distribution with shape ``C + 1`` and scale ``T_b``. Rejection at the
    energy floor matches the state-space restriction of direct bath exchange.
    """

    if size < 1 or heat_capacity <= 0.0 or bath_temperature <= 0.0:
        raise ValueError("invalid canonical background parameters")
    rng = np.random.default_rng(seed)
    values = rng.gamma(heat_capacity + 1.0, bath_temperature, size=size)
    invalid = values < energy_floor
    while np.any(invalid):
        values[invalid] = rng.gamma(
            heat_capacity + 1.0, bath_temperature, size=int(np.count_nonzero(invalid))
        )
        invalid = values < energy_floor
    return np.asarray(values, dtype=np.float32)


def initial_energy_distributions(
    energy: np.ndarray,
    exposure: np.ndarray,
    bath_reference: float,
    background: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Return equal-total actual, uniform, and single-hot-particle states."""

    energy = np.asarray(energy, dtype=np.float32)
    exposure = np.asarray(exposure, dtype=np.float32)
    if energy.ndim != 1 or exposure.shape != energy.shape or energy.size == 0:
        raise ValueError("energy and exposure must be nonempty matching vectors")
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(exposure)):
        raise ValueError("energy and exposure must be finite")
    if background is None:
        background = np.full(energy.shape, bath_reference, dtype=np.float32)
    else:
        background = np.asarray(background, dtype=np.float32)
    if background.shape != energy.shape or not np.all(np.isfinite(background)):
        raise ValueError("background must be a finite vector matching energy")

    total = float(np.sum(energy, dtype=np.float64))
    surplus = total - float(np.sum(background, dtype=np.float64))
    if surplus < 0.0:
        raise ValueError("concentrated quenches require energy above the background")

    shielded = int(np.argmin(exposure))
    exposed = int(np.argmax(exposure))
    uniform = _preserve_total(
        np.full(energy.shape, total / energy.size, dtype=np.float32), total
    )

    def concentrated(target: int) -> np.ndarray:
        values = background.copy()
        values[target] += np.float32(surplus)
        return _preserve_total(values, total, target)

    return (
        {
            "actual": _preserve_total(energy, total),
            "uniform": uniform,
            "concentrated_shielded": concentrated(shielded),
            "concentrated_exposed": concentrated(exposed),
            "canonical_background": background.copy(),
        },
        {"shielded": shielded, "exposed": exposed},
    )


def condition_grid() -> tuple[Condition, ...]:
    return tuple(
        Condition(distribution, conduction.name, bath.name)
        for distribution in DISTRIBUTION_NAMES
        for conduction in CONDUCTION_SETTINGS
        for bath in BATH_SETTINGS
    )


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def summarize_traces(
    traces: Mapping[str, np.ndarray],
    conditions: Sequence[Condition],
    replicates: int,
    initial_totals: Mapping[str, float],
    initial_participation: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Reduce traces and subtract each setting's matched thermal background."""

    total = np.asarray(traces["total_energy"], dtype=np.float64)
    heat = np.asarray(traces["bath_heat"], dtype=np.float64)
    early = min(256, total.shape[0])
    late = min(512, total.shape[0])
    lookup = {condition: index for index, condition in enumerate(conditions)}
    background_initial = initial_totals["canonical_background"]
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        start = condition_index * replicates
        indices = slice(start, start + replicates)
        background_index = lookup[
            Condition("canonical_background", condition.conduction, condition.bath)
        ]
        background_start = background_index * replicates
        background_indices = slice(background_start, background_start + replicates)
        condition_total = total[:, indices]
        background_total = total[:, background_indices]
        condition_heat = heat[:, indices]
        participation = np.asarray(
            traces["excess_participation"][:, indices], dtype=np.float64
        )
        initial_excess = initial_totals[condition.distribution] - background_initial
        matched_excess = condition_total - background_total
        mean_excess = np.mean(matched_excess, axis=1)
        reached = (
            np.flatnonzero(mean_excess <= 0.5 * initial_excess)
            if initial_excess > 0.0
            else np.asarray((), dtype=np.int64)
        )
        final_excess = float(np.mean(matched_excess[-1]))
        dissipated_excess = initial_excess - final_excess
        rows.append(
            {
                **asdict(condition),
                "replicates": replicates,
                "initial_total_energy": initial_totals[condition.distribution],
                "final_total_energy_mean": float(np.mean(condition_total[-1])),
                "final_total_energy_std": float(np.std(condition_total[-1])),
                "matched_background_final_energy": float(
                    np.mean(background_total[-1])
                ),
                "initial_matched_excess": initial_excess,
                "final_matched_excess": final_excess,
                "dissipated_matched_excess": dissipated_excess,
                "dissipated_matched_excess_fraction": (
                    dissipated_excess / initial_excess
                    if initial_excess > 0.0
                    else None
                ),
                "matched_excess_half_life_sweeps": (
                    int(reached[0] + 1) if reached.size else None
                ),
                "cumulative_bath_heat_mean": float(
                    np.mean(np.sum(condition_heat, axis=0))
                ),
                "cumulative_bath_heat_std": float(
                    np.std(np.sum(condition_heat, axis=0))
                ),
                "early_dissipation_per_sweep": float(
                    -np.mean(condition_heat[:early])
                ),
                "late_dissipation_per_sweep": float(
                    -np.mean(condition_heat[-late:])
                ),
                "late_exposed_bath_heat_per_sweep": float(
                    np.mean(traces["exposed_bath_heat"][-late:, indices])
                ),
                "late_shielded_bath_heat_per_sweep": float(
                    np.mean(traces["shielded_bath_heat"][-late:, indices])
                ),
                "conduction_throughput_per_sweep": float(
                    np.mean(traces["conduction_throughput"][:, indices])
                ),
                "initial_excess_participation": float(
                    initial_participation[condition.distribution]
                ),
                "final_excess_participation": float(np.mean(participation[-1])),
                "final_energetic_particles": float(
                    np.mean(traces["energetic_particles"][-1, indices])
                ),
                "cumulative_exposed_bath_heat": float(
                    np.mean(
                        np.sum(traces["exposed_bath_heat"][:, indices], axis=0)
                    )
                ),
                "cumulative_shielded_bath_heat": float(
                    np.mean(
                        np.sum(traces["shielded_bath_heat"][:, indices], axis=0)
                    )
                ),
                "participation_dissipation_correlation": _pearson(
                    participation, -condition_heat
                ),
            }
        )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_plot(
    path: Path,
    traces: Mapping[str, np.ndarray],
    conditions: Sequence[Condition],
    replicates: int,
    initial_totals: Mapping[str, float],
    initial_participation: Mapping[str, float],
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path(tempfile.gettempdir()) / "oe-tc-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lookup = {condition: index for index, condition in enumerate(conditions)}
    sweeps = np.arange(traces["total_energy"].shape[0] + 1)

    def curve(metric: str, condition: Condition, initial: float) -> np.ndarray:
        start = lookup[condition] * replicates
        values = np.mean(
            traces[metric][:, start : start + replicates], axis=1, dtype=np.float64
        )
        return np.concatenate(([initial], values))

    def matched_curve(condition: Condition) -> np.ndarray:
        background = Condition(
            "canonical_background", condition.conduction, condition.bath
        )
        return curve(
            "total_energy", condition, initial_totals[condition.distribution]
        ) - curve(
            "total_energy", background, initial_totals["canonical_background"]
        )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for distribution in DISTRIBUTION_NAMES[:-1]:
        condition = Condition(distribution, "default", "default")
        axes[0, 0].plot(
            sweeps, matched_curve(condition), label=distribution
        )
        axes[1, 0].plot(
            sweeps,
            curve(
                "excess_participation",
                condition,
                initial_participation[distribution],
            ),
            label=distribution,
        )
    for conduction in (setting.name for setting in CONDUCTION_SETTINGS):
        condition = Condition("concentrated_shielded", conduction, "default")
        axes[0, 1].plot(
            sweeps, matched_curve(condition), label=conduction
        )
    for bath in (setting.name for setting in BATH_SETTINGS):
        condition = Condition("concentrated_shielded", "default", bath)
        axes[1, 1].plot(
            sweeps, matched_curve(condition), label=bath
        )
    titles = (
        "Matched excess by initial distribution",
        "Matched excess: shielded hot spot by conduction",
        "Effective excess-energy carriers",
        "Matched excess: shielded hot spot by bath",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("thermal sweep")
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    axes[0, 0].set_ylabel("energy above matched background")
    axes[0, 1].set_ylabel("energy above matched background")
    axes[1, 0].set_ylabel("excess participation")
    axes[1, 1].set_ylabel("energy above matched background")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_experiment(
    run_directory: Path,
    output: Path,
    *,
    steps: int,
    replicates: int,
    chunk_size: int,
    seed: int,
    make_plot: bool,
    state_file: Path | None = None,
    quench_total: float | None = None,
) -> dict[str, Any]:
    if steps < 1 or replicates < 1 or chunk_size < 1:
        raise ValueError("steps, replicates, and chunk-size must be positive")
    if not 0 <= seed < 2**32 or seed + replicates > 2**32:
        raise ValueError("replicate seeds must fit unsigned 32-bit integers")

    import jax
    import jax.numpy as jnp

    from analyze import _load_state, _read_manifest
    from oe_tc.bath import direct_bath_exchange
    from oe_tc.conduction import randomized_conduction_sweep
    from oe_tc.geometry import exposure_fraction
    from oe_tc.random import Phase, phase_key

    manifest, config, params = _read_manifest(run_directory / "manifest.json")
    state_path = Path("checkpoint.h5") if state_file is None else state_file
    if not state_path.is_absolute():
        state_path = run_directory / state_path
    state = _load_state(state_path, config)
    positions = jax.device_put(jnp.asarray(state.R))
    lattice = jax.device_put(jnp.asarray(state.L))
    bonds = jax.device_put(jnp.asarray(state.bonds))
    exposure = exposure_fraction(positions, lattice, empty=config.num_particles)
    exposure_host = np.asarray(jax.device_get(exposure), dtype=np.float32)

    bath_mode = float(params.heat_capacity * params.bath_temperature)
    canonical_mean = float((params.heat_capacity + 1.0) * params.bath_temperature)
    background = canonical_bath_background(
        config.num_particles,
        params.heat_capacity,
        params.bath_temperature,
        params.energy_floor,
        seed ^ 0xB47BA7A,
    )
    source_energy = np.asarray(state.E, dtype=np.float32)
    source_total = float(np.sum(source_energy, dtype=np.float64))
    if quench_total is not None:
        if not np.isfinite(quench_total) or quench_total <= 0.0:
            raise ValueError("quench-total must be a finite positive number")
        source_energy = np.asarray(
            source_energy * (quench_total / source_total), dtype=np.float32
        )
        source_energy = _preserve_total(source_energy, quench_total)
    if np.any(source_energy < params.energy_floor):
        raise ValueError("quench energy falls below the model energy floor")
    distributions, targets = initial_energy_distributions(
        source_energy, exposure_host, canonical_mean, background
    )
    initial_total = float(np.sum(source_energy, dtype=np.float64))
    initial_totals = {
        name: float(np.sum(values, dtype=np.float64))
        for name, values in distributions.items()
    }
    conditions = condition_grid()
    conduction_by_name = {setting.name: setting for setting in CONDUCTION_SETTINGS}
    bath_by_name = {setting.name: setting for setting in BATH_SETTINGS}
    initial_participation = {}
    for name, values in distributions.items():
        excess = np.maximum(values.astype(np.float64) - canonical_mean, 0.0)
        denominator = float(np.sum(np.square(excess)))
        initial_participation[name] = (
            float(np.sum(excess) ** 2 / denominator) if denominator else 0.0
        )

    run_conditions = [condition for condition in conditions for _ in range(replicates)]
    energy = jax.device_put(
        jnp.asarray(
            np.stack([distributions[c.distribution] for c in run_conditions])
        )
    )
    contact = jax.device_put(
        jnp.asarray(
            [conduction_by_name[c.conduction].contact for c in run_conditions],
            dtype=jnp.float32,
        )
    )
    bond = jax.device_put(
        jnp.asarray(
            [conduction_by_name[c.conduction].bond for c in run_conditions],
            dtype=jnp.float32,
        )
    )
    kappa_base = jax.device_put(
        jnp.asarray(
            [bath_by_name[c.bath].base for c in run_conditions], dtype=jnp.float32
        )
    )
    kappa_exposure = jax.device_put(
        jnp.asarray(
            [bath_by_name[c.bath].exposure for c in run_conditions],
            dtype=jnp.float32,
        )
    )
    replicate_seeds = np.asarray(
        [seed + r for _ in conditions for r in range(replicates)], dtype=np.uint32
    )
    base_keys = jax.vmap(jax.random.key)(jnp.asarray(replicate_seeds))
    bath_reference_array = jnp.asarray(canonical_mean, dtype=energy.dtype)
    hot_threshold = bath_reference_array + 0.5 * params.bath_energy_quantum

    def thermal_step(current_energy, current_sweep):
        conduction_keys = jax.vmap(
            lambda key: phase_key(key, current_sweep, Phase.CONDUCTION_ORDER)
        )(base_keys)
        bath_keys = jax.vmap(
            lambda key: phase_key(key, current_sweep, Phase.BATH)
        )(base_keys)

        def one_run(particle_energy, conduction_key, bath_key, c0, c1, k0, k1):
            conducted = randomized_conduction_sweep(
                conduction_key,
                particle_energy,
                positions,
                lattice,
                bonds,
                c0,
                c1,
            )
            exchanged = direct_bath_exchange(
                bath_key,
                conducted.energy,
                positions,
                lattice,
                params.bath_energy_quantum,
                params.energy_floor,
                params.heat_capacity,
                params.bath_temperature,
                k0,
                k1,
            )
            excess = jnp.maximum(exchanged.energy - bath_reference_array, 0.0)
            excess_sum = jnp.sum(excess)
            participation = jnp.where(
                excess_sum > 0.0,
                jnp.square(excess_sum)
                / jnp.maximum(
                    jnp.sum(jnp.square(excess)), jnp.finfo(excess.dtype).tiny
                ),
                0.0,
            )
            metrics = jnp.stack(
                (
                    jnp.sum(exchanged.energy),
                    exchanged.heat,
                    jnp.sum(jnp.abs(conducted.edge_flux)),
                    participation,
                    jnp.sum(exchanged.energy > hot_threshold),
                    jnp.sum(
                        jnp.where(
                            exchanged.exposure > 0.0, exchanged.particle_heat, 0.0
                        )
                    ),
                    jnp.sum(
                        jnp.where(
                            exchanged.exposure == 0.0,
                            exchanged.particle_heat,
                            0.0,
                        )
                    ),
                    jnp.max(exchanged.energy),
                )
            )
            return exchanged.energy, metrics

        return jax.vmap(one_run)(
            current_energy,
            conduction_keys,
            bath_keys,
            contact,
            bond,
            kappa_base,
            kappa_exposure,
        )

    compiled_chunks: dict[int, Any] = {}

    def run_chunk(current_energy, start_sweep: int, count: int):
        if count not in compiled_chunks:

            def chunk_kernel(chunk_energy, chunk_start):
                def body(carry, _):
                    next_energy, metrics = thermal_step(carry[0], carry[1])
                    return (next_energy, carry[1] + 1), metrics

                return jax.lax.scan(
                    body,
                    (chunk_energy, jnp.asarray(chunk_start, dtype=jnp.uint32)),
                    xs=None,
                    length=count,
                )

            compiled_chunks[count] = jax.jit(chunk_kernel)
        (next_energy, _), metrics = compiled_chunks[count](current_energy, start_sweep)
        return next_energy, metrics

    started = time.perf_counter()
    chunks = []
    completed = 0
    while completed < steps:
        count = min(chunk_size, steps - completed)
        energy, metrics = run_chunk(energy, completed, count)
        chunks.append(np.asarray(jax.device_get(metrics), dtype=np.float32))
        completed += count
        print(f"Completed thermal sweep {completed}/{steps}", flush=True)

    elapsed = time.perf_counter() - started
    metric_array = np.concatenate(chunks, axis=0)
    traces = {
        name: metric_array[:, :, index]
        for index, name in enumerate(METRIC_NAMES)
    }
    equilibrium_total = float(np.sum(background, dtype=np.float64))
    bond_masks = np.asarray(state.bonds, dtype=np.uint8)
    bit_counts = np.asarray([value.bit_count() for value in range(16)], dtype=np.int32)
    bond_count = int(np.sum(bit_counts[bond_masks & 0x0F]) // 2)
    component_sizes = np.bincount(
        np.asarray(state.root, dtype=np.int32), minlength=config.num_particles
    )
    report = {
        "experiment": "frozen_geometry_thermal_quench",
        "source_run": str(run_directory.resolve()),
        "source_sweep": int(np.asarray(state.sweep)),
        "source_state_file": str(state_path.resolve()),
        "source_state_total_energy": source_total,
        "source_manifest": manifest,
        "steps": steps,
        "replicates": replicates,
        "seed": seed,
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "elapsed_seconds": elapsed,
        "thermal_particle_sweeps_per_second": (
            steps * len(run_conditions) * config.num_particles / elapsed
        ),
        "geometry": {
            "particles": config.num_particles,
            "bonds": bond_count,
            "molecules": int(np.count_nonzero(component_sizes)),
            "largest_molecule": int(np.max(component_sizes)),
            "exposed_particles": int(np.count_nonzero(exposure_host > 0.0)),
            "fully_shielded_particles": int(np.count_nonzero(exposure_host == 0.0)),
            "mean_exposure": float(np.mean(exposure_host)),
        },
        "initial_state": {
            "total_energy": initial_total,
            "bath_mode_per_particle": bath_mode,
            "canonical_mean_per_particle": canonical_mean,
            "canonical_background_total": equilibrium_total,
            "target_particles": targets,
            "target_exposure": {
                name: float(exposure_host[index]) for name, index in targets.items()
            },
            "distribution_totals": initial_totals,
            "excess_participation": initial_participation,
        },
        "conduction_settings": [asdict(setting) for setting in CONDUCTION_SETTINGS],
        "bath_settings": [asdict(setting) for setting in BATH_SETTINGS],
        "conditions": summarize_traces(
            traces,
            conditions,
            replicates,
            initial_totals,
            initial_participation,
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", report)
    np.savez_compressed(
        output / "traces.npz",
        **traces,
        condition_distribution=np.asarray([c.distribution for c in run_conditions]),
        condition_conduction=np.asarray([c.conduction for c in run_conditions]),
        condition_bath=np.asarray([c.bath for c in run_conditions]),
        replicate=np.asarray(
            [r for _ in conditions for r in range(replicates)], dtype=np.int32
        ),
        final_energy=np.asarray(jax.device_get(energy), dtype=np.float32),
    )
    if make_plot:
        _write_plot(
            output / "thermal_quench.png",
            traces,
            conditions,
            replicates,
            initial_totals,
            initial_participation,
        )
    return report


def _find_condition(
    report: Mapping[str, Any], distribution: str, conduction: str, bath: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in report["conditions"]
        if row["distribution"] == distribution
        and row["conduction"] == conduction
        and row["bath"] == bath
    )


def concise_report(report: Mapping[str, Any]) -> str:
    lines = [
        (
            f"Completed {report['steps']} sweeps x "
            f"{len(report['conditions'])} conditions x {report['replicates']} "
            f"replicates on {report['device']} in {report['elapsed_seconds']:.2f}s."
        ),
        "Mean cumulative direct-bath heat (negative means dissipation):",
    ]
    for distribution in DISTRIBUTION_NAMES:
        row = _find_condition(report, distribution, "default", "default")
        lines.append(
            f"  {distribution:24s} {row['cumulative_bath_heat_mean']:10.3f}"
        )
    lines.append("Shielded hot spot by conduction setting (default bath):")
    for setting in CONDUCTION_SETTINGS:
        row = _find_condition(
            report, "concentrated_shielded", setting.name, "default"
        )
        lines.append(
            f"  {setting.name:24s} {row['cumulative_bath_heat_mean']:10.3f} "
            f"N_eff={row['final_excess_participation']:.2f}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _bootstrap(arguments)
    args = build_parser().parse_args(arguments)
    try:
        report = run_experiment(
            args.run_directory,
            args.output,
            steps=args.steps,
            replicates=args.replicates,
            chunk_size=args.chunk_size,
            seed=args.seed,
            make_plot=not args.no_plot,
            state_file=args.state_file,
            quench_total=args.quench_total,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(concise_report(report), end="")
    print(f"Wrote {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
