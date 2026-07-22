"""Lean end-to-end throughput benchmark for OE-TC model.

This harness times the same fixed-length, donated chunk used by the production
runner.  Compilation and the first execution are reported separately from
steady-state execution; no host callbacks or profiler instrumentation are
inserted into the compiled simulation loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from oe_tc.backend import configure_environment


DEFAULT_N = 16
DEFAULT_DENSITY = 0.25
DEFAULT_CHUNK_SIZE = 4
DEFAULT_WARMUPS = 1
DEFAULT_REPEATS = 5


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing JAX or initializing a backend."""

    parser = argparse.ArgumentParser(
        description="Benchmark compiled OE-TC chunks and emit JSON."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"even lattice width (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_DENSITY,
        help=f"initial occupied-site fraction (default: {DEFAULT_DENSITY})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"sweeps per compiled invocation (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=DEFAULT_WARMUPS,
        help=f"extra untimed chunks after first execution (default: {DEFAULT_WARMUPS})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"number of timed steady-state chunks (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument("--seed", type=int, default=0, help="unsigned 32-bit seed")
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
        help="JAX platform selected before importing JAX",
    )
    parser.add_argument(
        "--no-preallocate",
        action="store_true",
        help="disable XLA GPU memory preallocation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; JSON is always printed to stdout",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.n < 4 or args.n % 2:
        raise ValueError("n must be an even integer of at least four")
    if not 0.0 < args.density <= 1.0:
        raise ValueError("density must lie in (0, 1]")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if args.warmups < 0:
        raise ValueError("warmups cannot be negative")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("seed must fit in an unsigned 32-bit integer")


def _block_until_ready(jax: Any, value: Any) -> Any:
    """Synchronize every array leaf without copying results to the host."""

    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


def _tree_nbytes(jax: Any, value: Any) -> int:
    """Estimate resident bytes from the logical array leaves in a pytree."""

    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return total


def _device_report(jax: Any, state: Any) -> dict[str, Any]:
    leaves = jax.tree_util.tree_leaves(state)
    device = getattr(leaves[0], "device", None)
    if callable(device):
        device = device()
    return {
        "backend": jax.default_backend(),
        "platform": getattr(device, "platform", None),
        "device_kind": getattr(device, "device_kind", None),
        "device_id": getattr(device, "id", None),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_report(seconds: Sequence[float], sweeps_per_chunk: int) -> dict[str, Any]:
    rates = [sweeps_per_chunk / duration for duration in seconds]
    return {
        "chunk_seconds": list(seconds),
        "sweeps_per_second": rates,
        "mean_sweeps_per_second": statistics.mean(rates),
        "median_sweeps_per_second": statistics.median(rates),
        "p05_sweeps_per_second": _percentile(rates, 0.05),
        "p95_sweeps_per_second": _percentile(rates, 0.95),
        "min_sweeps_per_second": min(rates),
        "max_sweeps_per_second": max(rates),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Compile, execute, synchronize, and summarize one benchmark case."""

    _validate_args(args)
    configure_environment(args.platform, args.no_preallocate)

    import jax

    from oe_tc.config import StaticConfig, default_params
    from oe_tc.initialization import initialize_state
    from oe_tc.simulation import run_chunk

    num_particles = max(1, min(args.n * args.n, round(args.density * args.n * args.n)))
    config = StaticConfig(n=args.n, num_particles=num_particles)
    params = default_params()
    seed_key = jax.random.key(args.seed)
    init_key = jax.random.fold_in(seed_key, 0)
    base_key = jax.random.fold_in(seed_key, 1)

    initialize_started = time.perf_counter()
    state = initialize_state(init_key, params, config)
    _block_until_ready(jax, state)
    initialize_seconds = time.perf_counter() - initialize_started
    state_bytes = _tree_nbytes(jax, state)
    device = _device_report(jax, state)

    def chunk(current_state: Any, key: Any) -> tuple[Any, Any]:
        return run_chunk(current_state, key, params, config, args.chunk_size)

    staged_chunk = jax.jit(chunk, donate_argnums=(0,))

    lower_started = time.perf_counter()
    lowered = staged_chunk.lower(state, base_key)
    lowering_seconds = time.perf_counter() - lower_started

    compile_started = time.perf_counter()
    compiled_chunk = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started

    first_started = time.perf_counter()
    state, metrics = compiled_chunk(state, base_key)
    _block_until_ready(jax, (state, metrics))
    first_execution_seconds = time.perf_counter() - first_started

    for _ in range(args.warmups):
        state, metrics = compiled_chunk(state, base_key)
        _block_until_ready(jax, (state, metrics))

    durations: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        state, metrics = compiled_chunk(state, base_key)
        _block_until_ready(jax, (state, metrics))
        durations.append(time.perf_counter() - started)

    completed_sweeps = int(jax.device_get(state.sweep))
    expected_sweeps = args.chunk_size * (1 + args.warmups + args.repeats)
    if completed_sweeps != expected_sweeps:
        raise RuntimeError(
            f"simulation advanced {completed_sweeps} sweeps; expected {expected_sweeps}"
        )

    return {
        "benchmark": "oe_tc_end_to_end_chunk",
        "jax_version": jax.__version__,
        "device": device,
        "case": {
            "n": args.n,
            "num_particles": num_particles,
            "requested_density": args.density,
            "actual_density": num_particles / (args.n * args.n),
            "chunk_size": args.chunk_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "memory": {
            "state_bytes_estimate": state_bytes,
            "state_bytes_per_particle": state_bytes / num_particles,
            "state_bytes_per_lattice_site": state_bytes / (args.n * args.n),
        },
        "startup": {
            "initialize_seconds": initialize_seconds,
            "lowering_seconds": lowering_seconds,
            "compile_seconds": compile_seconds,
            "first_execution_seconds": first_execution_seconds,
            "compile_and_first_run_seconds": (
                lowering_seconds + compile_seconds + first_execution_seconds
            ),
        },
        "steady_state": _timing_report(durations, args.chunk_size),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args)
    except ValueError as error:
        parser.error(str(error))

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
