"""Render state frames, movies, and diagnostics from a simulation run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render simulation snapshots with a direct NumPy/Pillow raster path "
            "and stream movies to FFmpeg."
        )
    )
    parser.add_argument("run_directory", type=Path, help="simulation run directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: RUN_DIRECTORY/visualization)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="maximum lattice image width in pixels (default: 1024)",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0, help="movie frame rate (default: 5)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="render every Kth eligible snapshot while retaining the last",
    )
    parser.add_argument("--start-sweep", type=int, help="first included sweep")
    parser.add_argument("--end-sweep", type=int, help="last included sweep")
    parser.add_argument(
        "--energy-percentile",
        type=float,
        default=99.0,
        help="robust global energy span percentile (default: 99)",
    )
    parser.add_argument(
        "--energy-span",
        type=float,
        help="fixed half-width around bath-equilibrium energy",
    )
    parser.add_argument("--no-bonds", action="store_true", help="hide bond lines")
    parser.add_argument("--no-movie", action="store_true", help="skip MP4 encoding")
    parser.add_argument(
        "--no-metrics", action="store_true", help="skip the metrics dashboard"
    )
    parser.add_argument(
        "--max-metric-points",
        type=int,
        default=10_000,
        help="maximum plotted samples per metric (default: 10000)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from oe_tc.visualization import visualize_run

        result = visualize_run(
            args.run_directory,
            output_directory=args.output,
            image_size=args.size,
            fps=args.fps,
            stride=args.stride,
            start_sweep=args.start_sweep,
            end_sweep=args.end_sweep,
            energy_percentile=args.energy_percentile,
            energy_span=args.energy_span,
            show_bonds=not args.no_bonds,
            movie=not args.no_movie,
            metrics=not args.no_metrics,
            max_metric_points=args.max_metric_points,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(result).items()
    }
    print(json.dumps(values, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())