from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

import visualize
from oe_tc.version import MODEL_VERSION, SCHEMA_VERSION
from oe_tc.visualization import (
    EnergyScale,
    HostFrame,
    _metric_smoothing_window,
    _smooth_metric,
    choose_energy_scale,
    discover_frames,
    energy_colors,
    load_frame,
    render_frame,
    visualize_run,
)


def _write_state(
    path: Path,
    sweep: int,
    *,
    positions: np.ndarray,
    energy: np.ndarray,
    bonds: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.attrs["model_version"] = MODEL_VERSION
        file.attrs["schema_version"] = SCHEMA_VERSION
        file.create_dataset("R", data=np.asarray(positions, dtype=np.int32))
        file.create_dataset("E", data=np.asarray(energy, dtype=np.float32))
        file.create_dataset("bonds", data=np.asarray(bonds, dtype=np.uint8))
        file.create_dataset("sweep", data=np.asarray(sweep, dtype=np.uint32))


def _write_metrics(path: Path, length: int = 4) -> None:
    fields = {
        "internal_energy",
        "configurational_energy",
        "source_energy",
        "bath_energy_direct",
        "bath_energy_structural",
        "conduction_energy_throughput",
        "num_molecules",
        "num_bonds",
        "accepted_molecule_moves",
        "accepted_bond_flips",
        "accepted_bath_exchanges",
        "accepted_conduction_exchanges",
        "molecule_conflicts",
        "molecule_unresolved",
        "mis_iterations",
        "component_iterations",
    }
    with h5py.File(path, "w") as file:
        file.attrs["model_version"] = MODEL_VERSION
        file.attrs["schema_version"] = SCHEMA_VERSION
        for offset, name in enumerate(sorted(fields)):
            file.create_dataset(name, data=np.arange(length, dtype=np.float32) + offset)


def _run(tmp_path: Path, *, snapshots: tuple[int, ...] = (1, 2, 3), final: int = 4) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    manifest = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "static": {"n": 8, "num_particles": 4},
        "params": {"heat_capacity": 10.0, "bath_temperature": 1.0},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    positions = np.asarray(((0, 0), (7, 0), (3, 3), (3, 4)), dtype=np.int32)
    bonds = np.asarray((2, 1, 4, 8), dtype=np.uint8)
    for sweep in snapshots:
        _write_state(
            run / "snapshots" / f"sweep_{sweep:010d}.h5",
            sweep,
            positions=positions,
            energy=np.asarray((8.0, 10.0, 12.0, 10.0 + sweep)),
            bonds=bonds,
        )
    _write_state(
        run / "checkpoint.h5",
        final,
        positions=positions,
        energy=np.asarray((8.0, 10.0, 12.0, 10.0 + final)),
        bonds=bonds,
    )
    _write_metrics(run / "metrics.h5")
    return run


def test_visualization_module_does_not_initialize_jax() -> None:
    code = "import sys; import oe_tc.visualization; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_visualization_defaults_and_metric_smoothing() -> None:
    assert visualize.build_parser().parse_args(("run",)).fps == 5.0
    assert _metric_smoothing_window(399) == 1
    assert _metric_smoothing_window(10_000) == 51
    raw = np.arange(10, dtype=np.float64)
    smoothed = _smooth_metric(raw, 5)
    assert smoothed.shape == raw.shape
    assert smoothed[5] == pytest.approx(5.0)
    np.testing.assert_array_equal(_smooth_metric(raw, 1), raw)


def test_discovery_filters_strides_and_retains_final_frame(tmp_path: Path) -> None:
    run_directory = _run(tmp_path)
    run = discover_frames(run_directory, stride=2)

    assert [frame.sweep for frame in run.frames] == [1, 3, 4]
    assert run.bath_energy == 10.0
    loaded = load_frame(run, run.frames[-1])
    assert loaded.sweep == 4
    np.testing.assert_array_equal(loaded.energy, (8.0, 10.0, 12.0, 14.0))

    selected = discover_frames(run_directory, start_sweep=2, end_sweep=3)
    assert [frame.sweep for frame in selected.frames] == [2, 3]
    with pytest.raises(ValueError, match="no state frames"):
        discover_frames(run_directory, start_sweep=10)


def test_energy_scale_is_global_and_centered_on_bath(tmp_path: Path) -> None:
    run = discover_frames(_run(tmp_path))
    scale = choose_energy_scale(run, percentile=100.0)

    assert scale.center == 10.0
    assert scale.span == 4.0
    colors = energy_colors(np.asarray((6.0, 10.0, 14.0)), scale)
    assert colors[0, 2] > colors[0, 0]
    assert abs(int(colors[1, 0]) - int(colors[1, 2])) < 5
    assert colors[2, 0] > colors[2, 2]


def test_renderer_orients_source_at_top_and_draws_periodic_bond() -> None:
    frame = HostFrame(
        positions=np.asarray(((7, 2), (0, 2)), dtype=np.int32),
        energy=np.asarray((8.0, 12.0), dtype=np.float32),
        bonds=np.asarray((1, 2), dtype=np.uint8),
        sweep=5,
    )
    image = render_frame(
        frame,
        n=8,
        scale=EnergyScale(center=10.0, span=2.0, percentile=100.0),
        image_size=128,
    )

    assert image.shape == (220, 128, 3)
    np.testing.assert_array_equal(image[40, 64], (239, 174, 45))
    np.testing.assert_array_equal(image[40 + 2 * 16 + 8, 0], (64, 68, 75))


def test_downsampled_render_has_bounded_output() -> None:
    frame = HostFrame(
        positions=np.asarray(((0, 0), (511, 511)), dtype=np.int32),
        energy=np.asarray((9.0, 11.0), dtype=np.float32),
        bonds=np.zeros(2, dtype=np.uint8),
        sweep=1,
    )
    image = render_frame(
        frame,
        n=512,
        scale=EnergyScale(center=10.0, span=1.0, percentile=100.0),
        image_size=128,
    )
    assert image.shape == (220, 128, 3)


def test_visualize_run_writes_png_movie_dashboard_and_metadata(tmp_path: Path) -> None:
    run_directory = _run(tmp_path, snapshots=(0, 1), final=2)
    output = tmp_path / "graphics"
    result = visualize_run(
        run_directory,
        output_directory=output,
        image_size=128,
        fps=5.0,
    )

    assert result.frame_count == 3
    assert result.initial_frame is not None and result.initial_frame.is_file()
    assert result.latest_frame.is_file()
    assert result.movie is not None and result.movie.stat().st_size > 0
    assert result.metrics is not None and result.metrics.stat().st_size > 0
    assert Image.open(result.latest_frame).size == (128, 220)
    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["movie_frame_count"] == 3
    assert metadata["outputs"]["initial_frame"] == "initial.png"
    assert metadata["energy_scale"]["center"] == 10.0


def test_visualize_cli_can_render_latest_frame_only(tmp_path: Path, capsys) -> None:
    run_directory = _run(tmp_path, snapshots=(), final=0)
    output = tmp_path / "graphics"
    assert (
        visualize.main(
            (
                str(run_directory),
                "--output",
                str(output),
                "--size",
                "128",
                "--no-movie",
                "--no-metrics",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["frame_count"] == 1
    assert payload["initial_frame"] == str(output / "initial.png")
    assert payload["movie"] is None
    assert (output / "initial.png").is_file()
    assert (output / "latest.png").is_file()
