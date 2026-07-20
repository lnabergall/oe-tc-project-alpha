"""Fast host-side rendering of simulation snapshots and metrics.

State frames are rasterized directly from particle positions, energies, and bond
bitmasks. The renderer never initializes JAX and never materializes Matplotlib
artists per particle; RGB arrays are streamed directly to FFmpeg instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator

import h5py
import numpy as np

from oe_tc.version import MODEL_VERSION, SCHEMA_VERSION


MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.h5"
CHECKPOINT_NAME = "checkpoint.h5"
SNAPSHOT_DIRECTORY = "snapshots"
_SNAPSHOT_PATTERN = re.compile(r"sweep_(\d+)\.h5\Z")

_POS_X_BIT = np.uint8(1 << 0)
_POS_Y_BIT = np.uint8(1 << 2)
_BACKGROUND = np.asarray((246, 247, 249), dtype=np.uint8)
_OUTLINE = np.asarray((41, 47, 54), dtype=np.uint8)
_BOND = np.asarray((64, 68, 75), dtype=np.uint8)
_SOURCE = np.asarray((239, 174, 45), dtype=np.uint8)
_PALETTE = np.asarray(
    ((59, 76, 192), (141, 176, 254), (221, 221, 221), (244, 152, 122), (180, 4, 38)),
    dtype=np.float32,
)


@dataclass(frozen=True)
class FrameReference:
    """One immutable state file and its recorded sweep."""

    path: Path
    sweep: int
    source: str


@dataclass(frozen=True)
class HostFrame:
    """Minimal host representation needed by the renderer."""

    positions: np.ndarray
    energy: np.ndarray
    bonds: np.ndarray
    sweep: int


@dataclass(frozen=True)
class RunFrames:
    """Validated run metadata and ordered frame references."""

    directory: Path
    n: int
    num_particles: int
    bath_energy: float
    frames: tuple[FrameReference, ...]


@dataclass(frozen=True)
class EnergyScale:
    """Symmetric color scale centered on bath-equilibrium energy."""

    center: float
    span: float
    percentile: float

    @property
    def minimum(self) -> float:
        return self.center - self.span

    @property
    def maximum(self) -> float:
        return self.center + self.span


@dataclass(frozen=True)
class VisualizationResult:
    """Artifacts produced by :func:`visualize_run`."""

    latest_frame: Path
    movie: Path | None
    metrics: Path | None
    metadata: Path
    frame_count: int


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"run manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("run manifest must contain one object")
    return value


def _validate_versions(attributes: h5py.AttributeManager, source: Path) -> None:
    if int(attributes.get("model_version", -1)) != MODEL_VERSION:
        raise ValueError(f"{source} belongs to a different model version")
    if int(attributes.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"{source} uses an unsupported schema")


def _state_sweep(path: Path) -> int:
    try:
        with h5py.File(path, "r") as file:
            _validate_versions(file.attrs, path)
            sweep = np.asarray(file["sweep"][()])
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"unable to read state file {path}: {error}") from error
    if sweep.shape != () or not np.issubdtype(sweep.dtype, np.integer):
        raise ValueError(f"state file {path} has an invalid sweep")
    value = int(sweep)
    if value < 0:
        raise ValueError(f"state file {path} has a negative sweep")
    return value


def discover_frames(
    run_directory: str | Path,
    *,
    stride: int = 1,
    start_sweep: int | None = None,
    end_sweep: int | None = None,
) -> RunFrames:
    """Discover ordered snapshots plus a newer final checkpoint.

    ``stride`` is applied after sweep filtering, and the final eligible frame is
    always retained. This bounds rendering work without omitting the final state.
    """

    directory = Path(run_directory)
    if not directory.is_dir():
        raise ValueError(f"run directory not found: {directory}")
    if stride < 1:
        raise ValueError("frame stride must be positive")
    if start_sweep is not None and start_sweep < 0:
        raise ValueError("start sweep cannot be negative")
    if end_sweep is not None and end_sweep < 0:
        raise ValueError("end sweep cannot be negative")
    if start_sweep is not None and end_sweep is not None and start_sweep > end_sweep:
        raise ValueError("start sweep cannot exceed end sweep")

    manifest = _read_json(directory / MANIFEST_NAME)
    if manifest.get("model_version") != MODEL_VERSION:
        raise ValueError("manifest belongs to a different model version")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest uses an unsupported schema")

    static = manifest.get("static")
    parameters = manifest.get("params")
    if not isinstance(static, dict) or not isinstance(parameters, dict):
        raise ValueError("manifest is missing simulation configuration")
    n = static.get("n")
    num_particles = static.get("num_particles")
    if type(n) is not int or n < 4 or n % 2:
        raise ValueError("manifest has an invalid lattice width")
    if type(num_particles) is not int or not 0 < num_particles <= n * n:
        raise ValueError("manifest has an invalid particle count")
    try:
        bath_energy = float(parameters["heat_capacity"]) * float(parameters["bath_temperature"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest has invalid thermal parameters") from error
    if not math.isfinite(bath_energy) or bath_energy <= 0.0:
        raise ValueError("manifest bath-equilibrium energy must be positive")

    references: list[FrameReference] = []
    snapshot_directory = directory / SNAPSHOT_DIRECTORY
    for path in sorted(snapshot_directory.glob("sweep_*.h5")):
        match = _SNAPSHOT_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid snapshot filename: {path.name}")
        filename_sweep = int(match.group(1))
        sweep = _state_sweep(path)
        if sweep != filename_sweep:
            raise ValueError(f"snapshot {path.name} records sweep {sweep}, not {filename_sweep}")
        if references and sweep <= references[-1].sweep:
            raise ValueError("snapshot sweeps must be strictly increasing")
        references.append(FrameReference(path, sweep, path.relative_to(directory).as_posix()))

    checkpoint = directory / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise ValueError(f"final checkpoint not found: {checkpoint}")
    checkpoint_sweep = _state_sweep(checkpoint)
    if references and checkpoint_sweep < references[-1].sweep:
        raise ValueError("final checkpoint precedes the latest snapshot")
    if not references or checkpoint_sweep > references[-1].sweep:
        references.append(FrameReference(checkpoint, checkpoint_sweep, CHECKPOINT_NAME))

    filtered = [
        reference
        for reference in references
        if (start_sweep is None or reference.sweep >= start_sweep)
        and (end_sweep is None or reference.sweep <= end_sweep)
    ]
    if not filtered:
        raise ValueError("no state frames fall within the requested sweep range")
    selected = filtered[::stride]
    if selected[-1] != filtered[-1]:
        selected.append(filtered[-1])

    return RunFrames(directory, n, num_particles, bath_energy, tuple(selected))


def load_frame(run: RunFrames, reference: FrameReference) -> HostFrame:
    """Load and validate only the arrays required to draw one frame."""

    try:
        with h5py.File(reference.path, "r") as file:
            _validate_versions(file.attrs, reference.path)
            positions = np.asarray(file["R"][()], dtype=np.int32)
            energy = np.asarray(file["E"][()], dtype=np.float32)
            bonds = np.asarray(file["bonds"][()], dtype=np.uint8)
            sweep_array = np.asarray(file["sweep"][()])
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"unable to load visualization frame {reference.path}: {error}") from error

    if positions.shape != (run.num_particles, 2):
        raise ValueError(f"state file {reference.path} has invalid positions")
    if energy.shape != (run.num_particles,) or not np.all(np.isfinite(energy)):
        raise ValueError(f"state file {reference.path} has invalid energy")
    if bonds.shape != (run.num_particles,) or np.any(bonds > np.uint8(15)):
        raise ValueError(f"state file {reference.path} has invalid bond masks")
    if np.any(positions < 0) or np.any(positions >= run.n):
        raise ValueError(f"state file {reference.path} has out-of-range positions")
    sweep = int(sweep_array)
    if sweep != reference.sweep:
        raise ValueError(f"state file {reference.path} changed during visualization")
    return HostFrame(positions, energy, bonds, sweep)


def choose_energy_scale(
    run: RunFrames,
    *,
    percentile: float = 99.0,
    span: float | None = None,
) -> EnergyScale:
    """Choose one stable color scale for every movie frame."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("energy percentile must lie in (0, 100]")
    if span is not None:
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("energy span must be a finite positive number")
        return EnergyScale(run.bath_energy, float(span), percentile)

    robust_span = 0.0
    for reference in run.frames:
        try:
            with h5py.File(reference.path, "r") as file:
                values = np.asarray(file["E"][()], dtype=np.float64)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"unable to read energies from {reference.path}") from error
        if values.shape != (run.num_particles,) or not np.all(np.isfinite(values)):
            raise ValueError(f"state file {reference.path} has invalid energy")
        robust_span = max(
            robust_span,
            float(np.percentile(np.abs(values - run.bath_energy), percentile)),
        )

    numerical_floor = max(abs(run.bath_energy) * 0.02, 1.0e-6)
    return EnergyScale(run.bath_energy, max(robust_span, numerical_floor), percentile)


def energy_colors(energy: np.ndarray, scale: EnergyScale) -> np.ndarray:
    """Map energies through a compact cool-to-warm lookup palette."""

    values = np.asarray(energy, dtype=np.float64)
    position = np.clip((values - scale.minimum) / (2.0 * scale.span), 0.0, 1.0)
    location = position * (_PALETTE.shape[0] - 1)
    lower = np.minimum(location.astype(np.intp), _PALETTE.shape[0] - 2)
    fraction = (location - lower)[..., None]
    colors = _PALETTE[lower] * (1.0 - fraction) + _PALETTE[lower + 1] * fraction
    return np.rint(colors).astype(np.uint8)


def _draw_bonds(
    canvas: np.ndarray,
    positions: np.ndarray,
    bonds: np.ndarray,
    cell: int,
) -> None:
    """Draw positive-direction bond segments behind particle disks."""

    if cell <= 1:
        return
    side = canvas.shape[0]
    center = cell // 2
    offsets = np.arange(cell + 1, dtype=np.int32)
    half_width = min(3, max(0, cell // 10))

    horizontal = positions[(bonds & _POS_X_BIT) != 0]
    if horizontal.size:
        x = (horizontal[:, 0, None] * cell + center + offsets[None, :]) % side
        y = horizontal[:, 1, None] * cell + center
        for width_offset in range(-half_width, half_width + 1):
            rows = np.clip(y + width_offset, 0, side - 1)
            canvas[rows, x] = _BOND

    vertical = positions[(bonds & _POS_Y_BIT) != 0]
    if vertical.size:
        x = vertical[:, 0, None] * cell + center
        y = vertical[:, 1, None] * cell + center + offsets[None, :]
        valid = y < side
        for width_offset in range(-half_width, half_width + 1):
            columns = np.clip(x + width_offset, 0, side - 1)
            rows = np.clip(y, 0, side - 1)
            canvas[rows[valid], np.broadcast_to(columns, y.shape)[valid]] = _BOND


def _render_resolved_lattice(
    frame: HostFrame,
    n: int,
    side: int,
    colors: np.ndarray,
    show_bonds: bool,
) -> np.ndarray:
    cell = side // n
    canvas = np.broadcast_to(_BACKGROUND, (side, side, 3)).copy()
    if show_bonds:
        _draw_bonds(canvas, frame.positions, frame.bonds, cell)

    occupied = np.zeros((n, n), dtype=np.bool_)
    grid_colors = np.zeros((n, n, 3), dtype=np.uint8)
    screen_y = frame.positions[:, 1]
    screen_x = frame.positions[:, 0]
    occupied[screen_y, screen_x] = True
    grid_colors[screen_y, screen_x] = colors
    expanded_occupied = np.repeat(np.repeat(occupied, cell, axis=0), cell, axis=1)
    expanded_colors = np.repeat(np.repeat(grid_colors, cell, axis=0), cell, axis=1)

    if cell < 4:
        canvas[expanded_occupied] = expanded_colors[expanded_occupied]
        return canvas

    coordinate = np.arange(cell, dtype=np.float32) - (cell - 1) / 2.0
    radius_squared = coordinate[:, None] ** 2 + coordinate[None, :] ** 2
    outer = radius_squared <= (0.46 * cell) ** 2
    inner = radius_squared <= (0.37 * cell) ** 2
    outer_mask = expanded_occupied & np.tile(outer, (n, n))
    inner_mask = expanded_occupied & np.tile(inner, (n, n))
    canvas[outer_mask] = _OUTLINE
    canvas[inner_mask] = expanded_colors[inner_mask]
    return canvas


def _render_downsampled_lattice(
    frame: HostFrame,
    n: int,
    side: int,
    colors: np.ndarray,
) -> np.ndarray:
    """Aggregate subpixel sites without allocating an ``n x n`` image."""

    canvas = np.broadcast_to(_BACKGROUND, (side, side, 3)).copy()
    x = np.minimum(side - 1, ((2 * frame.positions[:, 0] + 1) * side) // (2 * n))
    y = np.minimum(side - 1, ((2 * frame.positions[:, 1] + 1) * side) // (2 * n))
    flat = y * side + x
    totals = np.zeros((side * side, 3), dtype=np.float32)
    counts = np.zeros(side * side, dtype=np.int32)
    np.add.at(totals, flat, colors)
    np.add.at(counts, flat, 1)
    visible = counts > 0
    flattened = canvas.reshape(-1, 3)
    flattened[visible] = np.rint(
        totals[visible] / counts[visible, None]
    ).astype(np.uint8)
    return canvas


def _font(size: int):
    try:
        from PIL import ImageFont
    except ImportError as error:
        raise RuntimeError(
            'visualization dependencies are missing; install ".[visualization]"'
        ) from error
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_frame(
    frame: HostFrame,
    *,
    n: int,
    scale: EnergyScale,
    image_size: int = 1024,
    show_bonds: bool = True,
) -> np.ndarray:
    """Render one annotated RGB frame with ``y=0`` at the irradiated top."""

    if image_size < 64 or image_size % 2:
        raise ValueError("image size must be an even integer of at least 64")
    colors = energy_colors(frame.energy, scale)
    side = n * max(1, image_size // n) if n <= image_size else image_size
    if n <= side:
        lattice = _render_resolved_lattice(frame, n, side, colors, show_bonds)
    else:
        lattice = _render_downsampled_lattice(frame, n, side, colors)

    header = max(40, min(64, side // 16))
    footer = max(52, min(72, side // 14))
    header += header % 2
    footer += footer % 2
    result = np.broadcast_to(_BACKGROUND, (header + side + footer, side, 3)).copy()
    result[header : header + side] = lattice
    result[header : header + 3] = _SOURCE

    bar_left = max(12, side // 5)
    bar_right = min(side - 12, side - side // 5)
    bar_top = header + side + 12
    bar_height = 10
    if bar_right > bar_left:
        values = np.linspace(scale.minimum, scale.maximum, bar_right - bar_left)
        result[bar_top : bar_top + bar_height, bar_left:bar_right] = energy_colors(
            values, scale
        )[None]

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            'visualization dependencies are missing; install ".[visualization]"'
        ) from error
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    text_size = max(12, min(20, side // 52))
    font = _font(text_size)
    bond_count = int(
        np.count_nonzero(frame.bonds & _POS_X_BIT)
        + np.count_nonzero(frame.bonds & _POS_Y_BIT)
    )
    title = (
        f"sweep {frame.sweep:,}"
        if side < 256
        else (
            f"sweep {frame.sweep:,}    particles {frame.energy.size:,}    "
            f"bonds {bond_count:,}    mean E {float(np.mean(frame.energy)):.3g}"
        )
    )
    draw.text((12, max(4, (header - text_size) // 2)), title, fill=(27, 31, 36), font=font)
    label_y = bar_top + bar_height + 4
    draw.text((bar_left, label_y), f"{scale.minimum:.3g}", fill=(40, 44, 50), font=font)
    center_label = f"bath E {scale.center:.3g}"
    center_x = max(bar_left, side // 2 - len(center_label) * text_size // 4)
    draw.text((center_x, label_y), center_label, fill=(40, 44, 50), font=font)
    maximum_label = f"{scale.maximum:.3g}"
    maximum_x = max(bar_left, bar_right - len(maximum_label) * text_size // 2)
    draw.text((maximum_x, label_y), maximum_label, fill=(40, 44, 50), font=font)
    return np.asarray(image)


def iter_rendered_frames(
    run: RunFrames,
    scale: EnergyScale,
    *,
    image_size: int = 1024,
    show_bonds: bool = True,
) -> Iterator[np.ndarray]:
    """Stream frames while holding only one state and image in memory."""

    for reference in run.frames:
        yield render_frame(
            load_frame(run, reference),
            n=run.n,
            scale=scale,
            image_size=image_size,
            show_bonds=show_bonds,
        )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def write_png(path: str | Path, frame: np.ndarray) -> Path:
    """Atomically write one RGB frame as PNG."""

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            'visualization dependencies are missing; install ".[visualization]"'
        ) from error
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output)
    try:
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
            temporary, format="PNG", compress_level=6
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_movie(
    path: str | Path,
    frames: Iterator[np.ndarray],
    *,
    fps: float = 20.0,
) -> tuple[Path, int]:
    """Stream RGB frames directly into a portable H.264 MP4."""

    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("movie frame rate must be positive")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            'visualization dependencies are missing; install ".[visualization]"'
        ) from error

    iterator = iter(frames)
    try:
        first = np.ascontiguousarray(next(iterator), dtype=np.uint8)
    except StopIteration as error:
        raise ValueError("cannot encode an empty movie") from error
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("movie frames must be H x W x 3 RGB arrays")
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("movie frame dimensions must be even")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output)
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=2,
        ffmpeg_log_level="warning",
        output_params=["-preset", "fast", "-movflags", "+faststart"],
    )
    count = 0
    try:
        writer.send(None)
        writer.send(first)
        count = 1
        for frame in iterator:
            array = np.ascontiguousarray(frame, dtype=np.uint8)
            if array.shape != first.shape:
                raise ValueError("movie frame shape changed during rendering")
            writer.send(array)
            count += 1
        writer.close()
        os.replace(temporary, output)
    except BaseException:
        try:
            writer.close()
        finally:
            temporary.unlink(missing_ok=True)
        raise
    return output, count


def _metric_samples(
    file: h5py.File, max_points: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = {dataset.shape[0] for dataset in file.values()}
    if len(lengths) != 1:
        raise ValueError("metrics datasets have inconsistent lengths")
    length = lengths.pop() if lengths else 0
    if length == 0:
        raise ValueError("metrics file contains no samples")
    step = max(1, math.ceil(length / max_points))
    indices = np.arange(0, length, step, dtype=np.int64)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    values = {name: np.asarray(dataset[indices]) for name, dataset in file.items()}
    return indices + 1, values


def write_metrics_dashboard(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    max_points: int = 10_000,
) -> Path:
    """Create a compact four-panel diagnostic dashboard."""

    if max_points < 100:
        raise ValueError("maximum metric points must be at least 100")
    source = Path(metrics_path)
    try:
        with h5py.File(source, "r") as file:
            _validate_versions(file.attrs, source)
            sweep, values = _metric_samples(file, max_points)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"unable to load metrics file {source}: {error}") from error

    try:
        if "MPLCONFIGDIR" not in os.environ:
            cache = Path(tempfile.gettempdir()) / "oe-tc-matplotlib"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            'visualization dependencies are missing; install ".[visualization]"'
        ) from error

    required = {
        "internal_energy",
        "configurational_energy",
        "source_energy",
        "bath_energy_direct",
        "bath_energy_structural",
        "num_molecules",
        "num_bonds",
        "accepted_molecule_moves",
        "accepted_bond_flips",
        "accepted_bath_exchanges",
        "molecule_conflicts",
        "molecule_unresolved",
        "mis_iterations",
        "component_iterations",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"metrics file is missing fields: {sorted(missing)}")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        energy = axes[0, 0]
        energy.plot(sweep, values["internal_energy"], label="internal")
        energy.plot(sweep, values["configurational_energy"], label="configurational")
        energy.plot(sweep, values["source_energy"], label="source flux", alpha=0.8)
        bath_flux = values["bath_energy_direct"] + values["bath_energy_structural"]
        energy.plot(sweep, bath_flux, label="bath flux", alpha=0.8)
        energy.set_title("Energy and flux")
        energy.legend(loc="best", fontsize="small")

        structure = axes[0, 1]
        structure.plot(sweep, values["num_molecules"], label="molecules")
        structure.plot(sweep, values["num_bonds"], label="bonds")
        structure.set_title("Structure")
        structure.legend(loc="best", fontsize="small")

        events = axes[1, 0]
        events.plot(sweep, values["accepted_molecule_moves"], label="molecule moves")
        events.plot(sweep, values["accepted_bond_flips"], label="bond flips")
        events.plot(sweep, values["accepted_bath_exchanges"], label="bath exchanges")
        events.set_title("Accepted events per sweep")
        events.legend(loc="best", fontsize="small")

        scheduler = axes[1, 1]
        scheduler.plot(sweep, values["molecule_conflicts"], label="conflicts")
        scheduler.plot(sweep, values["molecule_unresolved"], label="unresolved")
        scheduler.plot(sweep, values["mis_iterations"], label="MIS iterations")
        scheduler.plot(
            sweep, values["component_iterations"], label="component iterations"
        )
        scheduler.set_title("Parallel scheduler")
        scheduler.legend(loc="best", fontsize="small")

        for axis in axes.flat:
            axis.set_xlabel("sweep")
        figure.suptitle("OE-TC simulation diagnostics", fontsize=15)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(output)
        try:
            figure.savefig(temporary, dpi=150, format="png")
            os.replace(temporary, output)
        finally:
            plt.close(figure)
            temporary.unlink(missing_ok=True)
    return output


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def visualize_run(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    image_size: int = 1024,
    fps: float = 20.0,
    stride: int = 1,
    start_sweep: int | None = None,
    end_sweep: int | None = None,
    energy_percentile: float = 99.0,
    energy_span: float | None = None,
    show_bonds: bool = True,
    movie: bool = True,
    metrics: bool = True,
    max_metric_points: int = 10_000,
) -> VisualizationResult:
    """Render the latest state, an optional movie, and metrics dashboard."""

    run = discover_frames(
        run_directory,
        stride=stride,
        start_sweep=start_sweep,
        end_sweep=end_sweep,
    )
    output = (
        Path(output_directory)
        if output_directory is not None
        else run.directory / "visualization"
    )
    output.mkdir(parents=True, exist_ok=True)
    scale = choose_energy_scale(run, percentile=energy_percentile, span=energy_span)

    last = render_frame(
        load_frame(run, run.frames[-1]),
        n=run.n,
        scale=scale,
        image_size=image_size,
        show_bonds=show_bonds,
    )
    latest_path = write_png(output / "latest.png", last)

    movie_path: Path | None = None
    movie_frames = 0
    if movie and len(run.frames) > 1:
        movie_path, movie_frames = write_movie(
            output / "movie.mp4",
            iter_rendered_frames(
                run,
                scale,
                image_size=image_size,
                show_bonds=show_bonds,
            ),
            fps=fps,
        )

    metrics_path: Path | None = None
    source_metrics = run.directory / METRICS_NAME
    if metrics and source_metrics.is_file():
        metrics_path = write_metrics_dashboard(
            source_metrics,
            output / "metrics.png",
            max_points=max_metric_points,
        )

    metadata_path = output / "visualization.json"
    metadata = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "renderer": "numpy-pillow-raster-v1",
        "run_directory": str(run.directory.resolve()),
        "frame_count": len(run.frames),
        "movie_frame_count": movie_frames,
        "first_sweep": run.frames[0].sweep,
        "last_sweep": run.frames[-1].sweep,
        "image_size": image_size,
        "fps": fps,
        "stride": stride,
        "show_bonds": show_bonds,
        "energy_scale": asdict(scale),
        "outputs": {
            "latest_frame": latest_path.name,
            "movie": movie_path.name if movie_path is not None else None,
            "metrics": metrics_path.name if metrics_path is not None else None,
        },
    }
    _atomic_json(metadata_path, metadata)
    return VisualizationResult(
        latest_path,
        movie_path,
        metrics_path,
        metadata_path,
        len(run.frames),
    )


__all__ = [
    "EnergyScale",
    "FrameReference",
    "HostFrame",
    "RunFrames",
    "VisualizationResult",
    "choose_energy_scale",
    "discover_frames",
    "energy_colors",
    "iter_rendered_frames",
    "load_frame",
    "render_frame",
    "visualize_run",
    "write_metrics_dashboard",
    "write_movie",
    "write_png",
]