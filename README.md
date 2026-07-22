# OE-TC project alpha

This repository implements a driven, dissipative lattice chemistry for
large-scale experiments in thermodynamic computing and open-ended
self-organization. Bond-connected components form rigid molecules. A top-down
source injects internal energy; contact and bond conduction transport it;
molecular motion and reversible bonding exchange internal, configurational,
and bath energy; and exposed particles exchange heat with an external bath.

The maintained implementation is fixed-shape and accelerator-first. It uses
particle vectors, a dense occupancy lattice, directional bond bitmasks,
component labels, counter-derived random streams, batched local kernels,
fixed bond/conduction color classes, and optimistic molecular proposals
filtered through lattice-resource arbitration. It does not construct an
`N x N` interaction matrix or variable-length molecule arrays.

Earlier experiments are preserved under `legacy/` and are not maintained.

## Setup

Python 3.12 or newer is required. For a CPU development environment:

```bash
python -m venv .venv
python -m pip --python .venv install --upgrade pip
python -m pip --python .venv install -e ".[dev]"
```

For NVIDIA acceleration, use the repository from the WSL ext4 filesystem
(such as `~/oe-tc-project-alpha`), not through `/mnt/c`. An up-to-date Windows
NVIDIA driver must expose the GPU to WSL, but neither a Linux display driver nor
a separately installed CUDA toolkit is required. From Ubuntu under WSL 2, run:

```bash
cd ~/oe-tc-project-alpha
bash scripts/setup_wsl_cuda.sh
source .venv/bin/activate
```

The setup script installs the pip-bundled JAX CUDA 13 runtime from the `cuda13`
extra and finishes by executing a GPU matrix workload. It deliberately fails
rather than silently accepting a CPU fallback. Avoid setting `LD_LIBRARY_PATH`
to another CUDA installation, which can override JAX's bundled libraries.
Re-run the script after dependency changes; it safely upgrades the existing
environment.

Validate the complete environment and measure the actual simulation kernel:

```bash
python -m pytest
python scripts/verify_cuda.py
python benchmarks/benchmark.py --platform gpu --no-preallocate \
  --n 64 --density 0.25 --chunk-size 16 --warmups 2 --repeats 10
```

The RTX 3050 Ti has 4 GiB of device memory, so begin at `n=64` or `n=128`, use
`--no-preallocate` while calibrating, and increase the lattice only after
measuring resident memory and throughput. The default dependency remains
CPU-capable JAX; `.[cuda13]` is intentionally opt-in. GitHub Actions runs the
full test suite on CPU for every push and pull request.

For a non-development installation with rendering support, use
`pip install -e ".[visualization]"`.

## Run a simulation

```bash
python run_simulation.py \
  --n 256 \
  --particles 16384 \
  --steps 10000 \
  --chunk-size 128 \
  --snapshot-every 1000 \
  --seed 12 \
  --output data/runs/run_001
```

The horizontal circumference `n` must be even and at least four. Complete
fixed-shape chunks execute on device; compact metrics and restartable state
cross to the host only at chunk boundaries. A run directory contains:

- `manifest.json`: immutable configuration, parameters, seed, and format
  versions;
- `metrics.h5`: append-only per-sweep flux, state, and scheduler diagnostics;
- `checkpoint.h5`: latest state, sweep counter, and base random key;
- `snapshots/sweep_*.h5`: optional immutable analysis snapshots.

Resume to a new total target sweep with:

```bash
python run_simulation.py --steps 20000 --resume data/runs/run_001
```

Chunk size may change on resume. Physical parameters, static shapes, seed, and
snapshot cadence remain immutable. Counter-derived random streams make the
trajectory independent of chunk partitioning.

Override parameters with an inline JSON object or a JSON file:

```bash
python run_simulation.py --n 64 --particles 1024 --steps 1000 \
  --output data/runs/calibration \
  --params-json '{"eta": 0.15, "catalysis_strength": 0.75}'
```

Use `--no-source` for source-off equilibrium validation. `--platform cpu`,
`--platform gpu`, and `--platform tpu` select a backend before JAX imports.

## Analyze a run

```bash
python analyze.py data/runs/run_001 \
  --output data/runs/run_001/analysis.json
```

The analyzer processes immutable snapshots in sweep order, includes a newer
checkpoint, and reports molecular size and type diversity, internal-energy
gradients, first appearances, and structural turnover. Signatures are invariant
to particle IDs and applicable lattice translations. Ordinary finite molecules
are also canonicalized across quarter-turn rotations. A molecule that wraps
around the periodic horizontal direction of the cylindrical lattice is treated
separately because it cannot generally be lifted to one finite planar shape.

## Visualize a run

```bash
python visualize.py data/runs/run_001
```

By default this writes `latest.png`, `movie.mp4`, `metrics.png`, and a
reproducibility manifest under `RUN_DIRECTORY/visualization`. The movie uses a
single global cool-to-warm energy scale centered on bath-equilibrium energy;
the gold upper boundary marks the incident source. Bond lines are drawn behind
particles, and `y = 0` appears at the physical top of the lattice.

Rendering is host-only and does not initialize JAX. It reads only positions,
energies, and bond masks from one snapshot at a time, rasterizes with
NumPy/Pillow, and streams raw frames directly to FFmpeg. This avoids the
per-particle Matplotlib artist and 300-DPI animation overhead of the legacy
renderer. Use `--stride` and `--size` to bound work for long or very large runs:

```bash
python visualize.py data/runs/run_001 --stride 10 --size 768 --fps 5
```

Use `--no-movie`, `--no-metrics`, or `--no-bonds` to suppress individual
outputs. `--energy-span` fixes the color range across comparable experiments;
otherwise the range is selected once from all included frames using
`--energy-percentile`. Metrics dashboards retain faint raw per-sweep traces and
overlay adaptive rolling means for readable long-run trends.

## Validate and benchmark

```bash
python -m pytest tests/simulation
python benchmarks/benchmark.py --n 64 --density 0.25 \
  --chunk-size 16 --warmups 2 --repeats 10
```

The benchmark separates lowering, compilation, first execution, and
synchronized steady-state throughput, and emits machine-readable JSON. Test at
the intended density and molecule-size distribution: molecular contact
accounting and conflict arbitration are expected to dominate mature large runs.

## Package layout

- `oe_tc/config.py`, `oe_tc/state.py`: immutable configuration and pytrees;
- `oe_tc/geometry.py`, `oe_tc/topology.py`: cylindrical lattice and bonded
  components;
- `oe_tc/source.py`, `oe_tc/conduction.py`, `oe_tc/bath.py`: local energy
  kernels;
- `oe_tc/molecules.py`, `oe_tc/scheduler.py`: rigid proposals and optimistic
  conflict resolution;
- `oe_tc/bonds.py`: reversible catalyzed bond dynamics in randomized fixed
  classes;
- `oe_tc/simulation.py`: audited sweep composition and deterministic chunk
  scans;
- `oe_tc/storage.py`, `oe_tc/runner.py`: versioned persistence and execution;
- `oe_tc/evaluation.py`: structural diversity and novelty analysis;
- `oe_tc/visualization.py`: streamed state rasterization, movie encoding, and
  metrics dashboards.

Each structural proposal samples either a bath or internal-energy channel and
uses a log-space Metropolis decision. Metrics include signed source and bath
fluxes, scheduler saturation, component convergence, and total-energy residual.
A run is not persisted when bounded component labeling fails to converge.

## Numerical and scaling notes

The default parameters are a calibrated dimensionless baseline, not a universal
physical calibration. The bonded-contact energy is the unit scale and the bath
energy mode is `0.5`; source and bath parameters avoid secular heating in short
calibration runs. Source input still scales with occupied columns, while bath
exchange and structural opportunities scale differently with particle count and
geometry. Revalidate source, bath, and kinetic parameters at the intended
lattice size and density before interpreting long-run behavior.

Internal energy uses `float32` for accelerator throughput. At very large
energies, fixed heat or interaction quanta approach the local floating-point
spacing and transitions lose resolution. Monitor `energy_residual`, energy
distributions, and source/sink balance, and choose nondimensional scales that
keep all operative quanta representable.

Dense occupancy and resource arbitration require `Theta(n^2)` storage even at
low density; the design targets large fixed-density systems. Checkpointing
transfers full state once per chunk, so larger chunks reduce host I/O but
increase recovery distance.

The optimistic molecule conflict filter is exact for conflict-free active
proposals but can perturb the global equilibrium kernel when accepted proposals
conflict. Source-off experiments should measure conflict rates and compare
small systems with a conservative reference before drawing equilibrium claims.