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

Python 3.12 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -e ".[dev]"
```

The default dependency installs the current CPU-capable JAX release. Follow
JAX's platform-specific installation instructions for CUDA or TPU hardware.

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
- `oe_tc/evaluation.py`: structural diversity and novelty analysis.

Each structural proposal samples either a bath or internal-energy channel and
uses a log-space Metropolis decision. Metrics include signed source and bath
fluxes, scheduler saturation, component convergence, and total-energy residual.
A run is not persisted when bounded component labeling fails to converge.

## Numerical and scaling notes

The default parameters are a runnable baseline, not a calibrated nonequilibrium
steady state. Source input scales with occupied columns, while bath exchange and
structural opportunities scale differently with particle count and geometry.
Calibrate source, bath, and kinetic parameters at the intended lattice size and
density before interpreting long-run behavior.

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