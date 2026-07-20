from __future__ import annotations

import jax
import numpy as np

import analyze
from oe_tc.config import StaticConfig, default_params
from oe_tc.state import State
from oe_tc.storage import save_checkpoint, save_manifest


def test_analyzer_accepts_a_cylinder_winding_molecule(tmp_path) -> None:
    run = tmp_path / "winding-run"
    config = StaticConfig(n=4, num_particles=4)
    positions = np.asarray([(x, 1) for x in range(4)], dtype=np.int32)
    lattice = np.full((4, 4), 4, dtype=np.int32)
    lattice[np.arange(4), 1] = np.arange(4)
    state = State(
        R=positions,
        E=np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32),
        L=lattice,
        bonds=np.full(4, 0b0011, dtype=np.uint8),
        root=np.zeros(4, dtype=np.int32),
        sweep=np.asarray(3, dtype=np.uint32),
    )
    key = jax.random.key(7)
    save_manifest(run / "manifest.json", config, default_params(), seed=7)
    save_checkpoint(run / "snapshots" / "sweep_0000000003.h5", state, key)
    save_checkpoint(run / "checkpoint.h5", state, key)

    report = analyze.analyze_run(run)

    assert report["observation_count"] == 1
    assert report["observations"][0]["summary"]["molecule_count"] == 1
    assert report["observations"][0]["summary"]["molecule_size_max"] == 4
