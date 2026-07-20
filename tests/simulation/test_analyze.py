from __future__ import annotations

import json

import jax
import numpy as np
import pytest

import analyze
from oe_tc.config import StaticConfig, default_params
from oe_tc.state import State
from oe_tc.storage import SCHEMA_VERSION, save_checkpoint, save_manifest


def _state(sweep: int, *, bonded: bool = False, energy=(1.0, 3.0)) -> State:
    positions = np.asarray(((0, 0), (1, 0)), dtype=np.int32)
    lattice = np.full((4, 4), 2, dtype=np.int32)
    lattice[0, 0] = 0
    lattice[1, 0] = 1
    bonds = np.asarray((1, 2) if bonded else (0, 0), dtype=np.uint8)
    return State(
        R=positions,
        E=np.asarray(energy, dtype=np.float32),
        L=lattice,
        bonds=bonds,
        root=np.asarray((0, 0) if bonded else (0, 1), dtype=np.int32),
        sweep=np.asarray(sweep, dtype=np.uint32),
    )


def _run_directory(tmp_path, final: State, snapshots=()):
    run = tmp_path / "run"
    config = StaticConfig(n=4, num_particles=2)
    save_manifest(run / "manifest.json", config, default_params(), seed=11)
    key = jax.random.key(11)
    for state in snapshots:
        save_checkpoint(
            run / "snapshots" / f"sweep_{int(state.sweep):010d}.h5",
            state,
            key,
        )
    save_checkpoint(run / "checkpoint.h5", final, key)
    return run


def test_analysis_uses_snapshots_and_deduplicates_final_checkpoint(tmp_path):
    first = _state(2)
    final = _state(4, bonded=True, energy=(2.0, 4.0))
    run = _run_directory(tmp_path, final, (first, final))

    report = analyze.analyze_run(run)

    assert report["observation_count"] == 2
    assert [row["sweep"] for row in report["observations"]] == [2, 4]
    assert report["observations"][0]["source"] == "snapshots/sweep_0000000002.h5"
    assert report["observations"][1]["source"] == "snapshots/sweep_0000000004.h5"
    assert report["observations"][0]["summary"]["molecule_count"] == 2
    assert report["observations"][1]["summary"]["molecule_count"] == 1
    assert report["observations"][0]["novelty"]["new_type_count"] == 1
    assert report["observations"][1]["novelty"]["new_type_count"] == 1
    assert report["cumulative_molecule_type_count"] == 2


def test_analysis_appends_a_later_final_checkpoint(tmp_path):
    run = _run_directory(tmp_path, _state(5, bonded=True), (_state(2),))

    report = analyze.analyze_run(run)

    assert [row["source"] for row in report["observations"]] == [
        "snapshots/sweep_0000000002.h5",
        "checkpoint.h5",
    ]
    assert report["last_sweep"] == 5


def test_same_sweep_snapshot_and_checkpoint_must_agree(tmp_path):
    run = _run_directory(
        tmp_path,
        _state(3, bonded=True),
        (_state(3, bonded=False),),
    )

    with pytest.raises(ValueError, match="disagrees"):
        analyze.analyze_run(run)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_version", 6, "model version"),
        ("schema_version", SCHEMA_VERSION + 1, "schema"),
        ("static", {"n": 5}, "static configuration"),
    ),
)
def test_analysis_rejects_invalid_manifest(tmp_path, field, value, message):
    run = _run_directory(tmp_path, _state(1))
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze.analyze_run(run)


def test_cli_stdout_and_optional_output_are_identical_and_deterministic(
    tmp_path, capsys
):
    run = _run_directory(tmp_path, _state(2), (_state(1),))
    output = tmp_path / "analysis.json"

    assert analyze.main((str(run), "--output", str(output))) == 0
    stdout = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == stdout
    assert stdout == analyze.report_json(analyze.analyze_run(run))
    parsed = json.loads(stdout)
    assert parsed["configuration"]["seed"] == 11
    assert parsed["observation_count"] == 2
