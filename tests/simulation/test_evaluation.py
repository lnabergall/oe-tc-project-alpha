from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from oe_tc.evaluation import (
    MoleculeEmbeddingError,
    NoveltyTracker,
    canonical_molecule_signatures,
    summarize_state,
)


DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE = (1, 0, 3, 2)


def make_state(
    sites: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    *,
    n: int = 8,
    energy: list[float] | None = None,
) -> SimpleNamespace:
    positions = np.asarray(sites, dtype=np.int32)
    bonds = np.zeros(len(sites), dtype=np.uint8)
    for first, second in edges:
        x0, y0 = sites[first]
        x1, y1 = sites[second]
        direction = None
        for candidate, (dx, dy) in enumerate(DIRECTIONS):
            if ((x0 + dx) % n, y0 + dy) == (x1, y1):
                direction = candidate
                break
        if direction is None:
            raise ValueError("test edge is not nearest-neighbor")
        bonds[first] |= np.uint8(1 << direction)
        bonds[second] |= np.uint8(1 << OPPOSITE[direction])

    lattice = np.full((n, n), len(sites), dtype=np.int32)
    for particle, (x, y) in enumerate(sites):
        lattice[x, y] = particle
    if energy is None:
        energy = [1.0] * len(sites)
    return SimpleNamespace(
        R=positions,
        E=np.asarray(energy, dtype=np.float32),
        L=lattice,
        bonds=bonds,
        root=np.arange(len(sites), dtype=np.int32),
        sweep=np.int32(0),
    )


def transform_state(
    state: SimpleNamespace,
    edges: list[tuple[int, int]],
    *,
    turns: int = 0,
    shift: tuple[int, int] = (0, 0),
    permutation: list[int] | None = None,
) -> SimpleNamespace:
    n = state.L.shape[0]
    points: list[tuple[int, int]] = []
    for x_value, y_value in np.asarray(state.R):
        x, y = int(x_value), int(y_value)
        for _ in range(turns % 4):
            x, y = -y, x
        points.append((x, y))
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    points = [
        ((x - min_x + shift[0]) % n, y - min_y + shift[1])
        for x, y in points
    ]

    if permutation is None:
        permutation = list(range(len(points)))
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted_points = [points[old] for old in permutation]
    permuted_edges = [(inverse[a], inverse[b]) for a, b in edges]
    energy = [float(state.E[old]) for old in permutation]
    return make_state(permuted_points, permuted_edges, n=n, energy=energy)


def test_signature_is_id_translation_and_rotation_invariant() -> None:
    sites = [(6, 2), (7, 2), (0, 2), (0, 3), (1, 3)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    original = make_state(sites, edges)
    translated = make_state(
        [((x + 3) % 8, y + 2) for x, y in sites], edges
    )
    assert canonical_molecule_signatures(original) == canonical_molecule_signatures(
        translated
    )

    planar = make_state([(1, 1), (2, 1), (3, 1), (3, 2), (4, 2)], edges)
    transformed = transform_state(
        planar,
        edges,
        turns=1,
        shift=(2, 1),
        permutation=[4, 1, 3, 0, 2],
    )
    assert canonical_molecule_signatures(planar) == canonical_molecule_signatures(
        transformed
    )


def test_signature_preserves_chirality() -> None:
    s_shape = make_state(
        [(2, 1), (3, 1), (1, 2), (2, 2)],
        [(0, 1), (0, 3), (2, 3)],
    )
    z_shape = make_state(
        [(1, 1), (2, 1), (2, 2), (3, 2)],
        [(0, 1), (1, 2), (2, 3)],
    )
    assert canonical_molecule_signatures(s_shape) != canonical_molecule_signatures(
        z_shape
    )


def test_signature_includes_bond_topology() -> None:
    sites = [(1, 1), (2, 1), (2, 2), (1, 2)]
    path = make_state(sites, [(0, 1), (1, 2), (2, 3)])
    cycle = make_state(sites, [(0, 1), (1, 2), (2, 3), (3, 0)])
    path_signature = canonical_molecule_signatures(path)[0]
    cycle_signature = canonical_molecule_signatures(cycle)[0]
    assert path_signature.sites == cycle_signature.sites
    assert path_signature.bonds != cycle_signature.bonds


def _decorated_winding_state() -> tuple[SimpleNamespace, list[tuple[int, int]]]:
    ring = [(x, 2) for x in range(8)]
    sites = ring + [(0, 3), (1, 3), (3, 3)]
    edges = [(x, (x + 1) % 8) for x in range(8)]
    edges += [(0, 8), (1, 9), (3, 10)]
    return make_state(sites, edges), edges


def test_winding_signature_is_translation_and_particle_id_invariant() -> None:
    original, edges = _decorated_winding_state()
    transformed = transform_state(
        original,
        edges,
        shift=(3, 1),
        permutation=[10, 4, 1, 8, 6, 0, 3, 9, 7, 2, 5],
    )

    signature = canonical_molecule_signatures(original)[0]
    assert signature == canonical_molecule_signatures(transformed)[0]
    assert any(abs(first[0] - second[0]) == 7 for first, second in signature.bonds)


def test_winding_signature_preserves_reflection_sensitivity() -> None:
    original, edges = _decorated_winding_state()
    reflected = make_state(
        [((-x) % 8, y) for x, y in original.R.tolist()],
        edges,
    )

    assert canonical_molecule_signatures(original) != canonical_molecule_signatures(
        reflected
    )


def test_inconsistent_directional_bond_is_rejected() -> None:
    state = make_state([(1, 1), (2, 1)], [(0, 1)])
    state.bonds[1] = 0
    with pytest.raises(MoleculeEmbeddingError, match="not reciprocal"):
        canonical_molecule_signatures(state)


def test_state_summary_reports_diversity_and_energy_gradient() -> None:
    state = make_state(
        [(1, 0), (2, 0), (4, 5), (6, 5)],
        [(0, 1)],
        n=8,
        energy=[1.0, 3.0, 5.0, 7.0],
    )
    summary = summarize_state(state)
    expected_entropy = -(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3)

    assert summary.molecule_count == 3
    assert summary.molecule_type_count == 2
    assert summary.molecule_size_mean == pytest.approx(4 / 3)
    assert summary.molecule_size_max == 2
    assert summary.molecule_size_entropy == pytest.approx(expected_entropy)
    assert summary.type_shannon_entropy == pytest.approx(expected_entropy)
    assert summary.effective_type_diversity == pytest.approx(np.exp(expected_entropy))
    assert summary.internal_energy_mean == pytest.approx(4.0)
    assert summary.internal_energy_std == pytest.approx(np.sqrt(5.0))
    assert summary.vertical_energy_gradient == pytest.approx(0.8)
    assert summary.top_bottom_energy_difference == pytest.approx(-4.0)


def test_novelty_tracker_records_first_seen_and_turnover() -> None:
    monomer = canonical_molecule_signatures(make_state([(1, 1)], []))[0]
    dimer = canonical_molecule_signatures(
        make_state([(1, 1), (2, 1)], [(0, 1)])
    )[0]
    trimer = canonical_molecule_signatures(
        make_state([(1, 1), (2, 1), (3, 1)], [(0, 1), (1, 2)])
    )[0]
    tracker = NoveltyTracker()

    first = tracker.observe((monomer, monomer, dimer))
    second = tracker.observe((monomer, trimer))

    assert first.snapshot_index == 0
    assert first.new_type_count == 2
    assert first.cumulative_type_count == 2
    assert first.type_turnover == 0.0
    assert tracker.first_seen[monomer] == 0
    assert tracker.first_seen[dimer] == 0
    assert second.new_types == (trimer,)
    assert second.new_type_count == 1
    assert second.cumulative_type_count == 3
    assert second.type_turnover == pytest.approx(2 / 3)
    assert second.abundance_turnover == pytest.approx(3 / 5)
    assert tracker.first_seen[trimer] == 1

    tracker.reset()
    assert tracker.cumulative_type_count == 0
