"""Host-side structural and thermodynamic evaluation for OE-TC model.

This module deliberately uses only NumPy and the Python standard library. It
is intended for checkpoint analysis, experiment dashboards, and validation;
none of its routines belongs inside a JAX-transformed simulation step.

Non-winding molecular types are canonicalized up to particle relabeling,
translation, and quarter-turn rotation. Components that wind around the
periodic direction instead retain their cylindrical orientation and are
canonicalized over cyclic horizontal translations. Reflections remain
distinct in both cases.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


Point = tuple[int, int]
Edge = tuple[Point, Point]

_DIRECTIONS: tuple[Point, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_OPPOSITE: tuple[int, ...] = (1, 0, 3, 2)


class MoleculeEmbeddingError(ValueError):
    """Raised when directional bonds do not define a cylindrical molecule."""


@dataclass(frozen=True, order=True)
class MoleculeSignature:
    """Canonical embedded bond graph of one molecule."""

    sites: tuple[Point, ...]
    bonds: tuple[Edge, ...]

    @property
    def size(self) -> int:
        return len(self.sites)


@dataclass(frozen=True)
class StateSummary:
    """Compact host-side description of one simulation snapshot.

    Entropies use natural logarithms. Molecules, rather than particles, are
    the samples in the size and molecular-type distributions. The lattice top
    is ``y=0``; ``top_bottom_energy_difference`` is mean energy on the
    minimum-y row minus mean energy on the maximum-y row.
    """

    molecule_count: int
    molecule_type_count: int
    molecule_size_mean: float
    molecule_size_max: int
    molecule_size_entropy: float
    type_shannon_entropy: float
    effective_type_diversity: float
    internal_energy_mean: float
    internal_energy_std: float
    vertical_energy_gradient: float
    top_bottom_energy_difference: float
    type_counts: tuple[tuple[MoleculeSignature, int], ...]
    size_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NoveltyRecord:
    """Novelty and turnover values produced for one observed snapshot."""

    snapshot_index: int
    present_type_count: int
    new_type_count: int
    cumulative_type_count: int
    type_turnover: float
    abundance_turnover: float
    new_types: tuple[MoleculeSignature, ...]


def _state_arrays(
    state: object, n: int | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract and validate positions and bond masks from a State-like object."""

    try:
        positions = np.asarray(getattr(state, "R"))
        bonds = np.asarray(getattr(state, "bonds"))
    except AttributeError as error:
        raise TypeError("state must provide R and bonds arrays") from error

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise MoleculeEmbeddingError("R must have shape (N, 2)")
    if not np.issubdtype(positions.dtype, np.integer):
        raise MoleculeEmbeddingError("particle coordinates must be integers")
    positions = positions.astype(np.int64, copy=False)

    num_particles = positions.shape[0]
    if bonds.shape != (num_particles,):
        raise MoleculeEmbeddingError("bonds must have shape (N,)")
    if not np.issubdtype(bonds.dtype, np.integer):
        raise MoleculeEmbeddingError("bond masks must be integers")
    bonds = bonds.astype(np.int64, copy=False)
    if np.any(bonds < 0) or np.any(bonds & ~0x0F):
        raise MoleculeEmbeddingError("bond masks may use only the low four bits")

    lattice = None
    if hasattr(state, "L"):
        lattice = np.asarray(getattr(state, "L"))
        if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
            raise MoleculeEmbeddingError("L must be a square occupancy array")
        inferred_n = int(lattice.shape[0])
        if n is None:
            n = inferred_n
        elif n != inferred_n:
            raise MoleculeEmbeddingError("n does not match the occupancy shape")

    if n is None:
        raise ValueError("n is required when state has no L array")
    if n < 1:
        raise ValueError("n must be positive")
    if np.any(positions < 0) or np.any(positions >= n):
        raise MoleculeEmbeddingError("particle coordinate is outside the lattice")

    expected = np.full((n, n), num_particles, dtype=np.int64)
    for particle, (x, y) in enumerate(positions):
        if expected[x, y] != num_particles:
            raise MoleculeEmbeddingError("multiple particles occupy one lattice site")
        expected[x, y] = particle
    if lattice is not None and not np.array_equal(
        lattice.astype(np.int64, copy=False), expected
    ):
        raise MoleculeEmbeddingError("L is inconsistent with particle positions")
    return positions, bonds, n


def _bond_graph(
    positions: np.ndarray, bonds: np.ndarray, n: int
) -> list[list[tuple[int, int]]]:
    """Validate directional bits and return directed bonded adjacency."""

    occupancy = {
        (int(x), int(y)): particle
        for particle, (x, y) in enumerate(positions)
    }
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(len(positions))
    ]
    for particle, (x_value, y_value) in enumerate(positions):
        x, y = int(x_value), int(y_value)
        for direction, (dx, dy) in enumerate(_DIRECTIONS):
            if not int(bonds[particle]) & (1 << direction):
                continue
            neighbor_y = y + dy
            if neighbor_y < 0 or neighbor_y >= n:
                raise MoleculeEmbeddingError(
                    f"particle {particle} has a bond through a closed vertical boundary"
                )
            other = occupancy.get(((x + dx) % n, neighbor_y))
            if other is None:
                raise MoleculeEmbeddingError(
                    f"particle {particle} has a bond pointing to an empty site"
                )
            if not int(bonds[other]) & (1 << _OPPOSITE[direction]):
                raise MoleculeEmbeddingError(
                    f"bond between particles {particle} and {other} is not reciprocal"
                )
            adjacency[particle].append((other, direction))
    return adjacency


def _unwrap_components(
    positions: np.ndarray,
    adjacency: list[list[tuple[int, int]]],
    n: int,
) -> list[tuple[dict[int, Point], tuple[tuple[int, int], ...], bool]]:
    """Unwrap components and distinguish winding from inconsistent cycles."""

    visited: set[int] = set()
    components: list[
        tuple[dict[int, Point], tuple[tuple[int, int], ...], bool]
    ] = []
    for start in range(len(positions)):
        if start in visited:
            continue
        coordinates: dict[int, Point] = {start: (0, 0)}
        queue: deque[int] = deque((start,))
        members: set[int] = set()
        winding = False

        while queue:
            particle = queue.popleft()
            if particle in members:
                continue
            members.add(particle)
            visited.add(particle)
            x, y = coordinates[particle]
            for other, direction in adjacency[particle]:
                dx, dy = _DIRECTIONS[direction]
                proposed = (x + dx, y + dy)
                existing = coordinates.get(other)
                if existing is None:
                    coordinates[other] = proposed
                    queue.append(other)
                elif existing != proposed:
                    difference = (
                        proposed[0] - existing[0], proposed[1] - existing[1]
                    )
                    if difference[1] == 0 and difference[0] % n == 0:
                        winding = True
                    else:
                        raise MoleculeEmbeddingError(
                            "directional bonds imply an inconsistent cylindrical embedding"
                        )

        start_position = positions[start]
        for particle, (x, y) in coordinates.items():
            projected = ((int(start_position[0]) + x) % n, int(start_position[1]) + y)
            if projected != tuple(map(int, positions[particle])):
                raise MoleculeEmbeddingError(
                    "unwrapped bonds do not reproduce the stored particle geometry"
                )

        component_edges = tuple(
            sorted(
                (particle, other)
                for particle in members
                for other, _ in adjacency[particle]
                if particle < other and other in members
            )
        )
        components.append((coordinates, component_edges, winding))
    return components


def _rotate(point: Point, quarter_turns: int) -> Point:
    x, y = point
    turns = quarter_turns % 4
    if turns == 0:
        return x, y
    if turns == 1:
        return -y, x
    if turns == 2:
        return -x, -y
    return y, -x


def _signature(
    coordinates: Mapping[int, Point], particle_edges: Iterable[tuple[int, int]]
) -> MoleculeSignature:
    """Return the least planar representation under four proper rotations."""

    candidates: list[MoleculeSignature] = []
    for turns in range(4):
        rotated = {
            particle: _rotate(point, turns)
            for particle, point in coordinates.items()
        }
        min_x = min(x for x, _ in rotated.values())
        min_y = min(y for _, y in rotated.values())
        normalized = {
            particle: (x - min_x, y - min_y)
            for particle, (x, y) in rotated.items()
        }
        candidates.append(
            MoleculeSignature(
                sites=tuple(sorted(normalized.values())),
                bonds=tuple(
                    sorted(
                        tuple(sorted((normalized[first], normalized[second])))
                        for first, second in particle_edges
                    )
                ),
            )
        )
    return min(candidates)


def _winding_signature(
    coordinates: Mapping[int, Point],
    particle_edges: Iterable[tuple[int, int]],
    n: int,
) -> MoleculeSignature:
    """Return the least cylindrical representation under horizontal shifts."""

    min_y = min(y for _, y in coordinates.values())
    projected = {
        particle: (x % n, y - min_y)
        for particle, (x, y) in coordinates.items()
    }
    candidates: list[MoleculeSignature] = []
    for shift in range(n):
        shifted = {
            particle: ((x + shift) % n, y)
            for particle, (x, y) in projected.items()
        }
        candidates.append(
            MoleculeSignature(
                sites=tuple(sorted(shifted.values())),
                bonds=tuple(
                    sorted(
                        tuple(sorted((shifted[first], shifted[second])))
                        for first, second in particle_edges
                    )
                ),
            )
        )
    return min(candidates)


def canonical_molecule_signatures(
    state: object, *, n: int | None = None
) -> tuple[MoleculeSignature, ...]:
    """Return sorted canonical signatures for all molecules in ``state``.

    Components are reconstructed from directional bonds rather than trusting
    cached labels. Non-winding components use planar C4 canonicalization;
    winding components use the translation symmetry of the cylinder.
    """

    positions, bonds, n = _state_arrays(state, n)
    components = _unwrap_components(positions, _bond_graph(positions, bonds, n), n)
    return tuple(
        sorted(
            _winding_signature(coordinates, edges, n)
            if winding
            else _signature(coordinates, edges)
            for coordinates, edges, winding in components
        )
    )


def molecule_type_counts(
    state: object, *, n: int | None = None
) -> Counter[MoleculeSignature]:
    """Count canonical molecular types in a State-like snapshot."""

    return Counter(canonical_molecule_signatures(state, n=n))


def _shannon_entropy(counts: Iterable[int]) -> float:
    values = np.asarray(tuple(counts), dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0.0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def summarize_state(state: object, *, n: int | None = None) -> StateSummary:
    """Compute structural diversity and energy-gradient diagnostics."""

    signatures = canonical_molecule_signatures(state, n=n)
    type_counter = Counter(signatures)
    size_counter = Counter(signature.size for signature in signatures)
    molecule_count = len(signatures)
    try:
        energy = np.asarray(getattr(state, "E"), dtype=np.float64)
        positions = np.asarray(getattr(state, "R"), dtype=np.float64)
    except AttributeError as error:
        raise TypeError("state must provide R, E, L, and bonds arrays") from error
    if energy.shape != (positions.shape[0],):
        raise ValueError("E must have shape (N,)")
    if not np.all(np.isfinite(energy)):
        raise ValueError("E must contain only finite values")

    if energy.size:
        energy_mean = float(np.mean(energy))
        energy_std = float(np.std(energy))
        y = positions[:, 1]
        centered_y = y - np.mean(y)
        denominator = float(np.dot(centered_y, centered_y))
        vertical_gradient = (
            float(np.dot(centered_y, energy - energy_mean) / denominator)
            if denominator > 0.0
            else 0.0
        )
        min_y, max_y = np.min(y), np.max(y)
        top_bottom = (
            float(np.mean(energy[y == min_y]) - np.mean(energy[y == max_y]))
            if max_y != min_y
            else 0.0
        )
    else:
        energy_mean = energy_std = vertical_gradient = top_bottom = 0.0

    type_entropy = _shannon_entropy(type_counter.values())
    sizes = np.asarray([signature.size for signature in signatures], dtype=np.float64)
    return StateSummary(
        molecule_count=molecule_count,
        molecule_type_count=len(type_counter),
        molecule_size_mean=float(np.mean(sizes)) if molecule_count else 0.0,
        molecule_size_max=int(np.max(sizes)) if molecule_count else 0,
        molecule_size_entropy=_shannon_entropy(size_counter.values()),
        type_shannon_entropy=type_entropy,
        effective_type_diversity=math.exp(type_entropy),
        internal_energy_mean=energy_mean,
        internal_energy_std=energy_std,
        vertical_energy_gradient=vertical_gradient,
        top_bottom_energy_difference=top_bottom,
        type_counts=tuple(sorted(type_counter.items())),
        size_counts=tuple(sorted(size_counter.items())),
    )


class NoveltyTracker:
    """Track first appearances and molecular-type turnover across snapshots."""

    def __init__(self) -> None:
        self.first_seen: dict[MoleculeSignature, int] = {}
        self._previous: Counter[MoleculeSignature] | None = None
        self._snapshot_index = 0

    @property
    def cumulative_type_count(self) -> int:
        return len(self.first_seen)

    @property
    def seen_types(self) -> frozenset[MoleculeSignature]:
        return frozenset(self.first_seen)

    def reset(self) -> None:
        self.first_seen.clear()
        self._previous = None
        self._snapshot_index = 0

    def observe(self, signatures: Iterable[MoleculeSignature]) -> NoveltyRecord:
        """Observe canonical signatures and return novelty for this snapshot."""

        current = Counter(signatures)
        new_types = tuple(sorted(set(current).difference(self.first_seen)))
        for signature in new_types:
            self.first_seen[signature] = self._snapshot_index

        if self._previous is None:
            type_turnover = abundance_turnover = 0.0
        else:
            previous_types = set(self._previous)
            current_types = set(current)
            union = previous_types | current_types
            type_turnover = (
                1.0 - len(previous_types & current_types) / len(union)
                if union
                else 0.0
            )
            numerator = sum(
                abs(current[signature] - self._previous[signature])
                for signature in union
            )
            denominator = sum(current.values()) + sum(self._previous.values())
            abundance_turnover = numerator / denominator if denominator else 0.0

        record = NoveltyRecord(
            snapshot_index=self._snapshot_index,
            present_type_count=len(current),
            new_type_count=len(new_types),
            cumulative_type_count=len(self.first_seen),
            type_turnover=float(type_turnover),
            abundance_turnover=float(abundance_turnover),
            new_types=new_types,
        )
        self._previous = current
        self._snapshot_index += 1
        return record

    def update(self, state: object, *, n: int | None = None) -> NoveltyRecord:
        """Canonicalize and observe all molecules in ``state``."""

        return self.observe(canonical_molecule_signatures(state, n=n))


molecule_signatures = canonical_molecule_signatures
state_summary = summarize_state


__all__ = [
    "MoleculeEmbeddingError",
    "MoleculeSignature",
    "StateSummary",
    "NoveltyRecord",
    "NoveltyTracker",
    "canonical_molecule_signatures",
    "molecule_signatures",
    "molecule_type_counts",
    "summarize_state",
    "state_summary",
]
