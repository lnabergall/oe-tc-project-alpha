"""Small NumPy reference routines used to validate compiled simulation kernels."""

from __future__ import annotations

from collections import deque

import numpy as np


DIRECTIONS = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=np.int32)
OPPOSITE = np.asarray((1, 0, 3, 2), dtype=np.int32)


def occupancy(R: np.ndarray, n: int) -> np.ndarray:
    empty = len(R)
    L = np.full((n, n), empty, dtype=np.int32)
    for particle, (x, y) in enumerate(R):
        if L[x % n, y] != empty:
            raise ValueError("duplicate particle position")
        L[x % n, y] = particle
    return L


def neighbor(L: np.ndarray, r: np.ndarray, direction: int) -> int:
    n = L.shape[0]
    x, y = r + DIRECTIONS[direction]
    if y < 0 or y >= n:
        return n * n
    return int(L[x % n, y])


def components(R: np.ndarray, bonds: np.ndarray, n: int) -> np.ndarray:
    """Return minimum-particle component roots for directional bond masks."""

    num_particles = len(R)
    L = occupancy(R, n)
    result = np.full(num_particles, -1, dtype=np.int32)

    for start in range(num_particles):
        if result[start] >= 0:
            continue
        queue = deque((start,))
        members: list[int] = []
        result[start] = start
        while queue:
            particle = queue.popleft()
            members.append(particle)
            for direction in range(4):
                if not (int(bonds[particle]) & (1 << direction)):
                    continue
                other = neighbor(L, R[particle], direction)
                if other >= num_particles:
                    raise ValueError("bond points to an empty or invalid site")
                if not (int(bonds[other]) & (1 << int(OPPOSITE[direction]))):
                    raise ValueError("asymmetric bond")
                if result[other] < 0:
                    result[other] = start
                    queue.append(other)
        root = min(members)
        result[members] = root
    return result


def configurational_energy(R: np.ndarray, bonds: np.ndarray, n: int, eta: float) -> float:
    """Count each occupied +x/+y contact once."""

    num_particles = len(R)
    L = occupancy(R, n)
    energy = 0.0
    for particle in range(num_particles):
        for direction in (0, 2):
            other = neighbor(L, R[particle], direction)
            if other >= num_particles:
                continue
            energy -= 1.0 if int(bonds[particle]) & (1 << direction) else eta
    return energy
