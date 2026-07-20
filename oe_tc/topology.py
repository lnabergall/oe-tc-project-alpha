"""Persistent bond topology for OE-TC model.

The lattice stores particle ids in ``L[x, y]`` and uses ``N`` (the number of
particles) as its empty-site sentinel.  Each particle stores four directional
bond bits using the shared order ``(+x, -x, +y, -y)``.  Every physical bond is
mirrored at its two endpoints.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


NUM_DIRECTIONS = 4
POS_X = 0
NEG_X = 1
POS_Y = 2
NEG_Y = 3
EAST = POS_X
WEST = NEG_X
NORTH = POS_Y
SOUTH = NEG_Y

DIRECTION_OFFSETS = jnp.asarray(
    ((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=jnp.int32
)
OPPOSITE_DIRECTIONS = jnp.asarray((NEG_X, POS_X, NEG_Y, POS_Y), dtype=jnp.int32)

_CCW_DIRECTION_PERMUTATIONS = jnp.asarray(
    (
        (POS_X, NEG_X, POS_Y, NEG_Y),
        (POS_Y, NEG_Y, NEG_X, POS_X),
        (NEG_X, POS_X, NEG_Y, POS_Y),
        (NEG_Y, POS_Y, POS_X, NEG_X),
    ),
    dtype=jnp.int32,
)


def direction_bit(direction: int | jax.Array) -> jax.Array:
    """Return the uint8 bit corresponding to ``direction``."""

    return jnp.left_shift(jnp.asarray(1, dtype=jnp.uint8), jnp.asarray(direction))


def opposite_direction(direction: int | jax.Array) -> jax.Array:
    """Return the direction at the other endpoint of an undirected edge."""

    return OPPOSITE_DIRECTIONS[jnp.asarray(direction, dtype=jnp.int32)]


def opposite_bit(direction: int | jax.Array) -> jax.Array:
    """Return the mirrored endpoint bit for an edge in ``direction``."""

    return direction_bit(opposite_direction(direction))


def has_bond(mask: jax.Array, direction: int | jax.Array) -> jax.Array:
    """Test a scalar or array of masks for a bond in ``direction``."""

    return jnp.bitwise_and(mask, direction_bit(direction)) != 0


bond_bit_is_set = has_bond


def set_bond_bit(
    mask: jax.Array, direction: int | jax.Array, value: bool | jax.Array = True
) -> jax.Array:
    """Set or clear one direction bit without changing the other bits."""

    bit = direction_bit(direction).astype(jnp.asarray(mask).dtype)
    return jnp.where(
        value,
        jnp.bitwise_or(mask, bit),
        jnp.bitwise_and(mask, jnp.bitwise_not(bit)),
    )


def clear_bond_bit(mask: jax.Array, direction: int | jax.Array) -> jax.Array:
    """Clear one direction bit."""

    return set_bond_bit(mask, direction, False)


def toggle_bond_bit(mask: jax.Array, direction: int | jax.Array) -> jax.Array:
    """Toggle one direction bit."""

    bit = direction_bit(direction).astype(jnp.asarray(mask).dtype)
    return jnp.bitwise_xor(mask, bit)


def rotate_direction(
    direction: int | jax.Array, quarter_turns: int | jax.Array
) -> jax.Array:
    """Rotate a direction anticlockwise by ``quarter_turns`` quarter turns."""

    turns = jnp.mod(jnp.asarray(quarter_turns, dtype=jnp.int32), 4)
    return _CCW_DIRECTION_PERMUTATIONS[
        turns, jnp.asarray(direction, dtype=jnp.int32)
    ]


def rotate_bond_mask(mask: jax.Array, quarter_turns: int | jax.Array) -> jax.Array:
    """Apply the geometric quarter-turn permutation to bond masks."""

    mask = jnp.asarray(mask)
    turns = jnp.mod(jnp.asarray(quarter_turns, dtype=jnp.int32), 4)
    permutation = _CCW_DIRECTION_PERMUTATIONS[turns]
    result = jnp.zeros_like(mask)
    for old_direction in range(NUM_DIRECTIONS):
        old_value = jnp.bitwise_and(
            jnp.right_shift(mask, old_direction), jnp.asarray(1, dtype=mask.dtype)
        )
        result = jnp.bitwise_or(
            result,
            jnp.left_shift(old_value, permutation[old_direction]).astype(mask.dtype),
        )
    return result


def rotate_bond_mask_ccw(
    mask: jax.Array, quarter_turns: int | jax.Array = 1
) -> jax.Array:
    return rotate_bond_mask(mask, quarter_turns)


def rotate_bond_mask_cw(
    mask: jax.Array, quarter_turns: int | jax.Array = 1
) -> jax.Array:
    return rotate_bond_mask(mask, -jnp.asarray(quarter_turns))


def build_occupancy(
    positions: jax.Array, n: int, empty: int | None = None
) -> jax.Array:
    """Build fixed-shape ``L[x,y]`` from unique canonical positions."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    num_particles = positions.shape[0]
    if empty is None:
        empty = num_particles
    lattice = jnp.full((n, n), empty, dtype=jnp.int32)
    particles = jnp.arange(num_particles, dtype=jnp.int32)
    return lattice.at[positions[:, 0], positions[:, 1]].set(particles)


occupancy_from_positions = build_occupancy


def neighbor_coordinates(
    positions: jax.Array, n: int
) -> tuple[jax.Array, jax.Array]:
    """Return safe four-neighbor coordinates and closed-y validity."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    raw = positions[:, None, :] + DIRECTION_OFFSETS[None, :, :]
    x = jnp.mod(raw[..., 0], n)
    valid = (raw[..., 1] >= 0) & (raw[..., 1] < n)
    y = jnp.clip(raw[..., 1], 0, n - 1)
    return jnp.stack((x, y), axis=-1), valid


def gather_lattice_neighbors(
    occupancy: jax.Array,
    positions: jax.Array,
    empty: int | None = None,
) -> jax.Array:
    """Gather four neighboring particle ids, padding vacancies/walls by N."""

    if empty is None:
        empty = positions.shape[0]
    coordinates, valid = neighbor_coordinates(positions, occupancy.shape[0])
    gathered = occupancy[coordinates[..., 0], coordinates[..., 1]]
    return jnp.where(valid, gathered, jnp.asarray(empty, dtype=gathered.dtype))


gather_neighbors = gather_lattice_neighbors
neighbor_particles = gather_lattice_neighbors


def gather_bonded_neighbors(
    occupancy: jax.Array,
    positions: jax.Array,
    bonds: jax.Array,
    empty: int | None = None,
    *,
    require_reciprocal: bool = True,
) -> jax.Array:
    """Gather bonded neighbors in direction slots, padding with ``empty``."""

    bonds = jnp.asarray(bonds)
    num_particles = bonds.shape[0]
    if empty is None:
        empty = num_particles
    candidates = gather_lattice_neighbors(occupancy, positions, empty)
    bits = jnp.left_shift(
        jnp.asarray(1, dtype=bonds.dtype),
        jnp.arange(NUM_DIRECTIONS, dtype=jnp.int32),
    )
    active = jnp.bitwise_and(bonds[:, None], bits[None, :]) != 0
    occupied = (candidates >= 0) & (candidates < num_particles)

    if require_reciprocal:
        padded = jnp.concatenate((bonds, jnp.zeros((1,), dtype=bonds.dtype)))
        safe_candidates = jnp.where(occupied, candidates, num_particles)
        target_masks = padded[safe_candidates]
        reciprocal_bits = bits[OPPOSITE_DIRECTIONS]
        reciprocal = jnp.bitwise_and(target_masks, reciprocal_bits[None, :]) != 0
        active = active & reciprocal

    return jnp.where(
        active & occupied, candidates, jnp.asarray(empty, dtype=candidates.dtype)
    )


bonded_neighbors = gather_bonded_neighbors


def _pointer_jump(parent: jax.Array) -> jax.Array:
    """Fully compress a parent forest in a fixed logarithmic number of steps."""

    num_particles = parent.shape[0]
    steps = max(1, math.ceil(math.log2(max(1, num_particles))) + 1)

    def jump(_: int, current: jax.Array) -> jax.Array:
        return current[current]

    return jax.lax.fori_loop(0, steps, jump, parent)


def connected_components_from_neighbors_with_status(
    neighbors: jax.Array,
    *,
    empty: int | None = None,
    max_iters: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return labels, executed rounds, and whether a fixed point was confirmed.

    ``neighbors`` has shape ``(N,K)``.  Simultaneous scatter-min hooks higher
    roots to lower roots, while pointer jumping compresses all trees after each
    round.  The storage cost is ``O(N + NK)``.  If the iteration cap stops a
    still-changing computation, ``root`` is a partial labeling and
    ``converged`` is false; callers must not treat it as a molecule partition.
    """

    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    num_particles = neighbors.shape[0]
    if empty is None:
        empty = num_particles
    del empty
    if max_iters is None:
        max_iters = max(1, num_particles)

    particles = jnp.arange(num_particles, dtype=jnp.int32)
    sources = jnp.broadcast_to(particles[:, None], neighbors.shape).reshape((-1,))
    targets = neighbors.reshape((-1,))
    valid_edges = (targets >= 0) & (targets < num_particles)
    safe_targets = jnp.where(valid_edges, targets, 0)

    def condition(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        iteration, _, changed = carry
        return (iteration < max_iters) & changed

    def hook_and_shortcut(
        carry: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        iteration, parent, _ = carry
        roots = _pointer_jump(parent)
        source_roots = roots[sources]
        target_roots = roots[safe_targets]
        high = jnp.maximum(source_roots, target_roots)
        low = jnp.minimum(source_roots, target_roots)

        safe_high = jnp.where(valid_edges, high, 0)
        safe_low = jnp.where(valid_edges, low, safe_high)
        hooks = particles.at[safe_high].min(safe_low)
        next_parent = _pointer_jump(hooks[roots])
        changed = jnp.any(next_parent != roots)
        return iteration + 1, next_parent, changed

    initial = (jnp.asarray(0, jnp.int32), particles, jnp.asarray(True))
    iterations, parent, changed = jax.lax.while_loop(
        condition, hook_and_shortcut, initial
    )
    return _pointer_jump(parent), iterations, jnp.logical_not(changed)


def connected_components_from_neighbors_with_iterations(
    neighbors: jax.Array,
    *,
    empty: int | None = None,
    max_iters: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Backward-compatible labels/iteration wrapper."""

    root, iterations, _ = connected_components_from_neighbors_with_status(
        neighbors, empty=empty, max_iters=max_iters
    )
    return root, iterations


def connected_components_from_neighbors(
    neighbors: jax.Array,
    *,
    empty: int | None = None,
    max_iters: int | None = None,
) -> jax.Array:
    """Backward-compatible labels-only wrapper."""

    return connected_components_from_neighbors_with_status(
        neighbors, empty=empty, max_iters=max_iters
    )[0]


def connected_components_with_status(
    occupancy: jax.Array,
    positions: jax.Array,
    bonds: jax.Array,
    max_iters: int | None = None,
    empty: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return molecule labels, executed rounds, and convergence status."""

    if empty is None:
        empty = bonds.shape[0]
    neighbors = gather_bonded_neighbors(occupancy, positions, bonds, empty)
    return connected_components_from_neighbors_with_status(
        neighbors, empty=empty, max_iters=max_iters
    )


def connected_components_with_iterations(
    occupancy: jax.Array,
    positions: jax.Array,
    bonds: jax.Array,
    max_iters: int | None = None,
    empty: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Backward-compatible labels/iteration wrapper."""

    root, iterations, _ = connected_components_with_status(
        occupancy, positions, bonds, max_iters=max_iters, empty=empty
    )
    return root, iterations


def connected_components(
    occupancy: jax.Array,
    positions: jax.Array,
    bonds: jax.Array,
    max_iters: int | None = None,
    empty: int | None = None,
) -> jax.Array:
    """Backward-compatible stable minimum-id molecule-label wrapper."""

    return connected_components_with_status(
        occupancy, positions, bonds, max_iters=max_iters, empty=empty
    )[0]


compute_components = connected_components
compute_molecule_roots = connected_components


def component_sizes(root: jax.Array) -> jax.Array:
    """Return an N-vector whose root entries contain molecule sizes."""

    root = jnp.asarray(root, dtype=jnp.int32)
    return jnp.bincount(root, length=root.shape[0])


def topology_is_valid(
    occupancy: jax.Array,
    positions: jax.Array,
    bonds: jax.Array,
    empty: int | None = None,
) -> jax.Array:
    """Check the occupancy bijection and reciprocal directional bond bits."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    bonds = jnp.asarray(bonds)
    num_particles = positions.shape[0]
    if empty is None:
        empty = num_particles
    n = occupancy.shape[0]

    coordinates_valid = (
        (positions[:, 0] >= 0)
        & (positions[:, 0] < n)
        & (positions[:, 1] >= 0)
        & (positions[:, 1] < n)
    )
    safe_positions = jnp.clip(positions, 0, n - 1)
    ids = jnp.arange(num_particles, dtype=occupancy.dtype)
    occupancy_valid = occupancy[
        safe_positions[:, 0], safe_positions[:, 1]
    ] == ids

    candidates = gather_lattice_neighbors(occupancy, safe_positions, empty)
    occupied = (candidates >= 0) & (candidates < num_particles)
    padded = jnp.concatenate((bonds, jnp.zeros((1,), dtype=bonds.dtype)))
    safe_candidates = jnp.where(occupied, candidates, num_particles)
    target_masks = padded[safe_candidates]
    bits = jnp.left_shift(
        jnp.asarray(1, dtype=bonds.dtype),
        jnp.arange(NUM_DIRECTIONS, dtype=jnp.int32),
    )
    local = jnp.bitwise_and(bonds[:, None], bits[None, :]) != 0
    reciprocal = jnp.bitwise_and(
        target_masks, bits[OPPOSITE_DIRECTIONS][None, :]
    ) != 0
    bonds_valid = jnp.all(local == (occupied & reciprocal))
    high_bits_clear = jnp.all(
        jnp.bitwise_and(bonds, jnp.asarray(0xF0, bonds.dtype)) == 0
    )
    return (
        jnp.all(coordinates_valid)
        & jnp.all(occupancy_valid)
        & bonds_valid
        & high_bits_clear
    )


validate_topology = topology_is_valid


__all__ = [
    "NUM_DIRECTIONS",
    "POS_X",
    "NEG_X",
    "POS_Y",
    "NEG_Y",
    "EAST",
    "WEST",
    "NORTH",
    "SOUTH",
    "DIRECTION_OFFSETS",
    "OPPOSITE_DIRECTIONS",
    "direction_bit",
    "opposite_direction",
    "opposite_bit",
    "has_bond",
    "bond_bit_is_set",
    "set_bond_bit",
    "clear_bond_bit",
    "toggle_bond_bit",
    "rotate_direction",
    "rotate_bond_mask",
    "rotate_bond_mask_ccw",
    "rotate_bond_mask_cw",
    "build_occupancy",
    "occupancy_from_positions",
    "neighbor_coordinates",
    "gather_lattice_neighbors",
    "gather_neighbors",
    "neighbor_particles",
    "gather_bonded_neighbors",
    "bonded_neighbors",
    "connected_components_from_neighbors_with_status",
    "connected_components_from_neighbors_with_iterations",
    "connected_components_from_neighbors",
    "connected_components_with_status",
    "connected_components_with_iterations",
    "connected_components",
    "compute_components",
    "compute_molecule_roots",
    "component_sizes",
    "topology_is_valid",
    "validate_topology",
]
