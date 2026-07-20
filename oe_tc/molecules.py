"""Rigid molecule proposals for OE-TC model.

All component data are root-indexed N-vectors.  Per-particle proposal gathers,
lexicographically grouped contact records, and pairwise site lookups keep both
storage and work sparse; no molecule-by-particle or root-by-lattice array is
materialized.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.config import Params, StaticConfig
from oe_tc.geometry import (
    DIRECTIONS,
    exposure_fraction,
    lattice_indices,
    neighbor_coordinates,
    neighbor_indices,
)
from oe_tc.state import State
from oe_tc.thermodynamics import (
    bath_channel_affinity,
    kinetic_metropolis_log_probability,
)
from oe_tc.topology import rotate_bond_mask, rotate_direction


CONTACT_EFFECT_RECORDS_PER_PARTICLE = 16


class MoleculeMetadata(NamedTuple):
    """Fixed-shape root-segment aggregate metadata."""

    segment_size: jax.Array
    is_root: jax.Array


class ContactEffects(NamedTuple):
    """Sparse endpoint energy updates grouped by event root and particle."""

    root: jax.Array
    particle: jax.Array
    delta: jax.Array
    valid: jax.Array


class MoleculeProposals(NamedTuple):
    """One independently sampled proposal for each active molecule root."""

    active: jax.Array
    translate: jax.Array
    direction: jax.Array
    pivot: jax.Array
    quarter_turns: jax.Array
    internal_channel: jax.Array
    candidate_R: jax.Array
    candidate_bonds: jax.Array
    legal: jax.Array
    potential_energy_change: jax.Array
    entropy_change: jax.Array
    kinetic_probability: jax.Array
    log_acceptance_probability: jax.Array
    accepted: jax.Array
    effects: ContactEffects


def molecule_metadata(root: jax.Array) -> MoleculeMetadata:
    """Build O(N) component sizes and present-root flags."""

    root = jnp.asarray(root, dtype=jnp.int32)
    size = jnp.bincount(root, length=root.shape[0])
    return MoleculeMetadata(segment_size=size, is_root=size > 0)


build_molecule_metadata = molecule_metadata
root_segment_metadata = molecule_metadata


def _segment_sum(values: jax.Array, root: jax.Array) -> jax.Array:
    output_shape = (root.shape[0],) + values.shape[1:]
    return jnp.zeros(output_shape, dtype=values.dtype).at[root].add(values)


def _segment_all(values: jax.Array, root: jax.Array) -> jax.Array:
    result = jnp.ones((root.shape[0],), dtype=jnp.int8)
    return result.at[root].min(values.astype(jnp.int8)).astype(jnp.bool_)


def minimum_image_displacement(
    positions: jax.Array, pivots: jax.Array, n: int
) -> tuple[jax.Array, jax.Array]:
    """Return cylindrical minimum-image offsets and lift-uniqueness flags.

    On an even cylinder, horizontal separation ``n/2`` has two shortest lifts.
    A rotation containing such an antipodal point is rejected rather than made
    dependent on an arbitrary branch.
    """

    positions = jnp.asarray(positions, dtype=jnp.int32)
    pivots = jnp.asarray(pivots, dtype=jnp.int32)
    raw = positions - pivots
    dx = jnp.mod(raw[..., 0] + n // 2, n) - n // 2
    displacement = raw.at[..., 0].set(dx)
    return displacement, jnp.abs(dx) != n // 2


def lexicographic_pair_membership(
    first: jax.Array,
    second: jax.Array,
    query_first: jax.Array,
    query_second: jax.Array,
) -> jax.Array:
    """Find int32 pair queries without overflow-prone packed scalar keys.

    Reference pairs are lexicographically sorted, then all queries perform a
    fixed-depth vectorized lower-bound search.  Query arrays may have any fixed
    shape and are returned with that shape.
    """

    first = jnp.asarray(first, dtype=jnp.int32)
    second = jnp.asarray(second, dtype=jnp.int32)
    query_first = jnp.asarray(query_first, dtype=jnp.int32)
    query_second = jnp.asarray(query_second, dtype=jnp.int32)
    count = first.shape[0]
    order = jnp.lexsort((second, first))
    sorted_first, sorted_second = first[order], second[order]
    q_first, q_second = query_first.reshape(-1), query_second.reshape(-1)
    low = jnp.zeros(q_first.shape, dtype=jnp.int32)
    high = jnp.full(q_first.shape, count, dtype=jnp.int32)

    def lower_bound_step(_: int, bounds: tuple[jax.Array, jax.Array]):
        lo, hi = bounds
        middle = (lo + hi) // 2
        safe_middle = jnp.minimum(middle, count - 1)
        middle_first = sorted_first[safe_middle]
        middle_second = sorted_second[safe_middle]
        less = (middle_first < q_first) | (
            (middle_first == q_first) & (middle_second < q_second)
        )
        less &= middle < count
        return jnp.where(less, middle + 1, lo), jnp.where(less, hi, middle)

    depth = max(1, math.ceil(math.log2(max(1, count))) + 1)
    low, _ = jax.lax.fori_loop(0, depth, lower_bound_step, (low, high))
    safe_low = jnp.minimum(low, count - 1)
    found = (
        (low < count)
        & (sorted_first[safe_low] == q_first)
        & (sorted_second[safe_low] == q_second)
    )
    return found.reshape(query_first.shape)


def _candidate_site_membership(
    candidate_R: jax.Array, root: jax.Array, query_coordinates: jax.Array, n: int
) -> jax.Array:
    candidate_site = lattice_indices(candidate_R, n).astype(jnp.int32)
    query_site = lattice_indices(query_coordinates, n).astype(jnp.int32)
    query_root = jnp.broadcast_to(root[..., None], query_site.shape)
    return lexicographic_pair_membership(
        root, candidate_site, query_root, query_site
    )


def _root_site_uniqueness(
    candidate_R: jax.Array, root: jax.Array, n: int
) -> jax.Array:
    """Check that no transform maps two same-root particles to one site."""

    count = root.shape[0]
    site = lattice_indices(candidate_R, n).astype(jnp.int32)
    order = jnp.lexsort((site, root))
    sorted_root, sorted_site = root[order], site[order]
    adjacent_duplicate = (sorted_root[1:] == sorted_root[:-1]) & (
        sorted_site[1:] == sorted_site[:-1]
    )
    duplicated = jnp.concatenate((adjacent_duplicate, jnp.asarray((False,)))) | jnp.concatenate(
        (jnp.asarray((False,)), adjacent_duplicate)
    )
    bad = jnp.zeros((count,), dtype=jnp.int8).at[sorted_root].max(
        duplicated.astype(jnp.int8)
    )
    return bad == 0


def _uniform_component_pivots(
    key: jax.Array, root: jax.Array
) -> jax.Array:
    """Choose a uniform particle pivot using O(N) scatter reductions."""

    count = root.shape[0]
    particle = jnp.arange(count, dtype=jnp.int32)
    priority = jax.random.bits(key, (count,), dtype=jnp.uint32)
    maximum = jnp.zeros((count,), dtype=jnp.uint32).at[root].max(priority)
    candidate = priority == maximum[root]
    return jnp.full((count,), count, dtype=jnp.int32).at[root].min(
        jnp.where(candidate, particle, count)
    )


def bond_geometry_preserved(
    state: State,
    candidate_R: jax.Array,
    translate: jax.Array,
    quarter_turns: jax.Array,
    n: int,
) -> jax.Array:
    """Check that a transform preserves every internal contact endpoint.

    Bonded contacts are included explicitly.  Requiring all old same-component
    contacts to reach the same endpoint in the rotated direction rejects
    inconsistent pointwise cylinder lifts; comparing directed contact counts
    also rejects rotations that create a spurious internal seam contact.
    """

    root = jnp.asarray(state.root, dtype=jnp.int32)
    count = root.shape[0]
    directions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (count, 4))
    particle_translate = translate[root]
    turns = quarter_turns[root]
    rotated_directions = jax.vmap(
        lambda turn: jax.vmap(lambda direction: rotate_direction(direction, turn))(
            jnp.arange(4, dtype=jnp.int32)
        )
    )(turns)
    candidate_direction = jnp.where(
        particle_translate[:, None], directions, rotated_directions
    )
    candidate_geometry = neighbor_coordinates(candidate_R, n)
    particle = jnp.arange(count, dtype=jnp.int32)[:, None]
    expected_coordinates = candidate_geometry.coordinates[
        particle, candidate_direction
    ]
    expected_valid = candidate_geometry.valid[particle, candidate_direction]

    original_neighbor = neighbor_indices(state.R, state.L, count)
    safe_neighbor = jnp.minimum(original_neighbor, count - 1)
    actual_coordinates = candidate_R[safe_neighbor]
    padded_root = jnp.concatenate((root, jnp.asarray((count,), jnp.int32)))
    original_internal = (original_neighbor < count) & (
        padded_root[jnp.minimum(original_neighbor, count)] == root[:, None]
    )
    bit = jnp.left_shift(jnp.asarray(1, dtype=state.bonds.dtype), directions)
    bonded = jnp.bitwise_and(state.bonds[:, None], bit) != 0
    endpoint_matches = (
        (original_neighbor < count)
        & expected_valid
        & jnp.all(expected_coordinates == actual_coordinates, axis=-1)
    )
    endpoints_preserved = _segment_all(
        jnp.all(~(original_internal | bonded) | endpoint_matches, axis=1), root
    )

    candidate_internal = candidate_geometry.valid & _candidate_site_membership(
        candidate_R, root, candidate_geometry.coordinates, n
    )
    old_contact_count = _segment_sum(
        jnp.sum(original_internal, axis=1, dtype=jnp.int32), root
    )
    new_contact_count = _segment_sum(
        jnp.sum(candidate_internal, axis=1, dtype=jnp.int32), root
    )
    return endpoints_preserved & (old_contact_count == new_contact_count)


internal_geometry_preserved = bond_geometry_preserved

def _group_endpoint_effects(
    raw_root: jax.Array,
    raw_particle: jax.Array,
    raw_delta: jax.Array,
    raw_valid: jax.Array,
    num_particles: int,
) -> ContactEffects:
    """Combine duplicate endpoint records using lexicographic pair grouping."""

    record_count = raw_root.shape[0]
    invalid = (~raw_valid).astype(jnp.int8)
    order = jnp.lexsort((raw_particle, raw_root, invalid))
    sorted_root = raw_root[order]
    sorted_particle = raw_particle[order]
    sorted_delta = raw_delta[order]
    sorted_valid = raw_valid[order]
    same_previous = jnp.concatenate(
        (
            jnp.asarray((False,)),
            (sorted_root[1:] == sorted_root[:-1])
            & (sorted_particle[1:] == sorted_particle[:-1])
            & sorted_valid[:-1],
        )
    )
    new_group = sorted_valid & ~same_previous
    group = jnp.cumsum(new_group.astype(jnp.int32)) - 1
    safe_group = jnp.maximum(group, 0)
    grouped_delta = jnp.zeros((record_count,), dtype=raw_delta.dtype).at[
        safe_group
    ].add(jnp.where(sorted_valid, sorted_delta, 0))
    grouped_root = jnp.full((record_count,), num_particles, dtype=jnp.int32).at[
        safe_group
    ].min(jnp.where(sorted_valid, sorted_root, num_particles))
    grouped_particle = jnp.full(
        (record_count,), num_particles, dtype=jnp.int32
    ).at[safe_group].min(jnp.where(sorted_valid, sorted_particle, num_particles))
    valid = grouped_root < num_particles
    return ContactEffects(
        root=jnp.where(valid, grouped_root, 0),
        particle=jnp.where(valid, grouped_particle, 0),
        delta=grouped_delta,
        valid=valid,
    )


def changed_contact_effects(
    state: State,
    candidate_R: jax.Array,
    internal_channel: jax.Array,
    active: jax.Array,
    eta: float | jax.Array,
) -> tuple[jax.Array, jax.Array, ContactEffects]:
    """Return delta-U, member delta-E, and exact grouped endpoint effects."""

    root = jnp.asarray(state.root, dtype=jnp.int32)
    count = root.shape[0]
    particle = jnp.arange(count, dtype=jnp.int32)
    padded_root = jnp.concatenate((root, jnp.asarray((count,), jnp.int32)))
    old_neighbor = neighbor_indices(state.R, state.L, count)
    new_neighbor = neighbor_indices(candidate_R, state.L, count)
    safe_old = jnp.minimum(old_neighbor, count)
    safe_new = jnp.minimum(new_neighbor, count)
    old_external = (old_neighbor < count) & (padded_root[safe_old] != root[:, None])
    new_external = (new_neighbor < count) & (padded_root[safe_new] != root[:, None])
    old_persists = jnp.any(
        (old_neighbor[..., None] == new_neighbor[:, None, :])
        & new_external[:, None, :],
        axis=-1,
    )
    new_persists = jnp.any(
        (new_neighbor[..., None] == old_neighbor[:, None, :])
        & old_external[:, None, :],
        axis=-1,
    )
    root_active = active[root]
    broken = old_external & ~old_persists & root_active[:, None]
    formed = new_external & ~new_persists & root_active[:, None]
    eta_value = jnp.asarray(eta, dtype=state.E.dtype)
    delta_u_particle = eta_value * (
        jnp.sum(broken, axis=1, dtype=state.E.dtype)
        - jnp.sum(formed, axis=1, dtype=state.E.dtype)
    )
    delta_u = _segment_sum(delta_u_particle, root)
    internal_particle = internal_channel[root]
    member_delta = jnp.where(internal_particle, -0.5 * delta_u_particle, 0)

    roots4 = jnp.broadcast_to(root[:, None], broken.shape)
    particles4 = jnp.broadcast_to(particle[:, None], broken.shape)
    half_eta = jnp.asarray(0.5, state.E.dtype) * eta_value
    old_delta = jnp.full(broken.shape, -half_eta, dtype=state.E.dtype)
    new_delta = jnp.full(formed.shape, half_eta, dtype=state.E.dtype)
    old_valid = broken & internal_particle[:, None]
    new_valid = formed & internal_particle[:, None]
    raw_root = jnp.concatenate(tuple(roots4.reshape(-1) for _ in range(4)))
    raw_particle = jnp.concatenate(
        (
            particles4.reshape(-1),
            jnp.minimum(old_neighbor, count - 1).reshape(-1),
            particles4.reshape(-1),
            jnp.minimum(new_neighbor, count - 1).reshape(-1),
        )
    )
    raw_delta = jnp.concatenate(
        (
            old_delta.reshape(-1),
            old_delta.reshape(-1),
            new_delta.reshape(-1),
            new_delta.reshape(-1),
        )
    )
    raw_valid = jnp.concatenate(
        (
            old_valid.reshape(-1),
            old_valid.reshape(-1),
            new_valid.reshape(-1),
            new_valid.reshape(-1),
        )
    )
    effects = _group_endpoint_effects(
        raw_root, raw_particle, raw_delta, raw_valid, count
    )
    return delta_u, member_delta, effects


def effect_thermodynamics(
    effects: ContactEffects,
    energy: jax.Array,
    heat_capacity: float | jax.Array,
    energy_floor: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute exact per-root entropy changes and endpoint floor validity."""

    count = energy.shape[0]
    before = energy[effects.particle]
    after = before + effects.delta
    floor = jnp.asarray(energy_floor, dtype=energy.dtype)
    valid_floor = (~effects.valid) | (after >= floor)
    floor_by_root = jnp.ones((count,), dtype=jnp.int8).at[effects.root].min(
        valid_floor.astype(jnp.int8)
    )
    safe_after = jnp.maximum(after, floor)
    entropy_record = jnp.where(
        effects.valid,
        jnp.asarray(heat_capacity, dtype=energy.dtype) * jnp.log(safe_after / before),
        0,
    )
    entropy_by_root = jnp.zeros((count,), dtype=energy.dtype).at[effects.root].add(
        entropy_record
    )
    return entropy_by_root, floor_by_root.astype(jnp.bool_)


def _post_move_exposure(
    state: State, candidate_R: jax.Array, root: jax.Array, n: int
) -> jax.Array:
    count = root.shape[0]
    geometry = neighbor_coordinates(candidate_R, n)
    gathered = state.L[geometry.coordinates[..., 0], geometry.coordinates[..., 1]]
    padded_root = jnp.concatenate((root, jnp.asarray((count,), jnp.int32)))
    safe = jnp.minimum(gathered, count)
    outside_occupied = (gathered < count) & (padded_root[safe] != root[:, None])
    own_occupied = _candidate_site_membership(candidate_R, root, geometry.coordinates, n)
    unoccupied = geometry.valid & ~outside_occupied & ~own_occupied
    return jnp.mean(unoccupied.astype(state.E.dtype), axis=-1)


def propose_molecule_moves(
    key: jax.Array,
    state: State,
    params: Params,
    static: StaticConfig,
    unresolved: jax.Array | None = None,
    metadata: MoleculeMetadata | None = None,
) -> MoleculeProposals:
    """Sample/evaluate one proposal for each unresolved molecule."""

    root = jnp.asarray(state.root, dtype=jnp.int32)
    count, n = root.shape[0], static.n
    metadata = molecule_metadata(root) if metadata is None else metadata
    unresolved = metadata.is_root if unresolved is None else unresolved
    active = jnp.asarray(unresolved, dtype=jnp.bool_) & metadata.is_root
    keys = jax.random.split(key, 7)
    translate = (
        jax.random.uniform(keys[0], (count,), dtype=state.E.dtype)
        < params.translation_probability
    ) | (metadata.segment_size == 1)
    direction = jax.random.randint(keys[1], (count,), 0, 4, dtype=jnp.int32)
    pivot = _uniform_component_pivots(keys[2], root)
    quarter_turns = jnp.where(
        jax.random.bernoulli(keys[3], 0.5, (count,)), 1, -1
    ).astype(jnp.int32)
    internal_channel = ~jax.random.bernoulli(
        keys[4], params.bath_channel_probability, (count,)
    )

    particle_translate = translate[root]
    particle_direction = direction[root]
    translation_R = state.R + DIRECTIONS[particle_direction]
    translation_valid = (translation_R[:, 1] >= 0) & (translation_R[:, 1] < n)
    translation_R = translation_R.at[:, 0].set(jnp.mod(translation_R[:, 0], n))
    translation_R = translation_R.at[:, 1].set(jnp.clip(translation_R[:, 1], 0, n - 1))

    pivot_R = state.R[pivot[root]]
    displacement, unique_lift = minimum_image_displacement(state.R, pivot_R, n)
    particle_turn = quarter_turns[root]
    dx, dy = displacement[:, 0], displacement[:, 1]
    rotated = jnp.stack(
        (
            jnp.where(particle_turn > 0, -dy, dy),
            jnp.where(particle_turn > 0, dx, -dx),
        ),
        axis=-1,
    )
    rotation_R = pivot_R + rotated
    rotation_valid = (rotation_R[:, 1] >= 0) & (rotation_R[:, 1] < n)
    rotation_R = rotation_R.at[:, 0].set(jnp.mod(rotation_R[:, 0], n))
    rotation_R = rotation_R.at[:, 1].set(jnp.clip(rotation_R[:, 1], 0, n - 1))
    # Minimum-image lifting must be unique in both states. Checking only the
    # source state admits one-way rotations: a vertical molecule can rotate
    # into a configuration containing an antipodal (n/2) horizontal offset,
    # while its nominal reverse is rejected during proposal construction.
    _, reverse_unique_lift = minimum_image_displacement(rotation_R, pivot_R, n)
    candidate_R = jnp.where(particle_translate[:, None], translation_R, rotation_R)
    candidate_bonds = jnp.where(
        particle_translate,
        state.bonds,
        jax.vmap(rotate_bond_mask)(state.bonds, particle_turn),
    ).astype(state.bonds.dtype)

    particle_geometry_valid = jnp.where(
        particle_translate,
        translation_valid,
        rotation_valid & unique_lift & reverse_unique_lift,
    )
    geometry_valid = _segment_all(particle_geometry_valid, root)
    unique_sites = _root_site_uniqueness(candidate_R, root, n)
    bonds_preserved = bond_geometry_preserved(
        state, candidate_R, translate, quarter_turns, n
    )
    occupant = state.L[candidate_R[:, 0], candidate_R[:, 1]]
    padded_root = jnp.concatenate((root, jnp.asarray((count,), jnp.int32)))
    safe_occupant = jnp.minimum(occupant, count)
    collision_free = _segment_all(
        (occupant == count) | (padded_root[safe_occupant] == root), root
    )
    legal = active & geometry_valid & unique_sites & bonds_preserved & collision_free

    delta_u, member_delta, effects = changed_contact_effects(
        state, candidate_R, internal_channel, active, params.eta
    )
    entropy_change, floor_valid = effect_thermodynamics(
        effects, state.E, params.heat_capacity, params.energy_floor
    )
    component_energy = _segment_sum(state.E, root)
    component_after = component_energy + _segment_sum(member_delta, root)
    size = jnp.maximum(metadata.segment_size, 1).astype(state.E.dtype)
    heat_capacity = jnp.asarray(params.heat_capacity, state.E.dtype)
    temperature_before = component_energy / (heat_capacity * size)
    temperature_after = component_after / (heat_capacity * size)
    symmetric_temperature = 0.5 * (temperature_before + temperature_after)

    exposure_before = exposure_fraction(state.R, state.L, count).astype(state.E.dtype)
    exposure_after = _post_move_exposure(state, candidate_R, root, n)
    gamma_before = jnp.asarray(params.gamma_base, state.E.dtype) + jnp.asarray(
        params.gamma_exposure, state.E.dtype
    ) * exposure_before
    gamma_after = jnp.asarray(params.gamma_base, state.E.dtype) + jnp.asarray(
        params.gamma_exposure, state.E.dtype
    ) * exposure_after
    translation_drag = 0.5 * (
        _segment_sum(gamma_before, root) + _segment_sum(gamma_after, root)
    )
    radius_squared = jnp.sum(displacement.astype(state.E.dtype) ** 2, axis=-1)
    rotation_drag = 0.5 * (
        _segment_sum(gamma_before * radius_squared, root)
        + _segment_sum(gamma_after * radius_squared, root)
    )
    tiny = jnp.finfo(state.E.dtype).tiny
    common = symmetric_temperature / jnp.asarray(params.bath_temperature, state.E.dtype)
    translation_kinetic = jnp.minimum(
        1.0,
        jnp.asarray(params.translation_frequency, state.E.dtype)
        * common
        * jnp.asarray(params.translation_drag_reference, state.E.dtype)
        / jnp.maximum(translation_drag, tiny),
    )
    rotation_kinetic = jnp.minimum(
        1.0,
        jnp.asarray(params.rotation_frequency, state.E.dtype)
        * common
        * jnp.asarray(params.rotation_drag_reference, state.E.dtype)
        / jnp.maximum(rotation_drag, tiny),
    )
    kinetic = jnp.where(translate, translation_kinetic, rotation_kinetic)
    affinity = jnp.where(
        internal_channel,
        entropy_change,
        bath_channel_affinity(delta_u, params.bath_temperature),
    )
    log_acceptance = kinetic_metropolis_log_probability(affinity, kinetic)
    uniform = jax.random.uniform(
        keys[5], (count,), dtype=state.E.dtype, minval=tiny, maxval=1.0
    )
    accepted = legal & (~internal_channel | floor_valid) & (
        jnp.log(uniform) < log_acceptance
    )
    return MoleculeProposals(
        active=active,
        translate=translate,
        direction=direction,
        pivot=pivot,
        quarter_turns=quarter_turns,
        internal_channel=internal_channel,
        candidate_R=candidate_R,
        candidate_bonds=candidate_bonds,
        legal=legal,
        potential_energy_change=delta_u,
        entropy_change=entropy_change,
        kinetic_probability=kinetic,
        log_acceptance_probability=log_acceptance,
        accepted=accepted,
        effects=effects,
    )


sample_molecule_proposals = propose_molecule_moves
propose_moves = propose_molecule_moves


def apply_selected_molecule_moves(
    state: State,
    proposals: MoleculeProposals,
    selected: jax.Array,
    static: StaticConfig,
) -> State:
    """Apply a nonconflicting accepted-root set simultaneously."""

    selected = jnp.asarray(selected, dtype=jnp.bool_)

    def apply_moves(_: None) -> State:
        particle_selected = selected[state.root]
        positions = jnp.where(
            particle_selected[:, None], proposals.candidate_R, state.R
        )
        bonds = jnp.where(
            particle_selected, proposals.candidate_bonds, state.bonds
        ).astype(state.bonds.dtype)
        effect_selected = proposals.effects.valid & selected[proposals.effects.root]
        energy_delta = jnp.zeros_like(state.E).at[proposals.effects.particle].add(
            jnp.where(effect_selected, proposals.effects.delta, 0)
        )

        # Update the existing occupancy in O(N) scatter work instead of filling
        # a fresh n-by-n lattice on every optimistic retry.  Clearing all moved
        # source sites before scattering every final particle id handles
        # rotations whose source and destination footprints overlap.
        particle_ids = jnp.arange(state.R.shape[0], dtype=state.L.dtype)
        source_values = jnp.where(
            particle_selected,
            jnp.asarray(static.empty, dtype=state.L.dtype),
            particle_ids,
        )
        lattice = state.L.at[state.R[:, 0], state.R[:, 1]].set(source_values)
        lattice = lattice.at[positions[:, 0], positions[:, 1]].set(particle_ids)
        return state._replace(
            R=positions, E=state.E + energy_delta, L=lattice, bonds=bonds
        )

    return jax.lax.cond(jnp.any(selected), apply_moves, lambda _: state, None)


__all__ = [
    "CONTACT_EFFECT_RECORDS_PER_PARTICLE",
    "MoleculeMetadata",
    "ContactEffects",
    "MoleculeProposals",
    "molecule_metadata",
    "build_molecule_metadata",
    "root_segment_metadata",
    "minimum_image_displacement",
    "lexicographic_pair_membership",
    "bond_geometry_preserved",
    "internal_geometry_preserved",
    "changed_contact_effects",
    "effect_thermodynamics",
    "propose_molecule_moves",
    "sample_molecule_proposals",
    "propose_moves",
    "apply_selected_molecule_moves",
]
