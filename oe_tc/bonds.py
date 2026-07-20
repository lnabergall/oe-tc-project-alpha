"""Persistent bond-flip dynamics for OE-TC model.

Candidate lattice edges are partitioned into four fixed classes: horizontal
or vertical orientation, crossed with the checkerboard parity of the canonical
(+x or +y) source site.  For even lattice width each class is a matching and
no edge in a class occurs in another edge's two-edge catalytic stencil.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.config import Params, StaticConfig
from oe_tc.state import State
from oe_tc.topology import (
    DIRECTION_OFFSETS,
    OPPOSITE_DIRECTIONS,
    POS_X,
    POS_Y,
    connected_components_with_status,
    direction_bit,
    gather_lattice_neighbors,
    has_bond,
)


NUM_BOND_CLASSES = 4
HORIZONTAL = 0
VERTICAL = 1


class CandidateEdges(NamedTuple):
    """Fixed-size candidate edge batch for one parallel class."""

    source: jax.Array
    target: jax.Array
    direction: jax.Array
    valid: jax.Array


class BondClassMetrics(NamedTuple):
    """Diagnostics accumulated by one edge-class subphase."""

    accepted_flips: jax.Array
    bath_energy: jax.Array
    configurational_delta: jax.Array


class BondPhaseMetrics(NamedTuple):
    """Compact diagnostics for one complete four-class bond phase."""

    accepted_flips: jax.Array
    bath_energy: jax.Array
    configurational_delta: jax.Array
    component_iterations: jax.Array
    components_converged: jax.Array
    class_order: jax.Array


class BondProposalTerms(NamedTuple):
    """Thermodynamic and kinetic terms for candidate flips."""

    delta_u: jax.Array
    endpoint_energy: jax.Array
    affinity: jax.Array
    log_kinetic: jax.Array
    log_acceptance: jax.Array
    internal_legal: jax.Array


def edge_class_masks(positions: jax.Array, n: int | None = None) -> jax.Array:
    """Return four orientation/checkerboard class masks, shape ``(4,N)``."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    parity = jnp.bitwise_and(positions[:, 0] + positions[:, 1], 1)
    parity_masks = jnp.stack((parity == 0, parity == 1), axis=0)
    horizontal = parity_masks
    vertical = parity_masks
    if n is not None:
        vertical = vertical & (positions[None, :, 1] < n - 1)
    return jnp.concatenate((horizontal, vertical), axis=0)


def edge_class_indices(positions: jax.Array) -> jax.Array:
    """Return horizontal and vertical class ids for every canonical source."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    parity = jnp.bitwise_and(positions[:, 0] + positions[:, 1], 1)
    return jnp.stack((parity, parity + 2), axis=-1)


def candidate_edges_for_class(
    occupancy: jax.Array,
    positions: jax.Array,
    class_index: int | jax.Array,
    empty: int | None = None,
) -> CandidateEdges:
    """Construct fixed-size occupied contacts in one bond edge class."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    num_particles = positions.shape[0]
    if empty is None:
        empty = num_particles
    class_index = jnp.asarray(class_index, dtype=jnp.int32)
    orientation = class_index // 2
    parity = jnp.bitwise_and(class_index, 1)
    direction = jnp.where(orientation == HORIZONTAL, POS_X, POS_Y)

    neighbors = gather_lattice_neighbors(occupancy, positions, empty)
    target = neighbors[:, direction]
    source = jnp.arange(num_particles, dtype=jnp.int32)
    in_class = (
        jnp.bitwise_and(positions[:, 0] + positions[:, 1], 1) == parity
    )
    occupied = (target >= 0) & (target < num_particles)
    return CandidateEdges(source, target, direction, in_class & occupied)


def _safe_lattice_gather(
    occupancy: jax.Array, coordinates: jax.Array, valid: jax.Array, empty: int
) -> jax.Array:
    """Gather L[x,y] after periodic-x/closed-y normalization."""

    n = occupancy.shape[0]
    x = jnp.mod(coordinates[..., 0], n)
    y_valid = (coordinates[..., 1] >= 0) & (coordinates[..., 1] < n)
    y = jnp.clip(coordinates[..., 1], 0, n - 1)
    gathered = occupancy[x, y]
    return jnp.where(valid & y_valid, gathered, jnp.asarray(empty, gathered.dtype))


def catalysis_count(
    occupancy: jax.Array,
    positions: jax.Array,
    bonds: jax.Array,
    direction: int | jax.Array,
    empty: int | None = None,
) -> jax.Array:
    """Count the exact two parallel bonds at perpendicular distance one."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    bonds = jnp.asarray(bonds)
    num_particles = bonds.shape[0]
    if empty is None:
        empty = num_particles
    direction = jnp.asarray(direction, dtype=jnp.int32)

    horizontal_offsets = jnp.asarray(((0, 1), (0, -1)), dtype=jnp.int32)
    vertical_offsets = jnp.asarray(((1, 0), (-1, 0)), dtype=jnp.int32)
    perpendicular = jnp.where(
        direction == POS_X, horizontal_offsets, vertical_offsets
    )
    anchors = positions[:, None, :] + perpendicular[None, :, :]
    endpoints = anchors + DIRECTION_OFFSETS[direction]
    all_valid = jnp.ones(anchors.shape[:-1], dtype=bool)
    anchor_ids = _safe_lattice_gather(occupancy, anchors, all_valid, empty)
    endpoint_ids = _safe_lattice_gather(occupancy, endpoints, all_valid, empty)
    occupied = (
        (anchor_ids >= 0)
        & (anchor_ids < num_particles)
        & (endpoint_ids >= 0)
        & (endpoint_ids < num_particles)
    )

    padded = jnp.concatenate((bonds, jnp.zeros((1,), dtype=bonds.dtype)))
    safe_anchor = jnp.where(occupied, anchor_ids, num_particles)
    safe_endpoint = jnp.where(occupied, endpoint_ids, num_particles)
    forward = has_bond(padded[safe_anchor], direction)
    reverse = has_bond(
        padded[safe_endpoint], OPPOSITE_DIRECTIONS[direction]
    )
    return jnp.sum(occupied & forward & reverse, axis=-1, dtype=jnp.int32)


parallel_bond_count = catalysis_count
catalytic_stencil_count = catalysis_count


def bond_energy_delta(is_bonded: jax.Array, eta: float | jax.Array) -> jax.Array:
    """Configurational energy change of toggling a contact bond."""

    dtype = jnp.result_type(eta, jnp.float32)
    indicator = jnp.asarray(is_bonded, dtype=dtype)
    return (2.0 * indicator - 1.0) * (1.0 - jnp.asarray(eta, dtype=dtype))


def internal_endpoint_update(
    endpoint_energy: jax.Array, delta_u: jax.Array
) -> jax.Array:
    """Exchange ``-delta_u`` equally between the two bond endpoints."""

    return endpoint_energy - delta_u[..., None] / 2.0


def bond_proposal_terms(
    endpoint_energy: jax.Array,
    is_bonded: jax.Array,
    catalyst_count: jax.Array,
    bath_channel: jax.Array,
    params: Params,
) -> BondProposalTerms:
    """Evaluate channel-resolved acceptance entirely in log space."""

    endpoint_energy = jnp.asarray(endpoint_energy)
    dtype = endpoint_energy.dtype
    bath_channel = jnp.asarray(bath_channel)
    delta_u = bond_energy_delta(is_bonded, params.eta).astype(dtype)
    internal_energy = internal_endpoint_update(endpoint_energy, delta_u)
    internal_legal = jnp.all(internal_energy >= params.energy_floor, axis=-1)
    proposed_energy = jnp.where(
        bath_channel[..., None], endpoint_energy, internal_energy
    )

    tiny = jnp.finfo(dtype).tiny
    safe_before = jnp.maximum(endpoint_energy, tiny)
    safe_after = jnp.maximum(internal_energy, tiny)
    entropy_delta = params.heat_capacity * jnp.sum(
        jnp.log(safe_after) - jnp.log(safe_before), axis=-1
    )
    bath_affinity = -delta_u / params.bath_temperature
    affinity = jnp.where(bath_channel, bath_affinity, entropy_delta)

    temperature_before = jnp.sum(endpoint_energy, axis=-1) / (
        2.0 * params.heat_capacity
    )
    temperature_after = jnp.sum(proposed_energy, axis=-1) / (
        2.0 * params.heat_capacity
    )
    symmetric_temperature = (temperature_before + temperature_after) / 2.0

    log_catalysis = jnp.minimum(
        jnp.log(jnp.asarray(params.catalysis_cap, dtype=dtype)),
        jnp.asarray(params.catalysis_strength, dtype=dtype)
        * jnp.asarray(catalyst_count, dtype=dtype),
    )
    log_raw_kinetic = (
        jnp.log(jnp.asarray(params.bond_frequency, dtype=dtype))
        + jnp.log(jnp.maximum(symmetric_temperature, tiny))
        - jnp.log(jnp.asarray(params.bath_temperature, dtype=dtype))
        + log_catalysis
    )
    log_kinetic = jnp.minimum(jnp.asarray(0.0, dtype=dtype), log_raw_kinetic)
    log_acceptance = log_kinetic + jnp.minimum(
        jnp.asarray(0.0, dtype=dtype), affinity
    )
    log_acceptance = jnp.where(
        bath_channel | internal_legal, log_acceptance, -jnp.inf
    )
    return BondProposalTerms(
        delta_u,
        proposed_energy,
        affinity,
        log_kinetic,
        log_acceptance,
        internal_legal,
    )


def _apply_flips(
    energy: jax.Array,
    bonds: jax.Array,
    edges: CandidateEdges,
    accepted: jax.Array,
    bath_channel: jax.Array,
    delta_u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply a matching of flips with symmetric endpoint updates."""

    num_particles = bonds.shape[0]
    bond_dtype = bonds.dtype
    safe_target = jnp.where(edges.valid, edges.target, 0)
    endpoint_delta = jnp.where(
        accepted & ~bath_channel, -delta_u / 2.0, 0.0
    )
    target_energy_delta = jnp.zeros_like(energy).at[safe_target].add(endpoint_delta)
    energy = energy + endpoint_delta + target_energy_delta

    source_toggle = jnp.where(
        accepted, direction_bit(edges.direction).astype(jnp.int32), 0
    )
    target_toggle = jnp.zeros((num_particles,), dtype=jnp.int32).at[safe_target].add(
        jnp.where(
            accepted,
            direction_bit(OPPOSITE_DIRECTIONS[edges.direction]).astype(jnp.int32),
            0,
        )
    )
    int_bonds = bonds.astype(jnp.int32)
    bonds = jnp.bitwise_xor(jnp.bitwise_xor(int_bonds, source_toggle), target_toggle)
    return energy, bonds.astype(bond_dtype)


def update_bond_class(
    key: jax.Array,
    positions: jax.Array,
    energy: jax.Array,
    occupancy: jax.Array,
    bonds: jax.Array,
    class_index: int | jax.Array,
    params: Params,
    empty: int | None = None,
) -> tuple[jax.Array, jax.Array, BondClassMetrics]:
    """Sample one independent edge class in parallel."""

    num_particles = bonds.shape[0]
    if empty is None:
        empty = num_particles
    edges = candidate_edges_for_class(occupancy, positions, class_index, empty)
    safe_target = jnp.where(edges.valid, edges.target, 0)

    source_bonded = has_bond(bonds, edges.direction)
    target_bonded = has_bond(
        bonds[safe_target], OPPOSITE_DIRECTIONS[edges.direction]
    )
    valid = edges.valid & (source_bonded == target_bonded)
    endpoint_energy = jnp.stack((energy, energy[safe_target]), axis=-1)
    catalyst_count = catalysis_count(
        occupancy, positions, bonds, edges.direction, empty
    )

    channel_key, accept_key = jax.random.split(key)
    bath_channel = jax.random.bernoulli(
        channel_key,
        p=jnp.asarray(params.bath_channel_probability),
        shape=(num_particles,),
    )
    terms = bond_proposal_terms(
        endpoint_energy,
        source_bonded,
        catalyst_count,
        bath_channel,
        params,
    )
    uniform = jax.random.uniform(
        accept_key,
        shape=(num_particles,),
        minval=jnp.finfo(energy.dtype).tiny,
        maxval=1.0,
        dtype=energy.dtype,
    )
    accepted = valid & (jnp.log(uniform) < terms.log_acceptance)

    valid_edges = CandidateEdges(edges.source, edges.target, edges.direction, valid)
    energy, bonds = _apply_flips(
        energy, bonds, valid_edges, accepted, bath_channel, terms.delta_u
    )
    accepted_delta = jnp.where(accepted, terms.delta_u, 0.0)
    metrics = BondClassMetrics(
        jnp.sum(accepted, dtype=jnp.int32),
        jnp.sum(jnp.where(accepted & bath_channel, terms.delta_u, 0.0)),
        jnp.sum(accepted_delta),
    )
    return energy, bonds, metrics


bond_class_update = update_bond_class


def bond_phase_arrays(
    key: jax.Array,
    positions: jax.Array,
    energy: jax.Array,
    occupancy: jax.Array,
    bonds: jax.Array,
    params: Params,
    static: StaticConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, BondPhaseMetrics]:
    """Run four random-ordered classes, then recompute components once."""

    split_keys = jax.random.split(key, NUM_BOND_CLASSES + 1)
    class_order = jax.random.permutation(split_keys[0], NUM_BOND_CLASSES)
    class_keys = split_keys[1:]

    def scan_class(
        carry: tuple[jax.Array, jax.Array],
        item: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], BondClassMetrics]:
        current_energy, current_bonds = carry
        class_index, class_key = item
        next_energy, next_bonds, metrics = update_bond_class(
            class_key,
            positions,
            current_energy,
            occupancy,
            current_bonds,
            class_index,
            params,
            static.empty,
        )
        return (next_energy, next_bonds), metrics

    (energy, bonds), class_metrics = jax.lax.scan(
        scan_class, (energy, bonds), (class_order, class_keys)
    )

    root, component_iterations, components_converged = (
        connected_components_with_status(
            occupancy,
            positions,
            bonds,
            max_iters=static.component_max_iters,
            empty=static.empty,
        )
    )
    metrics = BondPhaseMetrics(
        jnp.sum(class_metrics.accepted_flips, dtype=jnp.int32),
        jnp.sum(class_metrics.bath_energy),
        jnp.sum(class_metrics.configurational_delta),
        component_iterations,
        components_converged,
        class_order,
    )
    return energy, bonds, root, metrics


def bond_phase(
    key: jax.Array,
    state: State,
    params: Params,
    static: StaticConfig,
) -> tuple[State, BondPhaseMetrics]:
    """State-level wrapper for one complete bond dynamics phase."""

    energy, bonds, root, metrics = bond_phase_arrays(
        key, state.R, state.E, state.L, state.bonds, params, static
    )
    return state._replace(E=energy, bonds=bonds, root=root), metrics


__all__ = [
    "NUM_BOND_CLASSES",
    "HORIZONTAL",
    "VERTICAL",
    "CandidateEdges",
    "BondClassMetrics",
    "BondPhaseMetrics",
    "BondProposalTerms",
    "edge_class_masks",
    "edge_class_indices",
    "candidate_edges_for_class",
    "catalysis_count",
    "parallel_bond_count",
    "catalytic_stencil_count",
    "bond_energy_delta",
    "internal_endpoint_update",
    "bond_proposal_terms",
    "update_bond_class",
    "bond_class_update",
    "bond_phase_arrays",
    "bond_phase",
]
