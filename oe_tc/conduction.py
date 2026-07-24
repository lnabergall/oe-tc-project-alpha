"""Locally balanced nearest-neighbor internal-energy conduction."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.geometry import DIRECTIONS, POS_X, POS_Y
from oe_tc.thermodynamics import kinetic_metropolis_log_probability
from oe_tc.topology import has_bond


NUM_CONDUCTION_CLASSES = 4


class ConductionProposal(NamedTuple):
    """Thermodynamic data for one signed pair-exchange proposal."""

    source_energy: jax.Array
    target_energy: jax.Array
    flux: jax.Array
    affinity: jax.Array
    log_acceptance: jax.Array
    floor_valid: jax.Array
    coefficient: jax.Array


class ConductionClassResult(NamedTuple):
    """One matching-class update and its signed edge/particle fluxes."""

    energy: jax.Array
    particle_energy_change: jax.Array
    edge_flux: jax.Array
    edge_target: jax.Array
    active: jax.Array
    accepted: jax.Array
    signs: jax.Array
    affinity: jax.Array
    log_acceptance: jax.Array
    floor_valid: jax.Array
    heat: jax.Array
    throughput: jax.Array


class ConductionResult(NamedTuple):
    """A complete randomized four-matching stochastic conduction pass."""

    energy: jax.Array
    class_energy: jax.Array
    particle_energy_change: jax.Array
    edge_flux: jax.Array
    edge_target: jax.Array
    active: jax.Array
    accepted: jax.Array
    signs: jax.Array
    affinity: jax.Array
    log_acceptance: jax.Array
    floor_valid: jax.Array
    class_order: jax.Array
    class_heat: jax.Array
    class_throughput: jax.Array
    heat: jax.Array
    throughput: jax.Array


def pair_exchange_proposal(
    source_energy: jax.Array,
    target_energy: jax.Array,
    bonded: jax.Array,
    signs: jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
) -> ConductionProposal:
    """Return one signed, energy-conserving proposal per adjacent pair.

    A positive sign transfers one quantum from the canonical source endpoint
    to its target; a negative sign transfers it in the reverse direction.
    The symmetric contact/bond coefficient changes kinetics only. The
    Metropolis factor is set by the finite-reservoir entropy change, so the
    forward/reverse probability ratio obeys local detailed balance.
    """

    source = jnp.asarray(source_energy)
    target = jnp.asarray(target_energy, dtype=source.dtype)
    signs = jnp.asarray(signs, dtype=source.dtype)
    quantum = jnp.asarray(energy_quantum, dtype=source.dtype)
    floor = jnp.asarray(energy_floor, dtype=source.dtype)
    flux = signs * quantum
    source_candidate = source - flux
    target_candidate = target + flux
    floor_valid = (
        jnp.isfinite(source_candidate)
        & jnp.isfinite(target_candidate)
        & (source_candidate >= floor)
        & (target_candidate >= floor)
        & (source_candidate != source)
        & (target_candidate != target)
    )

    # Substitute unchanged energies before taking logarithms so invalid
    # proposals never expose XLA to nonpositive or nonfinite log arguments.
    safe_source = jnp.where(floor_valid, source_candidate, source)
    safe_target = jnp.where(floor_valid, target_candidate, target)
    capacity = jnp.asarray(heat_capacity, dtype=source.dtype)
    affinity = capacity * (
        jnp.log(safe_source / source) + jnp.log(safe_target / target)
    )
    coefficient = jnp.where(
        bonded,
        jnp.asarray(bond_coefficient, dtype=source.dtype),
        jnp.asarray(contact_coefficient, dtype=source.dtype),
    )
    log_acceptance = kinetic_metropolis_log_probability(affinity, coefficient)
    log_acceptance = jnp.where(
        floor_valid,
        log_acceptance,
        jnp.asarray(-jnp.inf, dtype=source.dtype),
    )
    return ConductionProposal(
        source_energy=source_candidate,
        target_energy=target_candidate,
        flux=flux,
        affinity=affinity,
        log_acceptance=log_acceptance,
        floor_valid=floor_valid,
        coefficient=coefficient,
    )


def _matching_edges(
    positions: jax.Array,
    lattice: jax.Array,
    class_id: int | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return active mask, safe target IDs, and positive edge direction."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    num_particles = positions.shape[0]
    n = lattice.shape[0]
    class_id = jnp.asarray(class_id, dtype=jnp.int32)
    horizontal = class_id < 2
    direction = jnp.where(horizontal, POS_X, POS_Y).astype(jnp.int32)
    candidate = positions + DIRECTIONS[direction]
    candidate = candidate.at[:, 0].set(jnp.mod(candidate[:, 0], n))
    vertical_valid = candidate[:, 1] < n
    safe_y = jnp.clip(candidate[:, 1], 0, n - 1)
    target = lattice[candidate[:, 0], safe_y]

    parity = jnp.bitwise_and(positions[:, 0] + positions[:, 1], 1)
    correct_class = parity == jnp.bitwise_and(class_id, 1)
    active = correct_class & (horizontal | vertical_valid) & (target < num_particles)
    safe_target = jnp.where(active, target, 0)
    return active, safe_target, direction


def conduction_class_from_draws(
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    bonds: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    class_id: int | jax.Array,
    signs: jax.Array,
    uniforms: jax.Array,
) -> ConductionClassResult:
    """Apply one matching class using caller-supplied random variates."""

    energy = jnp.asarray(energy)
    signs = jnp.asarray(signs, dtype=energy.dtype)
    uniforms = jnp.asarray(uniforms, dtype=energy.dtype)
    if signs.shape != energy.shape or uniforms.shape != energy.shape:
        raise ValueError("signs and uniforms must match the energy shape")

    num_particles = energy.shape[0]
    particle_ids = jnp.arange(num_particles, dtype=jnp.int32)
    active, safe_target, direction = _matching_edges(
        positions, lattice, class_id
    )
    bonded = has_bond(bonds, direction)
    proposal = pair_exchange_proposal(
        energy,
        energy[safe_target],
        bonded,
        signs,
        energy_quantum,
        energy_floor,
        heat_capacity,
        contact_coefficient,
        bond_coefficient,
    )
    tiny = jnp.finfo(energy.dtype).tiny
    accepted = active & (
        jnp.log(jnp.maximum(uniforms, tiny)) < proposal.log_acceptance
    )
    flux = jnp.where(accepted, proposal.flux, jnp.zeros_like(energy))
    particle_change = jnp.zeros_like(energy)
    particle_change = particle_change.at[particle_ids].add(-flux)
    particle_change = particle_change.at[safe_target].add(flux)
    return ConductionClassResult(
        energy=energy + particle_change,
        particle_energy_change=particle_change,
        edge_flux=flux,
        edge_target=jnp.where(active, safe_target, num_particles),
        active=active,
        accepted=accepted,
        signs=signs,
        affinity=jnp.where(active, proposal.affinity, jnp.zeros_like(energy)),
        log_acceptance=jnp.where(
            active,
            proposal.log_acceptance,
            jnp.asarray(-jnp.inf, dtype=energy.dtype),
        ),
        floor_valid=active & proposal.floor_valid,
        heat=jnp.sum(particle_change),
        throughput=jnp.sum(jnp.abs(flux)),
    )


def conduction_class(
    key: jax.Array,
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    bonds: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    class_id: int | jax.Array,
) -> ConductionClassResult:
    """Sample and apply all pair exchanges in one static matching."""

    energy = jnp.asarray(energy)
    draws = jax.random.uniform(key, energy.shape, dtype=energy.dtype)
    positive = draws < 0.5
    signs = jnp.where(positive, 1, -1).astype(energy.dtype)
    # Rescaling either half interval supplies the conditional acceptance
    # uniform, so direction and acceptance need only one random array.
    uniforms = jnp.where(positive, 2.0 * draws, 2.0 * draws - 1.0)
    return conduction_class_from_draws(
        energy,
        positions,
        lattice,
        bonds,
        contact_coefficient,
        bond_coefficient,
        energy_quantum,
        energy_floor,
        heat_capacity,
        class_id,
        signs,
        uniforms,
    )


conduct_class = conduction_class


def conduction_sweep(
    key: jax.Array,
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    bonds: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    class_order: jax.Array | None = None,
) -> ConductionResult:
    """Apply four stochastic matching classes sequentially."""

    initial_energy = jnp.asarray(energy)
    if class_order is None:
        class_order = jnp.arange(NUM_CONDUCTION_CLASSES, dtype=jnp.int32)
    else:
        class_order = jnp.asarray(class_order, dtype=jnp.int32)
    if class_order.shape != (NUM_CONDUCTION_CLASSES,):
        raise ValueError("class_order must have shape (4,)")
    class_keys = jax.random.split(key, NUM_CONDUCTION_CLASSES)

    def apply_class(
        current_energy: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, ConductionClassResult]:
        class_id, class_key = inputs
        result = conduction_class(
            class_key,
            current_energy,
            positions,
            lattice,
            bonds,
            contact_coefficient,
            bond_coefficient,
            energy_quantum,
            energy_floor,
            heat_capacity,
            class_id,
        )
        return result.energy, result

    final_energy, phases = jax.lax.scan(
        apply_class, initial_energy, (class_order, class_keys)
    )
    return ConductionResult(
        energy=final_energy,
        class_energy=phases.energy,
        particle_energy_change=phases.particle_energy_change,
        edge_flux=phases.edge_flux,
        edge_target=phases.edge_target,
        active=phases.active,
        accepted=phases.accepted,
        signs=phases.signs,
        affinity=phases.affinity,
        log_acceptance=phases.log_acceptance,
        floor_valid=phases.floor_valid,
        class_order=class_order,
        class_heat=phases.heat,
        class_throughput=phases.throughput,
        heat=jnp.sum(final_energy) - jnp.sum(initial_energy),
        throughput=jnp.sum(phases.throughput),
    )


conduct = conduction_sweep


def randomized_conduction_sweep(
    key: jax.Array,
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    bonds: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
) -> ConductionResult:
    """Apply one pass with a uniformly permuted matching-class order."""

    order_key, exchange_key = jax.random.split(key)
    order = jax.random.permutation(
        order_key,
        jnp.arange(NUM_CONDUCTION_CLASSES, dtype=jnp.int32),
        independent=True,
    )
    return conduction_sweep(
        exchange_key,
        energy,
        positions,
        lattice,
        bonds,
        contact_coefficient,
        bond_coefficient,
        energy_quantum,
        energy_floor,
        heat_capacity,
        order,
    )


__all__ = [
    "NUM_CONDUCTION_CLASSES",
    "ConductionProposal",
    "ConductionClassResult",
    "ConductionResult",
    "pair_exchange_proposal",
    "conduction_class_from_draws",
    "conduction_class",
    "conduct_class",
    "conduction_sweep",
    "conduct",
    "randomized_conduction_sweep",
]
