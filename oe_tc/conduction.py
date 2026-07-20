"""Four-class nearest-neighbor energy conduction."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.geometry import DIRECTIONS, POS_X, POS_Y
from oe_tc.topology import has_bond


NUM_CONDUCTION_CLASSES = 4


class ConductionClassResult(NamedTuple):
    """One matching-class update and its signed edge/particle fluxes."""

    energy: jax.Array
    particle_energy_change: jax.Array
    edge_flux: jax.Array
    edge_target: jax.Array
    active: jax.Array
    heat: jax.Array


class ConductionResult(NamedTuple):
    """A complete four-class pass with explicit intermediate energy fluxes."""

    energy: jax.Array
    class_energy: jax.Array
    particle_energy_change: jax.Array
    edge_flux: jax.Array
    edge_target: jax.Array
    active: jax.Array
    class_order: jax.Array
    class_heat: jax.Array
    heat: jax.Array


def pair_energy_flux(
    source_energy: jax.Array,
    target_energy: jax.Array,
    bonded: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
) -> jax.Array:
    """Return signed energy transferred from source to target on each edge."""

    source_energy = jnp.asarray(source_energy)
    target_energy = jnp.asarray(target_energy, dtype=source_energy.dtype)
    coefficient = jnp.where(
        bonded,
        jnp.asarray(bond_coefficient, dtype=source_energy.dtype),
        jnp.asarray(contact_coefficient, dtype=source_energy.dtype),
    )
    return 0.5 * coefficient * (source_energy - target_energy)


def conduction_class(
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    bonds: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
    class_id: int | jax.Array,
) -> ConductionClassResult:
    """Update all occupied edges in one static matching from one snapshot."""

    energy = jnp.asarray(energy)
    positions = jnp.asarray(positions, dtype=jnp.int32)
    bonds = jnp.asarray(bonds)
    num_particles = energy.shape[0]
    n = lattice.shape[0]
    particle_ids = jnp.arange(num_particles, dtype=jnp.int32)

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

    bonded = has_bond(bonds, direction)
    flux = pair_energy_flux(
        energy,
        energy[safe_target],
        bonded,
        contact_coefficient,
        bond_coefficient,
    )
    flux = jnp.where(active, flux, jnp.zeros_like(flux))
    particle_change = jnp.zeros_like(energy)
    particle_change = particle_change.at[particle_ids].add(-flux)
    particle_change = particle_change.at[safe_target].add(flux)
    return ConductionClassResult(
        energy=energy + particle_change,
        particle_energy_change=particle_change,
        edge_flux=flux,
        edge_target=jnp.where(active, target, num_particles),
        active=active,
        heat=jnp.sum(particle_change),
    )


conduct_class = conduction_class


def conduction_sweep(
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    bonds: jax.Array,
    contact_coefficient: float | jax.Array,
    bond_coefficient: float | jax.Array,
    class_order: jax.Array | None = None,
) -> ConductionResult:
    """Apply four matching classes sequentially in ``class_order``."""

    initial_energy = jnp.asarray(energy)
    if class_order is None:
        class_order = jnp.arange(NUM_CONDUCTION_CLASSES, dtype=jnp.int32)
    else:
        class_order = jnp.asarray(class_order, dtype=jnp.int32)
    if class_order.shape != (NUM_CONDUCTION_CLASSES,):
        raise ValueError("class_order must have shape (4,)")

    def apply_class(
        current_energy: jax.Array, class_id: jax.Array
    ) -> tuple[jax.Array, ConductionClassResult]:
        result = conduction_class(
            current_energy,
            positions,
            lattice,
            bonds,
            contact_coefficient,
            bond_coefficient,
            class_id,
        )
        return result.energy, result

    final_energy, phases = jax.lax.scan(apply_class, initial_energy, class_order)
    return ConductionResult(
        energy=final_energy,
        class_energy=phases.energy,
        particle_energy_change=phases.particle_energy_change,
        edge_flux=phases.edge_flux,
        edge_target=phases.edge_target,
        active=phases.active,
        class_order=class_order,
        class_heat=phases.heat,
        heat=jnp.sum(final_energy) - jnp.sum(initial_energy),
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
) -> ConductionResult:
    """Apply one pass with a uniformly permuted matching-class order."""

    order = jax.random.permutation(
        key, jnp.arange(NUM_CONDUCTION_CLASSES, dtype=jnp.int32), independent=True
    )
    return conduction_sweep(
        energy,
        positions,
        lattice,
        bonds,
        contact_coefficient,
        bond_coefficient,
        order,
    )


__all__ = [
    "NUM_CONDUCTION_CLASSES",
    "ConductionClassResult",
    "ConductionResult",
    "pair_energy_flux",
    "conduction_class",
    "conduct_class",
    "conduction_sweep",
    "conduct",
    "randomized_conduction_sweep",
]
