"""Exposure-dependent drag and direct bath heat exchange."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.geometry import exposure_fraction
from oe_tc.thermodynamics import kinetic_metropolis_log_probability


class ExposureProperties(NamedTuple):
    """Per-particle exposure, viscous collision rate, and bath coupling."""

    fraction: jax.Array
    gamma: jax.Array
    kappa: jax.Array


class BathProposal(NamedTuple):
    """Deterministic thermodynamic data for signed bath proposals."""

    energy: jax.Array
    particle_heat: jax.Array
    affinity: jax.Array
    log_acceptance: jax.Array
    floor_valid: jax.Array
    kappa: jax.Array


class BathExchangeResult(NamedTuple):
    """Updated energies and explicit accepted direct-bath heat flux."""

    energy: jax.Array
    heat: jax.Array
    particle_heat: jax.Array
    accepted: jax.Array
    signs: jax.Array
    proposed_energy: jax.Array
    affinity: jax.Array
    log_acceptance: jax.Array
    exposure: jax.Array
    kappa: jax.Array


def collision_coefficient(
    exposure: jax.Array,
    gamma_base: float | jax.Array,
    gamma_exposure: float | jax.Array,
) -> jax.Array:
    """Return ``gamma_0 + gamma_1 * exposure``."""

    exposure = jnp.asarray(exposure)
    return jnp.asarray(gamma_base, exposure.dtype) + jnp.asarray(
        gamma_exposure, exposure.dtype
    ) * exposure


def thermal_coupling(
    exposure: jax.Array,
    kappa_base: float | jax.Array,
    kappa_exposure: float | jax.Array,
) -> jax.Array:
    """Return ``kappa_0 + kappa_1 * exposure``."""

    exposure = jnp.asarray(exposure)
    return jnp.asarray(kappa_base, exposure.dtype) + jnp.asarray(
        kappa_exposure, exposure.dtype
    ) * exposure


def exposure_properties(
    positions: jax.Array,
    lattice: jax.Array,
    gamma_base: float | jax.Array,
    gamma_exposure: float | jax.Array,
    kappa_base: float | jax.Array,
    kappa_exposure: float | jax.Array,
) -> ExposureProperties:
    """Evaluate all exposure-dependent coefficients from the geometry."""

    exposure = exposure_fraction(positions, lattice, empty=positions.shape[0])
    return ExposureProperties(
        fraction=exposure,
        gamma=collision_coefficient(exposure, gamma_base, gamma_exposure),
        kappa=thermal_coupling(exposure, kappa_base, kappa_exposure),
    )


def molecule_translation_drag(gamma: jax.Array, roots: jax.Array) -> jax.Array:
    """Segment-sum particle drag into a fixed ``N`` vector indexed by root."""

    gamma = jnp.asarray(gamma)
    roots = jnp.asarray(roots, dtype=jnp.int32)
    return jnp.zeros_like(gamma).at[roots].add(gamma)


def bath_proposal(
    energy: jax.Array,
    exposure: jax.Array,
    signs: jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    bath_temperature: float | jax.Array,
    kappa_base: float | jax.Array,
    kappa_exposure: float | jax.Array,
) -> BathProposal:
    """Compute signed direct-bath proposal affinities in log space."""

    energy = jnp.asarray(energy)
    exposure = jnp.asarray(exposure, dtype=energy.dtype)
    signs = jnp.asarray(signs, dtype=energy.dtype)
    quantum = jnp.asarray(energy_quantum, dtype=energy.dtype)
    floor = jnp.asarray(energy_floor, dtype=energy.dtype)
    candidate = energy + signs * quantum
    realized_heat = candidate - energy
    floor_valid = (
        jnp.isfinite(candidate)
        & jnp.isfinite(realized_heat)
        & (realized_heat != 0.0)
        & (candidate >= floor)
    )

    # Invalid and unrepresentable proposals are replaced before log1p so XLA
    # never evaluates entropy from a nonfinite or nonpositive candidate.
    particle_heat = jnp.where(
        floor_valid, realized_heat, jnp.zeros_like(realized_heat)
    )
    delta_entropy = jnp.asarray(heat_capacity, energy.dtype) * jnp.log1p(
        particle_heat / energy
    )
    affinity = delta_entropy - particle_heat / jnp.asarray(
        bath_temperature, energy.dtype
    )
    kappa = thermal_coupling(exposure, kappa_base, kappa_exposure)
    log_acceptance = kinetic_metropolis_log_probability(affinity, kappa)
    log_acceptance = jnp.where(
        floor_valid,
        log_acceptance,
        jnp.asarray(-jnp.inf, dtype=energy.dtype),
    )
    return BathProposal(
        energy=candidate,
        particle_heat=particle_heat,
        affinity=affinity,
        log_acceptance=log_acceptance,
        floor_valid=floor_valid,
        kappa=kappa,
    )


def heat_exchange_from_draws(
    energy: jax.Array,
    exposure: jax.Array,
    signs: jax.Array,
    uniforms: jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    bath_temperature: float | jax.Array,
    kappa_base: float | jax.Array,
    kappa_exposure: float | jax.Array,
) -> BathExchangeResult:
    """Apply direct-bath proposals from supplied signs and uniform variates."""

    energy = jnp.asarray(energy)
    uniforms = jnp.asarray(uniforms, dtype=energy.dtype)
    proposal = bath_proposal(
        energy,
        exposure,
        signs,
        energy_quantum,
        energy_floor,
        heat_capacity,
        bath_temperature,
        kappa_base,
        kappa_exposure,
    )
    accepted = jnp.log(uniforms) < proposal.log_acceptance
    particle_heat = jnp.where(
        accepted, proposal.particle_heat, jnp.zeros_like(energy)
    )
    return BathExchangeResult(
        energy=jnp.where(accepted, proposal.energy, energy),
        heat=jnp.sum(particle_heat),
        particle_heat=particle_heat,
        accepted=accepted,
        signs=jnp.asarray(signs),
        proposed_energy=proposal.energy,
        affinity=proposal.affinity,
        log_acceptance=proposal.log_acceptance,
        exposure=jnp.asarray(exposure),
        kappa=proposal.kappa,
    )


def sample_heat_exchange(
    key: jax.Array,
    energy: jax.Array,
    exposure: jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    bath_temperature: float | jax.Array,
    kappa_base: float | jax.Array,
    kappa_exposure: float | jax.Array,
) -> BathExchangeResult:
    """Sample independent signed direct-bath proposals for all particles."""

    energy = jnp.asarray(energy)
    sign_key, uniform_key = jax.random.split(key)
    positive = jax.random.bernoulli(sign_key, shape=energy.shape)
    signs = jnp.where(positive, 1, -1).astype(energy.dtype)
    uniforms = jax.random.uniform(uniform_key, energy.shape, dtype=energy.dtype)
    return heat_exchange_from_draws(
        energy,
        exposure,
        signs,
        uniforms,
        energy_quantum,
        energy_floor,
        heat_capacity,
        bath_temperature,
        kappa_base,
        kappa_exposure,
    )


def direct_bath_exchange(
    key: jax.Array,
    energy: jax.Array,
    positions: jax.Array,
    lattice: jax.Array,
    energy_quantum: float | jax.Array,
    energy_floor: float | jax.Array,
    heat_capacity: float | jax.Array,
    bath_temperature: float | jax.Array,
    kappa_base: float | jax.Array,
    kappa_exposure: float | jax.Array,
) -> BathExchangeResult:
    """Compute exposure and perform one direct-bath exchange phase."""

    exposure = exposure_fraction(positions, lattice, empty=energy.shape[0])
    return sample_heat_exchange(
        key,
        energy,
        exposure,
        energy_quantum,
        energy_floor,
        heat_capacity,
        bath_temperature,
        kappa_base,
        kappa_exposure,
    )


__all__ = [
    "ExposureProperties",
    "BathProposal",
    "BathExchangeResult",
    "collision_coefficient",
    "thermal_coupling",
    "exposure_properties",
    "molecule_translation_drag",
    "bath_proposal",
    "heat_exchange_from_draws",
    "sample_heat_exchange",
    "direct_bath_exchange",
]
