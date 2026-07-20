"""Internal thermodynamics and log-space Metropolis helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EventThermodynamics(NamedTuple):
    """Energy/entropy fluxes and affinity for one structural proposal."""

    potential_energy_change: jax.Array
    internal_energy_change: jax.Array
    heat: jax.Array
    entropy_change: jax.Array
    affinity: jax.Array
    log_metropolis: jax.Array
    floor_valid: jax.Array


def temperature(energy: jax.Array, heat_capacity: float | jax.Array) -> jax.Array:
    """Return the derived particle temperature ``E / C``."""

    return jnp.asarray(energy) / jnp.asarray(heat_capacity)


def entropy(
    energy: jax.Array,
    heat_capacity: float | jax.Array,
    energy_floor: float | jax.Array,
) -> jax.Array:
    """Return ``S(E) = C log(E / E_floor)`` in units with ``k_B = 1``."""

    energy = jnp.asarray(energy)
    return jnp.asarray(heat_capacity, dtype=energy.dtype) * jnp.log(
        energy / jnp.asarray(energy_floor, dtype=energy.dtype)
    )


def entropy_change(
    energy_before: jax.Array,
    energy_after: jax.Array,
    heat_capacity: float | jax.Array,
    energy_floor: float | jax.Array,
    axis: int | tuple[int, ...] | None = -1,
) -> jax.Array:
    """Sum the entropy change along ``axis`` without materializing entropies."""

    before = jnp.asarray(energy_before)
    after = jnp.asarray(energy_after)
    # The floor cancels algebraically, but retaining it in the signature makes
    # the model definition explicit and gives it the same API as ``entropy``.
    del energy_floor
    change = jnp.asarray(heat_capacity, dtype=before.dtype) * jnp.log(after / before)
    return jnp.sum(change, axis=axis)


def inverse_bath_temperature(
    bath_temperature: float | jax.Array,
) -> jax.Array:
    """Return ``beta_b`` in the dimensionless convention ``k_B = 1``."""

    return jnp.reciprocal(jnp.asarray(bath_temperature))


def metropolis_log_probability(affinity: jax.Array) -> jax.Array:
    """Return ``log(min(1, exp(affinity)))`` without exponent overflow."""

    affinity = jnp.asarray(affinity)
    return jnp.minimum(affinity, jnp.zeros((), dtype=affinity.dtype))


def metropolis_probability(affinity: jax.Array) -> jax.Array:
    """Return the overflow-safe Metropolis acceptance probability."""

    return jnp.exp(metropolis_log_probability(affinity))


def kinetic_metropolis_log_probability(
    affinity: jax.Array, kinetic_probability: jax.Array | float
) -> jax.Array:
    """Combine a probability-valued kinetic prefactor with Metropolis in log space."""

    affinity = jnp.asarray(affinity)
    kinetic = jnp.asarray(kinetic_probability, dtype=affinity.dtype)
    clipped = jnp.minimum(kinetic, jnp.ones((), dtype=kinetic.dtype))
    log_kinetic = jnp.where(
        clipped > 0,
        jnp.log(clipped),
        jnp.asarray(-jnp.inf, dtype=kinetic.dtype),
    )
    return log_kinetic + metropolis_log_probability(affinity)


def metropolis_accept(
    key: jax.Array,
    affinity: jax.Array,
    kinetic_probability: jax.Array | float = 1.0,
) -> jax.Array:
    """Draw a Metropolis decision using a log-uniform comparison."""

    log_probability = kinetic_metropolis_log_probability(
        affinity, kinetic_probability
    )
    uniform = jax.random.uniform(
        key, shape=log_probability.shape, dtype=log_probability.dtype
    )
    return jnp.log(uniform) < log_probability


def event_thermodynamics(
    potential_energy_change: jax.Array | float,
    energy_before: jax.Array,
    energy_after: jax.Array,
    heat_capacity: float | jax.Array,
    energy_floor: float | jax.Array,
    bath_temperature: float | jax.Array,
    axis: int | tuple[int, ...] | None = -1,
) -> EventThermodynamics:
    """Evaluate ``Q = delta_U + delta_E`` and the structural affinity.

    The input arrays may carry a leading batch of independent events; ``axis``
    names the particle dimension(s) reduced for each event.
    """

    before = jnp.asarray(energy_before)
    after = jnp.asarray(energy_after)
    delta_u = jnp.asarray(potential_energy_change, dtype=before.dtype)
    delta_e = jnp.sum(after - before, axis=axis)
    heat = delta_u + delta_e
    delta_s = entropy_change(
        before, after, heat_capacity, energy_floor, axis=axis
    )
    affinity = delta_s - inverse_bath_temperature(bath_temperature) * heat
    floor = jnp.asarray(energy_floor, dtype=after.dtype)
    floor_valid = jnp.all(after >= floor, axis=axis)
    return EventThermodynamics(
        potential_energy_change=delta_u,
        internal_energy_change=delta_e,
        heat=heat,
        entropy_change=delta_s,
        affinity=affinity,
        log_metropolis=metropolis_log_probability(affinity),
        floor_valid=floor_valid,
    )


def event_affinity(
    potential_energy_change: jax.Array | float,
    energy_before: jax.Array,
    energy_after: jax.Array,
    heat_capacity: float | jax.Array,
    energy_floor: float | jax.Array,
    bath_temperature: float | jax.Array,
    axis: int | tuple[int, ...] | None = -1,
) -> jax.Array:
    """Return the general structural affinity ``delta_S - Q/T_b``."""

    return event_thermodynamics(
        potential_energy_change,
        energy_before,
        energy_after,
        heat_capacity,
        energy_floor,
        bath_temperature,
        axis,
    ).affinity


def bath_channel_affinity(
    potential_energy_change: jax.Array | float,
    bath_temperature: float | jax.Array,
) -> jax.Array:
    """Affinity for a bath-channel event, where ``delta_E = 0``."""

    return -jnp.asarray(potential_energy_change) * inverse_bath_temperature(
        bath_temperature
    )


def internal_channel_affinity(
    energy_before: jax.Array,
    energy_after: jax.Array,
    heat_capacity: float | jax.Array,
    energy_floor: float | jax.Array,
    axis: int | tuple[int, ...] | None = -1,
) -> jax.Array:
    """Affinity for an energy-conserving internal-channel event."""

    return entropy_change(
        energy_before, energy_after, heat_capacity, energy_floor, axis=axis
    )


__all__ = [
    "EventThermodynamics",
    "temperature",
    "entropy",
    "entropy_change",
    "inverse_bath_temperature",
    "metropolis_log_probability",
    "metropolis_probability",
    "kinetic_metropolis_log_probability",
    "metropolis_accept",
    "event_thermodynamics",
    "event_affinity",
    "bath_channel_affinity",
    "internal_channel_affinity",
]
