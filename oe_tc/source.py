"""Planck irradiation sampler and column-absorption kernels."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.geometry import topmost_particles


DEFAULT_PLANCK_TERMS = 64


class SourceResult(NamedTuple):
    """Updated energies and explicit incident/absorbed/escaped source fluxes."""

    energy: jax.Array
    heat: jax.Array
    particle_heat: jax.Array
    escaped_energy: jax.Array
    incident_energy: jax.Array
    packets: jax.Array
    targets: jax.Array
    absorbed: jax.Array


def planck_density(
    packet_energy: jax.Array, source_beta: float | jax.Array
) -> jax.Array:
    """Evaluate the normalized source packet density.

    ``f(q) = 15 beta**4 q**3 / (pi**4 expm1(beta*q))`` for ``q > 0``.
    """

    q = jnp.asarray(packet_energy)
    beta = jnp.asarray(source_beta, dtype=q.dtype)
    x = beta * q
    ratio = jnp.where(x > 0, q**3 / jnp.expm1(x), jnp.zeros_like(q))
    return jnp.asarray(15.0 / math.pi**4, dtype=q.dtype) * beta**4 * ratio


def planck_mode_probabilities(
    terms: int = DEFAULT_PLANCK_TERMS,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Return normalized weights proportional to ``m**-4`` for ``1..terms``."""

    if terms < 1:
        raise ValueError("terms must be positive")
    modes = jnp.arange(1, terms + 1, dtype=dtype)
    weights = modes ** jnp.asarray(-4.0, dtype=dtype)
    return weights / jnp.sum(weights)


def planck_tail_probability_bound(terms: int = DEFAULT_PLANCK_TERMS) -> float:
    """Upper-bound exact mixture mass omitted by truncation after ``terms``."""

    if terms < 1:
        raise ValueError("terms must be positive")
    zeta_four = math.pi**4 / 90.0
    return 1.0 / (3.0 * terms**3 * zeta_four)


def sample_planck_packets(
    key: jax.Array,
    source_beta: float | jax.Array,
    count: int,
    terms: int = DEFAULT_PLANCK_TERMS,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Sample a bounded zeta--Gamma approximation to the Planck law.

    Expanding ``1 / (exp(beta*q)-1)`` gives mixture weights proportional to
    ``m**-4`` and conditional law ``Gamma(shape=4, rate=beta*m)``.  A 64-term
    inverse-CDF draw has fixed storage and omits less than ``1.2e-6`` of the
    exact mixing mass.  The returned array always has shape ``(count,)``.
    """

    if count < 0:
        raise ValueError("count cannot be negative")
    probabilities = planck_mode_probabilities(terms, dtype)
    cdf = jnp.cumsum(probabilities).at[-1].set(jnp.asarray(1.0, dtype=dtype))
    mode_key, gamma_key = jax.random.split(key)
    uniforms = jax.random.uniform(mode_key, (count,), dtype=dtype)
    mode_index = jnp.searchsorted(cdf, uniforms, side="right", method="scan")
    modes = (mode_index + 1).astype(dtype)
    gamma_draw = jax.random.gamma(
        gamma_key,
        jnp.asarray(4.0, dtype=dtype),
        shape=(count,),
        dtype=dtype,
    )
    beta = jnp.asarray(source_beta, dtype=dtype)
    return gamma_draw / (beta * modes)


def absorb_packets(
    energy: jax.Array,
    lattice: jax.Array,
    packets: jax.Array,
    empty: int | None = None,
) -> SourceResult:
    """Deposit each column packet at its smallest-``y`` occupied site."""

    energy = jnp.asarray(energy)
    packets = jnp.asarray(packets, dtype=energy.dtype)
    num_particles = energy.shape[0]
    if packets.shape != (lattice.shape[0],):
        raise ValueError("packets must contain exactly one value per column")
    if empty is None:
        empty = num_particles

    surface = topmost_particles(lattice, empty)
    safe_targets = jnp.where(surface.occupied, surface.particle, 0)
    absorbed = jnp.where(surface.occupied, packets, jnp.zeros_like(packets))
    particle_heat = jnp.zeros_like(energy).at[safe_targets].add(absorbed)
    heat = jnp.sum(particle_heat)
    escaped = jnp.sum(jnp.where(surface.occupied, 0, packets))
    return SourceResult(
        energy=energy + particle_heat,
        heat=heat,
        particle_heat=particle_heat,
        escaped_energy=escaped,
        incident_energy=jnp.sum(packets),
        packets=packets,
        targets=surface.particle,
        absorbed=absorbed,
    )


apply_source_packets = absorb_packets


def irradiate(
    key: jax.Array,
    energy: jax.Array,
    lattice: jax.Array,
    source_beta: float | jax.Array,
    terms: int = DEFAULT_PLANCK_TERMS,
) -> SourceResult:
    """Sample one Planck packet per column and apply the source phase."""

    packets = sample_planck_packets(
        key,
        source_beta,
        count=lattice.shape[0],
        terms=terms,
        dtype=jnp.asarray(energy).dtype,
    )
    return absorb_packets(energy, lattice, packets, empty=energy.shape[0])


__all__ = [
    "DEFAULT_PLANCK_TERMS",
    "SourceResult",
    "planck_density",
    "planck_mode_probabilities",
    "planck_tail_probability_bound",
    "sample_planck_packets",
    "absorb_packets",
    "apply_source_packets",
    "irradiate",
]
