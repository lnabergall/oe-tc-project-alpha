"""Stable counter-based random streams for compiled simulation phases."""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp


class Phase(IntEnum):
    SOURCE = 1
    CONDUCTION_ORDER = 2
    STRUCTURAL_ORDER = 3
    MOLECULE = 4
    BOND = 5
    BATH = 6
    INITIALIZATION = 7


def phase_key(base_key: jax.Array, sweep: jax.Array, phase: Phase | int) -> jax.Array:
    """Derive a phase stream without consuming or returning a master key."""

    sweep_key = jax.random.fold_in(base_key, jnp.asarray(sweep, dtype=jnp.uint32))
    return jax.random.fold_in(sweep_key, int(phase))


def purpose_key(key: jax.Array, purpose: int, iteration: jax.Array | int = 0) -> jax.Array:
    """Derive stable purpose/retry streams inside one phase."""

    tagged = jax.random.fold_in(key, purpose)
    return jax.random.fold_in(tagged, jnp.asarray(iteration, dtype=jnp.uint32))


def log_uniform(
    key: jax.Array,
    shape: tuple[int, ...] = (),
    dtype=jnp.float32,
) -> jax.Array:
    """Sample ``log(U)`` safely for log-space Metropolis decisions."""

    values = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(jnp.maximum(values, jnp.finfo(dtype).tiny))
