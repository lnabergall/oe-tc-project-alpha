"""Cheap fixed-shape observables for OE-TC model."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.config import Params
from oe_tc.state import State
from oe_tc.topology import POS_X, POS_Y, gather_lattice_neighbors, has_bond


class StateObservables(NamedTuple):
    """Extensive state quantities used in diagnostics and energy accounting."""

    internal_energy: jax.Array
    configurational_energy: jax.Array
    num_molecules: jax.Array
    num_bonds: jax.Array


def configurational_energy(state: State, eta: float | jax.Array) -> jax.Array:
    """Return weak-contact plus bonded energy, counting each edge once.

    Canonical ``+x`` and ``+y`` edges cover every nearest-neighbour contact on
    the horizontally periodic, vertically closed lattice.  A weak contact has
    energy ``-eta`` and a bonded contact has energy ``-1``.
    """

    num_particles = state.R.shape[0]
    neighbors = gather_lattice_neighbors(state.L, state.R, num_particles)
    targets = neighbors[:, jnp.asarray((POS_X, POS_Y), dtype=jnp.int32)]
    occupied = targets < num_particles
    directions = jnp.asarray((POS_X, POS_Y), dtype=jnp.int32)
    bonded = has_bond(state.bonds[:, None], directions[None, :])
    dtype = state.E.dtype
    edge_energy = jnp.where(
        bonded,
        jnp.asarray(-1.0, dtype=dtype),
        -jnp.asarray(eta, dtype=dtype),
    )
    return jnp.sum(jnp.where(occupied, edge_energy, 0.0), dtype=dtype)


def count_bonds(state: State) -> jax.Array:
    """Count reciprocal undirected bonds from their canonical source bits."""

    positive_bits = jnp.asarray((1 << POS_X) | (1 << POS_Y), dtype=state.bonds.dtype)
    canonical = jnp.bitwise_and(state.bonds, positive_bits)
    return jnp.sum(
        jnp.bitwise_and(canonical, 1) + jnp.bitwise_and(canonical >> 2, 1),
        dtype=jnp.int32,
    )


def state_observables(state: State, params: Params) -> StateObservables:
    """Evaluate the compact state observables needed once per sweep."""

    particle_ids = jnp.arange(state.root.shape[0], dtype=state.root.dtype)
    return StateObservables(
        internal_energy=jnp.sum(state.E),
        configurational_energy=configurational_energy(state, params.eta),
        num_molecules=jnp.sum(state.root == particle_ids, dtype=jnp.int32),
        num_bonds=count_bonds(state),
    )


__all__ = [
    "StateObservables",
    "configurational_energy",
    "count_bonds",
    "state_observables",
]
