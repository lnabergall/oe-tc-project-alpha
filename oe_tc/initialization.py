"""Exclusion-safe initialization of fixed-shape simulation states."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from oe_tc.config import Params, StaticConfig
from oe_tc.geometry import build_lattice, lattice_coordinates
from oe_tc.state import State
from oe_tc.topology import connected_components


@partial(jax.jit, static_argnames=("config",))
def initialize_state(
    key: jax.Array,
    params: Params,
    config: StaticConfig,
) -> State:
    """Sample unique particle sites and initialize at the bath energy scale."""

    flat_sites = jax.random.choice(
        key,
        config.n * config.n,
        shape=(config.num_particles,),
        replace=False,
    )
    positions = lattice_coordinates(flat_sites, config.n).astype(jnp.int32)
    energy = jnp.full(
        (config.num_particles,),
        params.heat_capacity * params.bath_temperature,
        dtype=jnp.float32,
    )
    bonds = jnp.zeros((config.num_particles,), dtype=jnp.uint8)
    lattice = build_lattice(positions, config.n, config.empty)
    return State(
        R=positions,
        E=energy,
        L=lattice,
        bonds=bonds,
        root=jnp.arange(config.num_particles, dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )


def state_from_arrays(
    positions: jax.Array,
    energy: jax.Array,
    bonds: jax.Array,
    config: StaticConfig,
) -> State:
    """Construct a state from explicit arrays and derive its lattice/components."""

    positions = jnp.asarray(positions, dtype=jnp.int32)
    energy = jnp.asarray(energy, dtype=jnp.float32)
    bonds = jnp.asarray(bonds, dtype=jnp.uint8)
    expected = (config.num_particles,)
    if positions.shape != expected + (2,):
        raise ValueError(f"positions must have shape {expected + (2,)}")
    if energy.shape != expected or bonds.shape != expected:
        raise ValueError(f"energy and bonds must have shape {expected}")

    lattice = build_lattice(positions, config.n, config.empty)
    root = connected_components(
        lattice,
        positions,
        bonds,
        max_iters=config.component_max_iters,
        empty=config.empty,
    )
    return State(
        R=positions,
        E=energy,
        L=lattice,
        bonds=bonds,
        root=root,
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )
