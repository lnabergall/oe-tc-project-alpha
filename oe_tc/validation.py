"""Compiled invariants for debug runs and checkpoint validation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.config import Params, StaticConfig
from oe_tc.geometry import occupancy_is_consistent
from oe_tc.state import State
from oe_tc.topology import connected_components_with_status, topology_is_valid


class ValidationResult(NamedTuple):
    positions_and_lattice: jax.Array
    energy_floor: jax.Array
    bonds: jax.Array
    components: jax.Array
    components_converged: jax.Array
    valid: jax.Array


def validate_state(
    state: State,
    params: Params,
    config: StaticConfig,
    *,
    check_components: bool = True,
) -> ValidationResult:
    """Check all persistent-state invariants as JAX scalar booleans."""

    positions_and_lattice = occupancy_is_consistent(state.R, state.L, config.empty)
    energy_floor = jnp.all(jnp.isfinite(state.E)) & jnp.all(
        state.E >= params.energy_floor
    )
    bonds = topology_is_valid(state.L, state.R, state.bonds, config.empty)

    if check_components:
        expected, _, components_converged = connected_components_with_status(
            state.L,
            state.R,
            state.bonds,
            max_iters=config.component_max_iters,
            empty=config.empty,
        )
        components = components_converged & jnp.all(state.root == expected)
    else:
        components = jnp.asarray(True)
        components_converged = jnp.asarray(True)

    valid = (
        positions_and_lattice
        & energy_floor
        & bonds
        & components
        & components_converged
    )
    return ValidationResult(
        positions_and_lattice=positions_and_lattice,
        energy_floor=energy_floor,
        bonds=bonds,
        components=components,
        components_converged=components_converged,
        valid=valid,
    )
