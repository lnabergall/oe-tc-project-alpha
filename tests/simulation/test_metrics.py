from __future__ import annotations

import jax.numpy as jnp

from oe_tc.config import StaticConfig, default_params
from oe_tc.initialization import state_from_arrays
from oe_tc.metrics import configurational_energy, count_bonds, state_observables
from oe_tc.topology import NEG_X, POS_X, direction_bit


def test_configurational_energy_counts_contacts_and_bonds_once() -> None:
    config = StaticConfig(n=4, num_particles=3)
    positions = jnp.asarray(((0, 0), (1, 0), (2, 0)), dtype=jnp.int32)
    energy = jnp.asarray((2.0, 3.0, 4.0), dtype=jnp.float32)
    bonds = jnp.zeros((3,), dtype=jnp.uint8)
    bonds = bonds.at[0].set(direction_bit(POS_X))
    bonds = bonds.at[1].set(direction_bit(NEG_X))
    state = state_from_arrays(positions, energy, bonds, config)

    params = default_params()
    assert jnp.isclose(configurational_energy(state, params.eta), -1.1)
    assert int(count_bonds(state)) == 1

    observed = state_observables(state, params)
    assert jnp.isclose(observed.internal_energy, 9.0)
    assert jnp.isclose(observed.configurational_energy, -1.1)
    assert int(observed.num_molecules) == 2
    assert int(observed.num_bonds) == 1
