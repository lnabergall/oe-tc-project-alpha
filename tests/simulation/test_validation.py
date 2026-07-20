import jax.numpy as jnp

from oe_tc.config import StaticConfig, default_params
from oe_tc.geometry import build_lattice
from oe_tc.state import State
from oe_tc.validation import validate_state


def test_validation_accepts_unbonded_state():
    config = StaticConfig(n=4, num_particles=3)
    params = default_params()
    R = jnp.asarray(((0, 0), (1, 0), (3, 3)), dtype=jnp.int32)
    state = State(
        R=R,
        E=jnp.full((3,), params.heat_capacity * params.bath_temperature),
        L=build_lattice(R, config.n, config.empty),
        bonds=jnp.zeros((3,), dtype=jnp.uint8),
        root=jnp.arange(3, dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )

    assert bool(validate_state(state, params, config).valid)


def test_validation_rejects_energy_below_floor():
    config = StaticConfig(n=4, num_particles=1)
    params = default_params()
    R = jnp.asarray(((0, 0),), dtype=jnp.int32)
    state = State(
        R=R,
        E=jnp.asarray((params.energy_floor / 2,)),
        L=build_lattice(R, config.n, config.empty),
        bonds=jnp.zeros((1,), dtype=jnp.uint8),
        root=jnp.zeros((1,), dtype=jnp.int32),
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )

    result = validate_state(state, params, config)
    assert not bool(result.energy_floor)
    assert not bool(result.valid)
