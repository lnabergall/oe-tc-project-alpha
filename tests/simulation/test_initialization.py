import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.config import StaticConfig, default_params
from oe_tc.initialization import initialize_state
from oe_tc.validation import validate_state


def test_initialization_is_reproducible_and_valid():
    config = StaticConfig(n=8, num_particles=20)
    params = default_params()
    first = initialize_state(jax.random.key(11), params, config)
    second = initialize_state(jax.random.key(11), params, config)

    np.testing.assert_array_equal(first.R, second.R)
    assert bool(validate_state(first, params, config).valid)
    np.testing.assert_allclose(
        first.E,
        jnp.full((20,), params.heat_capacity * params.bath_temperature),
    )
