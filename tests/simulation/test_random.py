import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.random import Phase, log_uniform, phase_key, purpose_key


def test_streams_are_reproducible_and_separated():
    base = jax.random.key(8)
    source = phase_key(base, jnp.uint32(3), Phase.SOURCE)
    source_again = phase_key(base, jnp.uint32(3), Phase.SOURCE)
    bath = phase_key(base, jnp.uint32(3), Phase.BATH)

    np.testing.assert_array_equal(jax.random.key_data(source), jax.random.key_data(source_again))
    assert not np.array_equal(jax.random.key_data(source), jax.random.key_data(bath))
    assert not np.array_equal(
        jax.random.key_data(purpose_key(source, 1)),
        jax.random.key_data(purpose_key(source, 2)),
    )


def test_log_uniform_is_finite_and_nonpositive():
    values = log_uniform(jax.random.key(2), (1024,))
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values <= 0.0))
