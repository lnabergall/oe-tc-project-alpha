import jax
import jax.numpy as jnp

import os
print("XLA_FLAGS =", os.environ.get("XLA_FLAGS", "not set"))

@jax.jit
def big_kernel(x):
    # Something that forces a real XLA compile
    return jnp.fft.fft(x)

# Force a compile + execution
x = jnp.ones((10_000,))
y = big_kernel(x).block_until_ready()