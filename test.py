import jax
import jax.numpy as jnp

from log import jax_log_info
from utils import bfs


def outer_one(A):

    def loop_fn(i, A):
        x, y = A[i], A[i+1]
        z = inner_one(x, y)
        z = inner_two(z)
        A = A.at[i].set(z)
        return A

    return jax.lax.fori_loop(0, A.shape[0]-1, loop_fn, A)


def func():
    A = jnp.full((10000, 10000), 1)
    A = outer_one(A)
    return A
    #jax.debug.print("A: {}", A)


def inner_one(x, y):
    x = 10 * x
    return jnp.dot(x, y)


def inner_two(z):
    return z ** 2


if __name__ == '__main__':
    print("Using device: ", jax.devices()[0], "\n")

    #jax.profiler.start_trace("/tmp/jax-trace-test", create_perfetto_link=True)

    # edges = jnp.array([[-1, -1, -1, -1], [1, -1, -1, -1]], dtype=int)
    # i = 0
    # visited = bfs(edges, i, -1)
    # jax.debug.print("visited: {}", visited)

    jax_log_info("Starting test...")
    #jit_func = jax.jit(func)
    jax_log_info("Running...")
    A = func()
    A.block_until_ready()
    jax_log_info("Completed.")

    #jax.profiler.stop_trace()

