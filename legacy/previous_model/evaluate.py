import jax
import jax.numpy as jnp

from geometry import *


TOTAL_STATS = ["U_total", "K_total", "E_total", "S_total", "P_total", "bs_count", "bs_notfree_count"]
AVG_STATS = ["U_avg", "K_avg", "E_avg", "S_avg", "P_avg", "impulse_avg", 
             "external_field_avg", "brownian_field_avg", "external_field_hit_avg"]
BS_STATS = ["bs_size_avg", "bs_density"]


def calculate_eval_data(states):
    k = states.R.shape[1]
    pad_value = 2 * k

    norms = jax.tree.map(lattice_norm, (states.P, states.impulse, states.external_field, states.brownian_field))
    external_field_norm = norms[2]

    bs_funcs = map(jax.vmap, [sizes, count, count_bound, density])
    bs_stats = [fn(states.bound_states) for fn in bs_funcs]

    total_stats = jax.tree.map(partial(jnp.sum, axis=-1), 
        (states.U, states.K, states.E, states.S) + norms + (bs_stats[0],))

    avg_stats = jax.tree.map(lambda x: x / k, total_stats[:-1])
    avg_stats += (total_stats[-3] / jnp.sum(external_field_norm != 0.0, axis=-1),)
    bs_size_avg = total_stats[-1] / bs_stats[1]
    bs_stats[0] = bs_size_avg

    total_stats = tuple(zip(TOTAL_STATS, total_stats[:5] + (bs_stats[1], bs_stats[2])))
    avg_stats = tuple(zip(AVG_STATS, avg_stats))
    small_avg_stats = (avg_stats[3],) + avg_stats[5:]
    bs_stats = tuple(zip(BS_STATS, (bs_stats[0], bs_stats[3])))

    return total_stats, avg_stats, small_avg_stats, bs_stats