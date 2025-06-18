import math

import jax
import jax.numpy as jnp


PARAMS = {
    "d": [5, 10, 20, 40, 80, 160],
    "k": [12, 100, 500, 2500, 10000, 25000],
    "r_N": [1, 2, 3, 4, 5, 6, 8],    # number ratio

    "beta": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "gamma": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],

    "mu": [1.0],

    "alpha": [5.0, 10.0, 20.0, 40.0, 80.0, 160.0],

    "epsilon": [0.1, 0.2, 0.5, 1.0],
    "delta": [2, 3, 5, 10],
}


DEFAULTS = {
    "d": 2,
    "k": 0,
    "r_N": 3,

    "beta": 4,
    "gamma": -1,

    "mu": -1,

    "alpha": 3,

    "epsilon": -1,
    "delta": 0,
}


def build_config(param_indices):
    CONFIG = {key: PARAMS[key][idx] for key, idx in param_indices.items()}
    k, d, r_N = CONFIG["k"], CONFIG["d"], CONFIG["r_N"]
    del CONFIG["d"]
    del CONFIG["r_N"]

    nondefaults = {key: val for key, val in CONFIG.items() if param_indices[key] != DEFAULTS[key]}
    CONFIG["name"] = str(CONFIG["k"]) + "k_" + "_".join(
        str(val).replace(".", "-") + key.replace("_", "") for key, val in nondefaults.items() if key != "k")

    n_ = math.ceil(math.sqrt(d*k))
    denom = math.lcm(4, (r_N + 1))
    offset = denom if (n_ % denom) != 0 else 0
    n = n_ + offset - (n_ % denom)           # multiple of 4 and r_N + 1
    CONFIG["n"] = n

    CONFIG["t"] = 2
    CONFIG["T_M"] = jnp.array((1, 2))
    CONFIG["T_Q"] = jnp.array((-1, 1))

    k_p = k // (r_N + 1)
    CONFIG["N"] = jnp.array((r_N * k_p, k_p))

    CONFIG["pad_value"] = 2 * k
    CONFIG["emission_streams"] = 2 * math.floor(math.sqrt(k)) 
    CONFIG["boundstate_streams"] = CONFIG["emission_streams"]

    CONFIG["particle_limit"] = (k // 7) + (k // 20) + 4
    CONFIG["boundstate_limit"] = (CONFIG["particle_limit"] * 3) // 2
    CONFIG["field_preloads"] = 100

    return CONFIG