import jax
import jax.numpy as jnp


CONFIGS = {
    "toy_tiny": {
        "name": "toy_tiny",
        "n": 12,    # multiple of 4
        "k": 12,
        "t": 2,
        "N": jnp.array((9, 3)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 24,
        "emission_streams": 3,
        "boundstate_streams": 3,
        "particle_limit": 6,
        "boundstate_limit": 6,

        "field_preloads": 10,
    },
    "toy_small": {
        "name": "toy_small",
        "n": 40,    # multiple of 4
        "k": 100,
        "t": 2,
        "N": jnp.array((80, 20)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 200,
        "emission_streams": 5,
        "boundstate_streams": 5,
        "particle_limit": 30,
        "boundstate_limit": 40,

        "field_preloads": 10,
    },
    "toy_medium": {
        "name": "toy_medium",
        "n": 100,    # multiple of 4
        "k": 500,
        "t": 2,
        "N": jnp.array((400, 100)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 1000,
        "emission_streams": 20,
        "boundstate_streams": 20,
        "particle_limit": 120,
        "boundstate_limit": 160,

        "field_preloads": 10,
    },
    "toy_large": {
        "name": "toy_large",
        "n": 248,    # multiple of 4
        "k": 2500,
        "t": 2,
        "N": jnp.array((2000, 500)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 5000,
        "emission_streams": 100,
        "boundstate_streams": 100,
        "particle_limit": 500,
        "boundstate_limit": 800,

        "field_preloads": 10,
    },
    "toy_huge": {
        "name": "toy_huge",
        "n": 500,    # multiple of 4
        "k": 10000,
        "t": 2,
        "N": jnp.array((8000, 2000)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 20000,
        "emission_streams": 300,
        "boundstate_streams": 300,
        "particle_limit": 1800,
        "boundstate_limit": 3000,

        "field_preloads": 10,
    },
    "toy_massive": {
        "name": "toy_massive",
        "n": 800,    # multiple of 4
        "k": 25000,
        "t": 2,
        "N": jnp.array((20000, 5000)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 1.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 50000,
        "emission_streams": 600,
        "boundstate_streams": 600,
        "particle_limit": 4000,
        "boundstate_limit": 6000,

        "field_preloads": 10,
    },
    "base_1gamma_tiny": {
        "name": "base_1gamma_tiny",
        "n": 12,    # multiple of 4
        "k": 12,
        "t": 2,
        "N": jnp.array((9, 3)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 16.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 24,
        "emission_streams": 3,
        "boundstate_streams": 3,
        "particle_limit": 6,
        "boundstate_limit": 6,

        "field_preloads": 100,
    },
    "base_1gamma_medium": {
        "name": "base_1gamma_medium",
        "n": 100,    # multiple of 4
        "k": 500,
        "t": 2,
        "N": jnp.array((400, 100)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 16.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 1000,
        "emission_streams": 20,
        "boundstate_streams": 20,
        "particle_limit": 120,
        "boundstate_limit": 160,

        "field_preloads": 100,
    },
    "base_1gamma_large": {
        "name": "base_1gamma_large",
        "n": 248,    # multiple of 4
        "k": 2500,
        "t": 2,
        "N": jnp.array((2000, 500)),
        "T_M": jnp.array((1, 2)),
        "T_Q": jnp.array((-1, 1)),

        "beta": 16.0,
        "gamma": 1.0,

        "mu": 1.0,

        "alpha": 1.0,

        "epsilon": 1.0,
        "delta": 2,

        "pad_value": 5000,
        "emission_streams": 100,
        "boundstate_streams": 100,
        "particle_limit": 500,
        "boundstate_limit": 800,

        "field_preloads": 100,
    },
}