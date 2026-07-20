from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.config import StaticConfig, default_params
from oe_tc.geometry import build_lattice, occupancy_is_consistent
from oe_tc.molecules import propose_molecule_moves
from oe_tc.scheduler import (
    ProposalClaims,
    molecule_phase,
    proposal_claims,
    random_priority_mis,
)
from oe_tc.state import State


def _monomer_state(positions: jax.Array, n: int, energy: float = 100.0) -> State:
    count = positions.shape[0]
    return State(
        R=positions,
        E=jnp.full((count,), energy),
        L=build_lattice(positions, n),
        bonds=jnp.zeros((count,), jnp.uint8),
        root=jnp.arange(count, dtype=jnp.int32),
        sweep=jnp.asarray(0, jnp.uint32),
    )


def test_claims_are_exactly_source_destination_and_neighbor_shells() -> None:
    state = _monomer_state(jnp.asarray(((2, 2),), jnp.int32), 6)
    params = default_params()._replace(
        translation_frequency=1_000.0, bath_channel_probability=1.0
    )
    static = StaticConfig(6, 1)
    proposals = propose_molecule_moves(jax.random.key(3), state, params, static)
    claims = proposal_claims(state, proposals, static)
    assert claims.site.shape == claims.root.shape == claims.valid.shape == (1, 10)
    assert int(jnp.sum(claims.valid)) == 10
    assert int(claims.site[0, 0]) == 2 * 6 + 2
    destination_site = int(
        proposals.candidate_R[0, 0] * 6 + proposals.candidate_R[0, 1]
    )
    assert int(claims.site[0, 5]) == destination_site


def test_parallel_greedy_mis_is_maximal_and_early_exits() -> None:
    # Conflict path A--B--C with descending priorities. A wins first, B is
    # excluded, and C must then be selected to make the result maximal.
    claims = ProposalClaims(
        site=jnp.asarray(((0, 0), (0, 1), (1, 1)), jnp.int32),
        root=jnp.asarray(((0, 0), (1, 1), (2, 2)), jnp.int32),
        valid=jnp.ones((3, 2), jnp.bool_),
    )
    static = StaticConfig(4, 3, mis_max_iters=8)
    result = random_priority_mis(
        jax.random.key(0),
        claims,
        jnp.ones(3, jnp.bool_),
        static,
        priorities=jnp.asarray((3, 2, 1), jnp.uint32),
    )
    np.testing.assert_array_equal(result.selected, (True, False, True))
    np.testing.assert_array_equal(result.excluded, (False, True, False))
    assert int(result.iterations) == 2

    empty = jax.jit(
        lambda accepted: random_priority_mis(
            jax.random.key(1), claims, accepted, static
        )
    )(jnp.zeros(3, jnp.bool_))
    assert int(empty.iterations) == 0
    assert not bool(jnp.any(empty.selected | empty.excluded))


def test_molecule_phase_moves_nonconflicting_monomers_once_and_is_jittable() -> None:
    state = _monomer_state(jnp.asarray(((1, 2), (6, 5)), jnp.int32), 8)
    params = default_params()._replace(
        translation_probability=1.0,
        translation_frequency=1_000.0,
        bath_channel_probability=1.0,
    )
    static = StaticConfig(8, 2, molecule_max_retries=4, mis_max_iters=4)
    updated, metrics = jax.jit(
        lambda key: molecule_phase(key, state, params, static)
    )(jax.random.key(7))
    assert int(metrics.accepted_moves) == 2
    assert int(metrics.conflicts) == 0
    assert int(metrics.retries) == 0
    assert int(metrics.unresolved) == 0
    np.testing.assert_allclose(metrics.bath_energy, 0.0, atol=1e-7)
    np.testing.assert_allclose(metrics.configurational_delta, 0.0, atol=1e-7)
    np.testing.assert_allclose(updated.E, state.E)
    assert bool(occupancy_is_consistent(updated.R, updated.L, 2))
    for before, after in zip(np.asarray(state.R), np.asarray(updated.R)):
        raw = after - before
        raw[0] = (raw[0] + 4) % 8 - 4
        assert np.abs(raw).sum() == 1


def test_rejected_batch_resolves_without_empty_retry_rounds() -> None:
    state = _monomer_state(jnp.asarray(((1, 1),), jnp.int32), 4, energy=1.0)
    params = default_params()._replace(
        translation_frequency=0.0,
        bath_channel_probability=1.0,
    )
    static = StaticConfig(4, 1, molecule_max_retries=8, mis_max_iters=8)
    updated, metrics = molecule_phase(jax.random.key(11), state, params, static)
    np.testing.assert_array_equal(updated.R, state.R)
    assert int(metrics.accepted_moves) == 0
    assert int(metrics.mis_iterations) == 0
    assert int(metrics.retries) == 0
    assert int(metrics.unresolved) == 0
