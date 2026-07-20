"""Optimistic fixed-shape scheduler for OE-TC molecule moves."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oe_tc.config import Params, StaticConfig
from oe_tc.geometry import lattice_indices, neighbor_coordinates
from oe_tc.molecules import (
    MoleculeProposals,
    apply_selected_molecule_moves,
    molecule_metadata,
    propose_molecule_moves,
)
from oe_tc.state import State


CLAIMS_PER_PARTICLE = 10


class ProposalClaims(NamedTuple):
    """Source/destination closed-neighborhood claim records."""

    site: jax.Array
    root: jax.Array
    valid: jax.Array


class MISResult(NamedTuple):
    selected: jax.Array
    excluded: jax.Array
    iterations: jax.Array


class MoleculePhaseMetrics(NamedTuple):
    accepted_moves: jax.Array
    conflicts: jax.Array
    retries: jax.Array
    unresolved: jax.Array
    mis_iterations: jax.Array
    bath_energy: jax.Array
    configurational_delta: jax.Array


def proposal_claims(
    state: State, proposals: MoleculeProposals, static: StaticConfig
) -> ProposalClaims:
    """Build O(10N) source/destination footprints and neighbor shells."""

    source_shell = neighbor_coordinates(state.R, static.n)
    destination_shell = neighbor_coordinates(proposals.candidate_R, static.n)
    coordinates = jnp.concatenate(
        (
            state.R[:, None, :],
            source_shell.coordinates,
            proposals.candidate_R[:, None, :],
            destination_shell.coordinates,
        ),
        axis=1,
    )
    valid = jnp.concatenate(
        (
            jnp.ones((state.R.shape[0], 1), dtype=jnp.bool_),
            source_shell.valid,
            jnp.ones((state.R.shape[0], 1), dtype=jnp.bool_),
            destination_shell.valid,
        ),
        axis=1,
    )
    return ProposalClaims(
        site=lattice_indices(coordinates, static.n).astype(jnp.int32),
        root=jnp.broadcast_to(state.root[:, None], valid.shape).astype(jnp.int32),
        valid=valid,
    )


build_proposal_claims = proposal_claims


def random_priority_mis(
    key: jax.Array,
    claims: ProposalClaims,
    accepted: jax.Array,
    static: StaticConfig,
    priorities: jax.Array | None = None,
) -> MISResult:
    """Filter accepted proposals to a random-priority maximal independent set.

    Proposals that own every claimed site are selected, then they and all of
    their conflicts are removed.  The loop exits as soon as no candidates
    remain.  Equal uint32 priorities are broken by the smaller root id.
    """

    accepted = jnp.asarray(accepted, dtype=jnp.bool_)
    count, num_sites = accepted.shape[0], static.n * static.n
    priorities = (
        jax.random.bits(key, (count,), dtype=jnp.uint32)
        if priorities is None
        else jnp.asarray(priorities, dtype=jnp.uint32)
    )
    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        accepted,
        jnp.zeros_like(accepted),
    )

    def condition(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        iteration, remaining, _ = carry
        return (iteration < static.mis_max_iters) & jnp.any(remaining)

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        iteration, remaining, selected = carry
        active_claim = claims.valid & remaining[claims.root]
        claim_priority = priorities[claims.root]
        site_max = jnp.zeros((num_sites,), dtype=jnp.uint32).at[claims.site].max(
            jnp.where(active_claim, claim_priority, jnp.uint32(0))
        )
        contender = active_claim & (claim_priority == site_max[claims.site])
        site_owner = jnp.full((num_sites,), count, dtype=jnp.int32).at[
            claims.site
        ].min(jnp.where(contender, claims.root, count))
        lost_claim = active_claim & (site_owner[claims.site] != claims.root)
        lost = jnp.zeros((count,), dtype=jnp.int8).at[claims.root].max(
            lost_claim.astype(jnp.int8)
        )
        winner = remaining & (lost == 0)

        winner_claim = claims.valid & winner[claims.root]
        claimed_by_winner = jnp.zeros((num_sites,), dtype=jnp.int8).at[
            claims.site
        ].max(winner_claim.astype(jnp.int8))
        touches_winner = active_claim & claimed_by_winner[claims.site].astype(jnp.bool_)
        conflicted = jnp.zeros((count,), dtype=jnp.int8).at[claims.root].max(
            touches_winner.astype(jnp.int8)
        ).astype(jnp.bool_)
        conflicted &= remaining & ~winner
        return iteration + 1, remaining & ~winner & ~conflicted, selected | winner

    iterations, remaining, selected = jax.lax.while_loop(condition, body, initial)
    del remaining
    return MISResult(
        selected=selected,
        excluded=accepted & ~selected,
        iterations=iterations,
    )


select_proposal_mis = random_priority_mis


def molecule_phase(
    key: jax.Array,
    state: State,
    params: Params,
    static: StaticConfig,
) -> tuple[State, MoleculePhaseMetrics]:
    """Resolve every molecule once using optimistic conflict retries."""

    metadata = molecule_metadata(state.root)
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    zero_e = jnp.asarray(0, dtype=state.E.dtype)
    initial = (
        zero_i,
        state,
        metadata.is_root,
        zero_i,
        zero_i,
        zero_i,
        zero_i,
        zero_e,
        zero_e,
    )

    def condition(carry) -> jax.Array:
        retry, _, active, *_ = carry
        return (retry < static.molecule_max_retries) & jnp.any(active)

    def body(carry):
        (
            retry,
            current,
            active,
            accepted_count,
            conflict_count,
            retry_count,
            mis_count,
            bath_energy,
            configurational_delta,
        ) = carry
        retry_key = jax.random.fold_in(key, retry.astype(jnp.uint32))
        proposal_key, mis_key = jax.random.split(retry_key)
        proposals = propose_molecule_moves(
            proposal_key, current, params, static, active, metadata
        )
        mis = random_priority_mis(
            mis_key, proposal_claims(current, proposals, static), proposals.accepted, static
        )
        updated = apply_selected_molecule_moves(current, proposals, mis.selected, static)
        conflict_losers = proposals.accepted & ~mis.selected
        selected_delta = jnp.where(
            mis.selected, proposals.potential_energy_change, 0
        )
        return (
            retry + 1,
            updated,
            conflict_losers,
            accepted_count + jnp.sum(mis.selected, dtype=jnp.int32),
            conflict_count + jnp.sum(conflict_losers, dtype=jnp.int32),
            retry_count
            + jnp.where(retry > 0, jnp.sum(active, dtype=jnp.int32), zero_i),
            mis_count + mis.iterations,
            bath_energy
            + jnp.sum(
                jnp.where(
                    mis.selected & ~proposals.internal_channel,
                    proposals.potential_energy_change,
                    0,
                )
            ),
            configurational_delta + jnp.sum(selected_delta),
        )

    (
        _,
        final_state,
        unresolved,
        accepted,
        conflicts,
        retries,
        mis_iterations,
        bath_energy,
        configurational_delta,
    ) = jax.lax.while_loop(condition, body, initial)
    return final_state, MoleculePhaseMetrics(
        accepted_moves=accepted,
        conflicts=conflicts,
        retries=retries,
        unresolved=jnp.sum(unresolved, dtype=jnp.int32),
        mis_iterations=mis_iterations,
        bath_energy=bath_energy,
        configurational_delta=configurational_delta,
    )


run_molecule_phase = molecule_phase


__all__ = [
    "CLAIMS_PER_PARTICLE",
    "ProposalClaims",
    "MISResult",
    "MoleculePhaseMetrics",
    "proposal_claims",
    "build_proposal_claims",
    "random_priority_mis",
    "select_proposal_mis",
    "molecule_phase",
    "run_molecule_phase",
]
