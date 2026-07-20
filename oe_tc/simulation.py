"""Composition of OE-TC kernels into deterministic JAX sweeps."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from oe_tc.bath import direct_bath_exchange
from oe_tc.bonds import BondPhaseMetrics, bond_phase
from oe_tc.conduction import randomized_conduction_sweep
from oe_tc.config import Params, StaticConfig
from oe_tc.metrics import state_observables
from oe_tc.random import Phase, phase_key, purpose_key
from oe_tc.scheduler import MoleculePhaseMetrics, molecule_phase
from oe_tc.source import irradiate
from oe_tc.state import State, StepMetrics


def _source_phase(
    key: jax.Array,
    state: State,
    params: Params,
    config: StaticConfig,
) -> tuple[State, jax.Array, jax.Array, jax.Array]:
    """Apply or skip irradiation while retaining one fixed result signature."""

    zero = jnp.asarray(0.0, dtype=state.E.dtype)

    def enabled(_: None):
        result = irradiate(
            key,
            state.E,
            state.L,
            params.source_beta,
            terms=config.planck_terms,
        )
        return (
            state._replace(E=result.energy),
            result.heat,
            result.incident_energy,
            result.escaped_energy,
        )

    def disabled(_: None):
        return state, zero, zero, zero

    return jax.lax.cond(jnp.asarray(params.source_enabled), enabled, disabled, None)


def _conduction_phase(
    key: jax.Array,
    state: State,
    params: Params,
) -> State:
    result = randomized_conduction_sweep(
        key,
        state.E,
        state.R,
        state.L,
        state.bonds,
        params.conduction_contact,
        params.conduction_bond,
    )
    return state._replace(E=result.energy)


def _structural_phases(
    order_key: jax.Array,
    molecule_key: jax.Array,
    bond_key: jax.Array,
    state: State,
    params: Params,
    config: StaticConfig,
) -> tuple[State, MoleculePhaseMetrics, BondPhaseMetrics]:
    """Randomize block order while keeping each phase's RNG stream stable."""

    def bonds_first(current: State):
        current, bond_metrics = bond_phase(bond_key, current, params, config)
        current, molecule_metrics = molecule_phase(
            molecule_key, current, params, config
        )
        return current, molecule_metrics, bond_metrics

    def molecules_first(current: State):
        current, molecule_metrics = molecule_phase(
            molecule_key, current, params, config
        )
        current, bond_metrics = bond_phase(bond_key, current, params, config)
        return current, molecule_metrics, bond_metrics

    choose_bonds_first = jax.random.bernoulli(order_key)
    return jax.lax.cond(
        choose_bonds_first, bonds_first, molecules_first, state
    )


def sweep(
    state: State,
    base_key: jax.Array,
    params: Params,
    config: StaticConfig,
) -> tuple[State, StepMetrics]:
    """Advance one operational sweep and return compact audited diagnostics.

    The base key is immutable. Every random stream is derived from the
    checkpointed sweep counter and a stable phase tag, making results
    independent of host chunking and exactly reproducible after resume.
    """

    before = state_observables(state, params)
    source_key = phase_key(base_key, state.sweep, Phase.SOURCE)
    conduction_key = phase_key(base_key, state.sweep, Phase.CONDUCTION_ORDER)
    structural_order_key = phase_key(
        base_key, state.sweep, Phase.STRUCTURAL_ORDER
    )
    molecule_key = phase_key(base_key, state.sweep, Phase.MOLECULE)
    bond_key = phase_key(base_key, state.sweep, Phase.BOND)
    bath_key = phase_key(base_key, state.sweep, Phase.BATH)

    state, source_heat, source_incident, source_escaped = _source_phase(
        source_key, state, params, config
    )
    state = _conduction_phase(
        purpose_key(conduction_key, 0), state, params
    )
    state, molecule_metrics, bond_metrics = _structural_phases(
        structural_order_key,
        molecule_key,
        bond_key,
        state,
        params,
        config,
    )

    def second_conduction(current: State) -> State:
        return _conduction_phase(
            purpose_key(conduction_key, 1), current, params
        )

    state = jax.lax.cond(
        jnp.asarray(params.second_conduction),
        second_conduction,
        lambda current: current,
        state,
    )

    bath = direct_bath_exchange(
        bath_key,
        state.E,
        state.R,
        state.L,
        params.bath_energy_quantum,
        params.energy_floor,
        params.heat_capacity,
        params.bath_temperature,
        params.kappa_base,
        params.kappa_exposure,
    )
    state = state._replace(E=bath.energy)
    after = state_observables(state, params)
    structural_bath_heat = molecule_metrics.bath_energy + bond_metrics.bath_energy
    total_before = before.internal_energy + before.configurational_energy
    total_after = after.internal_energy + after.configurational_energy
    residual = (
        total_after
        - total_before
        - source_heat
        - bath.heat
        - structural_bath_heat
    )

    state = state._replace(
        sweep=state.sweep + jnp.asarray(1, dtype=state.sweep.dtype)
    )
    metrics = StepMetrics(
        source_energy=source_heat,
        source_incident_energy=source_incident,
        source_escaped_energy=source_escaped,
        bath_energy_direct=bath.heat,
        bath_energy_structural=structural_bath_heat,
        internal_energy=after.internal_energy,
        configurational_energy=after.configurational_energy,
        num_molecules=after.num_molecules,
        num_bonds=after.num_bonds,
        accepted_molecule_moves=molecule_metrics.accepted_moves,
        accepted_bond_flips=bond_metrics.accepted_flips,
        accepted_bath_exchanges=jnp.sum(bath.accepted, dtype=jnp.int32),
        molecule_conflicts=molecule_metrics.conflicts,
        molecule_retries=molecule_metrics.retries,
        molecule_unresolved=molecule_metrics.unresolved,
        mis_iterations=molecule_metrics.mis_iterations,
        component_iterations=bond_metrics.component_iterations,
        components_converged=bond_metrics.components_converged,
        energy_residual=residual,
    )
    return state, metrics


def run_chunk(
    state: State,
    base_key: jax.Array,
    params: Params,
    config: StaticConfig,
    num_steps: int,
) -> tuple[State, StepMetrics]:
    """Run a fixed-length device scan; the caller owns JIT and state donation."""

    def body(current: State, _: None) -> tuple[State, StepMetrics]:
        return sweep(current, base_key, params, config)

    return jax.lax.scan(body, state, xs=None, length=num_steps)


__all__ = ["sweep", "run_chunk"]
