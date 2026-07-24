"""Fixed-shape state and compact per-sweep diagnostics."""

from __future__ import annotations

from typing import NamedTuple

import jax


class State(NamedTuple):
    """Physical state carried through the compiled simulation."""

    R: jax.Array
    E: jax.Array
    L: jax.Array
    bonds: jax.Array
    root: jax.Array
    sweep: jax.Array


class StepMetrics(NamedTuple):
    """Small diagnostics returned by one sweep.

    Fluxes are signed positive into the particle system.  The three source
    fields distinguish incident, absorbed, and escaped energy without storing
    per-column packets.  Scheduler saturation and component convergence are
    explicit so bounded parallel algorithms cannot silently bias kinetics.
    """

    source_energy: jax.Array
    source_incident_energy: jax.Array
    source_escaped_energy: jax.Array
    bath_energy_direct: jax.Array
    bath_energy_structural: jax.Array
    conduction_energy_throughput: jax.Array
    internal_energy: jax.Array
    configurational_energy: jax.Array
    num_molecules: jax.Array
    num_bonds: jax.Array
    accepted_molecule_moves: jax.Array
    accepted_bond_flips: jax.Array
    accepted_bath_exchanges: jax.Array
    accepted_conduction_exchanges: jax.Array
    molecule_conflicts: jax.Array
    molecule_retries: jax.Array
    molecule_unresolved: jax.Array
    mis_iterations: jax.Array
    component_iterations: jax.Array
    components_converged: jax.Array
    energy_residual: jax.Array
