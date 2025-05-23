from functools import partial
from dataclasses import dataclass, field, asdict
from typing import NamedTuple
from operator import itemgetter

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from sample import *
from physics import *
from geometry import *
from utils import *
from log import *


class InternalData(NamedTuple):
    # system
    step: int                                   # sampling step

    # particle sampling
    P_particles: jax.Array = None               # independent Gibbs partition. 2D, 8xk.
    logdensities: jax.Array = None              # candidate state logdensities. 2D, kx5.
    probabilities: jax.Array = None             # candidate state probabilities. 2D, kx5.
    emission_indicator: jax.Array = None        # emission event indicator. 1D, k.

    # particle kinetics
    P_v: jax.Array = None                       # Viscosity-damped momenta. 2D, kx2.
    P_nv: jax.Array = None                      # Updated momenta without viscosity. 2D, kx2.
    P_ne: jax.Array = None                      # Updated momenta without emissions. 2D, kx2.
    Q_delta_mom: jax.Array = None               # Pure momentum terms for the heat released. 1D, k.
    E_emit: jax.Array = None                    # Energy emitted in the event of an emission. 1D, k.
    K_ne: jax.Array = None                      # Kinetic energies corresponding to P_ne. 1D, k.

    # bound state kinetics
    P_nv_bs: jax.Array = None                   # Updated net molecule momenta without viscosity. 2D, kx2.
    P_ne_bs: jax.Array = None                   # Updated net molecule momenta without emissions. 2D, kx2.
    Q_delta_mom_bs: jax.Array = None            # Pure molecule momentum terms for the heat released. 1D, k.


class SystemData(NamedTuple):
    # system
    R: jax.Array                            # particle positions. 2D, kx2. 
    L: jax.Array = None                     # labeled occupation lattice. 2D, nxn.
    LQ: jax.Array = None                    # charge-labeled occupation lattice. 2D, nxn.
    L_test: jax.Array = None                # labeled occupation lattice for test particles. 2D, nxn.
    P: jax.Array = None                     # particle momenta. 2D, kx2.
    U: jax.Array = None                     # particle potential energies. 1D, k.

    # fields
    external_fields: jax.Array = None       # top fields produced by external drive. 3D, axnx2.
    ef_idx: int = None                      # index of the external field for the current step
    external_field: jax.Array = None        # field produced by the external drive. 3D, nxnx2.
    emission_field: jax.Array = None        # field produced by emission events. 2D, kx2.
    net_field: jax.Array = None             # net field experienced by each particle. 2D, kx2.

    # bound states
    bound_states: jax.Array = None          # bound state index of each particle. 1D, k.
    masses: jax.Array = None                # mass of each bound state. 1D, k, 0-padded.
    coms: jax.Array = None                  # center of mass of each bound state. 2D, kx2, 0-padded.


def assign_properties(t, k, N, T_M, T_Q):
    I = jnp.arange(k)
    T = jnp.repeat(jnp.arange(t), N, total_repeat_length=k)         # particle types
    M = jnp.fromfunction(lambda i: T_M[T[i]], (k,), dtype=int)      # particle masses
    Q = jnp.fromfunction(lambda i: T_Q[T[i]], (k,), dtype=int)      # particle charges
    return I, T, M, Q


@register_pytree_node_class
@dataclass
class ParticleSystem:
    # system
    n: int                                  # number of lattice points along one dimension
    k: int                                  # number of particles
    t: int                                  # number of particle types
    N: jax.Array                            # number particles of each type, should sum to k. 1D, t.
    T_M: jax.Array                          # particle type masses. 1D, t.
    T_Q: jax.Array                          # particle type charges. 1D, t.

    # viscous heat bath
    beta: float                             # inverse temperature, 1/T
    gamma: float                            # collision coefficient

    # dynamics
    mu: float                               # constant converting force to impulse or momentum

    # external drive 
    alpha: float                            # scale factor in Wien approximation to Planck's law

    # emission
    epsilon: float                          # threshold constant of emission
    delta: float                            # range of emission events

    # bookkeeping
    pad_value: int                          # scalar used for padding most arrays
    charge_pad_value: int                   # scalar used for padding charge arrays
    emission_streams: int                   # number of emission events to parallel process
    boundstate_streams: int                 # number of parallel processes during bound state discovery 
    particle_limit: int                     # upper bound on the size of an independent set of particles
    boundstate_limit: int                   # upper bound on the size of an independent set of molecules

    # sampling
    field_preloads: int                     # number of external fields to preload, two per update step

    ### --- data fields with defaults ---

    # particles
    I: jax.Array = field(default=None)      # particle indices, to be initialized. 1D, k.
    T: jax.Array = field(default=None)      # particle types, to be initialized. 1D, k.
    M: jax.Array = field(default=None)      # particle masses, to be initialized. 1D, k.
    Q: jax.Array = field(default=None)      # particle charges, to be initialized. 1D, k.

    def __post_init__(self):
        if self.T is None:
            self._assign_properties()

    def _assign_properties(self):
        self.I, self.T, self.M, self.Q = assign_properties(self.t, self.k, self.N, self.T_M, self.T_Q)

    def tree_flatten(self):
        static_fields = asdict(self)
        return ((), static_fields)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def initialize(self, key):
        pass

    def run(self, key, steps):
        # initialize data
        key, key_init = jax.random.split(key)
        jax_log_info("Initializing...")
        data, internal_data = self.initialize(key_init)

        # start loop
        jax_log_info("Running the system...")
        step_fn = lambda i, args: self.step(i, *args)
        data, internal_data, key = jax.lax.fori_loop(0, steps, step_fn, (data, internal_data, key))
        jax_log_info("System run complete.")

        return key, data, internal_data

    def step(self, step, data, internal_data, key):
        jax_log_info("step {}...", step)

        key, key_drive, key_particle, key_partition, key_boundstate = jax.random.split(key, 5)

        # generate new drive fields if needed
        null_fn = lambda _: (data.external_fields, data.ef_idx)
        fields_consumed = data.ef_idx >= data.external_fields.shape[0]
        external_fields, ef_idx = jax.lax.cond(
            fields_consumed, self.generate_external_drives, null_fn, key_drive)
        data = data._replace(external_fields=external_fields, ef_idx=ef_idx)

        # get field and other precomputable data for current step
        (external_field, net_field, P_v, 
            P_nv, P_ne, Q_delta_mom, E_emit, K_ne) = self.system_update_data(data)
        data = data._replace(external_field=external_field, net_field=net_field)
        internal_data = internal_data._replace(P_v=P_v, P_nv=P_nv, P_ne=P_ne, Q_delta_mom=Q_delta_mom, 
                                               E_emit=E_emit, K_ne=K_ne)

        ### particle phase
        # assemble the partition
        P_particles = four_partition(data.R, data.L, self.particle_limit, self.pad_value)
        internal_data = internal_data._replace(step=step, P_particles=P_particles)

        # scan over partition
        particle_scan_fn = lambda carry, I: self.particle_gibbs_step(carry[0], carry[1], I, carry[2])
        (data, _), particle_step_info = jax.lax.scan(
            particle_scan_fn, (data, internal_data, key_particle), P_particles)

        # collect per-particle data
        compactify_fn = lambda V: compactify_partition(P_particles, V, self.k, self.pad_value)
        (logdensities, probabilities, emission_indicator, U_Inbhd, is_bound, no_move) = jax.tree.map(
            compactify_fn, particle_step_info)
        internal_data = internal_data._replace(logdensities=logdensities, probabilities=probabilities, 
                                               emission_indicator=emission_indicator)

        ### bound state phase
        # calculate bound states
        bound_states, masses, coms = self.determine_bound_states(data, no_move)
        data = data._replace(bound_states=bound_states, masses=masses, coms=coms)

        # get field and other precomputable data for current step
        (external_field, net_field, P_ne, P_nv_bs, 
            P_ne_bs, Q_delta_mom_bs) = self.boundstate_system_update_data(data)
        data = data._replace(external_field=external_field, net_field=net_field, P=P_ne)
        internal_data = internal_data._replace(
            P_nv_bs=P_nv_bs, P_ne_bs=P_ne_bs, Q_delta_mom_bs=Q_delta_mom_bs)

        # determine a partition as a coloring
        C_boundstates = four_group_partition(data.bound_states, data.L, key_partition, self.pad_value)

        # while loop over coloring
        cond_fn = lambda args: args[2] <= jnp.max(args[3])
        data, internal, _, _, _ = jax.lax.while_loop(
            cond_fn, self.boundstate_gibbs_step, (data, internal_data, 0, C_boundstates, key))

        # collect per-boundstate data?
        pass

        pass

        # reset and generate emission field, make optional
        emission_field = self.generate_emissions(data, internal_data, emission_indicator)
        data = data._replace(emission_field=emission_field)

        return data, internal_data, key 

    def particle_gibbs_step(self, data, internal_data, I, key):
        K_ne = internal_data.K_ne[I]

        L_test, R_Inbhd, U_Inbhd, U_I, is_bound = self.particle_gibbs_update_data(data, I, K_ne)
        P_ne = internal_data.P_ne[I]

        # determine emitters
        is_emitter = jax.vmap(determine_emissions, in_axes=(0, 0, 0, 0, None))(
            U_I, is_bound, internal_data.E_emit[I], K_ne, self.epsilon)

        probabilities, logdensities = calculate_probabilities(
            P_ne, internal_data.Q_delta_mom[I], U_I, U_Inbhd, self.mu, self.beta)
        next_indices, key = gibbs_sample(key, probabilities)

        # update states and data
        next_indices = jnp.expand_dims(jnp.expand_dims(next_indices, axis=-1), axis=-1)
        next_positions = jnp.take_along_axis(R_Inbhd, next_indices, axis=1)
        no_move = next_indices == 0
        emission_indicator = jnp.logical_and(is_emitter, no_move) 
        P_new = jnp.where(emission_indicator, internal_data.P_v[I], P_ne)

        I_indexer = replace(I, self.pad_value, 2*self.k)
        R = data.R.at[I_indexer].set(next_positions, mode="drop")
        P = data.P.at[I_indexer].set(P_new, mode="drop")
        L = add_to_lattice(L_test, I_indexer, R, self.pad_value)
        data = data._replace(R=R, P=P, L=L)

        return (data, key), (logdensities, probabilities, emission_indicator, U_Inbhd, is_bound, no_move)

    def boundstate_gibbs_step(self, data, internal_data, step, C_boundstates, key):
        I = jnp.nonzero(C_boundstates == step, size=self.boundstate_limit, fill_value=self.pad_value)[0]
        I_particles = get_classes_by_id(I, data.bound_states, self.pad_value)

        L_test, com_Inbhd, U_Inbhd, U_I = self.boundstate_gibbs_update_data(data, I, I_particles)
        P_ne = internal_data.P_ne_bs[I]
        
        probabilities, logdensities = calculate_probabilities(
            P_ne, internal_data.Q_delta_mom_bs[I], U_I, U_Inbhd, self.mu, self.beta)
        next_indices, key = gibbs_sample(key, probabilities)

        # update states and data
        new_shifts = shifts()[next_indices]
        I_indexer = replace(I, self.pad_value, 2*self.k)
        coms_new = data.coms.at[I_indexer].add(new_shifts, mode="drop")
        R = move(data.R, data.coms, coms_new)
        L = add_to_lattice(L_test, replace(I_particles, self.pad_value, 2*self.k), R, self.pad_value)
        data = data._replace(R=R, L=L)

        return data, internal_data, step + 1, C_boundstates, key 

    def generate_external_drives(self, key):
        """Generates external field values at the top of the lattice, plus the index of the starting set."""
        num_samples = self.n * self.field_preloads
        unmasked_samples = wien_approximation(key, self.beta, self.alpha, num_samples)
        external_fields = jnp.reshape(unmasked_samples, (self.field_preloads, self.n))
        return external_fields, 0

    def system_update_data(self, data):
        """Precomputable data needed for both phases of the update process."""
        # fields
        external_field = generate_drive_field(self.n, data.R, data.external_fields[data.ef_idx], masked)
        net_field = external_field[R[:, 0], R[:, 1]] + data.emission_field

        # kinetic terms
        P_v, P_nv, P_ne = calculate_momentum_vectors(data.P, net_field, self.Q, self.mu, self.gamma)

        return external_field, net_field, P_v, P_nv, P_ne

    def particle_system_update_data(self, data):
        """Precomputable data needed for the particle update phase."""
        external_field, net_field, P_v, P_nv, P_ne = self.system_update_data(data)
        Q_delta_momentum, E_emit, K_ne = jax.vmap(
            calculate_kinetic_factors)(data.P, P_nv, P_ne, self.M)

        return external_field, net_field, P_v, P_nv, P_ne, Q_delta_momentum, E_emit, K_ne

    def boundstate_system_update_data(self, data):
        external_field, net_field, _, P_nv, P_ne = self.system_update_data(data)

        P_nv_bs = compute_group_sums(P_nv, data.bound_states)
        P_ne_bs = compute_group_sums(P_ne, data.bound_states)

        Q_delta_momentum_bs, _, _ = jax.vmap(calculate_partial_kinetic_factors)(
            P_nv_bs, P_ne_bs, 2 * data.masses)

        return external_field, net_field, P_ne, P_nv_bs, P_ne_bs, Q_delta_momentum_bs

    def gibbs_update_data(self, data, I):
        # labeled occupation lattices without particles in I
        L_test = remove_from_lattice(data.L, I, self.pad_value)
        LQ_test = generate_property_lattice(L_test, self.Q, self.pad_value, self.charge_pad_value)

        # candidate positions
        R_Inbhd = jax.vmap(generate_neighborhood, in_axes=(0, None))(data.R[I], self.n)

        # potential energies
        potential_energy_func = make_potential_energy_func(LQ_test, self.charge_pad_value)
        U_Inbhd = jax.vmap(jax.vmap(potential_energy_func, in_axes=(0, None)))(R_Inbhd, self.Q[I])

        return L_test, R_Inbhd, U_Inbhd

    def particle_gibbs_update_data(self, data, I, K_ne):
        """Multi-particle data needed for a single particle-phase Gibbs update step."""
        L_test, R_Inbhd, U_Inbhd = self.gibbs_update_data(data, I)
        U_I = U_Inbhd[:, 0]

        # bound state containment
        E_I = U_I + K_ne
        is_bound = bound(E_I)

        return L_test, R_Inbhd, U_Inbhd, U_I, is_bound

    def boundstate_gibbs_update_data(self, data, I, I_particles):
        """Multi-particle data needed for a single bound state phase Gibbs update step."""
        L_test, _, U_Inbhd_particles = self.gibbs_update_data(data, I_particles)
        U_Inbhd = compute_group_sums(U_Inbhd_particles, I_particles, data.bound_states, I,
                                     self.boundstate_limit, self.pad_value)
        U_I = U_Inbhd[:, 0]

        # candidate centers of mass
        com_Inbhd = jax.vmap(generate_neighborhood, in_axes=(0, None))(data.coms[I], self.n)

        return L_test, com_Inbhd, U_Inbhd, U_I

    def generate_emissions(self, data, internal_data, emission_indicator):
        I = jnp.extract(emission_indicator, self.I, size=self.emission_streams, fill_value=2*self.k)
        field = jnp.zeros((self.n, self.n, 2), dtype=float)

        def emit_fn(args):
            I, field, emission_indicator = args
            R_I = data.R[I]
            pad_mask = I == self.pad_value

            excitations_fn = jax.vmap(calculate_emissions, 
                in_axes=(0, None, None, None, None, 0, None, None, None))
            field_arrays, position_arrays = excitations_fn(R_I, data.L, self.M, self.Q, 
                data.P, internal_data.E_emit[I], self.delta, self.mu, self.pad_value)
            field_arrays = field_arrays.at[pad_mask].set(0.0)

            field = field.at[:, :, :].add(merge_excitations(field_arrays, position_arrays, self.n))
            emission_indicator = emission_indicator.at[I].set(False, mode="drop")
            I = jnp.extract(emission_indicator, self.I, 
                            size=self.emission_streams, fill_value=2*self.k)
            return I, field, emission_indicator

        def cond_fn(args):
            emission_indicator = args[2]
            return jnp.any(emission_indicator)

        field = jax.lax.while_loop(cond_fn, emit_fn, (I, field, emission_indicator))
        emission_field = field[data.R[:, 0], data.R[:, 1]]

        return emission_field

    def determine_bound_states(self, data, no_move):
        # bound states to recompute
        bs_active = jnp.extract(~no_move, data.bound_states, size=self.k, fill_value=self.pad_value)
        bs_active = jnp.unique(bs_active, size=self.k, fill_value=self.pad_value)

        # particles in recomputed bound states
        active_indicator = jnp.isin(data.bound_states, bs_active)
        I = jnp.extract(active_indicator, self.I, size=self.k, fill_value=self.pad_value)

        # all bonds
        K = kinetic_energy(data.P, self.M)
        bond_data_fn = jax.vmap(compute_bond_data, in_axes=(0, 0, None, None, None, None))
        nbhds, factors, is_bound = bond_data_fn(self.I, K, data.R, data.L, self.Q, self.pad_value)
        bonds = jax.vmap(compute_bonds, in_axes=(0, 0, None, None))(
            factors, nbhds, is_bound, self.pad_value)

        # bound states
        visited = ~active_indicator
        bound_states = jnp.where(active_indicator, self.pad_value, data.bound_states)
        bound_states = compute_bound_states(I, bonds, bound_states, visited, 
                                            self.boundstate_streams, self.pad_value)
        coms, masses = centers_of_mass(R, self.M, bound_states)

        return bound_states, masses, coms
