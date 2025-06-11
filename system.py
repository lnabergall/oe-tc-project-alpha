from datetime import datetime
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from sample import *
from physics import *
from geometry import *
from utils import *
from log import *
from storage import *


class InternalData(NamedTuple):
    # particle sampling
    P_particles: jax.Array = None               # independent Gibbs partition. 2D, 8xm.
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

    # bound state helping data
    no_move: jax.Array = None                   # Recent particle movement indicator. 1D, k.


class SystemData(NamedTuple):
    # context
    step: int                               # sampling step

    # system
    R: jax.Array                            # particle positions. 2D, kx2. 
    L: jax.Array = None                     # labeled occupation lattice. 2D, nxn.
    P: jax.Array = None                     # particle momenta. 2D, kx2.

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
    drive: bool                             # toggles the external drive
    alpha: float                            # scale factor in Wien approximation to Planck's law

    # emission
    emissions: bool                         # toggles all emission events
    epsilon: float                          # threshold constant of emission
    delta: float                            # range of emission events

    # bookkeeping
    pad_value: int                          # scalar used for padding arrays, should be index-safe (> k,n)
    emission_streams: int                   # number of emission events to parallel process
    boundstate_streams: int                 # number of parallel processes during bound state discovery 
    particle_limit: int                     # upper bound on the size of an independent set of particles
    boundstate_limit: int                   # upper bound on the size of an independent set of molecules

    # sampling
    seed: int                               # seed for the PRNG
    field_preloads: int                     # number of external fields to preload, two per update step

    # additional information
    name: str                               # indexing name of the system configuration
    time: datetime                          # when the system was created, used for ID and/or seeding 

    # logging and storage
    logging: bool                           # toggles optional logging
    saving: bool                            # toggles data storage
    snapshot_period: int                    # number of update steps between snapshots

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

    def save_state(self, data):
        save_callback = partial(save_state, self.name, self.time)
        jax.experimental.io_callback(save_callback, (), data, ordered=True)

    def initialize(self, key):
        key, key_positions, key_fields = jax.random.split(key, 3)

        # data placeholders
        k2_zeros_float = jnp.zeros((self.k, 2), dtype=float)
        k_zeros_bool = jnp.zeros((self.k,), dtype=bool)
        bound_states_default = jnp.arange(self.k)

        R = sample_lattice_points(key_positions, self.n, self.k, replace=False)
        L = generate_lattice(R, self.n, self.pad_value)

        external_fields, ef_idx = self.generate_external_drives(key_fields)
        external_field = jnp.zeros((self.n, self.n, 2), dtype=float)
        emission_field = k2_zeros_float
        net_field = k2_zeros_float

        data = SystemData(step=0, R=R, L=L, P=k2_zeros_float, external_fields=external_fields, 
                          ef_idx=ef_idx, external_field=external_field, emission_field=emission_field,
                          net_field=net_field, bound_states=bound_states_default)

        if self.saving:
            initialize_hdf5(data, self.name, self.time)

        no_move_default = k_zeros_bool
        bound_states, masses, coms = self.determine_bound_states(data, no_move_default)
        data = data._replace(bound_states=bound_states, masses=masses, coms=coms)

        # internal data placeholders
        k_zeros_float = jnp.zeros((self.k,), dtype=float)
        k5_zeros_float = jnp.zeros((self.k, 5), dtype=float)
        _8m_padding = jnp.full((8, self.particle_limit), self.pad_value)

        internal_data = InternalData(
            P_particles=_8m_padding, logdensities=k5_zeros_float, probabilities=k5_zeros_float, 
            emission_indicator=k_zeros_bool, P_v=k2_zeros_float, P_nv=k2_zeros_float, 
            P_ne=k2_zeros_float, Q_delta_mom=k_zeros_float, E_emit=k_zeros_float, K_ne=k_zeros_float, 
            P_nv_bs=k2_zeros_float, P_ne_bs=k2_zeros_float, Q_delta_mom_bs=k_zeros_float)

        return data, internal_data

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
        if self.drive:
            null_fn = lambda _: (data.external_fields, data.ef_idx)
            fields_consumed = data.ef_idx >= data.external_fields.shape[0]
            external_fields, ef_idx = jax.lax.cond(
                fields_consumed, self.generate_external_drives, null_fn, key_drive)
            data = data._replace(external_fields=external_fields, ef_idx=ef_idx)

        # get field and other precomputable data for current step
        (external_field, ef_idx, net_field, P_v, 
            P_nv, P_ne, Q_delta_mom, E_emit, K_ne) = self.particle_system_update_data(data)
        data = data._replace(step=step, external_field=external_field, ef_idx=ef_idx, net_field=net_field)
        internal_data = internal_data._replace(P_v=P_v, P_nv=P_nv, P_ne=P_ne, Q_delta_mom=Q_delta_mom, 
                                               E_emit=E_emit, K_ne=K_ne)

        # save data if needed
        save = self.saving & ((step % self.snapshot_period) == 0)
        jax.lax.cond(save, self.save_state, lambda _: None, data)

        ### particle phase
        # assemble the partition
        P_particles = four_partition(data.R, data.L, self.particle_limit, self.pad_value)
        internal_data = internal_data._replace(P_particles=P_particles)

        if self.logging:
            jax.debug.print("R pre-particle: {}", data.R)

        # scan over partition
        particle_scan_fn = lambda carry, I: self.particle_gibbs_step(carry[0], carry[1], I, carry[2])
        (data, internal_data, _), particle_step_info = jax.lax.scan(
            particle_scan_fn, (data, internal_data, key_particle), P_particles)

        # collect per-particle data
        compactify_fn = lambda V: compactify_partition(P_particles, V, self.k)
        (logdensities, probabilities, emission_indicator, U_Inbhd, is_bound, no_move) = jax.tree.map(
            compactify_fn, particle_step_info)
        no_move = internal_data.no_move & no_move
        internal_data = internal_data._replace(logdensities=logdensities, probabilities=probabilities, 
                                               emission_indicator=emission_indicator, no_move=no_move)
        
        # update lattice
        L = generate_lattice(data.R, self.n, self.pad_value)
        data = data._replace(L=L)

        ### bound state phase
        # calculate bound states
        bound_states, masses, coms = self.determine_bound_states(data, no_move)
        data = data._replace(bound_states=bound_states, masses=masses, coms=coms)

        # get field and other precomputable data for current step
        (external_field, ef_idx, net_field, P_ne, P_nv_bs, 
            P_ne_bs, Q_delta_mom_bs) = self.boundstate_system_update_data(data)
        data = data._replace(external_field=external_field, ef_idx=ef_idx, net_field=net_field, P=P_ne)
        internal_data = internal_data._replace(
            P_nv_bs=P_nv_bs, P_ne_bs=P_ne_bs, Q_delta_mom_bs=Q_delta_mom_bs)

        # determine a partition as a coloring
        C_boundstates = four_group_partition(data.bound_states, data.L, key_partition, 
                                             self.boundstate_limit, self.pad_value)

        R_previous = data.R

        if self.logging:
            jax.debug.print("bound states: {}", data.bound_states)
            jax.debug.print("R pre-boundstate: {}", R_previous)

        # while loop over coloring
        C_bs_safe = replace(C_boundstates, self.pad_value, -1)
        cond_fn = lambda args: args[2] <= jnp.max(C_bs_safe)
        boundstate_loop_fn = lambda args: self.boundstate_gibbs_step(*args)
        data, internal_data, _, _, _ = jax.lax.while_loop(
            cond_fn, boundstate_loop_fn, (data, internal_data, 0, C_boundstates, key))

        no_move = data.R == R_previous
        internal_data = internal_data._replace(no_move=no_move)

        # update lattice
        L = generate_lattice(data.R, self.n, self.pad_value)
        data = data._replace(L=L)

        # reset and generate emission field if needed
        if self.emissions:
            emission_field = self.generate_emissions(data, internal_data, emission_indicator)
            data = data._replace(emission_field=emission_field)

        return data, internal_data, key 

    def particle_gibbs_step(self, data, internal_data, I, key):
        K_ne = internal_data.K_ne[I]

        R_Inbhd, U_Inbhd, U_I, is_bound = self.particle_gibbs_update_data(data, I, K_ne)
        P_ne = internal_data.P_ne[I]

        # determine emitters
        is_emitter = jax.vmap(determine_emissions, in_axes=(0, 0, 0, 0, None))(
            U_I, is_bound, internal_data.E_emit[I], K_ne, self.epsilon)

        boundary_mask = R_Inbhd[:, :, 1] != self.pad_value
        probabilities, logdensities = calculate_probabilities(
            P_ne, internal_data.Q_delta_mom[I], U_I, U_Inbhd, boundary_mask, self.mu, self.beta)
        next_indices, key = gibbs_sample(key, probabilities)

        # update states and data
        no_move = next_indices == 0
        emission_indicator = jnp.logical_and(is_emitter, no_move) 
        P_new = jnp.where(jnp.expand_dims(emission_indicator, axis=-1), internal_data.P_v[I], P_ne)
        next_indices = jnp.expand_dims(jnp.expand_dims(next_indices, axis=-1), axis=-1)
        next_positions = jnp.take_along_axis(R_Inbhd, next_indices, axis=1)

        R = data.R.at[I].set(jnp.squeeze(next_positions, axis=1), mode="drop")
        P = data.P.at[I].set(P_new, mode="drop")
        data = data._replace(R=R, P=P)

        return (data, internal_data, key), (logdensities, probabilities, 
            emission_indicator, U_Inbhd, is_bound, no_move)

    def boundstate_gibbs_step(self, data, internal_data, step, C_boundstates, key):
        I = jnp.nonzero(C_boundstates == step, size=self.boundstate_limit, fill_value=self.pad_value)[0]
        I_particles = get_classes_by_id(I, data.bound_states, self.pad_value)

        com_Inbhd, boundary_mask, U_Inbhd, U_I = self.boundstate_gibbs_update_data(data, I, I_particles)
        P_ne = internal_data.P_ne_bs[I]
        
        probabilities, logdensities = calculate_probabilities(
            P_ne, internal_data.Q_delta_mom_bs[I], U_I, U_Inbhd, boundary_mask, self.mu, self.beta)
        next_indices, key = gibbs_sample(key, probabilities)

        # update states and data
        new_shifts = get_shifts()[next_indices]
        coms_new = data.coms.at[I].add(new_shifts.astype(float), mode="drop")
        R = move(data.R, data.coms, coms_new, self.n)
        data = data._replace(R=R, coms=coms_new)

        return data, internal_data, step + 1, C_boundstates, key 

    def generate_external_drives(self, key):
        """Generates external field values at the top of the lattice, plus the index of the starting set."""
        if self.drive:
            num_samples = self.n * self.field_preloads
            unmasked_samples = wien_approximation(key, self.beta, self.alpha, num_samples)
            external_fields = jnp.reshape(unmasked_samples, (self.field_preloads, self.n))
        else:
            external_fields = jnp.zeros((self.field_preloads, self.n))

        return external_fields, 0

    def system_update_data(self, data):
        """Precomputable data needed for both phases of the update process."""
        # fields
        if self.drive:
            external_field = generate_drive_field(data.R, data.external_fields[data.ef_idx], masked, self.n)
        else:
            external_field = data.external_field
        net_field = external_field[data.R[:, 0], data.R[:, 1]] + data.emission_field

        # kinetic terms
        P_v, P_nv, P_ne = jax.vmap(calculate_momentum_vectors, in_axes=(0, 0, 0, None, None))(
            data.P, net_field, self.Q, self.mu, self.gamma)

        return external_field, data.ef_idx + 1, net_field, P_v, P_nv, P_ne

    def particle_system_update_data(self, data):
        """Precomputable data needed for the particle update phase."""
        external_field, ef_idx, net_field, P_v, P_nv, P_ne = self.system_update_data(data)
        Q_delta_momentum, E_emit, K_ne = jax.vmap(
            calculate_kinetic_factors)(data.P, P_nv, P_ne, self.M)

        return external_field, ef_idx, net_field, P_v, P_nv, P_ne, Q_delta_momentum, E_emit, K_ne

    def boundstate_system_update_data(self, data):
        external_field, ef_idx, net_field, _, P_nv, P_ne = self.system_update_data(data)

        P_nv_bs = compute_group_sums(P_nv, data.bound_states)
        P_ne_bs = compute_group_sums(P_ne, data.bound_states)

        Q_delta_momentum_bs, _, _ = jax.vmap(calculate_partial_kinetic_factors)(
            P_nv_bs, P_ne_bs, 2 * data.masses)

        return external_field, ef_idx, net_field, P_ne, P_nv_bs, P_ne_bs, Q_delta_momentum_bs

    def particle_gibbs_update_data(self, data, I, K_ne):
        """Multi-particle data needed for a single particle-phase Gibbs update step."""
        R_Inbhd = jax.vmap(generate_neighborhood, in_axes=(0, None, None))(
            data.R[I], self.n, self.pad_value)

        U_Inbhd = neighborhood_potential_energies(
            I, R_Inbhd, self.Q, data.R, self.n, self.pad_value)
        U_I = U_Inbhd[:, 0]
        
        # bound state containment
        E_I = U_I + K_ne
        is_bound = bound(E_I)

        return R_Inbhd, U_Inbhd, U_I, is_bound

    def boundstate_gibbs_update_data(self, data, I, I_particles):
        """Multi-particle data needed for a single bound state phase Gibbs update step."""
        R_Inbhd_particles = jax.vmap(generate_neighborhood, in_axes=(0, None, None))(
            data.R[I_particles], self.n, self.pad_value)
        buffer_size = self.boundstate_limit
        U_Inbhd_particles = neighborhood_potential_energies_dynamic(
            I_particles, R_Inbhd_particles, self.Q, data.R, self.I, 
            self.n, buffer_size, self.pad_value)

        U_Inbhd = compute_subgroup_sums(U_Inbhd_particles, I_particles, 
                                        data.bound_states, I, self.pad_value)
        U_I = U_Inbhd[:, 0]

        # candidate centers of mass
        com_Inbhd = jax.vmap(generate_neighborhood, in_axes=(0, None, None))(
            data.coms[I], self.n, self.pad_value)

        # boundary mask
        out_by_molecule = compute_subgroup_sums(
            (R_Inbhd_particles == self.pad_value).astype(int), I_particles, 
            data.bound_states, I, self.pad_value)
        boundary_mask = ~out_by_molecule[:, :, 1].astype(bool)

        return com_Inbhd, boundary_mask, U_Inbhd, U_I

    def generate_emissions(self, data, internal_data, emission_indicator):
        I = jnp.extract(emission_indicator, self.I, size=self.emission_streams, fill_value=self.pad_value)
        field = jnp.zeros((self.n, self.n, 2), dtype=float)

        def emit_fn(args):
            I, field, emission_indicator = args
            R_I = data.R[I]
            pad_mask = I == self.pad_value

            excitations_fn = jax.vmap(calculate_emissions, 
                in_axes=(0, None, None, None, None, 0, None, None, None))
            field_arrays, position_arrays = excitations_fn(R_I, data.L, self.M, self.Q, 
                data.P, internal_data.E_emit[I], self.delta, self.mu, self.pad_value)
            field_arrays = jnp.where(pad_mask.reshape((pad_mask.shape[0], 1, 1, 1)), 0.0, field_arrays)

            field = field.at[:, :, :].add(merge_excitations(field_arrays, position_arrays, self.n))
            emission_indicator = emission_indicator.at[I].set(False, mode="drop")
            I = jnp.extract(emission_indicator, self.I, 
                            size=self.emission_streams, fill_value=self.pad_value)
            return I, field, emission_indicator

        def cond_fn(args):
            emission_indicator = args[2]
            return jnp.any(emission_indicator)

        _, field, _ = jax.lax.while_loop(cond_fn, emit_fn, (I, field, emission_indicator))
        emission_field = field[data.R[:, 0], data.R[:, 1]]

        return emission_field

    def determine_bound_states(self, data, no_move):
        # bound states to recompute
        bs_active = jnp.extract(~no_move, data.bound_states, size=self.k, fill_value=self.pad_value)
        bs_active = jnp.unique(bs_active, size=self.k, fill_value=self.pad_value)

        # particles in recomputed bound states
        active_indicator = jnp.isin(data.bound_states, bs_active)

        # all bonds
        K = kinetic_energy(data.P, self.M)
        bond_data_fn = jax.vmap(compute_bond_data, in_axes=(0, 0, None, None, None, None))
        nbhds, factors, is_bound = bond_data_fn(self.I, K, data.R, data.L, self.Q, self.pad_value)
        bonds = jax.vmap(determine_bonds, in_axes=(0, 0, None, None))(
            factors, nbhds, is_bound, self.pad_value)

        # bound states
        visited = ~active_indicator
        bound_states = jnp.where(active_indicator, self.pad_value, data.bound_states)
        bound_states = compute_bound_states(self.I, bonds, bound_states, visited, 
                                            self.boundstate_streams, self.pad_value)
        coms, masses = centers_of_mass(data.R, self.M, bound_states)

        return bound_states, masses, coms

### NEED TO ACCOUNT FOR ALL MOVEMENT, INCLUDING DURING BOUND STATE PHASE