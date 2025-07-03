from datetime import datetime
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import NamedTuple
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.experimental import io_callback

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

    # bound state helping data
    no_move: jax.Array = None                   # Recent particle movement indicator. 1D, k.

    # timing
    t_start: float = None                       # model run start time


class SystemData(NamedTuple):
    # context
    step: int                               # sampling step

    # system
    R: jax.Array                            # particle positions. 2D, kx2. 
    L: jax.Array = None                     # labeled occupation lattice. 2D, nxn.
    P: jax.Array = None                     # particle momenta. 2D, kx2.

    # fields
    external_fields: jax.Array = None       # top field values for the charge-coupled external drive. 2D, axn.
    ef_idx: int = None                      # index of the external field for the current step
    brownian_fields: jax.Array = None       # fields produced by Brownian fluctuations. 3D, axnx2.
    bf_idx: int = None                      # index of the Brownian field for the current step
    external_field: jax.Array = None        # field produced by the charge-coupled external drive. 3D, nxnx2.
    brownian_field: jax.Array = None        # field produced by Brownian fluctuations. 2D, kx2.
    mass_field: jax.Array = None            # field produced by the mass-coupled external drive. 2D, kx2.
    emission_field: jax.Array = None        # field produced by emission events. 2D, kx2.
    net_force: jax.Array = None             # net force experienced by each particle. 2D, kx2.

    # bound states
    bound_states: jax.Array = None          # bound state index of each particle. 1D, k.
    masses: jax.Array = None                # mass of each bound state. 1D, k, 0-padded.
    coms: jax.Array = None                  # center of mass of each bound state. 2D, kx2, 0-padded.

    # energy
    U: jax.Array = None                     # particle potential energies. 1D, k.
    K: jax.Array = None                     # particle kinetic energies. 1D, k.
    E: jax.Array = None                     # particle energies. 1D, k.
    U_total: jax.Array = None               # total potential energy of the particle system. Scalar.
    K_total: jax.Array = None               # total kinetic energy of the particle system. Scalar.
    E_total: jax.Array = None               # total energy of the particle system. Scalar.

    # entropy
    S: jax.Array = None                     # particle conditional entropies. 1D, k.
    S_total: jax.Array = None               # total conditional entropy of the system. Scalar.
    S_avg: jax.Array = None                 # average particle conditional entropy. Scalar.


def assign_properties(t, k, N, T_M, T_Q):
    I = jnp.arange(k)
    T = jnp.repeat(jnp.arange(t), N, total_repeat_length=k)         # particle types
    M = jnp.fromfunction(lambda i: T_M[T[i]], (k,), dtype=int)      # particle masses
    Q = jnp.fromfunction(lambda i: T_Q[T[i]], (k,), dtype=int)      # particle charges
    return I, T, M, Q


def time_safe():
    scalar = jnp.float32(0.0)
    return io_callback(perf_counter, scalar)


@register_pytree_node_class
@dataclass
class ParticleSystem:
    # system
    n: int                                  # number of lattice points along one dimension
    k: int                                  # number of particles
    d: int                                  # particle density
    t: int                                  # number of particle types
    N: jax.Array                            # number particles of each type, should sum to k. 1D, t.
    r_N: int                                # particle number ratio
    T_M: jax.Array                          # particle type masses. 1D, t.
    T_Q: jax.Array                          # particle type charges. 1D, t.

    # viscous heat bath
    beta: float                             # inverse temperature, 1/T
    gamma: float                            # collision coefficient

    # dynamics
    mu: float                               # constant converting force to impulse or momentum

    # external drives
    drive: bool                             # toggles the external drives
    alpha: float                            # scale factor in Wien approximation to Planck's law
    kappa: float                            # relative strength of the charge-coupled field pair
    g: float                                # scalar value of the mass-coupled field

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
        k_zeros_float = jnp.zeros((self.k,), dtype=float)
        k_zeros_int = jnp.zeros((self.k,), dtype=int)
        k2_zeros_float = jnp.zeros((self.k, 2), dtype=float)
        k_zeros_bool = jnp.zeros((self.k,), dtype=bool)
        scalar_float = jnp.float32(0)
        scalar_int = jnp.int32(0)
        bound_states_default = jnp.arange(self.k)

        R = sample_lattice_points(key_positions, self.n, self.k, replace=False)
        L = generate_lattice(R, self.n, self.pad_value)

        external_fields, ef_idx, brownian_fields, bf_idx = self.generate_fields(key_fields)
        external_field = jnp.zeros((self.n, self.n, 2), dtype=float)
        mass_field = k2_zeros_float
        if self.drive:
            mass_field = mass_field.at[:, 1].set(self.g)    # does not vary 
        emission_field = k2_zeros_float
        net_force = k2_zeros_float

        log5 = jnp.log(5.0)
        S = jnp.full((self.k,), log5)
        S_total = self.k * log5
        S_avg = log5

        data = SystemData(step=0, R=R, L=L, P=k2_zeros_float, external_fields=external_fields, 
                          ef_idx=ef_idx, brownian_fields=brownian_fields, bf_idx=bf_idx, 
                          external_field=external_field, brownian_field=k2_zeros_float, 
                          mass_field=mass_field, emission_field=emission_field, net_force=net_force, 
                          bound_states=bound_states_default, U=k_zeros_int, K=k_zeros_float, 
                          E=k_zeros_float, U_total=scalar_int, K_total=scalar_float, 
                          E_total=scalar_float, S=S, S_total=S_total, S_avg=S_avg)

        if self.saving:
            initialize_hdf5(data, self.name, self.time)

        no_move_default = k_zeros_bool
        bound_states, masses, coms = self.determine_bound_states(data, no_move_default)
        data = data._replace(bound_states=bound_states, masses=masses, coms=coms)

        # internal data placeholders
        k5_zeros_float = jnp.zeros((self.k, 5), dtype=float)
        _8m_padding = jnp.full((8, self.particle_limit), self.pad_value)

        internal_data = InternalData(
            P_particles=_8m_padding, logdensities=k5_zeros_float, probabilities=k5_zeros_float, 
            emission_indicator=k_zeros_bool, no_move=~k_zeros_bool, t_start=0.0)

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
        jax_log_info("time per step: {:.3f}", (time_safe() - internal_data.t_start) / steps)
        jax_log_info("System run complete.")

        return key, data, internal_data

    def step(self, step, data, internal_data, key):
        jax_log_info("step {}...", step)
        t_start = jax.lax.cond(step == 0, time_safe, lambda : internal_data.t_start)

        key, key_drive, key_particle, key_partition, key_boundstate = jax.random.split(key, 5)

        # generate new drive and Brownian fields if needed
        null_fn = lambda _: (data.external_fields, data.ef_idx, data.brownian_fields, data.bf_idx)
        fields_consumed = data.bf_idx >= data.brownian_fields.shape[0]
        external_fields, ef_idx, brownian_fields, bf_idx = jax.lax.cond(
            fields_consumed, self.generate_fields, null_fn, key_drive)
        data = data._replace(external_fields=external_fields, ef_idx=ef_idx, 
                             brownian_fields=brownian_fields, bf_idx=bf_idx)

        # extra evaluation data
        LQ = generate_property_lattice(data.L, self.Q, self.pad_value)
        U = compute_potential_energies(data.R, self.Q, LQ, self.pad_value)
        K = compute_kinetic_energies(data.P, 2*self.M)
        E = compute_energies(U, K)
        U_total, K_total = jnp.sum(U), jnp.sum(K)
        E_total = compute_energies(U_total, K_total)

        # get field and other precomputable data for current step
        (external_field, ef_idx, brownian_field, bf_idx, net_force) = self.system_update_data(data)
        data = data._replace(step=step, external_field=external_field, ef_idx=ef_idx, 
                             brownian_field=brownian_field, bf_idx=bf_idx, net_force=net_force, 
                             U=U, K=K, E=E, U_total=U_total, K_total=K_total, E_total=E_total)

        # save data if needed
        save = self.saving & ((step % self.snapshot_period) == 0)
        jax.lax.cond(save, self.save_state, lambda _: None, data)

        ### particle phase
        # assemble the partition
        P_particles = four_partition(data.R, data.L, self.particle_limit, self.pad_value)
        internal_data = internal_data._replace(P_particles=P_particles, t_start=t_start)

        if self.logging:
            jax.debug.print("E_total: {}", E_total)
            jax.debug.print("R pre-particle: {}", data.R)

        # scan over partition
        particle_scan_fn = lambda carry, I: self.particle_gibbs_step(carry[0], I, carry[1])
        (data, _), particle_step_info = jax.lax.scan(particle_scan_fn, (data, key_particle), P_particles)

        # collect per-particle data
        compactify_fn = lambda V: compactify_partition(P_particles, V, self.k)
        (logdensities, probabilities, emission_indicator, is_bound, no_move) = jax.tree.map(
            compactify_fn, particle_step_info)
        no_move = internal_data.no_move & no_move
        internal_data = internal_data._replace(logdensities=logdensities, probabilities=probabilities, 
                                               emission_indicator=emission_indicator, no_move=no_move)

        # extra evaluation data
        S = jax.vmap(entropy)(probabilities)
        S_total = jnp.sum(S)
        S_avg = S_total / self.k
        
        # update lattice
        L = generate_lattice(data.R, self.n, self.pad_value)
        data = data._replace(L=L, S=S, S_total=S_total, S_avg=S_avg)

        ### bound state phase
        # calculate bound states
        bound_states, masses, coms = self.determine_bound_states(data, no_move)
        data = data._replace(bound_states=bound_states, masses=masses, coms=coms)

        # get field and other precomputable data for current step
        (external_field, ef_idx, brownian_field, bf_idx, net_force) = self.system_update_data(data)
        data = data._replace(external_field=external_field, ef_idx=ef_idx, brownian_field=brownian_field, 
                             bf_idx=bf_idx, net_force=net_force)

        # determine a partition as a coloring
        C_boundstates = four_group_partition(data.bound_states, data.L, key_partition, 
                                             self.boundstate_limit, self.pad_value)

        R_previous = data.R

        if self.logging:
            jax.debug.print("bound states: {}", data.bound_states)
            jax.debug.print("R pre-boundstate: {}", R_previous)

        # while loop over coloring
        C_bs_safe = replace(C_boundstates, self.pad_value, -1)
        cond_fn = lambda args: args[1] <= jnp.max(C_bs_safe)
        boundstate_loop_fn = lambda args: self.boundstate_gibbs_step(*args)
        data = jax.lax.while_loop(cond_fn, boundstate_loop_fn, (data, 0, C_boundstates, key))[0]

        no_move = jnp.all(data.R == R_previous, axis=-1)
        internal_data = internal_data._replace(no_move=no_move)

        # update lattice
        L = generate_lattice(data.R, self.n, self.pad_value)
        data = data._replace(L=L)

        # reset and generate emission field if needed
        if self.emissions:
            emission_field = self.generate_emissions(data, internal_data, emission_indicator)
            data = data._replace(emission_field=emission_field)

        return data, internal_data, key 

    def particle_gibbs_step(self, data, I, key):
        (R_Inbhd, U_Inbhd_diff, U_I, is_bound, 
            M, K, P_v, P_ne, K_ne, E_emit) = self.particle_gibbs_update_data(data, I)

        # determine emitters
        is_emitter = jax.vmap(determine_emissions, in_axes=(0, 0, 0, 0, None))(
            U_I, is_bound, E_emit, K_ne[:, 0], self.epsilon)

        boundary_mask = R_Inbhd[:, :, 1] != self.pad_value
        probabilities, logdensities = calculate_probabilities(
            K, P_ne, K_ne, U_Inbhd_diff, boundary_mask, M, self.mu, self.beta)
        next_indices, key = gibbs_sample(key, probabilities)

        # update states and data
        no_move = next_indices == 0
        emission_indicator = jnp.logical_and(is_emitter, no_move) 
        P_v, P_ne, next_positions = [
            jnp.take_along_axis(A, next_indices[..., None, None], axis=1).squeeze(1) 
            for A in (P_v, P_ne, R_Inbhd)]
        P_new = jnp.where(emission_indicator[..., None], P_v, P_ne)

        R = data.R.at[I].set(next_positions, mode="drop")
        P = data.P.at[I].set(P_new, mode="drop")
        data = data._replace(R=R, P=P)

        return (data, key), (logdensities, probabilities, emission_indicator, is_bound, no_move)

    def boundstate_gibbs_step(self, data, step, C_boundstates, key):
        I = jnp.nonzero(C_boundstates == step, size=self.boundstate_limit, fill_value=self.pad_value)[0]
        I_particles = get_classes_by_id(I, data.bound_states, self.pad_value)

        (P_ne_particle, boundary_mask, U_Inbhd_diff, 
            K, P_ne, K_ne) = self.boundstate_gibbs_update_data(data, I, I_particles)
        
        probabilities, logdensities = calculate_probabilities(
            K, P_ne, K_ne, U_Inbhd_diff, boundary_mask, data.masses[I], self.mu, self.beta)
        next_indices, key = gibbs_sample(key, probabilities)

        # update states and data
        new_shifts = get_shifts()[next_indices]
        bs_indices = get_class_indices(I, data.bound_states, self.pad_value)
        new_shifts = new_shifts.at[bs_indices].get(mode="fill", fill_value=0)
        R = move(data.R, new_shifts, self.n)
        next_indices = next_indices.at[bs_indices].get(mode="fill", fill_value=self.pad_value)
        next_indices = next_indices[I_particles]
        P_ne_particle = jnp.take_along_axis(
            P_ne_particle, next_indices[..., None, None], axis=1, fill_value=self.pad_value).squeeze(1)
        P = data.P.at[I_particles].set(P_ne_particle, mode="drop")
        data = data._replace(R=R, P=P)

        return data, step + 1, C_boundstates, key

    def generate_fields(self, key):
        """Generates external field values at the top of the lattice and Brownian fluctuation fields."""
        key_external, key_brownian = jax.random.split(key)

        if self.drive:
            num_drive_samples = self.n * self.field_preloads
            unmasked_samples = wien_approximation(key_external, self.alpha, num_drive_samples)
            external_fields = jnp.reshape(unmasked_samples, (self.field_preloads, self.n))
        else:
            external_fields = jnp.zeros((self.field_preloads, self.n))

        num_brownian_samples = 2 * self.k * self.field_preloads
        samples = brownian_noise(key_brownian, self.beta, self.gamma, num_brownian_samples)
        brownian_fields = jnp.reshape(samples, (self.field_preloads, self.k, 2))

        return external_fields, 0, brownian_fields, 0

    def system_update_data(self, data):
        """Precomputable data needed for both phases of the update process."""
        # fields
        if self.drive:
            external_field = generate_drive_field(
                data.external_fields[data.ef_idx], data.R, data.P, self.M, masked, self.n, self.k)
        else:
            external_field = data.external_field
        brownian_field = data.brownian_fields[data.bf_idx]
        net_charge_force = self.Q[..., None] * (
            external_field[data.R[:, 0], data.R[:, 1]] + data.emission_field)
        net_mass_force = self.M[..., None] * data.mass_field
        brownian_force = jnp.sqrt(self.M)[..., None] * brownian_field
        net_force = net_charge_force + net_mass_force + brownian_force

        return external_field, data.ef_idx + 1, brownian_field, data.bf_idx + 1, net_force

    def particle_gibbs_update_data(self, data, I):
        """Multi-particle data needed for a single particle-phase Gibbs update step."""
        R_Inbhd = jax.vmap(generate_neighborhood, in_axes=(0, None, None))(
            data.R[I], self.n, self.pad_value)

        # potential terms
        U_Inbhd = neighborhood_potential_energies(
            I, R_Inbhd, self.Q, data.R, self.n, self.pad_value)
        U_I, U_Inbhd_diff, U_grad = compute_potential_terms(U_Inbhd)

        # kinetic terms
        M, P, K, P_v, P_ne, K_ne, M_double, impulse = compute_kinetic_terms(
            I, data.P, data.net_force, U_grad, self.M, self.mu, self.gamma)
        E_emit = jax.vmap(emission_energy)(K, P, impulse[:, 0], M_double)

        # bound state containment
        E_I = compute_energies(U_I, K_ne[:, 0])
        is_bound = bound(E_I)

        return R_Inbhd, U_Inbhd_diff, U_I, is_bound, M, K, P_v, P_ne, K_ne, E_emit 

    def boundstate_gibbs_update_data(self, data, I, I_particles):
        """Multi-particle data needed for a single bound state phase Gibbs update step."""
        R_Inbhd_particles = jax.vmap(generate_neighborhood, in_axes=(0, None, None))(
            data.R[I_particles], self.n, self.pad_value)

        # particle potential terms
        buffer_size = self.boundstate_limit
        U_Inbhd_particles = neighborhood_potential_energies_dynamic(
            I_particles, R_Inbhd_particles, self.Q, data.R, self.I, 
            self.n, buffer_size, self.pad_value)
        U_grad_particles = compute_potential_terms(U_Inbhd_particles)[-1]
        
        # particle kinetic terms
        _, _, K, _, P_ne_particle, K_ne, _, _ = compute_kinetic_terms(
            I_particles, data.P, data.net_force, U_grad_particles, self.M, self.mu, self.gamma)
        
        # merge statistics to boundstate level
        out_indicator = (R_Inbhd_particles == self.pad_value).astype(int)
        K, P_ne, K_ne, U_Inbhd, out_by_molecule = [
            compute_subgroup_sums(A, I_particles, data.bound_states, I, self.pad_value) 
            for A in (K, P_ne_particle, K_ne, U_Inbhd_particles, out_indicator)]
        U_Inbhd_diff = U_Inbhd - U_Inbhd[:, 0][:, None]

        boundary_mask = ~out_by_molecule[:, :, 1].astype(bool)

        return P_ne_particle, boundary_mask, U_Inbhd_diff, K, P_ne, K_ne

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
        K = kinetic_energy(data.P, 2 * self.M)
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