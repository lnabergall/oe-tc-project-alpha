from functools import partial
from dataclasses import dataclass, field, asdict
from typing import NamedTuple
from operator import itemgetter

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from loguru import logger

from sample import *
from physics import *
from geometry import *
from utils import *
from log import *


class InternalData(NamedTuple):
    # system
    time: int                                       # time step

    # particle sampling
    P_particles: jax.Array = None                   # independent Gibbs partition. 2D, qxp. 
    proposed_positions: jax.Array = None            # proposed particle positions. 2D, kx2.
    logdensities: jax.Array = None                  # proposal logdensities. 1D, k.
    p_accepts: jax.Array = None                     # acceptance probabilities. 1D, k.
    accepts: jax.Array = None                       # acceptance indicators. 1D, k, bool.

    ## particle physics
    # kinetics
    velocity_bounds: jax.Array = None               # velocity bounds. 1D, k.
    paths: jax.Array = None                         # proposal paths. 3D, kxsx2.

    # energetics
    energies: jax.Array = None                      # energies. 1D, k.
    proposal_energies: jax.Array = None             # proposal energies. 1D, k.
    deltaEs: jax.Array = None                       # proposal energies - energies. 1D, k.
    B_paths: jax.Array = None                       # energy barriers encountered along the paths. 1D, k.

    # work
    transfer_works: jax.Array = None                # work done by transfer field. 1D, k.
    total_transfer_works: jax.Array = None          # total work done by transfer field this step. 1D, k.
    brownian_works: jax.Array = None                # work done by Brownian field. 1D, k.
    total_brownian_works: jax.Array = None          # total work done by Brownian field this step. 1D, k.
    drive_works: jax.Array = None                   # work done by the external drive. 1D, k.
    total_drive_works: jax.Array = None             # total work done by the external drive this step. 1D, k.
    excitation_works: jax.Array = None              # work done by the energy field. 1D, k.
    total_excitation_works: jax.Array = None        # total work done by the energy field this step. 1D, k.
    works: jax.Array = None                         # work done during the proposed updates. 1D, k.
    total_works: jax.Array = None                   # total work done this time step. 1D, k.
    excess_works: jax.Array = None                  # total work - work. 1D, k.

    # bound state sampling
    P_boundstates: jax.Array = None                 # independent Gibbs partition. 2D, uxv.
    proposed_coms: jax.Array = None                 # proposed centers of mass. 2D, kx2.
    proposed_orientations: jax.Array = None         # proposed orientations. 1D, k.
    boundstate_logdensities: jax.Array = None       # proposal logdensities. 1D, k.
    boundstate_p_accepts: jax.Array = None          # acceptance probabilities. 1D, k.
    boundstate_accepts: jax.Array = None            # acceptance indicators. 1D, k, bool.

    # bound state physics
    # kinetics
    linear_velocity_bounds: jax.Array = None        # linear velocity bounds. 1D, k.
    angular_velocity_bounds: jax.Array = None       # angular velocity bounds. 1D, k.
    com_paths: jax.Array = None                     # center of mass proposal paths. 3D, kxrx2.

    # energetics
    boundstate_deltaEs: jax.Array = None            # proposal energies - energies. 1D, k.
    boundstate_B_paths: jax.Array = None            # energy barriers encountered along the com paths. 1D, k. 

    # work
    transferred_excess_works: jax.Array = None      # excess work transferred to bound states. 1D, k.
    boundstate_excess_works: jax.Array = None       # excess work recieved by bound states. 1D, k.
    total_forces: jax.Array = None                  # total force experienced by bound states. 2D, kx2. 
    total_torques: jax.Array = None                 # total torque experienced by bound states. 1D, k.
    boundstate_works: jax.Array = None              # work done during the proposed updates. 1D, k.


class SystemData(NamedTuple):
    # system
    R: jax.Array                            # particle positions. 2D, kx2.
    L: jax.Array = None                     # labeled occupation lattice. 3D, nxnx3.
    LQ: jax.Array = None                    # charge-labeled occupation lattice. 3D, nxnx3.
    L_test: jax.Array = None                # labeled occupation lattice for test particles. 3D, nxnx3.
    LQ_test: jax.Array = None               # charge-labeled occupation lattice for test particles. 3D, nxnx3.
    potential_energies: jax.Array = None    # particle potential energies. 1D, k.  

    # fields
    potential: jax.Array = None             # particles potentials. 1D, k.
    brownian_fields: jax.Array = None       # fields produced by Brownian fluctuations. 4D, axnxnx2.
    bf_idx: int = None                      # index of the Brownian field for the current timestep 
    external_fields: jax.Array = None       # top fields produced by external drive. 3D, bxnx2.
    ef_idx: int = None                      # index of the external field for the current timestep
    interaction_field: jax.Array = None     # interaction field experienced by each particle. 2D, kx2.
    brownian_field: jax.Array = None        # field produced by Brownian fluctations. 2D, kx2. 
    external_field: jax.Array = None        # field produced by external drive. 3D, nxnx2. 
    energy_field: jax.Array = None          # energy field produced by excitations and collisions. 2D, kx2. 
    transfer_field: jax.Array = None        # field transferred to particles by other influences. 2D, kx2.
    net_field: jax.Array = None             # net non-transferred field experienced by each particle. 2D, kx2. 

    # bound states
    bound_states: jax.Array = None          # bound state index of each particle. 1D, k.
    masses: jax.Array = None                # mass of each bound state. 1D, k, 0-padded.
    coms: jax.Array = None                  # center of mass of each bound state. 2D, kx2, 0-padded.
    MOIs: jax.Array = None                  # moment of inertia of each bound state. 1D, k, 0-padded.


def determine_all_bound_states(system, data):
    """Determines all bound states."""
    R = data.R
    potential_energy_factors_fn = jax.vmap(
        potential_energy_factors, in_axes=(0, 0, 0, None, None, None, None, None, None))
    bonds_fn = jax.vmap(compute_bonds, in_axes=(0, 0, 0, None, None))
    LQ = generate_property_lattice(data.L, system.Q, system.pad_value, system.charge_pad_value)

    all_energy_factors, all_factor_indices = potential_energy_factors_fn(
        system.I, R, system.Q, data.L, LQ, system.rho, system.point_func, system.factor_limit, system.pad_value)
    all_bonds = bonds_fn(
        system.I, all_energy_factors, all_factor_indices, system.bond_energy, system.pad_value)

    #molecule = compute_bound_state(1, all_bonds, system.pad_value)
    # bound_states = compute_bound_states(all_bonds, system.pad_value)
    # coms, masses = centers_of_mass(R, system.M, bound_states)
    # MOIs = moments_of_inertia(R, system.M, coms, bound_states, system.n)

    # return LQ, bound_states, masses, coms, MOIs, all_energy_factors, all_factor_indices
    return LQ, all_energy_factors, all_factor_indices


# @jax.profiler.annotate_function
# @jax.named_scope("initialize")
def initialize(system, key):
    key, key_positions, key_fields = jax.random.split(key, 3)

    R = sample_lattice_points(key_positions, system.n, system.k, replace=False)
    L, _ = generate_lattice(R, system.n, system.pad_value)
    data = SystemData(R=R, L=L)

    _ = determine_all_bound_states(system, data)
    # LQ, bound_states, masses, coms, MOIs, _, _ = determine_all_bound_states(system, data)

    # brownian_fields, bf_idx, external_fields, ef_idx = system.generate_fields(key_fields)
    # brownian_field = jnp.zeros((system.k, 2))
    # external_field = jnp.zeros((system.n, system.n, 2))

    # potential = potential_all(system.I, R, L, LQ, system.rho, system.point_func, system.pad_value)
    # potential_energies = potential_energy_all(system.Q, potential)
    # interaction_field = generate_interaction_field(
    #     system.I, R, L, LQ, system.rho, system.point_func, system.pad_value)
    # net_field = jnp.zeros((system.k, 2))

    # data = SystemData(R=R, L=L, LQ=LQ, L_test=L, LQ_test=LQ, brownian_fields=brownian_fields, 
    #                   bf_idx=bf_idx, external_fields=external_fields, ef_idx=ef_idx, 
    #                   brownian_field=brownian_field, external_field=external_field, 
    #                   potential=potential, potential_energies=potential_energies, 
    #                   interaction_field=interaction_field, net_field=net_field, 
    #                   bound_states=bound_states, masses=masses, coms=coms, MOIs=MOIs)

    # # energy and transfer fields
    # data = system.reset_fields(data)

    # internal data placeholders
    k_zeros_float = jnp.zeros((system.k,), dtype=float)
    k_zeros_int = jnp.zeros((system.k,), dtype=int)
    k_zeros_bool = jnp.zeros((system.k,), dtype=bool)
    k2_zeros_float = jnp.zeros((system.k, 2), dtype=float)
    k2_zeros_int = jnp.zeros((system.k, 2), dtype=int)
    P_particles = jnp.full((partition_size(2 * system.speed_limit), system.particle_limit), system.pad_value)
    paths = jnp.full((system.k, system.speed_limit+1, 2), system.pad_value)
    P_boundstates = jnp.full((system.k, system.boundstate_limit), system.pad_value)
    com_paths = jnp.full((system.k, system.boundstate_speed_limit+1, 2), system.pad_value)

    internal_data = InternalData(
        time=0, P_particles=P_particles, proposed_positions=k2_zeros_int, logdensities=k_zeros_float, 
        p_accepts=k_zeros_float, accepts=k_zeros_bool, velocity_bounds=k_zeros_float, paths=paths, 
        energies=k_zeros_float, proposal_energies=k_zeros_float, deltaEs=k_zeros_float, 
        B_paths=k_zeros_float, transfer_works=k_zeros_float, total_transfer_works=k_zeros_float, 
        brownian_works=k_zeros_float, total_brownian_works=k_zeros_float, drive_works=k_zeros_float, 
        total_drive_works=k_zeros_float, excitation_works=k_zeros_float, 
        total_excitation_works=k_zeros_float, works=k_zeros_float, total_works=k_zeros_float,
        excess_works=k_zeros_float, P_boundstates=P_boundstates, proposed_coms=k2_zeros_float,
        proposed_orientations=k_zeros_int, boundstate_logdensities=k_zeros_float, 
        boundstate_p_accepts=k_zeros_float, boundstate_accepts=k_zeros_bool, 
        linear_velocity_bounds=k_zeros_float, angular_velocity_bounds=k_zeros_float, com_paths=com_paths, 
        boundstate_deltaEs=k_zeros_float, boundstate_B_paths=k_zeros_float, 
        transferred_excess_works=k_zeros_float, boundstate_excess_works=k_zeros_float, 
        total_forces=k2_zeros_float, total_torques=k_zeros_float, boundstate_works=k_zeros_float)

    return data, internal_data


#@jax.profiler.annotate_function
# @jax.named_scope("run")
def run(system, steps):
    # initialize data, e.g. positions
    key, key_init = jax.random.split(system.key)
    jax_log_info("Initializing...")
    data, internal_data = initialize(system, key_init)
    # jax_log_debug("initial R: {}", data.R)

    # # start loop
    # jax_log_info("Running the system...")
    # step_fn = lambda i, args: system.step(*args)

    # key, subkey = jax.random.split(key)
    # field_data = system.generate_fields(subkey)
    # field_data.block_until_ready()

    # data, internal_data, key = jax.lax.fori_loop(0, steps, step_fn, (data, internal_data, key))
    jax.block_until_ready((data, internal_data))
    # jax_log_debug("final R: {}", data.R)
    # jax_log_info("System run complete.")

    return system, data, internal_data, key
    #return system, key


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

    # kinetics
    time_unit: float                        # unit converting between velocity and distance in a timestep
    speed_limit: int                        # upper bound on the distance a particle can move in one timestep
    boundstate_speed_limit: int             # upper bound on the distance a molecule can move in one timestep

    # energy
    mu: float                               # discernability threshold for energy

    # potential energy 
    rho: int                                # range of the potential
    point_func: callable                    # point function of the potential
    energy_lower_bound: float               # lower bound for the potential energy of a particle
    factor_limit: int                       # upper bound for the number of particles within range of a point

    # bonding 
    bond_energy: float                      # maximum energy defining a bond

    # external drive
    alpha: float                            # scale factor in Wien approximation to Planck's law

    # radiation emission
    epsilon: float                          # threshold constant of radiation emission
    delta: int                              # range of radiation emission

    # bookkeeping
    pad_value: int                          # scalar used for padding most arrays
    charge_pad_value: int                   # scalar used for padding charge arrays
    particle_limit: int                     # upper bound on the size of an independent set of particles
    boundstate_limit: int                   # upper bound on the size of an independent set of molecules
    boundstate_nbhr_limit: int              # upper bound on the number of 'neighbors' of a molecule

    # sampling 
    key: jax.Array                          # pseudo-random number generator
    kappa: float                            # scale factor used to pseudo-normalize densities
    proposal_samples: int                   # number of samples to generate a valid proposal in samplers
    field_preloads: int                     # number of Brownian and external fields to preload

    ### --- data fields with defaults ---

    # system
    lattice: jax.Array = field(default=None)    # lattice indices/positions, to be initialized. 3D, nxnx2.

    # particles
    pauli_exclusion: bool = True            # Pauli exclusion indicator
    I: jax.Array = field(default=None)      # particle indices, to be initialized. 1D, k.
    T: jax.Array = field(default=None)      # particle types, to be initialized. 1D, k.
    M: jax.Array = field(default=None)      # particle masses, to be initialized. 1D, k.
    Q: jax.Array = field(default=None)      # particle charges, to be initialized. 1D, k.

    def __post_init__(self):
        if self.T is None:
            self._assign_properties()

    def _assign_properties(self):
        self.I, self.T, self.M, self.Q = assign_properties(self.t, self.k, self.N, self.T_M, self.T_Q)
        self.lattice = lattice_positions(self.n)

    def tree_flatten(self):
        fields = asdict(self)
        static_fields = {attr: val for attr, val in fields.items() if not isinstance(val, jax.Array)}
        other_fields = tuple(val for val in fields.values() if isinstance(val, jax.Array))
        return (other_fields, static_fields)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jax.profiler.annotate_function
    @jax.named_scope("initialize")
    def initialize(self, key):
        key, key_positions, key_fields = jax.random.split(key, 3)

        R = sample_lattice_points(key_positions, self.n, self.k, replace=False)
        L, _ = generate_lattice(R, self.n, self.pad_value)
        data = SystemData(R=R, L=L)

        LQ, bound_states, masses, coms, MOIs, _, _ = self.determine_all_bound_states(data)

        brownian_fields, bf_idx, external_fields, ef_idx = self.generate_fields(key_fields)
        brownian_field = jnp.zeros((self.k, 2))
        external_field = jnp.zeros((self.n, self.n, 2))

        potential = potential_all(self.I, R, L, LQ, self.rho, self.point_func, self.pad_value)
        potential_energies = potential_energy_all(self.Q, potential)
        interaction_field = generate_interaction_field(
            self.I, R, L, LQ, self.rho, self.point_func, self.pad_value)
        net_field = jnp.zeros((self.k, 2))

        data = SystemData(R=R, L=L, LQ=LQ, L_test=L, LQ_test=LQ, brownian_fields=brownian_fields, 
                          bf_idx=bf_idx, external_fields=external_fields, ef_idx=ef_idx, 
                          brownian_field=brownian_field, external_field=external_field, 
                          potential=potential, potential_energies=potential_energies, 
                          interaction_field=interaction_field, net_field=net_field, 
                          bound_states=bound_states, masses=masses, coms=coms, MOIs=MOIs)

        # energy and transfer fields
        data = self.reset_fields(data)

        # internal data placeholders
        k_zeros_float = jnp.zeros((self.k,), dtype=float)
        k_zeros_int = jnp.zeros((self.k,), dtype=int)
        k_zeros_bool = jnp.zeros((self.k,), dtype=bool)
        k2_zeros_float = jnp.zeros((self.k, 2), dtype=float)
        k2_zeros_int = jnp.zeros((self.k, 2), dtype=int)
        P_particles = jnp.full((partition_size(2 * self.speed_limit), self.particle_limit), self.pad_value)
        paths = jnp.full((self.k, self.speed_limit+1, 2), self.pad_value)
        P_boundstates = jnp.full((self.k, self.boundstate_limit), self.pad_value)
        com_paths = jnp.full((self.k, self.boundstate_speed_limit+1, 2), self.pad_value)

        internal_data = InternalData(
            time=0, P_particles=P_particles, proposed_positions=k2_zeros_int, logdensities=k_zeros_float, 
            p_accepts=k_zeros_float, accepts=k_zeros_bool, velocity_bounds=k_zeros_float, paths=paths, 
            energies=k_zeros_float, proposal_energies=k_zeros_float, deltaEs=k_zeros_float, 
            B_paths=k_zeros_float, transfer_works=k_zeros_float, total_transfer_works=k_zeros_float, 
            brownian_works=k_zeros_float, total_brownian_works=k_zeros_float, drive_works=k_zeros_float, 
            total_drive_works=k_zeros_float, excitation_works=k_zeros_float, 
            total_excitation_works=k_zeros_float, works=k_zeros_float, total_works=k_zeros_float,
            excess_works=k_zeros_float, P_boundstates=P_boundstates, proposed_coms=k2_zeros_float,
            proposed_orientations=k_zeros_int, boundstate_logdensities=k_zeros_float, 
            boundstate_p_accepts=k_zeros_float, boundstate_accepts=k_zeros_bool, 
            linear_velocity_bounds=k_zeros_float, angular_velocity_bounds=k_zeros_float, com_paths=com_paths, 
            boundstate_deltaEs=k_zeros_float, boundstate_B_paths=k_zeros_float, 
            transferred_excess_works=k_zeros_float, boundstate_excess_works=k_zeros_float, 
            total_forces=k2_zeros_float, total_torques=k_zeros_float, boundstate_works=k_zeros_float)

        return data, internal_data

    #@jax.profiler.annotate_function
    @jax.named_scope("run")
    def run(self, steps):
        # initialize data, e.g. positions
        key, key_init = jax.random.split(self.key)
        jax_log_info("Initializing...")
        data, internal_data = self.initialize(key_init)
        jax_log_debug("initial R: {}", data.R)

        # start loop
        jax_log_info("Running the system...")
        step_fn = lambda i, args: self.step(*args)

        key, subkey = jax.random.split(key)
        field_data = self.generate_fields(subkey)

        data, internal_data, key = jax.lax.fori_loop(0, steps, step_fn, (data, internal_data, key))
        data.R.block_until_ready()
        jax_log_debug("final R: {}", data.R)
        jax_log_info("System run complete.")

        return data, internal_data, key

    @jax.profiler.annotate_function
    @jax.named_scope("step")
    def step(self, data, internal_data, key):
        # generate new fields if needed
        key, key_fields, key_particle, key_boundstate = jax.random.split(key, 4)
        null_fn = lambda _: (data.brownian_fields, data.bf_idx, data.external_fields, data.ef_idx)
        fields_consumed = data.bf_idx >= data.brownian_fields.shape[0]
        brownian_fields, bf_idx, external_fields, ef_idx = jax.lax.cond(
            fields_consumed, self.generate_fields, null_fn, key_fields)
        data = data._replace(brownian_fields=brownian_fields, bf_idx=bf_idx, 
                             external_fields=external_fields, ef_idx=ef_idx)

        # get fields for current time
        brownian_field, external_field = self.system_update_data(data)
        data = data._replace(brownian_field=brownian_field, external_field=external_field)

        # particle phase
        # determine a partition
        P_particles = independent_partition(data.R, data.L, self.particle_limit, 
                                            2 * self.speed_limit, self.pad_value)
        internal_data = internal_data._replace(P_particles=P_particles)

        # scan over partition
        particle_scan_fn = lambda carry, I: self.particle_gibbs_step(carry[0], I, carry[1])
        (data, _), particle_step_info = jax.lax.scan(particle_scan_fn, (data, key_particle), P_particles)

        # collect per-particle data
        compactify_fn = lambda V: compactify_partition(P_particles, V, self.k, self.pad_value)
        (velocity_bounds, proposed_positions, logdensities, (p_accepts, accepts), works, 
            excess_works, total_works, (paths, energies, proposal_energies, deltaEs, B_paths, 
            transfer_works, total_transfer_works, brownian_works, total_brownian_works, 
            drive_works, total_drive_works, excitation_works, total_excitation_works)) = jax.tree.map(
            compactify_fn, particle_step_info)
        internal_data = internal_data._replace(
            velocity_bounds=velocity_bounds, proposed_positions=proposed_positions, logdensities=logdensities, 
            works=works, excess_works=excess_works, total_works=total_works, p_accepts=p_accepts,
            accepts=accepts, paths=paths, energies=energies, proposal_energies=proposal_energies, 
            deltaEs=deltaEs, B_paths=B_paths, transfer_works=transfer_works, 
            total_transfer_works=total_excitation_works, brownian_works=total_brownian_works, 
            drive_works=drive_works, total_drive_works=total_drive_works, 
            excitation_works=excitation_works, total_excitation_works=total_excitation_works)
        potential_energies = data.potential_energies

        # reset energy and transfer fields
        data = self.reset_fields(data)

        # bound state phase
        # determine a partition
        I = get_boundstates(data.bound_states, self.pad_value)
        P_boundstates = independent_group_partition(I, data.bound_states, data.R, data.L, 
                                                    self.boundstate_limit, 2 * self.boundstate_speed_limit, 
                                                    self.boundstate_nbhr_limit, self.pad_value)
        internal_data = internal_data._replace(P_boundstates=P_boundstates)

        # sum excess works by bound states
        transferred_excess_works = excess_works * transfer_mask(excess_works, potential_energies, self.epsilon)
        boundstate_excess_works = jax.ops.segment_sum(
            transferred_excess_works, data.bound_states, num_segments=self.k)
        internal_data = internal_data._replace(transferred_excess_works=transferred_excess_works,
                                               boundstate_excess_works=boundstate_excess_works)
        boundstate_excess_works = boundstate_excess_works.at[
            jnp.where(P_boundstates == self.pad_value, 2*self.k, P_boundstates)].get(mode="fill", fill_value=0.0)

        # scan over partition
        boundstate_scan_fn = lambda carry, s: self.boundstate_gibbs_step(
            carry[0], s[0], carry[1], s[1])
        (data, _), boundstate_step_info = jax.lax.scan(
            boundstate_scan_fn, (data, key_boundstate), (P_boundstates, boundstate_excess_works))

        # collect per-boundstate data
        compactify_fn = lambda V: compactify_partition(P_boundstates, V, self.k, self.pad_value)
        (linear_velocity_bounds, angular_velocity_bounds, proposed_coms, proposed_orientations, 
            logdensities, (p_accepts, accepts), works, total_forces, total_torques, 
            (com_paths, deltaEs, B_paths)) = jax.tree.map(compactify_fn, boundstate_step_info)

        internal_data = internal_data._replace(
            linear_velocity_bounds=linear_velocity_bounds, angular_velocity_bounds=angular_velocity_bounds,
            proposed_coms=proposed_coms, proposed_orientations=proposed_orientations, 
            boundstate_logdensities=logdensities, boundstate_p_accepts=p_accepts, boundstate_accepts=accepts,
            com_paths=com_paths, boundstate_works=works, total_forces=total_forces, 
            total_torques=total_torques, boundstate_deltaEs=deltaEs, boundstate_B_paths=B_paths)

        # generate excitation emissions
        energy_field = self.generate_excitations(data, excess_works, potential_energies)

        # reset fields, work, etc.?
        net_field = jnp.zeros((self.k, 2))

        data = data._replace(bf_idx=bf_idx+1, ef_idx=ef_idx+1, potential_energies=jnp.zeros(self.k), 
                             energy_field=energy_field, net_field=net_field)

        return data, internal_data, key

    @jax.profiler.annotate_function
    @jax.named_scope("particle_gibbs_step")
    def particle_gibbs_step(self, data, I, key):
        I_mask = jnp.isin(self.I, I)
        state = (I, data.R[I])

        interaction_field, L_test, LQ_test = self.gibbs_update_data(data, I)
        data = data._replace(interaction_field=interaction_field, L_test=L_test, LQ_test=LQ_test)

        update_data_fn = jax.vmap(self.particle_update_data, in_axes=(None, (0, 0)))
        velocity_bounds = update_data_fn(data, state)

        # generate proposals
        keys = jax.random.split(key, num=I.shape[0]+1)
        key, keys_proposal = keys[0], keys[1:]
        proposal_generator_fn = jax.vmap(self.particle_proposal_generator, in_axes=(None, (0, 0), 0, 0))
        proposed_positions = proposal_generator_fn(data, state, keys_proposal, velocity_bounds)

        # logdensities
        logdensity_fn = jax.vmap(self.particle_logdensity_function, in_axes=(None, (0, 0), 0, 0))
        (logdensities, works, total_works), extra_data = logdensity_fn(
            data, state, proposed_positions, velocity_bounds)
        excess_works = total_works - works

        # sample next positions
        keys = jax.random.split(key, num=I.shape[0]+1)
        key, keys_sample = keys[0], keys[1:]
        sample_fn = jax.vmap(sample, in_axes=(0, 0, 0, 0, None))
        accepted_positions, sample_info = sample_fn(
            keys_sample, logdensities, state[1], proposed_positions, self.kappa)
        R = data.R.at[jnp.where(I == self.pad_value, 2*self.k, I)].set(accepted_positions, mode="drop")
        data = data._replace(R=R)

        # compute particle potential energies
        potential = potential_all(self.I, data.R, data.L, data.LQ, self.rho, self.point_func, self.pad_value)
        potential_energies = potential_energy_all(self.Q, potential)
        
        # get bound states and transfer excess work
        data, LQ, bound_states, masses, coms, MOIs, _, _ = self.determine_bound_states(data, I)
        data = data._replace(LQ=LQ, bound_states=bound_states, masses=masses, coms=coms, MOIs=MOIs,
                             potential=potential, potential_energies=potential_energies)
        state_extra = (state[0], I_mask, state[1], accepted_positions)
        transfer_field, net_field = self.generate_transfer_field(
            data, state_extra, excess_works, potential_energies)
        data = data._replace(transfer_field=transfer_field, net_field=net_field)

        return (data, key), (velocity_bounds, proposed_positions, logdensities, 
                sample_info, works, excess_works, total_works, extra_data)

    @jax.profiler.annotate_function
    @jax.named_scope("boundstate_gibbs_step")
    def boundstate_gibbs_step(self, data, I, key, excess_works):
        I_particles = get_particles(I, data.bound_states, self.pad_value)
        particle_mask = I_particles != self.pad_value
        molecule_indices = bound_state_indices(I, data.bound_states, 2*self.k)
        state = (I, molecule_indices, I_particles, particle_mask)

        (interaction_field, L_test, LQ_test, linear_velocity_bounds, angular_velocity_bounds, 
            all_prev_energy_factors, all_prev_factor_indices) = self.boundstate_pregibbs_update_data(data, state)
        data = data._replace(interaction_field=interaction_field, L_test=L_test, LQ_test=LQ_test)

        # generate proposals
        keys = jax.random.split(key, num=I.shape[0]+1)
        key, keys_proposal = keys[0], keys[1:]
        proposal_generator_fn = jax.vmap(self.boundstate_proposal_sampler, in_axes=(None, 0, 0, 0, 0, None))
        proposed_coms, proposed_orientations = proposal_generator_fn(
            data, I, keys_proposal, linear_velocity_bounds, angular_velocity_bounds, self.proposal_samples)

        # particle positions of proposals
        R_proposed = self.boundstate_gibbs_update_data(data, state, proposed_coms, proposed_orientations)

        # logdensities
        (logdensities, works, total_forces, total_torques), extra_data = self.bulk_boundstate_logdensity_function(
            data, state, R_proposed, proposed_coms, proposed_orientations, excess_works)
        excess_works = excess_works - works

        # sample next positions
        keys = jax.random.split(key, num=I.shape[0]+1)
        key, keys_sample = keys[0], keys[1:]
        sample_fn = jax.vmap(sample, in_axes=(0, 0, 0, 0, None))
        accepted_coms, sample_info = sample_fn(
            keys_sample, logdensities, data.coms[I], proposed_coms, self.kappa)
        accepted_indicators = sample_info[1]
        accepted_orientations = jnp.where(accepted_indicators, proposed_orientations, 0)
        accept_mask = particle_mask * accepted_indicators[data.bound_states]
        R = jnp.where(jnp.expand_dims(accept_mask, axis=-1), R_proposed, data.R)
        data = data._replace(R=R)

        # determine new bound states
        prev_bound_states = data.bound_states
        (data, LQ, bound_states, masses, coms, MOIs, all_energy_factors, 
            all_factor_indices) = self.determine_bound_states(data, I_particles)
        data = data._replace(LQ=LQ, bound_states=bound_states, masses=masses, coms=coms, MOIs=MOIs)

        # transfer excess work
        energy_field = add_boundstate_energy_transfers(
            self.I, I, excess_works, total_forces, total_torques, coms, data.R, bound_states, 
            prev_bound_states, data.energy_field, all_energy_factors, all_factor_indices, 
            all_prev_energy_factors, all_prev_factor_indices, self.pad_value)
        data = data._replace(energy_field=energy_field)

        return (data, key), (linear_velocity_bounds, angular_velocity_bounds, 
                proposed_coms, proposed_orientations, logdensities, sample_info, works, 
                total_forces, total_torques, extra_data)

    def reset_fields(self, data):
        energy_field = jnp.zeros((self.k, 2))
        transfer_field = jnp.zeros((self.k, 2))
        data = data._replace(energy_field=energy_field, transfer_field=transfer_field)
        return data

    def generate_fields(self, key):
        key_brownian, key_external = jax.random.split(key)
        brownian_fields, bf_idx = self.generate_brownian_fields(key_brownian, self.field_preloads)
        external_fields, ef_idx = self.generate_external_drives(key_external, self.field_preloads)
        return brownian_fields, bf_idx, external_fields, ef_idx

    def generate_brownian_fields(self, key, steps=1):
        """Generates field values of Brownian fluctuations and the index of the starting set."""
        num_samples = 2 * self.k * steps
        samples = brownian_noise(key, self.beta, self.gamma, num_samples)
        brownian_fields = jnp.reshape(samples, (steps, self.k, 2))
        return brownian_fields, 0

    def generate_external_drives(self, key, steps=1):
        """Generates external field values at the top of the lattice and the index of the starting set."""
        num_samples = self.n * steps
        unmasked_samples = wien_approximation(key, self.beta, self.alpha, num_samples)
        external_fields = jnp.reshape(unmasked_samples, (steps, self.n))
        return external_fields, 0

    def system_update_data(self, data):
        """Particle-independent data needed for the update process."""
        brownian_field = data.brownian_fields[data.bf_idx]
        external_field = generate_drive_field(self.n, data.R, data.external_fields[data.ef_idx], masked)
        
        return brownian_field, external_field

    def gibbs_update_data(self, data, state):
        """Multi-particle data needed for a single Gibbs update step."""
        I = state

        interaction_field = generate_interaction_field(
            self.I, data.R, data.L, data.LQ, self.rho, self.point_func, self.pad_value)

        # labeled occupation lattices without particles in I
        L_test = remove_from_lattice(data.L, I, data.R, self.pad_value)
        LQ_test = generate_property_lattice(L_test, self.Q, self.pad_value, self.charge_pad_value)

        return interaction_field, L_test, LQ_test

    def particle_update_data(self, data, state):
        """Data needed for multiple particle-dependent functions of the update process."""
        i, position = state
        mass, charge = self.M[i], self.Q[i]
        molecule_mass = data.masses[data.bound_states[i]]

        velocity_bound = compute_particle_velocity_bound(i, position, data.interaction_field, 
            data.brownian_field, data.external_field, data.energy_field, data.transfer_field, 
            mass, charge, molecule_mass, self.gamma, self.time_unit)
        velocity_bound = jnp.floor(velocity_bound)
        velocity_bound = jnp.min(jnp.stack((velocity_bound, self.speed_limit)))

        return velocity_bound

    def boundstate_pregibbs_update_data(self, data, state):
        I, _, I_particles, _ = state

        interaction_field, L_test, LQ_test = self.gibbs_update_data(data, I_particles)

        potential_energy_factors_fn = jax.vmap(
            potential_energy_factors, in_axes=(0, 0, 0, None, None, None, None, None, None))
        all_energy_factors, all_factor_indices = potential_energy_factors_fn(
            self.I, data.R, self.Q, data.L, data.LQ, self.rho, 
            self.point_func, self.factor_limit, self.pad_value)
        
        linear_velocity_bounds, angular_velocity_bounds = compute_boundstate_velocity_bounds(
            I, data.R, data.bound_states, data.interaction_field, data.net_field, 
            data.masses, data.coms, self.M, self.Q, self.gamma, self.time_unit)
        linear_velocity_bounds = jnp.floor(linear_velocity_bounds)
        linear_velocity_bounds = jnp.min(jnp.stack((linear_velocity_bounds, 
            jnp.broadcast_to(jnp.array(self.speed_limit), I.shape)), axis=-1), axis=-1)

        return (interaction_field, L_test, LQ_test, linear_velocity_bounds, 
                angular_velocity_bounds, all_energy_factors, all_factor_indices)

    def boundstate_gibbs_update_data(self, data, state, proposed_coms, proposed_orientations):
        """Multi-molecule data needed for a single bound state Gibbs update step."""
        I = state[0]
        coms = data.coms[data.bound_states]
        proposed_coms = data.coms.at[jnp.where(I == self.pad_value, 2*self.k, I)].set(proposed_coms, mode="drop")
        proposed_coms = proposed_coms[data.bound_states]
        proposed_orientations = proposed_orientations[data.bound_states]

        R_moved = move(data.R, coms, proposed_coms)
        R_proposed = rotate(R_moved, proposed_coms, proposed_orientations)

        return R_proposed

    # @jax.profiler.annotate_function
    # @jax.named_scope("determine_bound_states")
    def determine_bound_states(self, data, I):
        """Determines all bound states."""
        L = add_to_lattice(data.L_test, I, data.R, self.pad_value)
        data = data._replace(L=L)

        bound_states_data = self.determine_all_bound_states(data)

        return data, *bound_states_data

    # @jax.profiler.annotate_function
    # @jax.named_scope("determine_all_bound_states")
    def determine_all_bound_states(self, data):
        """Determines all bound states."""
        R = data.R
        potential_energy_factors_fn = jax.vmap(
            potential_energy_factors, in_axes=(0, 0, 0, None, None, None, None, None, None))
        bonds_fn = jax.vmap(compute_bonds, in_axes=(0, 0, 0, None, None))
        LQ = generate_property_lattice(data.L, self.Q, self.pad_value, self.charge_pad_value)

        all_energy_factors, all_factor_indices = potential_energy_factors_fn(
            self.I, R, self.Q, data.L, LQ, self.rho, self.point_func, self.factor_limit, self.pad_value)
        all_bonds = bonds_fn(
            self.I, all_energy_factors, all_factor_indices, self.bond_energy, self.pad_value)

        bound_states = compute_bound_states(all_bonds, self.pad_value)
        coms, masses = centers_of_mass(R, self.M, bound_states)
        MOIs = moments_of_inertia(R, self.M, coms, bound_states, self.n)

        return LQ, bound_states, masses, coms, MOIs, all_energy_factors, all_factor_indices

    @jax.profiler.annotate_function
    @jax.named_scope("generate_transfer_field")
    def generate_transfer_field(self, data, state, excess_works, potential_energies):
        """Generate new transfer forces, which are added to the transfer field."""
        I, I_mask, positions, accepted_positions = state
        I_mask_expand = jnp.broadcast_to(jnp.expand_dims(I_mask, axis=-1), I_mask.shape + (2,))
        I_alt = jnp.where(I == self.pad_value, 2*self.k, I)
        zeros = jnp.zeros(self.k, dtype=float)
        excess_works = zeros.at[I_alt].set(excess_works, mode="drop")

        end_field_fn = jax.vmap(calculate_end_field, in_axes=(None, None, None, None, 0, 0, 0, 0, None))
        end_field = end_field_fn(data.brownian_field, data.external_field, data.energy_field, 
            data.transfer_field, I, positions, accepted_positions, self.M[I], self.gamma)
        net_field = data.net_field.at[I_alt].set(end_field, mode="drop")
        new_transfer_field = jnp.where(I_mask_expand, 0.0, data.transfer_field)

        transfer_field = generate_full_transferred_field(
            I, net_field, data.R, data.bound_states, data.masses, 
            data.coms, data.MOIs, self.I, self.pad_value)
        net_field = jnp.where(I_mask_expand, 0.0, net_field)
        transfer_field = transfer_field.at[:, :].multiply(
            jnp.expand_dims(transfer_mask(excess_works, potential_energies, self.epsilon), axis=-1))
        net_field = net_field.at[:, :].add(transfer_field)
        transfer_field = transfer_field.at[:, :].add(new_transfer_field) 

        return transfer_field, net_field

    @jax.profiler.annotate_function
    @jax.named_scope("generate_excitations")
    def generate_excitations(self, data, excess_works, potential_energies):
        """
        Generate new excitations, which are added to the energy field. 
        Occurs at the end of a timestep, after all state updates.
        """
        L = generate_unlabeled_lattice(data.R, self.n)
        excitations_fn = jax.vmap(calculate_excitations, in_axes=(0, None, None, 0, None))

        # generate excitations for *all* particles, consider changing to loop if vast majority don't excite
        excitation_arrays, position_arrays = excitations_fn(self.I, data.R, L, excess_works, self.delta)

        # filter for those that generate excitations (and are not padding)
        mask = excitation_mask(excess_works, potential_energies, self.epsilon)
        mask = jnp.expand_dims(jnp.expand_dims(mask, axis=-1), axis=-1)
        excitation_arrays = excitation_arrays.at[:, :, :].multiply(mask)
        
        # merge into an energy field
        energy_field = merge_excitations(data.R, L, excitation_arrays, position_arrays)
        energy_field = energy_field[data.R[:, 0], data.R[:, 1]]
        
        return energy_field

    @jax.profiler.annotate_function
    @jax.named_scope("particle_logdensity_function")
    def particle_logdensity_function(self, data, state, proposed_position, velocity_bound):
        i, position = state
        mass, charge = self.M[i], self.Q[i]

        # compute particle path
        path = canonical_shortest_path(position, proposed_position, self.speed_limit+1, self.n, self.pad_value)

        # assemble potential function
        potential_energy_func = make_potential_energy_func(
            data.L_test, data.LQ_test, self.rho, self.point_func, self.pad_value, self.charge_pad_value)

        # energy difference
        energy = potential_energy_func(position, charge)
        proposal_energy = potential_energy_func(proposed_position, charge)
        deltaE = proposal_energy - energy

        # energy barrier
        max_boundary_energy = jnp.max(jnp.array((energy, proposal_energy)))
        B_path = energy_barrier(path, potential_energy_func, max_boundary_energy, 
                                charge, self.energy_lower_bound)

        # work
        transfer_work, total_transfer_work = calculate_work_step(
            position, proposed_position, data.transfer_field[i], mass)
        brownian_work, total_brownian_work = calculate_work_step(
            position, proposed_position, data.brownian_field[i], jnp.sqrt(mass))
        drive_work, total_drive_work = calculate_work(path, data.external_field, mass)
        excitation_work, total_excitation_work = calculate_excitation_work(
            i, position, proposed_position, data.energy_field)
        work = brownian_work + transfer_work + drive_work + excitation_work
        total_work = calculate_total_work(
            total_brownian_work + total_transfer_work + total_drive_work + total_excitation_work, 
            data.brownian_field, data.external_field, data.transfer_field, 
            i, mass, position, proposed_position, velocity_bound)

        logdensity = self.beta * (work - deltaE - B_path)
        # return both, for excess work and work applied even if proposal rejected
        return (logdensity, work, total_work), (path, energy, proposal_energy, deltaE, B_path,
                transfer_work, total_transfer_work, brownian_work, total_brownian_work,
                drive_work, total_drive_work, excitation_work, total_excitation_work)

    @jax.profiler.annotate_function
    @jax.named_scope("bulk_boundstate_logdensity_function")
    def bulk_boundstate_logdensity_function(self, data, state, R_proposed, proposed_coms, 
                                            proposed_orientations, work_budgets):
        I, molecule_indices, _, particle_mask = state
        masses, coms, MOIs = data.masses[I], data.coms[I], data.MOIs[I]

        # compute molecule path, first translation then add rotation at the end
        com_paths = jax.vmap(canonical_shortest_path, in_axes=(0, 0, None, None, None))(
            coms, proposed_coms, self.boundstate_speed_limit+1, self.n, self.pad_value)
        com_paths_byparticle = com_paths[molecule_indices]
        particle_paths_nospin_nofilter = jax.vmap(shift_path)(com_paths_byparticle, data.R)
        particle_paths_nofilter = jax.vmap(append_on_path)(particle_paths_nospin_nofilter, R_proposed)

        # assemble potential function
        potential_energy_func = make_potential_energy_func(
            data.L_test, data.LQ_test, self.rho, self.point_func, self.pad_value, self.charge_pad_value)
        bulk_potential_energy_func = jax.vmap(potential_energy_func)

        # energy difference, all particles 
        energies = bulk_potential_energy_func(data.R, self.Q)
        proposal_energies = bulk_potential_energy_func(R_proposed, self.Q)
        deltaEs_by_particle = particle_mask * (proposal_energies - energies)     # mask probably not needed
        deltaEs = jnp.zeros(I.shape).at[molecule_indices].add(deltaEs_by_particle, mode="drop")

        # energy barrier, all particle paths
        energy_barrier_fn = jax.vmap(energy_barrier, in_axes=(0, None, 0, 0, None))
        max_boundary_energies = jnp.max(jnp.stack((energies, proposal_energies), axis=-1), axis=-1)
        B_paths_by_particle = energy_barrier_fn(
            particle_paths_nofilter, potential_energy_func, 
            max_boundary_energies, self.Q, self.energy_lower_bound)
        B_paths_by_particle = B_paths_by_particle.at[:].multiply(particle_mask)
        B_paths = jnp.zeros(I.shape).at[molecule_indices].add(B_paths_by_particle, mode="drop")

        # work, com + orientation only
        work_fn = jax.vmap(calculate_boundstate_work, in_axes=(0, None, None, None, None, 0, 0, 0))
        works, total_forces, total_torques = work_fn(I, data.bound_states, data.net_field, data.R, 
                                                     self.M, coms, proposed_coms, proposed_orientations)
        works = jnp.max(jnp.stack((works, work_budgets), axis=-1), axis=-1)

        logdensities = self.beta * (works - deltaEs - B_paths)
        return (logdensities, works, total_forces, total_torques), (com_paths, deltaEs, B_paths)

    @jax.profiler.annotate_function
    @jax.named_scope("particle_proposal_sampler")
    def particle_proposal_sampler(self, data, state, key, range_, num_samples=1):
        """
        Samples 'num_samples' particle position proposals uniformly at random within range 
        and selects the first proposal that does not violate Pauli exclusion.  
        """
        i, position = state
        q = self.Q[i]
        proposed_positions = uniform_proposal_generator(key, position, range_, num_samples)
        invalid_fn = lambda r: violates_pauli_exclusion(data.LQ_test[tuple(r)], q)
        invalids = jax.vmap(invalid_fn)(proposed_positions)
        proposed_position = proposed_positions[jnp.argmin(invalids)]
        return proposed_position

    @jax.profiler.annotate_function
    @jax.named_scope("particle_proposal_generator")
    def particle_proposal_generator(self, data, state, key, range_):
        """
        Samples a particle position proposal uniformly at random from the positions within range
        that do not violate Pauli exclusion.  
        """
        key, subkey = jax.random.split(key)
        i, position = state
        q = self.Q[i]
        Lr_square, Lr_square_positions = centered_square(data.L_test, position, self.speed_limit)
        Qr_square, _ = centered_square(data.LQ_test, position, self.speed_limit)
        distances = lattice_distances_2D(position, Lr_square_positions, self.n)

        # determine possible positions
        invalid_fn = jax.vmap(jax.vmap(
            violates_pauli_exclusion, in_axes=(0, None, None)), in_axes=(0, None, None))
        valid_mask = (distances <= range_) & ~invalid_fn(Qr_square, q, self.charge_pad_value)
        valid_count = jnp.sum(valid_mask)

        # sample proposal index
        proposal_idx = jax.random.randint(subkey, (), 0, valid_count)

        # convert index to position
        flat_valid_mask = jnp.ravel(valid_mask)
        proposal_idx = ith_nonzero_index(flat_valid_mask, proposal_idx)
        proposal_indices = unravel_2Dindex(proposal_idx, Lr_square_positions.shape[1])
        proposed_position = Lr_square_positions[tuple(proposal_indices)]

        return proposed_position

    @jax.profiler.annotate_function
    @jax.named_scope("boundstate_proposal_sampler")
    def boundstate_proposal_sampler(self, data, state, key, range_, angular_range, num_samples=1):
        """
        Samples 'num_samples' bound state centers-of-mass and orientation proposals uniformly at random 
        within range and selects the first proposal that does not violate strong Pauli exclusion."""
        i = state
        com = data.coms[i]
        Rs_proposed = jnp.broadcast_to(data.R, (num_samples,) + data.R.shape)
        curr_coms = jnp.broadcast_to(jnp.expand_dims(data.coms[data.bound_states], axis=0), 
                                     (num_samples,) + data.bound_states.shape + (2,))

        # generate proposals
        proposed_coms, proposed_orientations = uniform_boundstate_proposal_generator(
            key, com, range_, num_samples, angular_range)

        # move particles by proposed coms
        new_coms = jnp.broadcast_to(data.coms, (num_samples,) + data.coms.shape).at[:, i, :].set(proposed_coms)
        new_coms = new_coms[:, data.bound_states, :]
        Rs_proposed = move(Rs_proposed, curr_coms, new_coms)

        # rotate particles by proposed orientations
        new_orientations = jnp.zeros((num_samples, self.k), dtype=int)
        new_orientations = new_orientations.at[:, i].set(proposed_orientations)
        new_orientations = new_orientations[:, data.bound_states]
        Rs_proposed = jax.vmap(rotate)(Rs_proposed, new_coms, standardize_angle(new_orientations))

        # get strong Pauli valid or first
        invalids = jax.vmap(violates_strong_pauli_exclusion)(Rs_proposed)
        proposed_idx = jnp.argmin(invalids)

        return proposed_coms[proposed_idx], proposed_orientations[proposed_idx]

### filter out small work?

### generate initial lattice L and any other needed data
### generate excitation emissions after all state updates, phase 1 and 2

### need to cutoff velocity bound?
### following that, cuttoff paths and consider vmap version of work functions, etc.

### use sparse arrays to speed up bound state discovery, among other things
### vmap over particles (all? independent/dense set?), finding each molecule, then merge


def assign_properties(t, k, N, T_M, T_Q):
    I = jnp.arange(k)
    T = jnp.repeat(jnp.arange(t), N, total_repeat_length=k)         # particle types
    M = jnp.fromfunction(lambda i: T_M[T[i]], (k,), dtype=int)      # particle masses
    Q = jnp.fromfunction(lambda i: T_Q[T[i]], (k,), dtype=int)      # particle charges
    return I, T, M, Q