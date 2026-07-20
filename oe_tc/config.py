"""Configuration objects for OE-TC model.

Shape-affecting values are kept in :class:`StaticConfig`. Physical parameters
live in the ``NamedTuple`` :class:`Params`, so changing them does not force JAX
to compile a new executable when their shapes and dtypes stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class StaticConfig:
    """Shape and scheduler limits that are static during a compiled run."""

    n: int
    num_particles: int
    component_max_iters: int = 64
    molecule_max_retries: int = 8
    mis_max_iters: int = 32
    planck_terms: int = 64

    def __post_init__(self) -> None:
        integer_names = (
            "n",
            "num_particles",
            "component_max_iters",
            "molecule_max_retries",
            "mis_max_iters",
            "planck_terms",
        )
        for name in integer_names:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be a host integer")

        # At n=2 the periodic +x and -x neighbors coincide, invalidating the
        # four-direction bond mask and checkerboard matching assumptions.
        if self.n < 4 or self.n % 2:
            raise ValueError("n must be an even integer of at least four")
        if self.n * self.n > 2_147_483_647:
            raise ValueError("n**2 must fit in signed 32-bit lattice indices")
        if not 0 < self.num_particles <= self.n * self.n:
            raise ValueError("num_particles must be in [1, n**2]")
        if self.component_max_iters < 1:
            raise ValueError("component_max_iters must be positive")
        if self.molecule_max_retries < 1:
            raise ValueError("molecule_max_retries must be positive")
        if self.mis_max_iters < 1:
            raise ValueError("mis_max_iters must be positive")
        if self.planck_terms < 1:
            raise ValueError("planck_terms must be positive")

    @property
    def empty(self) -> int:
        """Sentinel stored at unoccupied lattice sites."""

        return self.num_particles


class Params(NamedTuple):
    """Dimensionless physical and per-sweep kinetic parameters."""

    eta: float
    energy_floor: float
    heat_capacity: float
    bath_temperature: float
    source_beta: float

    gamma_base: float
    gamma_exposure: float
    kappa_base: float
    kappa_exposure: float

    conduction_contact: float
    conduction_bond: float
    bath_energy_quantum: float

    bath_channel_probability: float
    translation_probability: float
    translation_frequency: float
    rotation_frequency: float
    translation_drag_reference: float
    rotation_drag_reference: float

    bond_frequency: float
    catalysis_strength: float
    catalysis_cap: float

    source_enabled: bool
    second_conduction: bool


def default_params() -> Params:
    """Return conservative numerical defaults for smoke tests and calibration.

    These values define a runnable baseline, not a claimed physical calibration.
    Production experiments should record all values in their run manifest.
    """

    return Params(
        eta=0.1,
        energy_floor=0.1,
        heat_capacity=10.0,
        bath_temperature=1.0,
        source_beta=4.0,
        gamma_base=1.0,
        gamma_exposure=1.0,
        kappa_base=0.01,
        kappa_exposure=0.04,
        conduction_contact=0.02,
        conduction_bond=0.10,
        bath_energy_quantum=0.1,
        bath_channel_probability=0.5,
        translation_probability=0.8,
        translation_frequency=0.25,
        rotation_frequency=0.10,
        translation_drag_reference=1.0,
        rotation_drag_reference=1.0,
        bond_frequency=0.02,
        catalysis_strength=0.5,
        catalysis_cap=4.0,
        source_enabled=True,
        second_conduction=False,
    )


def validate_params(params: Params) -> None:
    """Validate parameter values at the float32 precision used by JAX."""

    boolean_names = ("source_enabled", "second_conduction")
    for name in boolean_names:
        if type(getattr(params, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")

    converted: dict[str, float] = {}
    for name in params._fields:
        if name in boolean_names:
            continue
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        try:
            host_value = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite real number") from exc
        if not math.isfinite(host_value):
            raise ValueError(f"{name} must be finite")

        # Params enter jitted kernels as float32 values. Validate that cast
        # explicitly so a finite host value cannot silently become infinity or
        # zero on device.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            device_value = np.float32(host_value)
        if not np.isfinite(device_value):
            raise ValueError(f"{name} must remain finite in float32")
        if host_value != 0.0 and device_value == 0.0:
            raise ValueError(f"{name} underflows to zero in float32")
        converted[name] = float(device_value)

    # All physical constraints must hold after the same rounding that occurs
    # at the compiled-kernel boundary, not merely for the host inputs.
    device = params._replace(**converted)

    if not 0.0 < device.eta < 1.0:
        raise ValueError("eta must lie strictly between zero and one")
    if device.energy_floor <= 0.0:
        raise ValueError("energy_floor must be positive")
    if device.heat_capacity <= 0.0 or device.bath_temperature <= 0.0:
        raise ValueError("heat_capacity and bath_temperature must be positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        bath_equilibrium = np.float32(
            np.float32(device.heat_capacity) * np.float32(device.bath_temperature)
        )
    if not np.isfinite(bath_equilibrium):
        raise ValueError("bath-equilibrium energy must remain finite in float32")
    if float(bath_equilibrium) < device.energy_floor:
        raise ValueError("bath-equilibrium energy must not be below energy_floor")
    if device.source_beta <= 0.0:
        raise ValueError("source_beta must be positive")
    if device.gamma_base <= 0.0 or device.gamma_exposure < 0.0:
        raise ValueError("invalid viscous coupling")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kappa_peak = np.float32(
            np.float32(device.kappa_base) + np.float32(device.kappa_exposure)
        )
    if not 0.0 <= device.kappa_base <= float(kappa_peak) <= 1.0:
        raise ValueError("kappa must remain in [0, 1]")
    if not 0.0 <= device.conduction_contact < device.conduction_bond <= 1.0:
        raise ValueError("conduction must satisfy 0 <= contact < bond <= 1")
    if device.bath_energy_quantum <= 0.0:
        raise ValueError("bath_energy_quantum must be positive")
    if not 0.0 <= device.bath_channel_probability <= 1.0:
        raise ValueError("bath_channel_probability must lie in [0, 1]")
    if not 0.0 <= device.translation_probability <= 1.0:
        raise ValueError("translation_probability must lie in [0, 1]")
    if min(
        device.translation_frequency,
        device.rotation_frequency,
        device.bond_frequency,
    ) < 0.0:
        raise ValueError("kinetic frequencies cannot be negative")
    if (
        device.translation_drag_reference <= 0.0
        or device.rotation_drag_reference <= 0.0
    ):
        raise ValueError("reference drags must be positive")
    if device.catalysis_strength < 0.0 or device.catalysis_cap < 1.0:
        raise ValueError("invalid catalytic parameters")
