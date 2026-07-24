import math

import pytest

from oe_tc.config import StaticConfig, default_params, validate_params


def test_default_configuration_is_valid():
    config = StaticConfig(n=16, num_particles=32)
    params = default_params()
    validate_params(params)

    assert config.empty == 32
    bath_mode = params.heat_capacity * params.bath_temperature
    bond_breaking_cost = 1.0 - params.eta
    assert bath_mode == pytest.approx(0.5)
    assert params.energy_floor < bath_mode < bond_breaking_cost
    assert params.conduction_energy_quantum < bath_mode
    assert params.bath_energy_quantum < bath_mode
    assert params.conduction_contact == pytest.approx(0.04)
    assert params.conduction_bond == pytest.approx(0.20)
    assert params.kappa_base + params.kappa_exposure == pytest.approx(0.5)


@pytest.mark.parametrize("n", [1, 2, 3, 15])
def test_static_configuration_requires_even_n_of_at_least_four(n):
    with pytest.raises(ValueError):
        StaticConfig(n=n, num_particles=1)


@pytest.mark.parametrize(
    "name",
    (
        "n",
        "num_particles",
        "component_max_iters",
        "molecule_max_retries",
        "mis_max_iters",
        "planck_terms",
    ),
)
@pytest.mark.parametrize("value", [4.0, True])
def test_static_configuration_requires_host_integers(name, value):
    kwargs = {"n": 4, "num_particles": 4}
    kwargs[name] = value
    with pytest.raises(ValueError, match=rf"{name}.*integer"):
        StaticConfig(**kwargs)


def test_particle_count_cannot_exceed_lattice():
    with pytest.raises(ValueError):
        StaticConfig(n=4, num_particles=17)


@pytest.mark.parametrize("name,value", [("energy_floor", math.nan), ("gamma_base", math.inf)])
def test_nonfinite_parameters_are_rejected(name, value):
    with pytest.raises(ValueError, match="finite"):
        validate_params(default_params()._replace(**{name: value}))


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("heat_capacity", 1e300, "finite in float32"),
        ("source_beta", 1e-300, "underflows to zero"),
    ],
)
def test_parameters_must_survive_float32_conversion(name, value, message):
    with pytest.raises(ValueError, match=message):
        validate_params(default_params()._replace(**{name: value}))


def test_constraints_are_rechecked_after_float32_rounding():
    params = default_params()._replace(eta=1.0 - 1e-10)
    with pytest.raises(ValueError, match="eta"):
        validate_params(params)

    params = default_params()._replace(
        conduction_contact=0.1 + 1e-3, conduction_bond=0.1
    )
    with pytest.raises(ValueError, match="conduction"):
        validate_params(params)


def test_equal_and_zero_conduction_frequencies_are_valid():
    validate_params(
        default_params()._replace(conduction_contact=0.1, conduction_bond=0.1)
    )
    validate_params(
        default_params()._replace(conduction_contact=0.0, conduction_bond=0.0)
    )


def test_derived_bath_energy_must_remain_finite_in_float32():
    params = default_params()._replace(heat_capacity=1e30, bath_temperature=1e30)
    with pytest.raises(ValueError, match="bath-equilibrium.*finite"):
        validate_params(params)


@pytest.mark.parametrize(
    "name", ("conduction_energy_quantum", "bath_energy_quantum")
)
def test_energy_quanta_must_be_positive(name):
    with pytest.raises(ValueError, match=name):
        validate_params(default_params()._replace(**{name: 0.0}))


def test_boolean_parameters_are_strict():
    with pytest.raises(ValueError, match="boolean"):
        validate_params(default_params()._replace(source_enabled="false"))


def test_initial_bath_energy_must_respect_floor():
    params = default_params()._replace(
        heat_capacity=0.2, bath_temperature=1.0, energy_floor=0.3
    )
    with pytest.raises(ValueError, match="bath-equilibrium"):
        validate_params(params)
