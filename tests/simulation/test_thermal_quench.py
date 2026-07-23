from __future__ import annotations

import numpy as np

import thermal_quench


def test_canonical_background_is_reproducible_and_respects_floor():
    first = thermal_quench.canonical_bath_background(100, 2.5, 0.2, 0.05, 7)
    second = thermal_quench.canonical_bath_background(100, 2.5, 0.2, 0.05, 7)

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert np.all(first >= 0.05)


def test_initial_distributions_preserve_total_and_target_exposure_extrema():
    energy = np.asarray((0.5, 1.5, 2.0, 0.5), dtype=np.float32)
    exposure = np.asarray((0.5, 0.0, 1.0, 0.25), dtype=np.float32)

    distributions, targets = thermal_quench.initial_energy_distributions(
        energy, exposure, bath_reference=0.5
    )

    assert tuple(distributions) == thermal_quench.DISTRIBUTION_NAMES
    assert targets == {"shielded": 1, "exposed": 2}
    expected_total = np.sum(energy, dtype=np.float64)
    for name, values in distributions.items():
        assert values.dtype == np.float32
        expected = 2.0 if name == "canonical_background" else expected_total
        assert np.sum(values, dtype=np.float64) == expected
    np.testing.assert_array_equal(
        distributions["canonical_background"],
        np.full(energy.shape, 0.5, dtype=np.float32),
    )
    assert np.count_nonzero(distributions["concentrated_shielded"] > 0.5) == 1
    assert np.count_nonzero(distributions["concentrated_exposed"] > 0.5) == 1
    assert distributions["concentrated_shielded"][1] > 0.5
    assert distributions["concentrated_exposed"][2] > 0.5


def test_condition_grid_is_complete_and_stably_ordered():
    conditions = thermal_quench.condition_grid()

    assert len(conditions) == 5 * 3 * 3
    assert len(set(conditions)) == len(conditions)
    assert conditions[0] == thermal_quench.Condition("actual", "default", "default")
    assert conditions[-1] == thermal_quench.Condition(
        "canonical_background", "off", "exposed_only"
    )


def test_trace_summary_reports_matched_excess_half_life():
    conditions = (
        thermal_quench.Condition("uniform", "default", "default"),
        thermal_quench.Condition("canonical_background", "default", "default"),
    )
    traces = {
        name: np.zeros((3, 4), dtype=np.float32)
        for name in thermal_quench.METRIC_NAMES
    }
    traces["total_energy"][:, :2] = ((9.0, 9.0), (7.0, 7.0), (5.0, 5.0))
    traces["total_energy"][:, 2:] = 4.0
    traces["bath_heat"][:, :2] = -1.0
    traces["exposed_bath_heat"][:, :2] = -0.75
    traces["shielded_bath_heat"][:, :2] = -0.25
    traces["excess_participation"][:, :2] = 2.0
    traces["energetic_particles"][:, :2] = 2.0

    row = thermal_quench.summarize_traces(
        traces,
        conditions,
        replicates=2,
        initial_totals={"uniform": 10.0, "canonical_background": 4.0},
        initial_participation={"uniform": 2.0, "canonical_background": 1.0},
    )[0]

    assert row["cumulative_bath_heat_mean"] == -3.0
    assert row["cumulative_exposed_bath_heat"] == -2.25
    assert row["cumulative_shielded_bath_heat"] == -0.75
    assert row["matched_excess_half_life_sweeps"] == 2
    assert row["dissipated_matched_excess"] == 5.0