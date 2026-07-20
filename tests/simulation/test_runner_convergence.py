from typing import NamedTuple

import numpy as np

from oe_tc import runner


class _Metrics(NamedTuple):
    components_converged: np.ndarray
    component_iterations: np.ndarray


def test_explicit_convergence_at_iteration_cap_is_authoritative():
    config = runner.StaticConfig(n=4, num_particles=2, component_max_iters=4)
    metrics = _Metrics(
        components_converged=np.asarray((True,), dtype=bool),
        component_iterations=np.asarray((config.component_max_iters,), dtype=np.int32),
    )
    runner._guard_component_convergence(metrics, config)
