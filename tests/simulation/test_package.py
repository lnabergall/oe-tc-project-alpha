from __future__ import annotations

import subprocess
import sys


def test_runner_import_does_not_initialize_jax() -> None:
    code = "import sys; import oe_tc.runner; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_state_exports_remain_available() -> None:
    code = (
        "from oe_tc import Params, State, StepMetrics; "
        "assert (Params.__name__, State.__name__, StepMetrics.__name__) "
        "== ('Params', 'State', 'StepMetrics')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
