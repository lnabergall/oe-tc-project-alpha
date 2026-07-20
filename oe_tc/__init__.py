"""OE-TC model: a driven, dissipative lattice chemistry implemented in JAX.

The package root deliberately avoids importing JAX. This lets command-line
entry points select the accelerator before the runtime is initialized while
retaining convenient lazy exports for the dynamic state types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oe_tc.config import Params, StaticConfig, default_params

if TYPE_CHECKING:
    from oe_tc.state import State, StepMetrics


def __getattr__(name: str) -> Any:
    if name in {"State", "StepMetrics"}:
        from oe_tc import state

        return getattr(state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Params", "State", "StaticConfig", "StepMetrics", "default_params"]
