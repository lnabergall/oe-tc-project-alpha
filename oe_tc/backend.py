"""Backend selection before JAX initializes an accelerator runtime."""

from __future__ import annotations

import os


def jax_platform_name(platform: str) -> str:
    """Translate the stable CLI vocabulary to JAX backend names."""

    return "cuda" if platform == "gpu" else platform


def configure_environment(platform: str, no_preallocate: bool) -> None:
    """Configure JAX without importing it or initializing a backend."""

    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = jax_platform_name(platform)
    if no_preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"