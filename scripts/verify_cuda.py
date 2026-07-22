"""Fail-fast verification that JAX can execute work on an NVIDIA GPU."""

from __future__ import annotations

import json
import math
import os

# Fail during backend initialization instead of silently falling back to CPU.
os.environ.setdefault("JAX_PLATFORMS", "gpu")

import jax
import jax.numpy as jnp


def main() -> int:
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(f"JAX initialized without a GPU: {devices}")

    device = gpu_devices[0]
    x = jnp.linspace(-1.0, 1.0, 2048, dtype=jnp.float32).reshape(2048, 1)
    matrix = jnp.sin(x + x.T)
    checksum = jnp.sum(matrix @ matrix.T).block_until_ready().item()
    if not math.isfinite(checksum):
        raise RuntimeError(f"GPU workload produced a non-finite checksum: {checksum}")

    report = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "local_device_count": jax.local_device_count(),
        "workload": "2048x2048 float32 matrix product",
        "checksum": checksum,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())