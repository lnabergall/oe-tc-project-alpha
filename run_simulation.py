"""Executable wrapper for the OE-TC JAX runner.

The small bootstrap parser configures the requested accelerator before importing
any module that imports JAX.  All substantive CLI behavior lives in
``oe_tc.runner`` and remains importable for tests.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def _bootstrap(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu", "tpu"), default="auto")
    parser.add_argument("--no-preallocate", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.platform != "auto":
        os.environ["JAX_PLATFORMS"] = args.platform
    if args.no_preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _bootstrap(arguments)
    from oe_tc.runner import main as runner_main

    return runner_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
