# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Command line: process a session folder with the default pipeline.

    python -m reak_core --session <dir> [--model KEY] [--no-nlms] [--force] [--extensions]
    python -m reak_core --list-models

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import argparse
import logging
import sys

import reak_core
from reak_core.models import available_models
from reak_core.pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reak-core", description="RE-AK open processing core")
    parser.add_argument("--session", help="session folder (contains rawData/)")
    parser.add_argument("--model", default=None, help="emotion model key or alias (default: standard-latest)")
    parser.add_argument("--no-nlms", action="store_true", help="skip NLMS decorrelation before cognition")
    parser.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    parser.add_argument("--extensions", action="store_true", help="also run installed extension steps and exporters")
    parser.add_argument("--list-models", action="store_true", help="list the available emotion models and exit")
    parser.add_argument("--quiet", action="store_true", help="only print warnings and errors")
    parser.add_argument("--version", action="version", version=f"reak_core {reak_core.__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.list_models:
        print("\n".join(available_models()))
        return 0
    if not args.session:
        parser.error("--session is required")

    pipeline = Pipeline.default(emotion_model=args.model, apply_nlms=not args.no_nlms, force=args.force)
    if args.extensions:
        pipeline.with_installed_extensions()
    result = pipeline.run(args.session)
    for step in result.steps:
        line = f"{step.name:22s} {step.status:8s} {step.seconds:7.1f} s"
        print(line + (f"  {step.error}" if step.error else ""))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
