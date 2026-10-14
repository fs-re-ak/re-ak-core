# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Run the current NucleusKit processing steps that are in the core's scope on one session.

Same steps and order as ``NucleusKitProcessingToolkit.configureAsDefault`` minus the
wrapper-only ones (video rotation, POV conversion, UWB, GPS).

Usage: python run_nk.py --repo <NucleusKit checkout> --session <dir> --model <release key>

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import block_network, parse_args, run_steps, setup_io  # noqa: E402


def main() -> int:
    args = parse_args()
    setup_io()
    block_network()
    sys.path.insert(0, os.path.abspath(args.repo))

    from nucleuskit_toolkit.events import seedPlaybackAnnotations
    from nucleuskit_toolkit.events.processor import eventProcessor
    from nucleuskit_toolkit.hermes.processor.cognition_processor import computeCognitiveIndexes
    from nucleuskit_toolkit.hermes.processor.emotions_processor import computeEmotions
    from nucleuskit_toolkit.session.layout import ensure_session_rawdata_layout, prepareDirectory
    from nucleuskit_toolkit.session.meta.info import extractMetaInfo
    from nucleuskit_toolkit.shimmer import computeArousal, computeHeartDynamics

    session = os.path.abspath(args.session) + os.sep
    ensure_session_rawdata_layout(session)
    steps = [
        ("prepareDirectory", prepareDirectory),
        ("extractMetaInfo", extractMetaInfo),
        ("computeHeartDynamics", computeHeartDynamics),
        ("computeArousal", computeArousal),
        ("computeCognitiveIndexes", lambda p: computeCognitiveIndexes(p, apply_nlms=True)),
        ("computeEmotions", lambda p: computeEmotions(p, apply_nlms=True, model_name=args.model)),
        ("eventProcessor", eventProcessor),
        ("seedPlaybackAnnotations", seedPlaybackAnnotations),
    ]
    return run_steps(steps, session, os.path.join(session, "reference_run.json"), {"system": "nk", "model": args.model})


if __name__ == "__main__":
    sys.exit(main())
