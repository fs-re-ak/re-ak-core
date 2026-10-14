# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Run the current Prime processing steps that are in the core's scope on one session.

Same steps and order as ``HermesAnalysisPipeline.configureAsDefault`` minus the
wrapper-only ones (POV conversion, UWB, GPS, automatic zones, labeled dataframe). The
automatic-zones step is excluded on purpose: without a local configuration it falls back
to the production API. Network access is also blocked for the whole process.

Prime selects the emotion model from the ``RECIPE_CONFIG`` event; the runner rewrites the
``model`` field of that event before running.

Usage: python run_prime.py --repo <primeV3 checkout> --session <dir> --model <release key>

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import block_network, parse_args, run_steps, setup_io  # noqa: E402

_EVENT_FILES = ("rawEvents.csv", "event.csv", "events_0.csv")


def set_recipe_model(session: str, model: str) -> None:
    for name in _EVENT_FILES:
        path = os.path.join(session, "rawData", name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        for i, line in enumerate(lines):
            parts = line.split(",", 2)
            if len(parts) == 3 and parts[1] == "RECIPE_CONFIG":
                payload = json.loads(parts[2])
                payload["model"] = model
                lines[i] = f"{parts[0]},RECIPE_CONFIG,{json.dumps(payload, separators=(',', ':'))}"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines))
        return


def main() -> int:
    args = parse_args()
    setup_io()
    block_network()
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)

    from sessionProcessor.analyticsEngine.devices.hermesHeadset.processor.cognition_processor import (
        computeCognitiveIndexes,
    )
    from sessionProcessor.analyticsEngine.devices.hermesHeadset.processor.emotions_processor import computeEmotions
    from sessionProcessor.analyticsEngine.devices.shimmer.processor.eda import computeArousal
    from sessionProcessor.analyticsEngine.devices.shimmer.processor.heart import computeHeartDynamics
    from sessionProcessor.analyticsEngine.sessionInfra.eventsProcessor import seedPlaybackAnnotations
    from sessionProcessor.analyticsEngine.sessionInfra.eventsTools import eventProcessor
    from sessionProcessor.analyticsEngine.sessionInfra.fileTools import prepareDirectory
    from sessionProcessor.analyticsEngine.sessionInfra.metaInfoTools import extractMetaInfo

    session = os.path.abspath(args.session) + os.sep
    set_recipe_model(session, args.model)
    steps = [
        ("prepareDirectory", prepareDirectory),
        ("extractMetaInfo", extractMetaInfo),
        ("computeArousal", computeArousal),
        ("computeHeartDynamics", computeHeartDynamics),
        ("computeCognitiveIndexes", lambda p: computeCognitiveIndexes(p, apply_nlms=True)),
        ("computeEmotions", lambda p: computeEmotions(p, apply_nlms=True)),
        ("eventProcessor", eventProcessor),
        ("seedPlaybackAnnotations", seedPlaybackAnnotations),
    ]
    return run_steps(
        steps, session, os.path.join(session, "reference_run.json"), {"system": "prime", "model": args.model}
    )


if __name__ == "__main__":
    sys.exit(main())
