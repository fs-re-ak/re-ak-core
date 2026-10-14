# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Built-in steps of the default pipeline, in their default order.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from reak_core.eda import compute_arousal
from reak_core.eeg import compute_cognition
from reak_core.femg import compute_emotions
from reak_core.io import write_metainfo
from reak_core.io.session import Session
from reak_core.pipeline.base import RunContext, StepResult
from reak_core.ppg import compute_heart_dynamics
from reak_core.sync import process_feature_events, seed_playback_annotations


@dataclass
class FunctionStep:
    """A step backed by a function ``fn(session, context) -> dict | None`` (detail of the result)."""

    name: str
    fn: Callable[[Session, RunContext], dict | None]
    outputs: tuple[str, ...] = ()
    """Outputs that, when all present, let the step be skipped (unless ``force``)."""

    def run(self, session: Session, context: RunContext) -> StepResult:
        if self.outputs and not context.force and all(os.path.isfile(session.path(p)) for p in self.outputs):
            return StepResult(self.name, "skipped", detail={"reason": "outputs already present"})
        detail = self.fn(session, context) or {}
        return StepResult(self.name, detail.pop("_status", "ok"), detail=detail)


def _remove(session: Session, outputs: tuple[str, ...]) -> None:
    for rel in outputs:
        path = session.path(rel)
        if os.path.isfile(path):
            os.remove(path)


def _prepare(session: Session, context: RunContext) -> dict:
    session.prepare()
    return {"modalities": {k: (os.path.basename(v) if v else None) for k, v in session.modalities().items()}}


def _metainfo(session: Session, context: RunContext) -> dict:
    return write_metainfo(session)


def _forced(outputs: tuple[str, ...], fn: Callable[[str], object]) -> Callable[[Session, RunContext], dict]:
    """Wrap a NucleusKit-style function that skips itself when its outputs exist."""

    def run(session: Session, context: RunContext) -> dict:
        if context.force:
            _remove(session, outputs)
        fn(session.root)
        missing = [p for p in outputs if not os.path.isfile(session.path(p))]
        return {"_status": "skipped", "missing": missing} if missing else {}

    return run


HEART_OUTPUTS = ("results/HeartDynamics.csv",)
AROUSAL_OUTPUTS = ("results/Arousal.csv",)
COGNITION_OUTPUTS = ("results/Cognition.csv", "features/cognition/powerBands.csv")


def _cognition(session: Session, context: RunContext) -> dict:
    if context.force:
        _remove(session, COGNITION_OUTPUTS)
    compute_cognition(session.root, apply_nlms=context.apply_nlms)
    missing = [p for p in COGNITION_OUTPUTS[:1] if not os.path.isfile(session.path(p))]
    return {"_status": "skipped", "missing": missing} if missing else {"nlms": context.apply_nlms}


def _emotions(session: Session, context: RunContext) -> dict:
    run = compute_emotions(session.root, context.emotion_model, force=context.force, **context.postprocessing)
    if run is None:
        return {"_status": "skipped", "reason": "no usable EXG"}
    detail = dict(run.__dict__)
    if detail.pop("skipped"):
        detail["_status"] = "skipped"
    return detail


def _feature_events(session: Session, context: RunContext) -> dict:
    counts = process_feature_events(session.root)
    if counts is None:
        return {"_status": "skipped", "reason": "no raw events"}
    return {"feature_events": counts[0], "web_events": counts[1]}


def _annotations(session: Session, context: RunContext) -> dict:
    return {} if seed_playback_annotations(session.root) else {"_status": "skipped"}


def default_steps() -> list[FunctionStep]:
    """The built-in steps, in NucleusKit's order."""
    return [
        FunctionStep("prepare", _prepare),
        FunctionStep("metainfo", _metainfo),
        FunctionStep("heart_dynamics", _forced(HEART_OUTPUTS, compute_heart_dynamics), HEART_OUTPUTS),
        FunctionStep("arousal", _forced(AROUSAL_OUTPUTS, compute_arousal), AROUSAL_OUTPUTS),
        FunctionStep("cognition", _cognition, COGNITION_OUTPUTS[:1]),
        FunctionStep("emotions", _emotions),
        FunctionStep("feature_events", _feature_events),
        FunctionStep("playback_annotations", _annotations),
    ]
