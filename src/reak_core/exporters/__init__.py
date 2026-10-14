# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Built-in exporters: the per-session emotions report (decision D19)."""

from __future__ import annotations

import os

from reak_core.exporters.emotions_report import generate_report
from reak_core.io.session import Session
from reak_core.pipeline.base import RunContext, StepResult


class EmotionsReport:
    """Jitter plots of the per-channel RMS pattern per emotion and the emotion distribution.

    Writes ``features/emotions/report_jitter_<emotion>.png`` and ``report_emotion_distribution.png``.
    Runs when the emotion step produced or kept its outputs.
    """

    name = "emotions_report"

    def export(self, session: Session, context: RunContext, results: list[StepResult]) -> None:
        if os.path.isfile(session.path("features", "emotions", "emotionClassifierInputs.csv")):
            generate_report(session.root)


__all__ = ["EmotionsReport", "generate_report"]
