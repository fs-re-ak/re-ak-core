# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Extension contracts: processing steps and exporters.

A wrapper or a proprietary module adds behaviour without modifying the core:

- a :class:`Step` runs on a session folder and writes outputs (for example a facial-video analysis);
- an :class:`Exporter` runs once all steps are done and turns the outputs into something else
  (a report, an upload, a summary).

Both are discovered through entry points (``reak_core.steps`` and ``reak_core.exporters``): each entry
point is a callable returning an instance. The default pipeline uses built-in steps only.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from reak_core.io.session import Session


@dataclass
class RunContext:
    """Options shared by every step of one pipeline run."""

    emotion_model: str | None = None
    """Model key or alias; ``None`` uses ``standard-latest`` (D9)."""
    apply_nlms: bool = True
    """NLMS decorrelation before cognition (never applied to emotions)."""
    force: bool = False
    """Recompute outputs that already exist."""
    postprocessing: dict[str, Any] = field(default_factory=dict)
    """Overrides of the emotion post-processing defaults (D8), e.g. ``{"high_rms_threshold_uv": 250}``."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Free options for extension steps."""


@dataclass
class StepResult:
    name: str
    status: str  # "ok", "skipped" or "error"
    seconds: float = 0.0
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Step(Protocol):
    """A processing step. Must not raise for missing inputs: return a ``skipped`` result instead."""

    name: str

    def run(self, session: Session, context: RunContext) -> StepResult: ...


@runtime_checkable
class Exporter(Protocol):
    """Runs after all steps; receives the step results."""

    name: str

    def export(self, session: Session, context: RunContext, results: list[StepResult]) -> None: ...
