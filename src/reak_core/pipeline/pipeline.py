# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Ordered processing pipeline.

``Pipeline.default()`` runs the built-in steps. Extensions are added without modifying the core,
either explicitly (``with_step``) or from installed packages (``with_installed_extensions``, through
the ``reak_core.steps`` and ``reak_core.exporters`` entry points). A failing step is recorded and the
following steps still run, as in NucleusKit and Prime. Every run writes ``results/processing.json``.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import entry_points

import reak_core
from reak_core.io.session import Session
from reak_core.models import resolve_key
from reak_core.pipeline.base import Exporter, RunContext, Step, StepResult
from reak_core.pipeline.steps import default_steps
from reak_core.quality import session_quality

logger = logging.getLogger(__name__)

PROCESSING_JSON = os.path.join("results", "processing.json")
STEP_ENTRY_POINTS = "reak_core.steps"
EXPORTER_ENTRY_POINTS = "reak_core.exporters"


@dataclass
class PipelineResult:
    session: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.status != "error" for s in self.steps)

    def step(self, name: str) -> StepResult | None:
        return next((s for s in self.steps if s.name == name), None)


class Pipeline:
    """An ordered list of steps and exporters sharing one :class:`RunContext`."""

    def __init__(self, steps: list[Step], exporters: list[Exporter] | None = None, context: RunContext | None = None):
        self.steps = list(steps)
        self.exporters = list(exporters or [])
        self.context = context or RunContext()

    @classmethod
    def default(cls, **options) -> Pipeline:
        """Built-in steps; ``options`` are :class:`RunContext` fields (``emotion_model``, ``apply_nlms``, …)."""
        return cls(default_steps(), context=RunContext(**options))

    # -- composition ------------------------------------------------------------------------------
    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def with_step(self, step: Step, *, after: str | None = None, before: str | None = None) -> Pipeline:
        """Insert a step (at the end by default). Returns ``self`` for chaining."""
        if not isinstance(step, Step):
            raise TypeError(f"{step!r} does not implement the Step protocol")
        names = self.step_names()
        if step.name in names:
            raise ValueError(f"a step named {step.name!r} is already in the pipeline")
        if after is not None:
            self.steps.insert(names.index(after) + 1, step)
        elif before is not None:
            self.steps.insert(names.index(before), step)
        else:
            self.steps.append(step)
        return self

    def without(self, *names: str) -> Pipeline:
        """Remove steps by name. Returns ``self``."""
        self.steps = [s for s in self.steps if s.name not in names]
        return self

    def with_exporter(self, exporter: Exporter) -> Pipeline:
        if not isinstance(exporter, Exporter):
            raise TypeError(f"{exporter!r} does not implement the Exporter protocol")
        self.exporters.append(exporter)
        return self

    def with_installed_extensions(self) -> Pipeline:
        """Append the steps and exporters registered by installed packages (entry points)."""
        for ep in entry_points(group=STEP_ENTRY_POINTS):
            self.with_step(ep.load()())
        for ep in entry_points(group=EXPORTER_ENTRY_POINTS):
            self.with_exporter(ep.load()())
        return self

    # -- execution --------------------------------------------------------------------------------
    def run(self, session: Session | str) -> PipelineResult:
        if not isinstance(session, Session):
            session = Session.open(session)
        result = PipelineResult(session.root)
        started = datetime.now(timezone.utc)
        for step in self.steps:
            result.steps.append(self._run_step(step, session))
        for exporter in self.exporters:
            result.steps.append(self._run_exporter(exporter, session, result.steps))
        self._write_processing_json(session, result, started)
        return result

    def _run_step(self, step: Step, session: Session) -> StepResult:
        t0 = time.time()
        try:
            res = step.run(session, self.context)
        except Exception as exc:  # noqa: BLE001 — a failing step must not stop the others
            logger.error("Step %s failed: %s\n%s", step.name, exc, traceback.format_exc())
            res = StepResult(step.name, "error", error=f"{type(exc).__name__}: {exc}")
        res.seconds = round(time.time() - t0, 3)
        logger.info("Step %s: %s (%.1f s)", res.name, res.status, res.seconds)
        return res

    def _run_exporter(self, exporter: Exporter, session: Session, results: list[StepResult]) -> StepResult:
        t0 = time.time()
        try:
            exporter.export(session, self.context, results)
            res = StepResult(f"export:{exporter.name}", "ok")
        except Exception as exc:  # noqa: BLE001
            logger.error("Exporter %s failed: %s\n%s", exporter.name, exc, traceback.format_exc())
            res = StepResult(f"export:{exporter.name}", "error", error=f"{type(exc).__name__}: {exc}")
        res.seconds = round(time.time() - t0, 3)
        return res

    def _write_processing_json(self, session: Session, result: PipelineResult, started: datetime) -> None:
        document = {
            "core_version": reak_core.__version__,
            "started_at": started.isoformat(),
            "emotion_model": resolve_key(self.context.emotion_model),
            "nlms": self.context.apply_nlms,
            "postprocessing_overrides": self.context.postprocessing,
            "steps": [
                {k: v for k, v in s.__dict__.items() if v not in (None, {}, "")} | {"status": s.status}
                for s in result.steps
            ],
            "quality": session_quality(session),
        }
        os.makedirs(session.path("results"), exist_ok=True)
        with open(session.path(PROCESSING_JSON), "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2, default=str)
