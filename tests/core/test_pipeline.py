# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Tests of reak_core.pipeline, reak_core.quality and the command line."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest

import reak_core
from reak_core.__main__ import main
from reak_core.io.session import Session
from reak_core.pipeline import Pipeline, RunContext, StepResult

DEFAULT_STEPS = [
    "prepare",
    "metainfo",
    "heart_dynamics",
    "arousal",
    "cognition",
    "emotions",
    "feature_events",
    "playback_annotations",
]


class MarkerStep:
    """Example extension step: writes a file, proving it ran with the shared context."""

    name = "marker"

    def run(self, session: Session, context: RunContext) -> StepResult:
        with open(session.path("features", "marker.txt"), "w", encoding="utf-8") as fh:
            fh.write(str(context.extra.get("label")))
        return StepResult(self.name, "ok")


class FailingStep:
    name = "failing"

    def run(self, session, context):
        raise RuntimeError("boom")


class CollectingExporter:
    name = "collector"

    def __init__(self):
        self.seen = []

    def export(self, session, context, results):
        self.seen = [r.name for r in results]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the core opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_default_pipeline_runs_and_writes_processing_json(session_copy, no_network):
    root = session_copy("baseline")
    result = Pipeline.default().run(root)
    assert [s.name for s in result.steps] == DEFAULT_STEPS and result.ok
    with open(os.path.join(root, "results", "processing.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["core_version"] == reak_core.__version__ and doc["emotion_model"] == "classical-emotion-2.6.0"
    assert doc["quality"]["exg"]["samples"] == 37500 and doc["quality"]["emotions"]["null_windows"] == 0
    assert doc["quality"]["uwb"] is None
    for rel in ("results/Emotions.csv", "results/Cognition.csv", "results/Arousal.csv", "results/HeartDynamics.csv"):
        assert os.path.isfile(os.path.join(root, rel)), rel


def test_second_run_reuses_outputs_unless_forced(session_copy):
    root = session_copy("baseline")
    Pipeline.default().run(root)
    statuses = {s.name: s.status for s in Pipeline.default().run(root).steps}
    assert {statuses[n] for n in ("heart_dynamics", "arousal", "cognition", "emotions")} == {"skipped"}
    forced = {s.name: s.status for s in Pipeline.default(force=True).run(root).steps}
    assert {forced[n] for n in ("heart_dynamics", "arousal", "cognition", "emotions")} == {"ok"}


def test_extension_step_and_exporter(session_copy):
    root = session_copy("baseline")
    exporter = CollectingExporter()
    pipeline = Pipeline.default(extra={"label": "ext"}).without("cognition", "emotions", "heart_dynamics", "arousal")
    pipeline.with_step(MarkerStep(), after="metainfo").with_exporter(exporter)
    result = pipeline.run(root)
    assert pipeline.step_names()[2] == "marker" and result.ok
    with open(os.path.join(root, "features", "marker.txt"), encoding="utf-8") as fh:
        assert fh.read() == "ext"
    assert exporter.seen[:3] == ["prepare", "metainfo", "marker"]
    assert result.step("export:collector").status == "ok"


def test_a_failing_step_does_not_stop_the_others(session_copy):
    root = session_copy("baseline")
    pipeline = Pipeline.default().without("cognition", "emotions", "heart_dynamics", "arousal")
    result = pipeline.with_step(FailingStep(), after="prepare").run(root)
    assert result.step("failing").status == "error" and "boom" in result.step("failing").error
    assert result.step("feature_events").status == "ok" and not result.ok


def test_steps_must_implement_the_protocol():
    with pytest.raises(TypeError):
        Pipeline.default().with_step(object())
    with pytest.raises(ValueError):
        Pipeline.default().with_step(MarkerStep()).with_step(MarkerStep())


def test_session_without_shimmer_or_exg(tmp_path):
    root = tmp_path / "empty"
    (root / "rawData").mkdir(parents=True)
    (root / "rawData" / "notes.txt").write_text("nothing to process", encoding="utf-8")
    result = Pipeline.default().run(str(root))
    statuses = {s.name: s.status for s in result.steps}
    assert result.ok and statuses["emotions"] == "skipped" and statuses["cognition"] == "skipped"


def test_command_line(session_copy, capsys):
    root = session_copy("baseline")
    assert main(["--session", root, "--quiet", "--model", "classical-emotion-2.5.0"]) == 0
    out = capsys.readouterr().out
    assert "emotions" in out and "ok" in out
    with open(os.path.join(root, "results", "emotion_model.json"), encoding="utf-8") as fh:
        assert json.load(fh)["model"] == "classical-emotion-2.5.0"


def test_import_does_not_load_wrapper_modules():
    # neurokit2 imports requests for its dataset helpers, which the core never calls; the no_network
    # fixture proves that no connection is opened during a run.
    code = (
        "import sys, reak_core, reak_core.pipeline, reak_core.models, reak_core.femg; "
        "wrappers = ('nucleuskit_toolkit', 'sessionProcessor', 'dbAPI', 'utils.printUtils'); "
        "bad = [m for m in wrappers if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""
