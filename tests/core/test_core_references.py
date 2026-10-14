# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
The core against the Phase 2 references, case by case (see expectations.py).

For every synthetic case, the core steps implemented so far run on a fresh copy of the session with
the primary model (2.6.0); the emotion step then runs again with 2.5.0 on a copy. Each output is
compared with the reference of the system the decisions designate.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest
from expectations import ALTERNATES, EXCEPTIONS, EXPECTATIONS

from reak_core.eda import compute_arousal
from reak_core.eeg import compute_cognition
from reak_core.femg import compute_emotions
from reak_core.io import Session, write_metainfo
from reak_core.ppg import compute_heart_dynamics
from reak_core.sync import process_feature_events, seed_playback_annotations

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import refkit  # noqa: E402

CASES = [c.name for c in refkit.CASES]
PRIMARY, SECONDARY = refkit.MODELS


def run_core(root: str, model: str) -> None:
    session = Session.open(root)
    session.prepare()
    write_metainfo(session)
    compute_heart_dynamics(session.root)
    compute_arousal(session.root)
    compute_cognition(session.root, apply_nlms=True)
    compute_emotions(session.root, model)
    process_feature_events(session.root)
    seed_playback_annotations(session.root)


@pytest.fixture(scope="module")
def core_outputs(tmp_path_factory):
    """Run the core once per case and model; collect the outputs like the references were collected."""
    work = tmp_path_factory.mktemp("core")
    collected = {}
    for case in CASES:
        primary = str(work / PRIMARY / case)
        refkit.generate(case, primary)
        run_core(primary, PRIMARY)
        secondary = str(work / SECONDARY / case)
        shutil.copytree(primary, secondary)
        compute_emotions(secondary, SECONDARY)
        for model, root in ((PRIMARY, primary), (SECONDARY, secondary)):
            dest = str(work / "collected" / model / case)
            replacements = {"<session>": root, "<source>": primary, "<home>": os.path.expanduser("~")}
            refkit.collect_outputs(root, dest, tuple(EXPECTATIONS), replacements)
            collected[(model, case)] = dest
    return collected


def _comparisons():
    for output, (rule, _, models) in EXPECTATIONS.items():
        if rule == "new":
            continue
        for model in models:
            for case in CASES:
                yield pytest.param(output, model, case, id=f"{output.split('/')[-1]}-{model[-5:]}-{case}")


@pytest.mark.parametrize("output, model, case", list(_comparisons()))
def test_core_matches_designated_reference(core_outputs, output, model, case):
    system = ALTERNATES.get(output, {}).get(case, EXPECTATIONS[output][0])
    changed_by = EXCEPTIONS.get(output, {}).get(case)
    if changed_by:
        pytest.skip(f"changed by {', '.join(changed_by)}")
    reference = refkit.reference_dir(system, model, case)
    name = output.replace("/", "__")
    if not any(os.path.isfile(os.path.join(reference, name + ext)) for ext in ("", ".gz", ".summary.json")):
        pytest.skip(f"{system} writes no {output} for {case}")
    diffs = refkit.compare_dirs(reference, core_outputs[(model, case)], outputs=(output,))
    assert not diffs, "\n".join(f"{d.output}: {d.detail}" for d in diffs)


def _new_outputs():
    for output, (rule, _, models) in EXPECTATIONS.items():
        if rule == "new":
            for model in models:
                yield pytest.param(output, model, id=f"{output.split('/')[-1]}-{model[-5:]}")


@pytest.mark.parametrize("output, model", list(_new_outputs()))
def test_core_writes_new_rule_outputs(core_outputs, output, model):
    name = output.replace("/", "__")
    missing = [
        case
        for case in CASES
        if not any(os.path.isfile(os.path.join(core_outputs[(model, case)], name + ext)) for ext in ("", ".gz"))
    ]
    assert not missing, f"core wrote no {output} for {missing}"
