# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
The core against the Phase 2 references, case by case (see expectations.py).

For every synthetic case, the core steps implemented so far run on a fresh copy of the session;
each output is then compared with the reference of the system the decisions designate.
"""

from __future__ import annotations

import os
import sys

import pytest
from expectations import EXCEPTIONS, EXPECTATIONS

from reak_core.io import Session, write_metainfo
from reak_core.sync import process_feature_events, seed_playback_annotations

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import refkit  # noqa: E402

MODEL = refkit.PRIMARY_MODEL
CASES = [c.name for c in refkit.CASES]


def run_core(root: str) -> None:
    session = Session.open(root)
    session.prepare()
    write_metainfo(session)
    process_feature_events(session.root)
    seed_playback_annotations(session.root)


@pytest.fixture(scope="module")
def core_outputs(tmp_path_factory):
    """Run the core once per case and collect its outputs like the references were collected."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    work = tmp_path_factory.mktemp("core")
    collected = {}
    for case in CASES:
        root = str(work / "sessions" / case)
        refkit.generate(case, root)
        run_core(root)
        dest = str(work / "collected" / case)
        refkit.collect_outputs(root, dest, tuple(EXPECTATIONS), {"<session>": root, "<home>": os.path.expanduser("~")})
        collected[case] = dest
    return collected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("output", [o for o, (rule, _) in EXPECTATIONS.items() if rule != "new"])
def test_core_matches_designated_reference(core_outputs, case, output):
    system, decisions = EXPECTATIONS[output]
    if (output, case) in EXCEPTIONS:
        pytest.skip(f"changed by {', '.join(EXCEPTIONS[(output, case)])}")
    reference = refkit.reference_dir(system, MODEL, case)
    in_reference = any(
        os.path.isfile(os.path.join(reference, output.replace("/", "__") + ext)) for ext in ("", ".gz", ".summary.json")
    )
    if not in_reference:
        pytest.skip(f"{system} writes no {output} for {case}")
    diffs = refkit.compare_dirs(reference, core_outputs[case], outputs=(output,))
    assert not diffs, "\n".join(f"{d.output}: {d.detail}" for d in diffs)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("output", [o for o, (rule, _) in EXPECTATIONS.items() if rule == "new"])
def test_core_writes_new_rule_outputs(core_outputs, case, output):
    stored = os.path.join(core_outputs[case], output.replace("/", "__"))
    assert os.path.isfile(stored) or os.path.isfile(stored + ".gz"), f"core wrote no {output}"
