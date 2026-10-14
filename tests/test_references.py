# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Integrity of the synthetic generator and of the stored references.

- The generator must reproduce byte for byte the raw files the references were recorded
  from (hashes in ``tests/references/manifest.json``); otherwise the references no longer
  describe the generated sessions.
- Stored references must not contain local paths (they are published).

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import refkit  # noqa: E402

MANIFEST = os.path.join(refkit.REFERENCES_DIR, "manifest.json")


def _manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def test_every_case_has_references():
    recorded = set(_manifest()["cases"])
    assert recorded == {c.name for c in refkit.CASES}
    for case in refkit.CASES:
        for system in ("nk", "prime"):
            for model in refkit.MODELS:
                assert os.path.isfile(
                    os.path.join(refkit.reference_dir(system, model, case.name), "reference_run.json")
                ), f"missing reference: {system}/{model}/{case.name}"


@pytest.mark.parametrize("case", [c.name for c in refkit.CASES])
def test_generator_reproduces_recorded_raw_files(case, tmp_path):
    expected = _manifest()["cases"][case]["raw_sha256"]
    assert refkit.generate(case, str(tmp_path / case)) == expected


def test_references_contain_no_local_paths():
    for dirpath, _, files in os.walk(refkit.REFERENCES_DIR):
        for name in files:
            path = os.path.join(dirpath, name)
            if name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    text = fh.read()
            else:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            refkit.assert_no_local_paths(text, os.path.relpath(path, refkit.REFERENCES_DIR))
