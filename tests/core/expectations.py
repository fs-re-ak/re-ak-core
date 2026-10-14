# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
What the core must reproduce, output by output (open-core decisions D1–D25).

``EXPECTATIONS`` maps a session output to a rule:

- ``("nk", decisions)`` / ``("prime", decisions)``: the core must match that system's reference
  (``tests/references/<system>/classical-emotion-2.6.0/<case>/``) within the reference tolerance,
  except for the cases listed in ``EXCEPTIONS``.
- ``("new", decisions)``: the core follows a new rule. The output is not compared; its difference
  with both systems is measured and recorded in the CHANGELOG.

``EXCEPTIONS`` lists ``(output, case)`` pairs where a decision changes the result of an otherwise
matching output, with the decisions responsible.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

EXPECTATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "features/events/playback_annotations.json": ("nk", ()),
    "features/events/processedFeatureEvents.csv": ("prime", ("D3", "D16")),
    "features/metainfo.json": ("new", ("D25",)),
}

EXCEPTIONS: dict[tuple[str, str], tuple[str, ...]] = {}

# Outputs written by the steps implemented so far (grows with each Phase 3 PR).
IMPLEMENTED_STEPS = ("metainfo", "feature_events", "playback_annotations")
