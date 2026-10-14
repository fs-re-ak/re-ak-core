# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Tests of reak_core.sync: resampling, feature events (D3, D16), playback annotations."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pandas as pd

from reak_core.sync import (
    process_feature_events,
    resample_to_grid,
    resample_to_timebase,
    seed_playback_annotations,
)


def test_resample_to_grid_keeps_long_gaps_nan(tmp_path):
    t = np.concatenate([np.arange(0, 10, 1 / 51.2), np.arange(20, 30, 1 / 51.2)])
    grid, values, gap = resample_to_grid(t, np.sin(t), "ppg", str(tmp_path))
    assert np.isnan(values[gap]).all() and gap.sum() > 0
    assert os.path.isfile(tmp_path / "ppg_resampled.csv")
    assert np.allclose(np.diff(grid), 1 / 51.2)


def test_resample_to_timebase_does_not_bridge_nan_gaps():
    df = pd.DataFrame({"Timestamp": np.arange(0, 10, 0.25), "v": np.arange(40, dtype=float)})
    df.loc[(df.Timestamp > 3) & (df.Timestamp < 6), "v"] = np.nan
    out = resample_to_timebase(df)
    assert list(out.Timestamp[:3]) == [0.0, 0.5, 1.0]
    assert out.loc[out.Timestamp == 4.5, "v"].isna().all()
    assert out.loc[out.Timestamp == 2.0, "v"].item() == 8.0


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_feature_events_from_event_csv(session_copy):
    root = session_copy("baseline")
    assert process_feature_events(root) == (3, 0)
    rows = _rows(os.path.join(root, "features", "events", "processedFeatureEvents.csv"))
    assert [r[0] for r in rows] == ["Button pressed"] * 3  # admin events dropped (D3)
    assert [r[1] for r in rows] == ["PROCESSED"] * 3


def test_feature_events_legacy_file(session_copy):
    root = session_copy("legacy_file_names")
    assert process_feature_events(root) == (3, 0)


def test_playback_annotations_seeded_once(session_copy):
    root = session_copy("baseline")
    assert seed_playback_annotations(root) is True
    path = os.path.join(root, "features", "events", "playback_annotations.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert [p["t"] for p in doc["points"]] == [30.0, 75.0, 120.0]
    assert seed_playback_annotations(root) is False
