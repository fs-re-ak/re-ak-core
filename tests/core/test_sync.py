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
    snap_to_timebase,
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


# ---------------------------------------------------------------------------
# snap_to_timebase
# ---------------------------------------------------------------------------

def test_snap_to_timebase_no_blending():
    """Labels at 0.5015 s intervals (like the Hermes clock) must never produce fractional values.

    Linear interpolation blends adjacent one-hot rows and drifts with the headset clock, creating
    the ~6.5 minute triangle artefact.  Nearest-neighbour snapping must produce only 0.0, 1.0,
    or NaN — no blended fractions.
    """
    # Simulate alternating labels at 0.5015 s intervals (Hermes clock drift scenario).
    n = 100
    src_ts = np.arange(n) * 0.5015  # ~249.7 Hz effective clock
    neutral = np.tile([1.0, 0.0], n // 2 + 1)[:n]
    sadness = 1.0 - neutral
    df = pd.DataFrame({"Timestamp": src_ts, "Neutral": neutral, "Sadness": sadness})

    out = snap_to_timebase(df)
    for col in ("Neutral", "Sadness"):
        vals = out[col].dropna().to_numpy()
        assert np.all((vals == 0.0) | (vals == 1.0)), (
            f"snap_to_timebase produced fractional values in {col}: {vals[~((vals==0)|(vals==1))]}"
        )

    # Linear interpolation on the same data does produce fractions (documents the regression).
    out_linear = resample_to_timebase(df)
    for col in ("Neutral", "Sadness"):
        vals_lin = out_linear[col].dropna().to_numpy()
        assert np.any((vals_lin > 0.0) & (vals_lin < 1.0)), (
            f"Expected linear blend to produce fractions in {col} (regression guard)"
        )


def test_snap_to_timebase_gap_becomes_nan():
    """A gap wider than one output step stays NaN — no filling across dropouts.

    Source segments: [0, 4.5] and [10, 14.5], each sampled every 0.5 s.  The grid covers 0–14.5 at
    0.5 s steps.  Grid points that are more than one step from any source row (i.e. > 0.5 s away)
    must stay NaN.  Grid points within one step of a source edge (5.0 snaps to 4.5; 9.5 snaps to
    10.0) are valid snaps and should NOT be NaN.
    """
    src_ts = np.concatenate([np.arange(0, 5, 0.5), np.arange(10, 15, 0.5)])
    y = np.concatenate([np.ones(10), np.zeros(10)])
    df = pd.DataFrame({"Timestamp": src_ts, "v": y})
    out = snap_to_timebase(df)
    # Strictly interior gap points — more than one step from any source row.
    interior_gap = (out["Timestamp"] > 5.0) & (out["Timestamp"] < 9.5)
    assert out.loc[interior_gap, "v"].isna().all(), "Interior gap region should be NaN"
    assert not out.loc[out["Timestamp"] <= 5.0, "v"].isna().any(), "Before gap should be filled"
    assert not out.loc[out["Timestamp"] >= 9.5, "v"].isna().any(), "After gap should be filled"


def test_snap_to_timebase_selects_nearest():
    """Each output point maps to the closest source row, not the interpolated midpoint."""
    # Source points at 0.1 and 0.9; output grid at 0.0, 0.5, 1.0.
    df = pd.DataFrame({"Timestamp": [0.1, 0.9], "v": [10.0, 20.0]})
    out = snap_to_timebase(df, target_interval=0.5)
    # 0.0 → nearest is 0.1 (dist 0.1 < one step 0.5) → 10
    # 0.5 → equidistant; both are 0.4 away → either is within one step → pick 0.1 (left wins tie)
    # 1.0 → nearest is 0.9 (dist 0.1 < one step 0.5) → 20
    assert out.loc[out["Timestamp"] == 0.0, "v"].item() == 10.0
    assert out.loc[out["Timestamp"] == 1.0, "v"].item() == 20.0


def test_snap_to_timebase_session_spot_check():
    """Reproduce the 486.5 s soft-Sadness artefact and confirm snap removes it.

    The real session had PredictedLabel alternating Sadness / Neutral every 0.5015 s.
    Linear resampling blended them, giving Sadness ≈ 0.905 at 486.5 s.
    Nearest-neighbour resampling must give exactly 1.0 or 0.0.
    """
    # Reproduce the classifier timestamps from the real session around 486.5 s.
    classifier_ts = np.array([485.054, 485.549, 486.055, 486.546, 487.049, 487.552, 488.054])
    labels = ["Surprise", "Surprise", "Surprise", "Sadness", "Neutral", "Sadness", "Neutral"]
    emotion_cols = ["Neutral", "Sadness", "Surprise"]
    rows = []
    for t, lab in zip(classifier_ts, labels):
        row = {"Timestamp": t}
        for c in emotion_cols:
            row[c] = 1.0 if c == lab else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)

    # Linear: 486.5 falls between 486.046 (Surprise=1) and 487.049 (Neutral=1).
    # With 0.5015 step, the blend is significant → Sadness should be 0.
    out_linear = resample_to_timebase(df)
    row_486 = out_linear.loc[np.isclose(out_linear["Timestamp"], 486.5)]
    assert not row_486.empty
    # Nearest: 486.5 falls within 0.5 s of 486.546 (Sadness) → Sadness = 1.0, not a blend.
    out_snap = snap_to_timebase(df)
    row_486s = out_snap.loc[np.isclose(out_snap["Timestamp"], 486.5)]
    assert not row_486s.empty
    for col in emotion_cols:
        val = row_486s[col].item()
        assert val in (0.0, 1.0), f"snap produced fractional {col}={val} at 486.5 s"
    assert row_486s["Sadness"].item() == 1.0, "Nearest to 486.5 s should be Sadness (486.546 s)"


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
