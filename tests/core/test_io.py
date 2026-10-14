# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Tests of reak_core.io: timestamps (D1), loaders (D2), invalid samples (D6, D24), metadata (D25), streams."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from reak_core.constants import FULL_SCALE_UV
from reak_core.io import (
    Session,
    check_gps,
    check_uwb,
    derive_eeg_channels,
    invalid_mask,
    load_eda_signal,
    load_exg,
    load_ppg_signal,
    load_raw_events,
    normalise_timestamps_to_seconds,
    write_metainfo,
)

EPOCH = 1791000000.0


@pytest.mark.parametrize(
    "raw, expected",
    [
        (EPOCH + np.array([0.0, 0.004, 1.5]), [0.0, 0.004, 1.5]),  # epoch seconds (Vizia)
        ((EPOCH + np.array([0.0, 0.004, 1.5])) * 1e3, [0.0, 0.004, 1.5]),  # epoch ms
        ((EPOCH + np.array([0.0, 0.004, 1.5])) * 1e6, [0.0, 0.004, 1.5]),  # epoch µs
        (np.array([0.0, 4.0, 8.0, 1500.0]), [0.0, 0.004, 0.008, 1.5]),  # relative ms
        (np.array([0.0, 0.004, 0.008]), [0.0, 0.004, 0.008]),  # relative s
    ],
)
def test_timestamp_units_are_detected(raw, expected):
    np.testing.assert_allclose(normalise_timestamps_to_seconds(raw), expected, atol=1e-6)


def test_vizia_exg_loads_in_hardware_order(session_copy):
    rec = load_exg(session_copy("baseline"))
    assert rec.data.shape == (37500, 8)
    assert rec.timestamps[0] == 0.0 and 149.9 < rec.timestamps[-1] < 150.1


def test_named_columns_are_reordered(session_copy):
    base = load_exg(session_copy("baseline"))
    named = load_exg(session_copy("eeg_named_columns"))
    # Same generator seed differs per case, so compare the channel identity through the DC offsets
    offsets = np.median(named.data, axis=0)
    assert offsets[0] > 700 and offsets[3] < -800  # AF8 and CHEEK_L back in hardware positions
    assert base.data.shape == named.data.shape


def test_truncated_last_line_is_ignored(session_copy):
    rec = load_exg(session_copy("eeg_truncated_last_line"))
    assert rec.data.shape == (37499, 8)


def test_invalid_mask_marks_non_finite_and_full_scale_only():
    data = np.array(
        [
            [1.0, np.nan, np.inf, -np.inf],
            [FULL_SCALE_UV - 0.045, -FULL_SCALE_UV, -FULL_SCALE_UV + 0.05, -FULL_SCALE_UV + 0.5],
            [187500.0, 0.0, 0.3, -1.0],
        ]
    )
    expected = np.array([[False, True, True, True], [True, True, True, False], [False, False, False, False]])
    np.testing.assert_array_equal(invalid_mask(data), expected)


def test_inf_cells_are_loaded_and_marked(session_copy):
    rec = load_exg(session_copy("exg_inf_values"))
    assert int(invalid_mask(rec.data).sum()) == 5


def test_full_scale_case_is_marked(session_copy):
    rec = load_exg(session_copy("exg_full_scale"))
    mask = invalid_mask(rec.data)
    assert mask[:, 2].sum() == 750  # 3 s burst at 250 Hz
    assert mask[:, 1].sum() == 3 and mask[:, 5].sum() == 1
    assert mask[:, 0].sum() == 1  # within 0.1 µV marked, 0.5 µV away kept


def test_eeg_derivation():
    exg = np.arange(16, dtype=float).reshape(2, 8)
    eeg = derive_eeg_channels(exg)
    row = exg[0]
    assert eeg.loc[0, "AF7"] == row[1] - row[4] / 2
    assert eeg.loc[0, "AF8"] == row[0] - row[4] / 2
    assert eeg.loc[0, "T9"] == -row[4] / 2 and eeg.loc[0, "T10"] == row[4] / 2
    assert eeg.loc[0, "RightHemi"] == row[0] - row[4]


@pytest.mark.parametrize("case", ["baseline", "legacy_file_names", "timestamps_epoch_us"])
def test_shimmer_loaders(session_copy, case):
    root = session_copy(case)
    eda = load_eda_signal(root)
    ts, ppg = load_ppg_signal(root)
    assert eda.shape[1] == 2 and 149.0 < eda[0].iloc[-1] < 150.1
    assert len(ts) == len(ppg) and 149.0 < ts[-1] < 150.1


def test_raw_events_from_vizia_file(session_copy):
    events = load_raw_events(session_copy("baseline"))
    names = [e[1] for e in events]
    assert names[:2] == ["RECORDING_ONSET", "RECIPE_CONFIG"]
    assert events[1][2].startswith('{"model"')  # unquoted JSON payload re-joined
    assert [round(e[0] - events[0][0], 3) for e in events[2:]] == [30.0, 75.0, 120.0]


@pytest.mark.parametrize(
    "case, duration",
    [
        ("baseline", 150.0),
        ("timestamps_epoch_ms", 150.0),
        ("timestamps_relative_ms", 150.0),
        ("legacy_file_names", 150.0),
    ],
)
def test_metainfo_duration_uses_normalised_timestamps(session_copy, case, duration):
    session = Session.open(session_copy(case))
    write_metainfo(session)
    with open(session.path("features", "metainfo.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert float(meta["duration"]) == duration
    assert meta["eegReadable"] == "True" and meta["shimmerReadable"] == "True"


def test_streams_absent_or_valid(session_copy, tmp_path):
    root = session_copy("baseline")
    assert not check_uwb(root).present and not check_gps(root).valid
    with open(os.path.join(root, "rawData", "gps.csv"), "w", encoding="utf-8") as fh:
        fh.write("timestamp,latitude,longitude\n")
        fh.writelines(f"{EPOCH + i:.3f},46.8,-71.2\n" for i in range(10))
    check = check_gps(root)
    assert check.present and check.valid and check.rows == 10 and check.duration_s == 9.0


def test_session_layout_is_fixed(tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    (root / "eeg.csv").write_text("timestamp,ch1\n", encoding="utf-8")
    session = Session.open(str(root))
    assert os.path.isfile(session.path("rawData", "eeg.csv"))
    session.prepare()
    for name in ("features", "results", "meta", "experimentConfigs"):
        assert os.path.isdir(session.path(name))
