# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Tests of reak_core.eeg (D4, D5, D23), reak_core.eda and reak_core.ppg beyond the reference comparisons."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from reak_core.constants import FULL_SCALE_UV
from reak_core.eda import compute_arousal
from reak_core.eeg import compute_cognition
from reak_core.eeg.cognition import bandpass_filter
from reak_core.io import Session
from reak_core.ppg import compute_heart_dynamics


def _prepared(session_copy, case):
    root = session_copy(case)
    Session.open(root).prepare()
    return root


def test_no_notch_after_the_cognition_bandpass():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 10, (5000, 2)), columns=["T9", "T10"])
    b, a = butter(4, [0.5 / 125, 30 / 125], btype="band")
    expected = filtfilt(b, a, df.to_numpy(), axis=0)
    np.testing.assert_allclose(bandpass_filter(df, 250).to_numpy(), expected)


def test_dead_unused_channel_does_not_block_cognition(session_copy):
    root = _prepared(session_copy, "baseline")
    path = os.path.join(root, "rawData", "eeg.csv")
    df = pd.read_csv(path)
    df["ch8"] = FULL_SCALE_UV  # NOSE saturated for the whole recording; cognition does not use it (D23)
    df.to_csv(path, index=False, float_format="%.4f")
    compute_cognition(root, apply_nlms=False)
    cognition = pd.read_csv(os.path.join(root, "results", "Cognition.csv"))
    assert list(cognition.columns) == ["Timestamp", "Engagement"]  # D5
    assert cognition["Engagement"].notna().mean() > 0.9


def test_cognition_writes_nucleuskit_diagnostics(session_copy):
    root = _prepared(session_copy, "baseline")
    compute_cognition(root)
    for name in (
        "powerBands.csv",
        "artefactStats.csv",
        "epochMetrics.csv",
        "temporalBandPowers.csv",
        "filteredEEG.csv",
    ):
        assert os.path.isfile(os.path.join(root, "features", "cognition", name)), name
    assert os.path.isfile(os.path.join(root, "features", "cognition", "eegArtefactPlot.png"))


def test_arousal_and_heart_outputs(session_copy):
    root = _prepared(session_copy, "baseline")
    compute_arousal(root)
    compute_heart_dynamics(root)
    arousal = pd.read_csv(os.path.join(root, "results", "Arousal.csv"))
    heart = pd.read_csv(os.path.join(root, "results", "HeartDynamics.csv"))
    assert arousal["Timestamp"].diff().dropna().round(6).eq(0.5).all()
    assert 60 < np.nanmedian(heart["mean_hr"]) < 85  # synthetic heart rate is 72 bpm
    assert os.path.isfile(os.path.join(root, "features", "ppg", "ppg_rejected.png"))
