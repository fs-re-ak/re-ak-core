# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Tests of reak_core.tools (D20) and reak_core.exporters (D19)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from reak_core.constants import OUTPUT_RMS_COLUMNS
from reak_core.pipeline import Pipeline
from reak_core.tools import ChannelFixModel, fix_ppg_session, fix_session, revert_working_to_original


@pytest.fixture
def processed(session_copy):
    root = session_copy("baseline")
    result = Pipeline.default().without("cognition").run(root)
    assert result.ok
    return root


def _rms(root):
    return pd.read_csv(os.path.join(root, "features", "emotions", "rmsSignals.csv"))


def test_emotions_report_is_exported_by_default(processed):
    files = os.listdir(os.path.join(processed, "features", "emotions"))
    assert "report_emotion_distribution.png" in files
    assert any(f.startswith("report_jitter_") for f in files)


@pytest.mark.parametrize("target", range(8))
def test_channel_fix_models_load(target):
    model = ChannelFixModel.load(target)
    assert model.target_channel == target and len(model.input_channels) == 7 and target not in model.input_channels


def test_channel_fixer_repairs_only_the_target(processed):
    before = _rms(processed)
    report = fix_session(processed, 3)
    after = _rms(processed)
    assert list(after.columns) == ["Timestamp"] + list(OUTPUT_RMS_COLUMNS)
    others = [c for i, c in enumerate(OUTPUT_RMS_COLUMNS) if i != 3]
    valid = before[others].notna().all(axis=1)
    np.testing.assert_allclose(after.loc[valid, others], before.loc[valid, others])
    model = ChannelFixModel.load(3)
    expected = model.predict(before[others].to_numpy())
    in_range = (expected >= 0) & (expected <= 30) & valid.to_numpy()
    np.testing.assert_allclose(after.loc[in_range, "CHEEK_L_RMS"], expected[in_range], rtol=1e-9)
    assert report.is_file()
    original = os.path.join(processed, "features", "emotions", "original")
    assert os.path.isfile(os.path.join(original, "rmsSignals.csv"))
    with open(os.path.join(original, "operations.log"), encoding="utf-8") as fh:
        assert "channel_fix target=CHEEK_L" in fh.read()
    revert_working_to_original(processed)
    pd.testing.assert_frame_equal(_rms(processed), before)


def test_channel_fixer_rejects_unsupported_inputs(processed):
    with pytest.raises(IndexError):
        fix_session(processed, 8)
    with pytest.raises(ValueError):
        fix_session(processed, 2, ignored_channels=[0])


def test_ppg_fixer_reruns_heart_dynamics(processed):
    path = os.path.join(processed, "results", "HeartDynamics.csv")
    before = pd.read_csv(path)
    fix_ppg_session(processed)
    after = pd.read_csv(path)
    assert list(after.columns) == list(before.columns) and len(after) == len(before)
