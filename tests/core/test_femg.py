# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Tests of reak_core.models and reak_core.femg (D6, D7, D8, D9, D10, D24)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from reak_core.constants import FULL_SCALE_UV, OUTPUT_RMS_COLUMNS
from reak_core.femg import compute_emotions, interpolate_invalid, prepare, windows
from reak_core.io import Session, invalid_mask, load_exg
from reak_core.models import (
    DEFAULT_MODEL,
    EMOTION_COLUMNS,
    PreprocessSpec,
    ReleaseIntegrityError,
    available_models,
    builtin_releases,
    load_model,
    resolve_key,
)


def test_default_model_and_aliases():
    assert resolve_key(None) == DEFAULT_MODEL == "classical-emotion-2.6.0"
    assert resolve_key("standard-latest") == "classical-emotion-2.6.0"
    assert resolve_key("classical-emotion") == "classical-emotion-2.5.0"
    assert {"classical-emotion-2.5.0", "classical-emotion-2.6.0"} <= set(available_models())


@pytest.mark.parametrize("key", ["classical-emotion-2.5.0", "classical-emotion-2.6.0"])
def test_releases_verify_and_declare_sklearn(key):
    release = builtin_releases()[key]
    release.verify()
    assert release.manifest["sklearn_version"] == "1.6.1"


def test_tampered_release_is_rejected(tmp_path):
    release = builtin_releases()["classical-emotion-2.6.0"]
    import shutil

    from reak_core.models.registry import Release

    copy = tmp_path / release.key
    shutil.copytree(release.path, copy)
    (copy / "labels.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ReleaseIntegrityError):
        Release(release.key, copy, release.manifest).verify()


def test_release_preprocessing_and_postprocessing():
    m25, m26 = load_model("classical-emotion-2.5.0"), load_model("classical-emotion-2.6.0")
    assert (m25.preprocess.bandpass_low, m25.preprocess.bandpass_high, m25.preprocess.notch_hz) == (15.0, 40.0, None)
    assert (m26.preprocess.bandpass_low, m26.preprocess.bandpass_high, m26.preprocess.notch_hz) == (20.0, 110.0, 60.0)
    assert not m25.postprocess.one_hot and m25.postprocess.low_confidence_enabled
    assert m26.postprocess.one_hot and not m26.postprocess.low_confidence_enabled
    assert m25.postprocess.high_rms_threshold_uv == m26.postprocess.high_rms_threshold_uv == 200.0


def test_interpolation_only_bridges_channels_with_valid_samples():
    data = np.array([[0.0, np.nan], [np.nan, np.nan], [2.0, np.nan]])
    out = interpolate_invalid(data, np.isnan(data))
    np.testing.assert_allclose(out[:, 0], [0.0, 1.0, 2.0])
    assert np.isnan(out[:, 1]).all()


def test_one_invalid_sample_voids_every_window_containing_it():
    rng = np.random.default_rng(1)
    raw = rng.normal(0, 10, (2000, 8))
    raw[1000, 3] = FULL_SCALE_UV
    filtered, mask = prepare(raw, PreprocessSpec())
    assert np.isnan(filtered[1000, 3]) and mask.sum() == 1
    ts = np.arange(2000) / 250.0
    nulls = [w for w in windows(filtered, ts) if w.is_null]
    covering = [s for s in range(0, 2000 - 500 + 1, 125) if s <= 1000 < s + 500]
    assert len(nulls) == len(covering) == 4


def _run(session_copy, case, model, **kwargs):
    root = session_copy(case)
    Session.open(root).prepare()
    return root, compute_emotions(root, model, **kwargs)


def test_outputs_and_schema(session_copy):
    root, run = _run(session_copy, "baseline", None)
    assert run.model == "classical-emotion-2.6.0" and run.windows == 297 and run.null_windows == 0
    emotions = pd.read_csv(os.path.join(root, "results", "Emotions.csv"), na_values=["NULL"])
    assert list(emotions.columns) == ["Timestamp"] + EMOTION_COLUMNS
    # one-hot per window for 2.6.0; the 2 Hz resampling interpolates between windows, rows still sum to 1
    np.testing.assert_allclose(emotions[EMOTION_COLUMNS].dropna().sum(axis=1), 1.0, atol=1e-9)
    inputs = pd.read_csv(os.path.join(root, "features", "emotions", "emotionClassifierInputs.csv"))
    assert list(inputs.columns[-9:]) == list(OUTPUT_RMS_COLUMNS) + ["HighRmsArtifact"]
    assert inputs[list(OUTPUT_RMS_COLUMNS)].to_numpy().min() > 0.5  # raw µV, not L2-normalised
    rms = pd.read_csv(os.path.join(root, "features", "emotions", "rmsSignals.csv"))
    assert list(rms.columns) == ["Timestamp"] + list(OUTPUT_RMS_COLUMNS) and len(rms) == 297
    with open(os.path.join(root, "results", "emotion_model.json"), encoding="utf-8") as fh:
        assert json.load(fh) == {
            "model": "classical-emotion-2.6.0",
            "model_name": "classical-emotion",
            "model_version": "2.6.0",
            "feature_set": "extended_nm_a",
            "kind": "classical",
        }
    assert not os.path.exists(os.path.join(root, "features", "emotions", "model_info.json"))


def test_existing_outputs_are_kept_unless_forced(session_copy):
    root, _ = _run(session_copy, "baseline", None)
    assert compute_emotions(root).skipped
    assert not compute_emotions(root, force=True).skipped
    assert not compute_emotions(root, "classical-emotion-2.5.0").skipped  # other model: recomputed


def test_high_rms_heuristic_threshold_is_a_parameter(session_copy):
    _, default = _run(session_copy, "emg_rms_threshold", None)
    _, raised = _run(session_copy, "emg_rms_threshold", None, high_rms_threshold_uv=1000.0)
    _, disabled = _run(session_copy, "emg_rms_threshold", None, high_rms_enabled=False)
    assert default.relabeled_high_rms > 0 and raised.relabeled_high_rms == 0 and disabled.relabeled_high_rms == 0


def test_low_confidence_only_for_2_5_0(session_copy):
    _, m25 = _run(session_copy, "baseline", "classical-emotion-2.5.0")
    _, m26 = _run(session_copy, "baseline", "classical-emotion-2.6.0")
    assert m25.relabeled_low_confidence > 0 and m26.relabeled_low_confidence == 0


@pytest.mark.parametrize("case", ["exg_inf_values", "exg_full_scale"])
def test_invalid_samples_void_their_windows(session_copy, case):
    root, run = _run(session_copy, case, None)
    invalid_rows = np.where(invalid_mask(load_exg(root).data).any(axis=1))[0]
    starts = range(0, 37500 - 500 + 1, 125)
    expected = sum(1 for s in starts if ((invalid_rows >= s) & (invalid_rows < s + 500)).any())
    assert expected > 0 and run.null_windows == expected


def test_old_disconnect_value_inside_windows_is_not_invalid(session_copy):
    _, run = _run(session_copy, "exg_single_invalid", None)
    assert run.null_windows == 0  # the case writes 187 500 µV, no longer invalid (D6)


def test_legacy_disconnect_value_is_not_invalid(session_copy):
    _, run = _run(session_copy, "exg_dead_channel", None)
    assert run.null_windows == 0  # 187 500 µV dates from before the gain change (D6)
