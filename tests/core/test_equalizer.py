# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""RMS equalizer: the stored-feature rescale must equal re-extracting features from scaled samples."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from reak_core.femg import compute_emotions
from reak_core.femg.equalizer import apply_channel_eq, channel_scales, score, scoring_spec
from reak_core.models import load_model

MODEL = "classical-emotion-2.14.0"
FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "eq_parity.json")
_CASES = (
    ([0.0] * 8, [0.0] * 8),
    ([6.0, -3.0, 0.0, 2.5, -12.0, 4.0, 0.0, -1.5], [0.0] * 8),
    ([0.0] * 8, [2.0, -1.0, 0.5, 0.0, -10.0, 3.0, -0.25, 1.0]),
    ([-6.0, 6.0, 12.0, -12.0, 3.0, -3.0, 9.0, -9.0], [-2.0, 4.0, -10.0, 10.0, 0.0, -0.5, 2.5, -5.0]),
)


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL)


@pytest.fixture(scope="module")
def spec(model):
    return scoring_spec(model)


def _windows(n: int = 40, seed: int = 7) -> np.ndarray:
    """Synthetic EMG-like windows ``(n, 8, 500)`` with per-window, per-channel amplitudes of 1–40 µV."""
    rng = np.random.default_rng(seed)
    amp = rng.uniform(1.0, 40.0, size=(n, 8, 1))
    t = np.arange(500) / 250.0
    tone = np.sin(2 * np.pi * rng.uniform(40.0, 90.0, size=(n, 8, 1)) * t)
    return amp * (rng.standard_normal((n, 8, 500)) + 0.5 * tone)


def _features(model, windows: np.ndarray) -> np.ndarray:
    return np.stack([model._predictor._model._window_features(w)[0] for w in windows])


def _rms(windows: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(windows**2, axis=2))


def test_spec_only_for_models_scored_from_stored_features():
    assert scoring_spec(load_model("classical-emotion-2.11.0")) is not None
    for key in ("classical-emotion-2.5.0", "classical-emotion-2.9.0", "classical-emotion-2.10.0"):
        assert scoring_spec(load_model(key)) is None


def test_spec_describes_the_release(model, spec):
    assert spec["model_key"] == MODEL and spec["columns"] == list(model.feature_columns)
    assert len(spec["classifier"]["coef"]) == len(spec["classifier"]["classes"]) == 9
    assert spec["labels"]["CONTEMPT_LEFT"] == spec["labels"]["CONTEMPT_RIGHT"] == "Contempt"
    assert spec["postprocessing"]["high_rms"] == {"enabled": True, "threshold_uv": 200.0, "max_gap_windows": 6}
    assert len(spec["neutral_reference"]["channel_rms_median"]) == 8
    json.dumps(spec)  # plain JSON


def test_score_matches_release_predict(model, spec):
    windows = _windows()
    labels, confidence = score(_features(model, windows), spec)
    for w, label, conf in zip(windows, labels, confidence):
        result = model.predict(w)
        assert label == result.label
        assert conf == pytest.approx(result.confidence, abs=1e-9)


@pytest.mark.parametrize("gains, offsets", _CASES)
def test_rescale_equals_scaled_samples(model, spec, gains, offsets):
    windows = _windows()
    rms = _rms(windows)
    k = channel_scales(rms, gains, offsets)
    expected = _features(model, windows * k[:, :, None])
    vectors, new_rms = apply_channel_eq(_features(model, windows), rms, gains, offsets, spec)
    np.testing.assert_allclose(new_rms, _rms(windows * k[:, :, None]), rtol=1e-9)
    np.testing.assert_allclose(vectors, expected, rtol=1e-7, atol=1e-9)
    assert score(vectors, spec)[0] == [model.predict(w).label for w in windows * k[:, :, None]]


def test_offset_floor_keeps_channels_positive(spec):
    rms = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0]])
    k = channel_scales(rms, [0.0] * 8, [-10.0] * 8)
    assert np.all(rms[0, :7] * k[0, :7] == pytest.approx(spec["eq"]["epsilon_uv"]))
    assert k[0, 7] == 1.0  # a silent channel cannot be amplified


def test_scoring_model_sidecar_reproduces_stored_labels(session_copy):
    root = session_copy("baseline")
    compute_emotions(root, force=True)
    with open(os.path.join(root, "features", "emotions", "scoring_model.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    inputs = pd.read_csv(os.path.join(root, "features", "emotions", "emotionClassifierInputs.csv"))
    labels, _ = score(inputs[spec["columns"]].to_numpy(), spec)
    assert labels == list(inputs["PredictedLabel"])


def _fixture(model, spec) -> dict:
    windows = _windows(n=24, seed=11)
    vectors, rms = _features(model, windows), _rms(windows)
    cases = []
    for gains, offsets in _CASES:
        eq_vectors, eq_rms = apply_channel_eq(vectors, rms, gains, offsets, spec)
        labels, confidence = score(eq_vectors, spec)
        cases.append(
            {
                "gains_db": gains,
                "offsets_uv": offsets,
                "vectors": eq_vectors.tolist(),
                "channel_rms": eq_rms.tolist(),
                "labels": labels,
                "confidence": confidence.tolist(),
            }
        )
    return {"spec": spec, "vectors": vectors.tolist(), "channel_rms": rms.tolist(), "cases": cases}


def test_parity_fixture_is_current(model, spec):
    """``tests/fixtures/eq_parity.json`` pins the C# implementation (RE-AK-FullWebsite) to this one.

    Regenerate with ``REAK_WRITE_EQ_FIXTURE=1 pytest tests/core/test_equalizer.py`` and copy the file to
    ``RE-AK-FullWebsite/Tests/REAK.CoreTest/Fixtures/eq_parity.json``.
    """
    current = _fixture(model, spec)
    if os.environ.get("REAK_WRITE_EQ_FIXTURE"):
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=1)
    with open(FIXTURE, encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored["spec"]["classifier"] == current["spec"]["classifier"]
    assert stored["spec"]["eq"] == current["spec"]["eq"]
    for got, want in zip(stored["cases"], current["cases"]):
        assert got["labels"] == want["labels"]
        np.testing.assert_allclose(got["vectors"], want["vectors"], rtol=1e-12)
