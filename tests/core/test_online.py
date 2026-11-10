# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Online (causal) processors: building blocks, and replay of a synthetic session against the offline outputs."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from reak_core.eeg.cognition import computeCognitiveIndexes
from reak_core.eeg.nlms import CausalNLMSFilter
from reak_core.femg.emotions import CLASSIFIER_INPUTS, compute_emotions
from reak_core.femg.signal import prepare, windows
from reak_core.io.exg import load_exg
from reak_core.models import load_model
from reak_core.online import (
    OnlineEmotionClassifier,
    OnlineEngagement,
    PercentileMinMaxNormalizer,
    SampleClock,
    Windower,
)

BLOCK = 250  # 1 s packets, as Vizia publishes them


@pytest.fixture(scope="module")
def model():
    return load_model()


def _replay(processor, data: np.ndarray, block: int = BLOCK) -> list:
    out = []
    for start in range(0, len(data), block):
        out += processor.push(data[start : start + block], start)
    return out


# -- Normaliser --------------------------------------------------------------------------------------


def test_normalizer_warmup_then_percentile_scale():
    norm = PercentileMinMaxNormalizer(window_size=240, min_samples=60)
    values = np.linspace(0.0, 1.0, 59)
    assert all(norm.update(v) is None for v in values)
    assert not norm.ready
    first = norm.update(0.5)
    assert norm.ready
    assert norm.q_low == pytest.approx(np.quantile(np.append(values, 0.5), 0.10))
    assert first == pytest.approx(norm.scale(0.5))


def test_normalizer_folds_outliers_and_smooths():
    norm = PercentileMinMaxNormalizer(min_samples=10, window_size=50, alpha=0.35)
    for v in np.linspace(0.0, 1.0, 50):
        norm.update(v)
    above, below = norm.scale(1.3), norm.scale(-0.3)
    assert 0.99 < above < 1.0 and 0.0 < below < 0.01  # folded back inside (0, 1), not clipped
    assert norm.scale(0.5) == pytest.approx(0.5, abs=0.02)  # identity inside the band
    before = norm.update(0.5)
    after = norm.update(1.0)
    assert after == pytest.approx(before + 0.35 * (norm.scale(1.0) - before))


def test_normalizer_state_round_trip():
    norm = PercentileMinMaxNormalizer(min_samples=5, window_size=20)
    for v in range(12):
        norm.update(float(v))
    clone = PercentileMinMaxNormalizer(min_samples=5, window_size=20)
    clone.load_state_dict(norm.state_dict())
    assert clone.update(7.0) == pytest.approx(norm.update(7.0))


# -- Stream building blocks --------------------------------------------------------------------------


def test_sample_clock_overlap_short_gap_and_restart():
    clock = SampleClock(max_fill_samples=100)
    ones = np.ones((50, 8))
    first = clock.feed(ones, 0)
    assert first.restart and first.start == 0
    assert clock.feed(ones[:30], 20) is None  # already seen
    overlap = clock.feed(ones, 40)  # samples 40..49 already seen
    assert overlap.start == 50 and len(overlap.data) == 40 and not overlap.restart
    gap = clock.feed(ones, 150)  # 60 missing samples are filled with NaN
    assert gap.start == 90 and len(gap.data) == 110 and np.isnan(gap.data[:60]).all() and not gap.restart
    long_gap = clock.feed(ones, 1000)
    assert long_gap.restart and long_gap.start == 1000


def test_windower_is_aligned_on_offline_grid():
    windower = Windower(window=500, step=125, warmup=250)
    data, mask = np.zeros((2000, 1)), np.zeros(2000, dtype=bool)
    starts = [s for s, _, _ in windower.feed(37, data, mask, restart=True)]
    assert starts and all(s % 125 == 0 for s in starts)
    assert starts[0] == 375  # first multiple of 125 at or after 37 + 250
    assert np.all(np.diff(starts) == 125)


def test_nlms_output_channels_match_full_filter():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)) * 10
    full = CausalNLMSFilter(8).push_batch(x)
    subset = CausalNLMSFilter(8, output_channels=[4]).push_batch(x)
    np.testing.assert_allclose(subset[:, 4], full[:, 4])
    np.testing.assert_array_equal(subset[:, 0], x[:, 0])


# -- Emotions ----------------------------------------------------------------------------------------


def test_online_emotions_block_size_does_not_change_results(session_copy, model):
    data = load_exg(session_copy("baseline")).data[: 60 * 250]
    a = OnlineEmotionClassifier(model, min_calibration_windows=20)
    b = OnlineEmotionClassifier(model, min_calibration_windows=20)
    ra, rb = _replay(a, data, 250), _replay(b, data, 77)
    assert [(r.sample, r.status, r.label) for r in ra] == [(r.sample, r.status, r.label) for r in rb]


def test_online_emotions_track_offline(session_copy, model):
    root = session_copy("baseline")
    compute_emotions(root, model, force=True)
    offline = pd.read_csv(os.path.join(root, CLASSIFIER_INPUTS))
    recording = load_exg(root)
    offline_by_sample = {
        int(round((ts - recording.timestamps[0]) * 250 / 125)) * 125: label
        for ts, label in zip(offline["Timestamp"], offline["PredictedLabel"])
    }

    online = OnlineEmotionClassifier(model, calibration_windows=10_000, min_calibration_windows=60)
    results = _replay(online, recording.data)
    assert all(r.sample % 125 == 0 for r in results)
    assert [r.status for r in results[:5]] == ["calibrating"] * 5
    ok = [r for r in results if r.status == "ok"]
    assert ok and all(sum(r.intensities.values()) == pytest.approx(1.0) for r in ok)

    # Once calibration has seen half the session, labels agree with the whole-session offline run.
    late = [r for r in ok if r.time_s > recording.data.shape[0] / 250 / 2]
    matched = [(r.label, offline_by_sample.get(r.sample)) for r in late if r.sample in offline_by_sample]
    agreement = np.mean([a == b for a, b in matched])
    assert len(matched) > 100
    assert agreement >= 0.9, agreement


def test_online_emotions_filters_match_offline_with_same_calibration(session_copy, model):
    root = session_copy("baseline")
    compute_emotions(root, model, force=True)
    offline = pd.read_csv(os.path.join(root, CLASSIFIER_INPUTS))
    recording = load_exg(root)
    filtered, _ = prepare(recording.data, model.preprocess)
    model.reset()
    model.set_subject_context(np.stack([w.samples for w in windows(filtered, recording.timestamps) if not w.is_null]))
    inner = model._predictor._model
    frozen = (inner._subject_mean, inner._subject_std, inner._subject_wl_mean, inner._subject_wl_std)

    online = OnlineEmotionClassifier(model, min_calibration_windows=1)
    online._update_stats = lambda: setattr(online, "_stats", frozen)
    labels = {r.sample: r.label for r in _replay(online, recording.data) if r.status == "ok"}
    offline_by_sample = {
        int(round((ts - recording.timestamps[0]) * 250 / 125)) * 125: label
        for ts, label in zip(offline["Timestamp"], offline["PredictedLabel"])
    }
    common = sorted(set(labels) & set(offline_by_sample))
    agreement = np.mean([labels[k] == offline_by_sample[k] for k in common])
    assert agreement >= 0.93, agreement


def test_online_emotions_lost_packets_and_reconnect(session_copy, model):
    data = load_exg(session_copy("baseline")).data[: 90 * 250]
    online = OnlineEmotionClassifier(model, min_calibration_windows=20)
    results = []
    for start in range(0, 40 * 250, BLOCK):
        if start == 30 * 250:
            continue  # one lost 1 s packet: filled, windows over it are null
        results += online.push(data[start : start + BLOCK], start)
    nulls = [r for r in results if r.status == "null"]
    assert nulls and all(28 * 250 <= r.sample <= 33 * 250 for r in nulls)
    calibrated = online.n_calibration_windows
    after = online.push(data[60 * 250 : 62 * 250], 60 * 250)  # 20 s gap: restart, calibration kept
    assert online.n_calibration_windows >= calibrated
    assert all(r.sample >= 60 * 250 + 250 for r in after)


def test_online_emotions_high_rms_rule_is_past_only(model):
    online = OnlineEmotionClassifier(model, min_calibration_windows=1)
    rng = np.random.default_rng(3)
    quiet = rng.normal(scale=5.0, size=(20 * 250, 8))
    online.push(quiet, 0)
    loud = rng.normal(scale=400.0, size=(10 * 250, 8))
    results = online.push(loud, 20 * 250)
    reasons = [r.relabel_reason for r in results if r.high_rms_artifact]
    assert "high_rms_borrowed" in reasons and "high_rms_neutral" in reasons
    assert reasons.index("high_rms_neutral") > reasons.index("high_rms_borrowed")


def test_online_emotions_checkpoint_round_trip(session_copy, model):
    data = load_exg(session_copy("baseline")).data[: 60 * 250]
    a = OnlineEmotionClassifier(model, min_calibration_windows=20)
    _replay(a, data[: 40 * 250])
    b = OnlineEmotionClassifier(model, min_calibration_windows=20)
    b.load_state_dict(a.state_dict())
    assert b.n_calibration_windows == a.n_calibration_windows and b.calibrated


# -- Engagement --------------------------------------------------------------------------------------


def test_online_engagement_tracks_offline(session_copy):
    root = session_copy("baseline")
    computeCognitiveIndexes(root)
    bands = pd.read_csv(os.path.join(root, "features", "cognition", "powerBands.csv"))
    # pivot (not pivot_table) keeps the all-NaN rows of null windows, so row k is window k.
    t9 = bands[bands["channel"] == "T9"].pivot(index="Timestamp", columns="band", values="power").sort_index()
    offline = (t9["beta"] / (t9["alpha"] + t9["theta"] + 1e-8)).to_numpy()
    offline_by_sample = {125 * k + 250: v for k, v in enumerate(offline)}

    online = OnlineEngagement()
    results = _replay(online, load_exg(root).data)
    values = {r.sample: r.engagement for r in results if r.engagement is not None}
    common = [k for k in sorted(values) if np.isfinite(offline_by_sample.get(k, np.nan))]
    a = np.array([values[k] for k in common])
    b = np.array([offline_by_sample[k] for k in common])
    assert len(common) > 200
    assert np.corrcoef(a, b)[0, 1] > 0.95
    assert np.median(a / b) == pytest.approx(1.0, abs=0.05)

    statuses = [r.status for r in results]
    assert statuses.index("ok") >= 59  # 60 valid windows before the first calibrated score
    scores = [r.engagement_calibrated for r in results if r.status == "ok"]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_online_engagement_rejects_amplitude_artifacts():
    online = OnlineEngagement()
    rng = np.random.default_rng(5)
    x = rng.normal(scale=5.0, size=(20 * 250, 8))
    x[12 * 250 : 13 * 250, 4] += 3000.0 * np.sin(np.linspace(0, 20 * np.pi, 250))
    results = online.push(x, 0)
    rejected = [r for r in results if r.status == "null"]
    assert rejected and {r.reject_reason for r in rejected} <= {"peak_amp", "ptp"}
