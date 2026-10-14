# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Emotion inference from fEMG.

Outputs (decision D10):

- ``results/Emotions.csv``: one column per emotion on the 2 Hz timebase (probabilities, or a one-hot
  of the emitted label for releases that declare it), ``NULL`` for null windows.
- ``features/emotions/emotionClassifierInputs.csv``: per classified window, the features, the raw
  classifier label and confidence, the raw per-channel RMS (``HEAD_*_RMS``, µV) and
  ``HighRmsArtifact``.
- ``features/emotions/rmsSignals.csv``: raw per-channel RMS of every window (NaN for null windows).
- ``features/emotions/avgRms.csv`` and ``relabeled-samples.csv``: heuristic diagnostics.
- ``results/emotion_model.json``: the model that produced the outputs (the only sidecar).

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reak_core.constants import N_CHANNELS, OUTPUT_RMS_COLUMNS
from reak_core.femg import heuristics
from reak_core.femg.signal import prepare, windows
from reak_core.io.exg import load_exg
from reak_core.models import EMOTION_COLUMNS, EmotionModel, load_model
from reak_core.models.base import EmotionWindowResult
from reak_core.sync.resample import resample_to_timebase

logger = logging.getLogger(__name__)

EMOTIONS_CSV = os.path.join("results", "Emotions.csv")
SIDECAR = os.path.join("results", "emotion_model.json")
EMOTIONS_DIR = os.path.join("features", "emotions")
CLASSIFIER_INPUTS = os.path.join(EMOTIONS_DIR, "emotionClassifierInputs.csv")
RMS_SIGNALS = os.path.join(EMOTIONS_DIR, "rmsSignals.csv")
AVG_RMS = os.path.join(EMOTIONS_DIR, "avgRms.csv")
RELABELED = os.path.join(EMOTIONS_DIR, "relabeled-samples.csv")
_LEGACY_SIDECAR = os.path.join(EMOTIONS_DIR, "model_info.json")

RMS_COLUMNS = list(OUTPUT_RMS_COLUMNS)


@dataclass
class EmotionRun:
    """Summary of one emotion computation."""

    model: str
    windows: int
    null_windows: int
    relabeled_high_rms: int
    relabeled_low_confidence: int
    skipped: bool = False


def _intensities(result: EmotionWindowResult, one_hot: bool) -> list[float]:
    if not one_hot:
        return [float(result.probabilities.get(c, 0.0)) for c in EMOTION_COLUMNS]
    winner = result.label
    if winner not in EMOTION_COLUMNS:
        winner = max(EMOTION_COLUMNS, key=lambda c: float(result.probabilities.get(c, 0.0)))
    return [1.0 if c == winner else 0.0 for c in EMOTION_COLUMNS]


def _confidence(result: EmotionWindowResult, one_hot: bool) -> float:
    """Confidence used by the heuristics and written as ``PredictedConfidence`` (Prime behaviour, D8).

    One-hot releases (2.6.0) report the probability of the emitted label. Others report the highest
    class probability: in 2.5.0 the label is chosen among 9 classes before CONTEMPT_LEFT / RIGHT are
    merged, so it is not always the class with the highest merged probability.
    """
    if one_hot:
        return float(result.probabilities.get(result.label, max(result.probabilities.values())))
    return float(max(result.probabilities.values()))


def outputs_are_current(session_root: str, model_key: str) -> bool:
    """True when the emotion outputs exist and the sidecar names ``model_key``."""
    needed = (EMOTIONS_CSV, CLASSIFIER_INPUTS, RMS_SIGNALS, SIDECAR)
    if not all(os.path.isfile(os.path.join(session_root, p)) for p in needed):
        return False
    try:
        with open(os.path.join(session_root, SIDECAR), encoding="utf-8") as fh:
            return json.load(fh).get("model") == model_key
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def write_sidecar(session_root: str, model: EmotionModel) -> None:
    info = model.info
    sidecar = {
        "model": info.key,
        "model_name": info.name,
        "model_version": info.version,
        "feature_set": info.feature_set,
        "kind": info.kind,
    }
    os.makedirs(os.path.join(session_root, "results"), exist_ok=True)
    with open(os.path.join(session_root, SIDECAR), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    legacy = os.path.join(session_root, _LEGACY_SIDECAR)
    if os.path.isfile(legacy):
        os.remove(legacy)


def compute_emotions(
    session_root: str,
    model: str | EmotionModel | None = None,
    *,
    force: bool = False,
    high_rms_threshold_uv: float | None = None,
    high_rms_max_gap_windows: int | None = None,
    low_confidence_threshold: float | None = None,
    high_rms_enabled: bool | None = None,
    low_confidence_enabled: bool | None = None,
) -> EmotionRun | None:
    """Compute and write the emotion outputs of a session.

    ``model`` is a model key or alias (default ``standard-latest``, decision D9) or a loaded model.
    The post-processing parameters override the release defaults (decision D8). Existing outputs
    produced by the same model are kept unless ``force`` is set. Returns ``None`` when the session
    has no usable EXG.
    """
    if not isinstance(model, EmotionModel):
        model = load_model(model)
    post = model.postprocess.with_overrides(
        high_rms_threshold_uv=high_rms_threshold_uv,
        high_rms_max_gap_windows=high_rms_max_gap_windows,
        low_confidence_threshold=low_confidence_threshold,
        high_rms_enabled=high_rms_enabled,
        low_confidence_enabled=low_confidence_enabled,
    )
    if not force and outputs_are_current(session_root, model.info.key):
        logger.info("Emotion outputs of %s already present, skipping", model.info.key)
        return EmotionRun(model.info.key, 0, 0, 0, 0, skipped=True)

    recording = load_exg(session_root)
    if recording is None:
        return None
    if recording.data.ndim != 2 or recording.data.shape[1] != N_CHANNELS:
        logger.error("Expected %d EXG channels, got shape %s", N_CHANNELS, recording.data.shape)
        return None

    filtered, _ = prepare(recording.data, model.preprocess)
    all_windows = list(windows(filtered, recording.timestamps))
    if not all_windows:
        logger.error("Recording too short for a 2.0 s window")
        return None

    model.reset()
    valid = [w.samples for w in all_windows if not w.is_null]
    if valid:
        model.set_subject_context(np.stack(valid))
    else:
        logger.warning("No valid window for subject calibration")

    emotion_rows, rms_rows, model_rows = [], [], []
    for window in all_windows:
        if window.is_null:
            emotion_rows.append([window.timestamp] + [np.nan] * len(EMOTION_COLUMNS))
            rms_rows.append([window.timestamp] + [np.nan] * N_CHANNELS)
            continue
        result = model.predict(window.samples)
        emotion_rows.append([window.timestamp] + _intensities(result, post.one_hot))
        rms_rows.append([window.timestamp] + result.channel_rms.tolist())
        model_rows.append(
            [window.timestamp]
            + result.model_features.tolist()
            + [result.label, _confidence(result, post.one_hot)]
            + result.channel_rms.tolist()
        )

    emotions_dir = os.path.join(session_root, EMOTIONS_DIR)
    os.makedirs(emotions_dir, exist_ok=True)
    os.makedirs(os.path.join(session_root, "results"), exist_ok=True)

    emo_df = pd.DataFrame(emotion_rows, columns=["Timestamp"] + EMOTION_COLUMNS)
    inputs_columns = ["Timestamp"] + model.feature_columns + ["PredictedLabel", "PredictedConfidence"] + RMS_COLUMNS
    inputs_df = pd.DataFrame(model_rows, columns=inputs_columns)
    inputs_df["HighRmsArtifact"] = inputs_df[RMS_COLUMNS].sum(axis=1).gt(post.high_rms_threshold_uv).to_numpy()

    model_df = inputs_df[["Timestamp"] + RMS_COLUMNS + ["PredictedLabel", "PredictedConfidence"]].copy()
    audit = []
    if post.high_rms_enabled and not model_df.empty:
        model_df, emo_df, rel = heuristics.apply_high_rms(
            model_df, emo_df, RMS_COLUMNS, post.high_rms_threshold_uv, post.high_rms_max_gap_windows
        )
        audit.append(rel)
    if post.low_confidence_enabled and not model_df.empty:
        model_df, emo_df, rel = heuristics.apply_low_confidence(model_df, emo_df, post.low_confidence_threshold)
        audit.append(rel)
    non_empty = [a for a in audit if not a.empty]
    relabeled = (
        pd.concat(non_empty, ignore_index=True)
        if non_empty
        else pd.DataFrame(columns=list(model_df.columns) + heuristics.AUDIT_COLUMNS)
    )

    inputs_df.to_csv(os.path.join(session_root, CLASSIFIER_INPUTS), index=False)
    pd.DataFrame(np.asarray(rms_rows, dtype=float), columns=["Timestamp"] + RMS_COLUMNS).to_csv(
        os.path.join(session_root, RMS_SIGNALS), index=False
    )
    relabeled.to_csv(os.path.join(session_root, RELABELED), index=False)
    avg = model_df[["Timestamp", "PredictedLabel", "PredictedConfidence"]].copy()
    avg.insert(1, "AvgRMS", model_df[RMS_COLUMNS].mean(axis=1))
    avg.to_csv(os.path.join(session_root, AVG_RMS), index=False)
    resample_to_timebase(emo_df).to_csv(os.path.join(session_root, EMOTIONS_CSV), na_rep="NULL", index=False)
    write_sidecar(session_root, model)

    n_null = sum(1 for w in all_windows if w.is_null)
    reasons = relabeled["RelabelReason"] if "RelabelReason" in relabeled else pd.Series(dtype=str)
    run = EmotionRun(
        model=model.info.key,
        windows=len(all_windows),
        null_windows=n_null,
        relabeled_high_rms=int(reasons.astype(str).str.startswith("high_rms").sum()),
        relabeled_low_confidence=int((reasons == "low_confidence").sum()),
    )
    logger.info(
        "Emotions with %s: %d windows, %d null, %d high-RMS and %d low-confidence relabels",
        run.model,
        run.windows,
        run.null_windows,
        run.relabeled_high_rms,
        run.relabeled_low_confidence,
    )
    return run
