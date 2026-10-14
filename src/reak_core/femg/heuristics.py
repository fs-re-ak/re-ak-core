# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Post-classifier heuristics (decision D8, Prime behaviour), applied after the classifier and before
resampling. They change ``Emotions.csv``, ``avgRms.csv`` and ``relabeled-samples.csv``;
``emotionClassifierInputs.csv`` keeps the raw classifier output.

- High-RMS artifact: a window whose sum of per-channel raw RMS exceeds the threshold takes the
  classification of the nearest non-artifact window within ``max_gap`` windows (either side),
  otherwise Neutral.
- Low confidence: a window whose emitted probability is below the threshold becomes Neutral
  (``Neutral = 1 - p``, the original label keeps ``p``). Only for releases before 2.6.0.

``model_df`` holds one row per classified (non-null) window: ``Timestamp``, the raw RMS columns,
``PredictedLabel`` and ``PredictedConfidence``.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reak_core.models.base import EMOTION_COLUMNS

AUDIT_COLUMNS = ["OriginalLabel", "OriginalConfidence", "RelabelReason"]


def _emo_index(emo_df: pd.DataFrame, ts: float) -> pd.Index:
    idx = emo_df.index[emo_df["Timestamp"] == ts]
    return idx if not idx.empty else pd.Index([(emo_df["Timestamp"] - ts).abs().idxmin()])


def apply_high_rms(
    model_df: pd.DataFrame, emo_df: pd.DataFrame, rms_cols: list[str], threshold_uv: float, max_gap: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace high-RMS artifact windows; returns ``(model_df, emo_df, relabeled)``."""
    artifact = model_df[rms_cols].sum(axis=1).gt(threshold_uv)
    if not artifact.any():
        return model_df, emo_df, pd.DataFrame(columns=list(model_df.columns) + AUDIT_COLUMNS)
    model_df, emo_df = model_df.copy(), emo_df.copy()
    artifact_pos = np.where(artifact.values)[0]
    donor_pos = np.where(~artifact.values)[0]
    relabeled = model_df[artifact].copy()
    relabeled["OriginalLabel"] = relabeled["PredictedLabel"]
    relabeled["OriginalConfidence"] = relabeled["PredictedConfidence"]
    labels = model_df.index.values
    new_labels, new_confs, reasons = [], [], []
    for pos in artifact_pos:
        idx = labels[pos]
        emo_idx = _emo_index(emo_df, model_df.loc[idx, "Timestamp"])
        if donor_pos.size == 0 or np.abs(donor_pos - pos).min() > max_gap:
            model_df.loc[idx, "PredictedLabel"] = "Neutral"
            model_df.loc[idx, "PredictedConfidence"] = 1.0
            emo_df.loc[emo_idx, EMOTION_COLUMNS] = 0.0
            emo_df.loc[emo_idx, "Neutral"] = 1.0
            new_labels.append("Neutral")
            new_confs.append(1.0)
            reasons.append("high_rms_neutral")
        else:
            donor = labels[donor_pos[int(np.abs(donor_pos - pos).argmin())]]
            label = model_df.loc[donor, "PredictedLabel"]
            conf = model_df.loc[donor, "PredictedConfidence"]
            model_df.loc[idx, "PredictedLabel"] = label
            model_df.loc[idx, "PredictedConfidence"] = conf
            donor_emo = _emo_index(emo_df, model_df.loc[donor, "Timestamp"])
            emo_df.loc[emo_idx, EMOTION_COLUMNS] = emo_df.loc[donor_emo, EMOTION_COLUMNS].values
            new_labels.append(label)
            new_confs.append(float(conf))
            reasons.append("high_rms_borrowed")
    relabeled["PredictedLabel"] = new_labels
    relabeled["PredictedConfidence"] = new_confs
    relabeled["RelabelReason"] = reasons
    return model_df, emo_df, relabeled


def apply_low_confidence(
    model_df: pd.DataFrame, emo_df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Relabel low-confidence windows as Neutral; returns ``(model_df, emo_df, relabeled)``."""
    low = model_df["PredictedConfidence"].lt(threshold)
    if not low.any():
        return model_df, emo_df, pd.DataFrame(columns=list(model_df.columns) + AUDIT_COLUMNS)
    model_df, emo_df = model_df.copy(), emo_df.copy()
    relabeled = model_df[low].copy()
    relabeled.insert(len(relabeled.columns) - 2, "OriginalLabel", relabeled["PredictedLabel"])
    relabeled.insert(len(relabeled.columns) - 2, "OriginalConfidence", relabeled["PredictedConfidence"])
    originals = {
        r["Timestamp"]: (r["PredictedLabel"], float(r["PredictedConfidence"])) for _, r in model_df[low].iterrows()
    }
    model_df.loc[low, "PredictedLabel"] = "Neutral"
    model_df.loc[low, "PredictedConfidence"] = 1.0 - model_df.loc[low, "PredictedConfidence"]
    relabeled["PredictedLabel"] = "Neutral"
    relabeled["PredictedConfidence"] = 1.0 - relabeled["OriginalConfidence"].values
    relabeled["RelabelReason"] = "low_confidence"
    for ts, (label, conf) in originals.items():
        idx = _emo_index(emo_df, ts)
        emo_df.loc[idx, EMOTION_COLUMNS] = 0.0
        if label in EMOTION_COLUMNS:
            emo_df.loc[idx, label] = conf
        emo_df.loc[idx, "Neutral"] = 1.0 - conf
    return model_df, emo_df, relabeled
