# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Per-session signal-quality summary, written into ``results/processing.json``.

Built only from the raw files and the outputs the steps already write; no signal value is copied.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from reak_core.constants import CHANNEL_NAMES, EEG_SOURCE_CHANNELS
from reak_core.io.exg import invalid_mask, load_exg
from reak_core.io.positioning import check_gps, check_uwb
from reak_core.io.session import Session


def _read(session: Session, rel: str) -> pd.DataFrame | None:
    path = session.path(rel)
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path, na_values=["NULL"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def _fraction(numerator: float, denominator: float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def exg_quality(session: Session) -> dict | None:
    recording = load_exg(session.root)
    if recording is None:
        return None
    mask = invalid_mask(recording.data)
    n = len(mask)
    names = list(CHANNEL_NAMES[: mask.shape[1]])
    steps = np.diff(recording.timestamps)
    return {
        "samples": n,
        "duration_s": round(float(recording.timestamps[-1]), 3) if n else 0.0,
        "invalid_fraction": {name: _fraction(mask[:, i].sum(), n) for i, name in enumerate(names)},
        "eeg_invalid_fraction": _fraction(mask[:, list(EEG_SOURCE_CHANNELS)].any(axis=1).sum(), n),
        "time_gaps_over_5s": int(np.sum(steps >= 5.0)),
    }


def emotions_quality(session: Session) -> dict | None:
    rms = _read(session, os.path.join("features", "emotions", "rmsSignals.csv"))
    inputs = _read(session, os.path.join("features", "emotions", "emotionClassifierInputs.csv"))
    if rms is None:
        return None
    null = rms.drop(columns=["Timestamp"]).isna().all(axis=1)
    out = {"windows": len(rms), "null_windows": int(null.sum()), "null_fraction": _fraction(null.sum(), len(rms))}
    if inputs is not None and "HighRmsArtifact" in inputs:
        out["high_rms_windows"] = int(inputs["HighRmsArtifact"].sum())
    return out


def cognition_quality(session: Session) -> dict | None:
    stats = _read(session, os.path.join("features", "cognition", "artefactStats.csv"))
    cognition = _read(session, os.path.join("results", "Cognition.csv"))
    if stats is None and cognition is None:
        return None
    out = {}
    if stats is not None and len(stats):
        row = stats.iloc[0]
        out["epochs"] = int(row["n_epochs_total"])
        out["rejected_epoch_fraction"] = _fraction(row["n_epochs_rejected"], row["n_epochs_total"])
    if cognition is not None and "Engagement" in cognition:
        out["engagement_missing_fraction"] = _fraction(cognition["Engagement"].isna().sum(), len(cognition))
    return out


def ppg_quality(session: Session) -> dict | None:
    ppg = _read(session, os.path.join("features", "ppg", "ppg_resampled.csv"))
    heart = _read(session, os.path.join("results", "HeartDynamics.csv"))
    if ppg is None and heart is None:
        return None
    out = {}
    if ppg is not None and "Value" in ppg:
        out["rejected_fraction"] = _fraction(ppg["Value"].isna().sum(), len(ppg))
    if heart is not None and "mean_hr" in heart:
        out["heart_rate_missing_fraction"] = _fraction(heart["mean_hr"].isna().sum(), len(heart))
    return out


def eda_quality(session: Session) -> dict | None:
    arousal = _read(session, os.path.join("results", "Arousal.csv"))
    if arousal is None or arousal.shape[1] < 2:
        return None
    values = arousal.drop(columns=["Timestamp"], errors="ignore")
    return {"missing_fraction": _fraction(values.isna().all(axis=1).sum(), len(values))}


def session_quality(session: Session) -> dict:
    """Quality summary of every stream present in the session (``None`` for absent streams)."""
    uwb, gps = check_uwb(session.root), check_gps(session.root)
    return {
        "exg": exg_quality(session),
        "emotions": emotions_quality(session),
        "cognition": cognition_quality(session),
        "ppg": ppg_quality(session),
        "eda": eda_quality(session),
        "uwb": uwb.as_dict() if uwb.present else None,
        "gps": gps.as_dict() if gps.present else None,
    }
