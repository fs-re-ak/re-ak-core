# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Shimmer wristband loading: GSR (EDA) and PPG columns from the raw files.

``shimmer.csv`` (Vizia) and ``rawShimmer_0.csv`` (legacy) are multi-channel exports
(timestamp, accel x/y/z, GSR in kΩ, PPG in mV, optional gyro); ``gsr.*`` / ``ppg.*`` are
two-column legacy dumps. A header row is optional (decision D2). Timestamps are normalised
with decision D1.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from reak_core.constants import EDA_FILES, PPG_FILES, SHIMMER_TIMESTAMP_COLUMN
from reak_core.io.csvfiles import has_text_header
from reak_core.io.timestamps import normalise_timestamps_to_seconds

logger = logging.getLogger(__name__)


def _read(path: str) -> pd.DataFrame | None:
    try:
        skiprows = 1 if has_text_header(path) else 0
        return pd.read_csv(path, header=None, skiprows=skiprows)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def find_shimmer_file(session_root: str) -> str | None:
    """Return the first Shimmer file found (EDA candidates first), or ``None``."""
    for name, _ in EDA_FILES + PPG_FILES:
        path = os.path.join(session_root, "rawData", name)
        if os.path.isfile(path):
            return path
    return None


def load_eda_signal(session_root: str) -> pd.DataFrame | None:
    """Return a two-column frame (0 = seconds from start, 1 = GSR in kΩ), or ``None``.

    Conversion to conductance (µS) happens after resampling, in :mod:`reak_core.eda`.
    """
    raw_dir = os.path.join(session_root, "rawData")
    for name, column in EDA_FILES:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        df = _read(path)
        if df is None:
            continue
        if df.shape[1] <= column:
            logger.warning("%s has fewer than %d columns; cannot read EDA (column %d)", path, column + 1, column)
            continue
        signal = df[[SHIMMER_TIMESTAMP_COLUMN, column]].copy()
        signal.columns = [0, 1]
        signal[0] = normalise_timestamps_to_seconds(signal[0].values)
        return signal
    logger.warning("No Shimmer EDA file in %s; Shimmer was likely not used for this recording", raw_dir)
    return None


def load_ppg_signal(session_root: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(timestamps_s, ppg)`` from the raw Shimmer file, or ``(None, None)``."""
    raw_dir = os.path.join(session_root, "rawData")
    for name, column in PPG_FILES:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        df = _read(path)
        if df is None or df.shape[1] <= column:
            continue
        timestamps = df[SHIMMER_TIMESTAMP_COLUMN].values.astype(float)
        ppg = df[column].values.astype(float)
        return normalise_timestamps_to_seconds(timestamps), ppg
    logger.warning("No Shimmer PPG file in %s; Shimmer was likely not used for this recording", raw_dir)
    return None, None
