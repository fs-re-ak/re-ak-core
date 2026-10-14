# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Hermès EXG (EEG / fEMG) loading, invalid-sample marking and EEG channel derivation.

- Loading follows decision D2 (NucleusKit): optional text header, named columns reordered into
  hardware order, incomplete last line ignored.
- Invalid samples follow decisions D6 and D24: a raw sample is invalid when it is non-finite or
  at ADC full scale (absolute value within 0.1 µV of 375 000 µV). There is no near-zero rule.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reak_core.constants import (
    CHANNEL_NAMES,
    EXG_FILENAMES,
    FULL_SCALE_TOLERANCE_UV,
    FULL_SCALE_UV,
    N_CHANNELS,
    hardware_channel_index,
)
from reak_core.io.csvfiles import loadtxt_drop_last_if_incomplete, read_header
from reak_core.io.timestamps import normalise_timestamps_to_seconds

logger = logging.getLogger(__name__)

_TIMESTAMP_HEADERS = {"timestamp", "time", "t"}


@dataclass
class ExgRecording:
    """Raw EXG of one session."""

    path: str
    timestamps: np.ndarray  # seconds from the first sample
    data: np.ndarray  # (n_samples, n_channels) µV, hardware order, unmarked


def find_exg_file(session_root: str) -> str | None:
    """Return the first raw EXG file found under ``rawData/``, in EXG_FILENAMES order."""
    for name in EXG_FILENAMES:
        path = os.path.join(session_root, "rawData", name)
        if os.path.isfile(path):
            return path
    return None


def _reorder_named_exg(data: np.ndarray, header: list[str]) -> np.ndarray:
    """Place named channel columns into hardware order; fall back to file order when ambiguous."""
    if data.shape[1] != len(header):
        return data
    ts_idx = 0
    for i, name in enumerate(header):
        if name.lower() in _TIMESTAMP_HEADERS:
            ts_idx = i
            break
    placed: dict[int, int] = {}
    for i, name in enumerate(header):
        if i == ts_idx:
            continue
        hw = hardware_channel_index(name)
        if hw is None or hw in placed:
            return data
        placed[hw] = i
    if set(placed) != set(range(N_CHANNELS)):
        return data
    ordered = np.empty((data.shape[0], 1 + N_CHANNELS), dtype=float)
    ordered[:, 0] = data[:, ts_idx]
    for hw, src in placed.items():
        ordered[:, 1 + hw] = data[:, src]
    return ordered


def load_exg_matrix(filepath: str) -> np.ndarray:
    """Load a raw EXG CSV as ``[timestamp, ch1..ch8]`` in hardware order, timestamps left raw."""
    header = read_header(filepath)
    data = np.atleast_2d(loadtxt_drop_last_if_incomplete(filepath))
    if header is None:
        return data
    return _reorder_named_exg(data, header)


def load_exg(session_root: str) -> ExgRecording | None:
    """Load the session's raw EXG, or ``None`` when no usable file exists."""
    path = find_exg_file(session_root)
    if path is None:
        logger.warning("No EXG file in %s (looked for %s)", os.path.join(session_root, "rawData"), EXG_FILENAMES)
        return None
    try:
        matrix = load_exg_matrix(path)
    except (OSError, ValueError) as exc:
        logger.error("Could not load EXG file %s: %s", path, exc)
        return None
    if matrix.ndim < 2 or matrix.shape[1] < 2:
        logger.error("EXG file %s has an unexpected shape %s", path, matrix.shape)
        return None
    timestamps = normalise_timestamps_to_seconds(matrix[:, 0])
    return ExgRecording(path=path, timestamps=timestamps, data=matrix[:, 1:].astype(float, copy=True))


def invalid_mask(data: np.ndarray) -> np.ndarray:
    """Boolean mask of invalid raw samples (D6, D24): non-finite or at ADC full scale."""
    data = np.asarray(data, dtype=float)
    with np.errstate(invalid="ignore"):
        full_scale = np.abs(np.abs(data) - FULL_SCALE_UV) < FULL_SCALE_TOLERANCE_UV
    return ~np.isfinite(data) | full_scale


def mark_invalid(data: np.ndarray) -> np.ndarray:
    """Copy of ``data`` with invalid samples (see :func:`invalid_mask`) set to NaN."""
    marked = np.array(data, dtype=float, copy=True)
    marked[invalid_mask(marked)] = np.nan
    return marked


def derive_eeg_channels(exg: np.ndarray) -> pd.DataFrame | None:
    """Derive the EEG channels from the 8 hardware channels (midpoint re-reference on EAR_R).

    ``ch5`` (EAR_R) carries the T10 − T9 differential, so T9 = −ch5 / 2 and T10 = ch5 / 2::

        AF7       = ch2 − ch5 / 2      AF8       = ch1 − ch5 / 2
        T9        = −ch5 / 2           T10       = ch5 / 2
        LeftHemi  = ch2                RightHemi = ch1 − ch5
    """
    if exg is None or exg.ndim != 2 or exg.shape[1] < 5:
        logger.error("Cannot derive EEG channels from an array of shape %s", getattr(exg, "shape", None))
        return None
    mid_ref = exg[:, 4] / 2.0
    return pd.DataFrame(
        {
            "AF7": exg[:, 1] - mid_ref,
            "AF8": exg[:, 0] - mid_ref,
            "T9": -mid_ref,
            "T10": mid_ref,
            "LeftHemi": exg[:, 1],
            "RightHemi": exg[:, 0] - exg[:, 4],
        }
    )


__all__ = [
    "CHANNEL_NAMES",
    "ExgRecording",
    "derive_eeg_channels",
    "find_exg_file",
    "invalid_mask",
    "load_exg",
    "load_exg_matrix",
    "mark_invalid",
]
