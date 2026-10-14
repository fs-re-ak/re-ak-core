# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
fEMG signal preparation and windowing (decisions D6, D7, D24).

1. Invalid raw samples (non-finite or at ADC full scale) are marked and their mask is kept.
2. Each channel's invalid samples are linearly interpolated, only so the filters do not spread
   NaN; a channel with no valid sample is left as is.
3. The release's notch (if any) and bandpass are applied to the whole recording, zero-phase.
4. The mask is reapplied: invalid samples are NaN again.
5. Windows of 2.0 s (500 samples) every 0.5 s; a window containing any non-finite value is null.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from reak_core.constants import EXG_SAMPLING_RATE
from reak_core.io.exg import invalid_mask
from reak_core.models.base import PreprocessSpec

WINDOW_SAMPLES = 500
STEP_SAMPLES = 125
NOTCH_Q = 30.0
FILTER_ORDER = 4


def interpolate_invalid(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate masked samples per channel; channels without valid samples are unchanged."""
    out = np.array(data, dtype=float, copy=True)
    out[mask] = np.nan
    index = np.arange(out.shape[0])
    for ch in range(out.shape[1]):
        bad = mask[:, ch]
        if bad.any() and not bad.all():
            out[bad, ch] = np.interp(index[bad], index[~bad], out[~bad, ch])
    return out


def filter_exg(data: np.ndarray, spec: PreprocessSpec, fs: float = EXG_SAMPLING_RATE) -> np.ndarray:
    """Zero-phase notch (when declared) then zero-phase Butterworth bandpass over the whole recording."""
    nyq = fs / 2.0
    x = np.asarray(data, dtype=np.float64)
    if spec.notch_hz is not None and spec.notch_hz > 0:
        if spec.notch_hz >= nyq:
            raise ValueError(f"notch_hz {spec.notch_hz} must be below Nyquist {nyq}")
        b, a = iirnotch(spec.notch_hz / nyq, NOTCH_Q)
        x = filtfilt(b, a, x, axis=0)
    sos = butter(FILTER_ORDER, [spec.bandpass_low / nyq, spec.bandpass_high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def prepare(raw: np.ndarray, spec: PreprocessSpec, fs: float = EXG_SAMPLING_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(filtered, mask)``: the filtered recording with invalid samples set back to NaN, and the mask."""
    mask = invalid_mask(raw)
    filtered = filter_exg(interpolate_invalid(raw, mask), spec, fs)
    filtered[mask] = np.nan
    return filtered, mask


@dataclass
class Window:
    timestamp: float
    """Median sample time of the window (seconds)."""
    samples: np.ndarray | None
    """Filtered window, shape (8, 500); ``None`` for a null window."""

    @property
    def is_null(self) -> bool:
        return self.samples is None


def windows(
    filtered: np.ndarray,
    timestamps: np.ndarray,
    window: int = WINDOW_SAMPLES,
    step: int = STEP_SAMPLES,
) -> Iterator[Window]:
    """Slide over the filtered recording; a window with any non-finite value is null."""
    n = filtered.shape[0]
    for start in range(0, n - window + 1, step):
        chunk = filtered[start : start + window]
        ts = float(np.median(timestamps[start : start + window]))
        if np.all(np.isfinite(chunk)):
            yield Window(ts, chunk.T.astype(np.float64, copy=True))
        else:
            yield Window(ts, None)
