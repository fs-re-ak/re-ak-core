# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Online (causal) engagement of one headset stream.

Online counterpart of :func:`reak_core.eeg.cognition.computeCognitiveIndexes`:

1. Invalid raw samples (D6, D24) are masked; a sample is invalid for cognition when ch1, ch2 or ch5 is
   (D23).
2. Causal NLMS decorrelation (same filter as offline). Only EAR_R (ch5) is decorrelated: T9 = −ch5/2 and
   T10 = ch5/2 have the same band powers, so the bilateral average only needs ch5. Weights depend only
   on the raw input, so this gives exactly the offline values for that channel.
3. 0.5–30 Hz 4th-order Butterworth, forward only (offline: zero-phase).
4. Windows of 2.0 s every 0.5 s on the offline grid. A window is null when more than 25 % of its samples
   are invalid, or when either 1 s half fails the offline epoch criteria (> 20 % invalid, peak or
   peak-to-peak above 500 µV). Offline judges fixed 1 s epochs from session start; online judges the
   window's own halves, which are the same epochs every other window.
5. Welch PSD (``nperseg=256``), delta 0–4, theta 4–8, alpha 8–13, beta 13–22 Hz, and
   ``Engagement = beta / (alpha + theta)`` as offline.
6. ``EngagementCalibrated``: :class:`PercentileMinMaxNormalizer` on ``log10(Engagement)`` (RE-AK online
   calibration specification), 0–1 relative to the wearer's last two minutes.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, welch

from reak_core.constants import EEG_SOURCE_CHANNELS, EXG_SAMPLING_RATE, N_CHANNELS
from reak_core.eeg.nlms import CausalNLMSFilter
from reak_core.io.exg import invalid_mask
from reak_core.online.normalizer import PercentileMinMaxNormalizer
from reak_core.online.stream import CausalSosFilter, SampleClock, Windower

EAR_R = 4
WINDOW_SAMPLES = 500
STEP_SAMPLES = 125
BANDPASS_HZ = (0.5, 30.0)
BANDS = {"delta": (0.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 22.0)}
MAX_WINDOW_INVALID = 0.25
MAX_EPOCH_INVALID = 0.20
MAX_AMPLITUDE_UV = 500.0
FILTER_WARMUP_SAMPLES = 2 * EXG_SAMPLING_RATE
"""Samples skipped after a (re)start while the 0.5 Hz high-pass settles (2 s)."""


@dataclass
class OnlineEngagementResult:
    sample: int
    """Sample index of the window centre, from session start."""
    time_s: float
    status: str
    """``"ok"``, ``"calibrating"`` (raw value only) or ``"null"`` (rejected window)."""
    engagement: float | None = None
    """``beta / (alpha + theta)``, same definition and scale as ``Cognition.csv``."""
    engagement_calibrated: float | None = None
    """0–1 score relative to the last two minutes (``None`` while calibrating or for null windows)."""
    reject_reason: str | None = None


def band_powers(x: np.ndarray, fs: float = EXG_SAMPLING_RATE) -> dict[str, float]:
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs <= hi)
        out[name] = float(np.trapezoid(psd[sel], freqs[sel])) if sel.any() else 0.0
    return out


class OnlineEngagement:
    """Stateful causal engagement for one headset."""

    def __init__(
        self,
        *,
        fs: float = EXG_SAMPLING_RATE,
        apply_nlms: bool = True,
        normalizer: PercentileMinMaxNormalizer | None = None,
        max_fill_samples: int = 500,
    ) -> None:
        self.fs = fs
        self.apply_nlms = apply_nlms
        nyq = fs / 2.0
        sos = butter(4, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="bandpass", output="sos")
        sos = np.vstack([sos, sos])  # twice forward = the |H|² magnitude of offline zero-phase filtfilt
        self._clock = SampleClock(max_fill_samples)
        self._nlms = CausalNLMSFilter(n_channels=N_CHANNELS, fs=fs, output_channels=[EAR_R])
        self._filter = CausalSosFilter(sos, 1)
        self._windower = Windower(WINDOW_SAMPLES, STEP_SAMPLES, warmup=FILTER_WARMUP_SAMPLES)
        self.normalizer = normalizer or PercentileMinMaxNormalizer()

    def reset(self) -> None:
        """Forget everything: calibration, NLMS weights, filters and windows."""
        self.normalizer.reset()
        self._nlms.reset()
        self._clock.reset()
        self._filter.reset()
        self._windower.reset()

    @property
    def calibrated(self) -> bool:
        return self.normalizer.ready

    def push(self, samples: np.ndarray, first_sample: int | None = None) -> list[OnlineEngagementResult]:
        """Add raw µV samples ``(n, 8)``; see :meth:`OnlineEmotionClassifier.push` for ``first_sample``."""
        block = self._clock.feed(samples, first_sample)
        if block is None:
            return []
        if block.data.shape[1] != N_CHANNELS:
            raise ValueError(f"expected {N_CHANNELS} channels, got {block.data.shape[1]}")
        if block.restart:
            self._filter.reset()
        invalid = invalid_mask(block.data)
        raw = np.where(invalid, 0.0, block.data)  # NLMS input: zero-filled like nlms_nan_safe
        if self.apply_nlms:
            raw = self._nlms.push_batch(raw)
        bad = invalid[:, list(EEG_SOURCE_CHANNELS)].any(axis=1)
        t10 = np.where(bad, np.nan, raw[:, EAR_R] / 2.0)[:, None]
        filtered = self._filter.process(t10)[:, 0:1]
        return [
            self._window(start, chunk[:, 0], mask)
            for start, chunk, mask in self._windower.feed(block.start, filtered, bad, block.restart)
        ]

    def _window(self, start: int, x: np.ndarray, bad: np.ndarray) -> OnlineEngagementResult:
        centre = start + WINDOW_SAMPLES // 2
        base = {"sample": centre, "time_s": centre / self.fs}
        reason = None
        if bad.mean() > MAX_WINDOW_INVALID:
            reason = "invalid_samples"
        else:
            half = WINDOW_SAMPLES // 2
            for lo in (0, half):
                seg, seg_bad = x[lo : lo + half], bad[lo : lo + half]
                if seg_bad.mean() > MAX_EPOCH_INVALID:
                    reason = "invalid_samples"
                elif np.max(np.abs(seg)) > MAX_AMPLITUDE_UV:
                    reason = "peak_amp"
                elif np.ptp(seg) > MAX_AMPLITUDE_UV:
                    reason = "ptp"
                if reason:
                    break
        if reason:
            return OnlineEngagementResult(status="null", reject_reason=reason, **base)
        p = band_powers(x, self.fs)
        engagement = p["beta"] / (p["alpha"] + p["theta"] + 1e-8)
        calibrated = self.normalizer.update(float(np.log10(max(engagement, 1e-12))))
        return OnlineEngagementResult(
            status="ok" if calibrated is not None else "calibrating",
            engagement=float(engagement),
            engagement_calibrated=calibrated,
            **base,
        )

    def state_dict(self) -> dict:
        return {"normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.reset()
        self.normalizer.load_state_dict(state.get("normalizer", {}))
