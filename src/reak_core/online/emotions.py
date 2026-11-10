# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Online (causal) emotion classification of one headset stream.

Online counterpart of :func:`reak_core.femg.emotions.compute_emotions`, for live use. Differences from
the offline computation, all forced by causality:

- Filters: the release's notch and bandpass run forward only (``sosfilt`` with state) instead of
  zero-phase over the whole recording. RMS, WL and MDF depend on magnitude, so the change is small.
- Calibration: the per-subject amplitude statistics come from a ring buffer of the last
  ``calibration_windows`` valid windows (default 600 = 5 min at 2 Hz), recomputed at every window with
  the release's own math (gated p95 + cold-start prior for 2.6.0, mean / std before), instead of every
  window of the recording. No label is emitted until ``min_calibration_windows`` windows (default 60 =
  30 s) have been seen.
- High-RMS artifact rule: a window whose summed RMS exceeds the threshold takes the label of the latest
  clean window within ``max_gap`` windows **before** it (never after), otherwise Neutral.

Windows are 2.0 s every 0.5 s on the offline grid, so results are on the 2 Hz timebase. Several streams
may share one loaded model: each classifier injects its own calibration before predicting.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos

from reak_core.constants import EXG_SAMPLING_RATE, N_CHANNELS
from reak_core.femg.emotions import _confidence, _intensities
from reak_core.femg.signal import FILTER_ORDER, NOTCH_Q, STEP_SAMPLES, WINDOW_SAMPLES
from reak_core.io.exg import invalid_mask
from reak_core.models import EMOTION_COLUMNS, EmotionModel, load_model
from reak_core.online.stream import CausalSosFilter, SampleClock, Windower

FILTER_WARMUP_SAMPLES = EXG_SAMPLING_RATE
"""Samples skipped after a (re)start before the first window, while the filters settle (1 s)."""


@dataclass
class OnlineEmotionResult:
    """One window on the 2 Hz timebase."""

    sample: int
    """Sample index of the window centre, from session start."""
    time_s: float
    """``sample / fs``: seconds since session start."""
    status: str
    """``"ok"``, ``"calibrating"`` (not enough windows yet) or ``"null"`` (invalid samples in the window)."""
    label: str | None = None
    confidence: float | None = None
    intensities: dict[str, float] = field(default_factory=dict)
    """Values written to ``Emotions.csv`` (one-hot of the label, or probabilities), empty unless ``ok``."""
    probabilities: dict[str, float] = field(default_factory=dict)
    channel_rms: np.ndarray | None = None
    high_rms_artifact: bool = False
    relabel_reason: str | None = None
    """``"high_rms_borrowed"``, ``"high_rms_neutral"`` or ``"low_confidence"`` when the label was changed."""


def _filter_sos(model: EmotionModel, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    spec = model.preprocess
    sections = []
    if spec.notch_hz is not None and spec.notch_hz > 0:
        b, a = iirnotch(spec.notch_hz / nyq, NOTCH_Q)
        sections.append(tf2sos(b, a))
    sections.append(
        butter(FILTER_ORDER, [spec.bandpass_low / nyq, spec.bandpass_high / nyq], btype="bandpass", output="sos")
    )
    single = np.vstack(sections)
    # Zero-phase filtering applies the filter twice (|H|²). Running the cascade twice forward gives the same
    # magnitude response, which is what RMS, WL and MDF depend on; only the phase differs.
    return np.vstack([single, single])


class OnlineEmotionClassifier:
    """Stateful causal emotion classifier for one headset."""

    def __init__(
        self,
        model: str | EmotionModel | None = None,
        *,
        fs: float = EXG_SAMPLING_RATE,
        calibration_windows: int = 600,
        min_calibration_windows: int = 60,
        max_fill_samples: int = 500,
    ) -> None:
        if not isinstance(model, EmotionModel):
            model = load_model(model)
        if not hasattr(model, "set_amplitude_stats"):
            raise TypeError(f"model {model.info.key} does not support online calibration")
        if not 0 < min_calibration_windows <= calibration_windows:
            raise ValueError("expected 0 < min_calibration_windows <= calibration_windows")
        self.model = model
        self.fs = fs
        self.post = model.postprocess
        self.calibration_windows = calibration_windows
        self.min_calibration_windows = min_calibration_windows
        self._sos = _filter_sos(model, fs)
        self._emg = list(model.emg_indices)
        self._clock = SampleClock(max_fill_samples)
        self._filter = CausalSosFilter(self._sos, N_CHANNELS)
        self._windower = Windower(WINDOW_SAMPLES, STEP_SAMPLES, warmup=FILTER_WARMUP_SAMPLES)
        self.reset()

    # -- Lifecycle ----------------------------------------------------------------------------------

    def reset(self) -> None:
        """Forget everything: calibration, filters and windows (new session or new wearer)."""
        self._rms: deque[np.ndarray] = deque(maxlen=self.calibration_windows)
        self._wl: deque[np.ndarray] = deque(maxlen=self.calibration_windows)
        self._stats: tuple[np.ndarray, ...] | None = None
        self._last_clean: tuple[int, OnlineEmotionResult] | None = None
        self._window_index = 0
        self._clock.reset()
        self._filter.reset()
        self._windower.reset()

    @property
    def n_calibration_windows(self) -> int:
        return len(self._rms)

    @property
    def calibrated(self) -> bool:
        return len(self._rms) >= self.min_calibration_windows

    # -- Processing ---------------------------------------------------------------------------------

    def push(self, samples: np.ndarray, first_sample: int | None = None) -> list[OnlineEmotionResult]:
        """Add raw µV samples ``(n, 8)`` starting at sample ``first_sample`` of the session.

        ``first_sample`` lets the classifier place blocks after lost packets: a gap of up to
        ``max_fill_samples`` becomes invalid samples (null windows), a longer one restarts the filters and
        windows (calibration is kept). Without it, blocks are assumed contiguous. Returns the windows
        completed by this block.
        """
        block = self._clock.feed(samples, first_sample)
        if block is None:
            return []
        if block.data.shape[1] != N_CHANNELS:
            raise ValueError(f"expected {N_CHANNELS} channels, got {block.data.shape[1]}")
        if block.restart:
            self._filter.reset()
            self._last_clean = None
        invalid = invalid_mask(block.data)
        filtered = self._filter.process(np.where(invalid, np.nan, block.data))
        any_invalid = invalid.any(axis=1)
        return [
            self._classify(start, chunk, bad)
            for start, chunk, bad in self._windower.feed(block.start, filtered, any_invalid, block.restart)
        ]

    def _classify(self, start: int, chunk: np.ndarray, bad: np.ndarray) -> OnlineEmotionResult:
        self._window_index += 1
        centre = start + WINDOW_SAMPLES // 2
        base = {"sample": centre, "time_s": centre / self.fs}
        if bad.any() or not np.all(np.isfinite(chunk)):
            return OnlineEmotionResult(status="null", **base)
        window = chunk.T.astype(np.float64, copy=True)  # (8, 500), offline layout
        rms = np.sqrt(np.mean(window**2, axis=1))
        wl = np.sum(np.abs(np.diff(window[self._emg], axis=1)), axis=1)
        self._rms.append(rms)
        self._wl.append(wl)
        if not self.calibrated:
            return OnlineEmotionResult(status="calibrating", channel_rms=rms, **base)
        self._update_stats()
        self.model.set_amplitude_stats(*self._stats)
        result = self.model.predict(window)
        label = result.label
        confidence = _confidence(result, self.post.one_hot)
        intensities = dict(zip(EMOTION_COLUMNS, _intensities(result, self.post.one_hot)))
        artifact = bool(rms.sum() > self.post.high_rms_threshold_uv)
        out = OnlineEmotionResult(
            status="ok",
            label=label,
            confidence=confidence,
            intensities=intensities,
            probabilities=dict(result.probabilities),
            channel_rms=rms,
            high_rms_artifact=artifact,
            **base,
        )
        if artifact and self.post.high_rms_enabled:
            self._apply_high_rms(out)
        elif self.post.low_confidence_enabled and confidence < self.post.low_confidence_threshold:
            out.relabel_reason = "low_confidence"
            out.intensities = {c: 0.0 for c in EMOTION_COLUMNS}
            if label in EMOTION_COLUMNS:
                out.intensities[label] = confidence
            out.intensities["Neutral"] = 1.0 - confidence
            out.label, out.confidence = "Neutral", 1.0 - confidence
        if not artifact:
            self._last_clean = (self._window_index, out)
        return out

    def _apply_high_rms(self, out: OnlineEmotionResult) -> None:
        """Past-only high-RMS rule: borrow the latest clean window within ``max_gap`` windows, else Neutral."""
        if (
            self._last_clean is not None
            and self._window_index - self._last_clean[0] <= self.post.high_rms_max_gap_windows
        ):
            donor = self._last_clean[1]
            out.label, out.confidence = donor.label, donor.confidence
            out.intensities = dict(donor.intensities)
            out.relabel_reason = "high_rms_borrowed"
        else:
            out.label, out.confidence = "Neutral", 1.0
            out.intensities = {c: (1.0 if c == "Neutral" else 0.0) for c in EMOTION_COLUMNS}
            out.relabel_reason = "high_rms_neutral"

    def _update_stats(self) -> None:
        rms = np.stack(self._rms)
        wl = np.stack(self._wl)
        if self.model.amplitude_calibration == "gated_prior":
            self._stats = tuple(self.model.gated_prior_stats(rms, wl))
        else:
            self._stats = (rms.mean(axis=0), rms.std(axis=0), wl.mean(axis=0), wl.std(axis=0))

    # -- Checkpointing ------------------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Calibration history, JSON-serialisable. Filters and windows restart after a reload."""
        return {
            "model": self.model.info.key,
            "rms": np.stack(self._rms).tolist() if self._rms else [],
            "wl": np.stack(self._wl).tolist() if self._wl else [],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("model") != self.model.info.key:
            raise ValueError(f"checkpoint is for model {state.get('model')}, not {self.model.info.key}")
        self.reset()
        self._rms.extend(np.asarray(r, dtype=np.float64) for r in state.get("rms", []))
        self._wl.extend(np.asarray(w, dtype=np.float64) for w in state.get("wl", []))
