# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Online calibration of an index into a 0–1 score relative to the wearer's recent history.

Specification: RE-AK online calibration of cognitive indexes (percentile min-max). For every new value:

1. Append it to a ring buffer of the last ``window_size`` updates (240 ≈ 2 min at 2 Hz).
2. Once ``min_samples`` values are buffered (60 ≈ 30 s), take the ``low`` and ``high`` quantiles
   (10th / 90th); collapsed quantiles are split by a tiny epsilon.
3. Map ``y = (x - q_low) / (q_high - q_low)`` (no margin).
4. Inside [0, 1] keep ``y``; outside, fold it back with ``1 / (1 + exp(-k (y - 0.5)))`` (k = 10), then
   smooth with ``y = y_prev + alpha (y_new - y_prev)`` (alpha = 0.35).

Difference from the reference implementation: before the warm-up is complete :meth:`update` returns
``None`` (the dashboard shows "calibrating") instead of 0, and the smoother starts at the first calibrated
value instead of being pulled up from 0.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from collections import deque

import numpy as np

_EPS = 1e-6


class PercentileMinMaxNormalizer:
    """Ring-buffer percentile min-max normaliser with sigmoid fold and output smoothing."""

    def __init__(
        self,
        low: float = 0.10,
        high: float = 0.90,
        window_size: int = 240,
        min_samples: int = 60,
        sigmoid_k: float = 10.0,
        alpha: float = 0.35,
    ) -> None:
        if not 0.0 < low < high < 1.0:
            raise ValueError("expected 0 < low < high < 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("expected 0 < alpha <= 1")
        if not 0 < min_samples <= window_size:
            raise ValueError("expected 0 < min_samples <= window_size")
        self.low = low
        self.high = high
        self.window_size = window_size
        self.min_samples = min_samples
        self.sigmoid_k = sigmoid_k
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._buffer: deque[float] = deque(maxlen=self.window_size)
        self.q_low: float | None = None
        self.q_high: float | None = None
        self._y: float | None = None

    @property
    def ready(self) -> bool:
        return self.q_low is not None

    @property
    def n_samples(self) -> int:
        return len(self._buffer)

    def _recompute(self) -> None:
        if len(self._buffer) < self.min_samples:
            self.q_low = self.q_high = None
            return
        arr = np.fromiter(self._buffer, dtype=float)
        q_low, q_high = (float(v) for v in np.quantile(arr, [self.low, self.high]))
        if q_high <= q_low:
            mid = (q_low + q_high) / 2.0
            q_low, q_high = mid - _EPS, mid + _EPS
        self.q_low, self.q_high = q_low, q_high

    def scale(self, x: float) -> float | None:
        """Unsmoothed score of ``x`` against the current range (``None`` while calibrating)."""
        if self.q_low is None or self.q_high is None:
            return None
        y = (x - self.q_low) / (self.q_high - self.q_low)
        if 0.0 <= y <= 1.0:
            return float(y)
        return float(1.0 / (1.0 + np.exp(-self.sigmoid_k * (y - 0.5))))

    def update(self, x: float) -> float | None:
        """Add ``x`` to the history and return its smoothed score (``None`` while calibrating).

        Non-finite values are ignored and return the last smoothed score.
        """
        if not np.isfinite(x):
            return self._y
        self._buffer.append(float(x))
        self._recompute()
        y = self.scale(float(x))
        if y is None:
            return None
        self._y = y if self._y is None else self._y + self.alpha * (y - self._y)
        return self._y

    # -- Checkpointing -------------------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"buffer": list(self._buffer), "y": self._y}

    def load_state_dict(self, state: dict) -> None:
        self.reset()
        self._buffer.extend(float(v) for v in state.get("buffer", []))
        self._recompute()
        self._y = state.get("y")
