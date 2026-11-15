# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Building blocks shared by the online (causal, streaming) processors.

- :class:`SampleClock` places incoming blocks on the stream's sample index (counted from session start),
  drops repeated samples, fills short gaps with NaN rows and reports a restart after a long gap.
- :class:`CausalSosFilter` filters block by block with persistent state; invalid samples are held at the
  last valid value on the filter input (the caller keeps their mask), the online counterpart of the
  offline interpolate-filter-remask sequence.
- :class:`Windower` cuts the stream into windows aligned on the offline grid (window starts are
  multiples of ``step`` samples from session start), so online and offline windows coincide.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi


@dataclass
class Block:
    start: int
    """Sample index of the first row."""
    data: np.ndarray
    """Samples ``(n, channels)``; gap rows are NaN."""
    restart: bool
    """True on the first block and after a gap too long to fill: filters and windows start over."""


class SampleClock:
    """Places blocks on the stream's sample index."""

    def __init__(self, max_fill_samples: int = 500) -> None:
        self.max_fill_samples = max_fill_samples
        self.next: int | None = None

    def reset(self) -> None:
        self.next = None

    def feed(self, data: np.ndarray, first_sample: int | None = None) -> Block | None:
        """Return the block to process, or ``None`` when every sample was already seen."""
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"expected (n_samples, n_channels), got shape {data.shape}")
        if first_sample is None:
            first_sample = self.next if self.next is not None else 0
        first_sample = int(first_sample)
        if self.next is None:
            self.next = first_sample + len(data)
            return Block(first_sample, data, restart=True)
        if first_sample < self.next:
            overlap = self.next - first_sample
            if overlap >= len(data):
                return None
            data, first_sample = data[overlap:], self.next
        gap = first_sample - self.next
        if gap == 0:
            block = Block(first_sample, data, restart=False)
        elif gap <= self.max_fill_samples:
            filler = np.full((gap, data.shape[1]), np.nan)
            block = Block(self.next, np.vstack([filler, data]), restart=False)
        else:
            block = Block(first_sample, data, restart=True)
        self.next = first_sample + len(data)
        return block


def hold_invalid(x: np.ndarray, last_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-finite samples by the previous valid one per channel; returns ``(held, new_last_valid)``.

    Leading invalid samples take ``last_valid`` (from the previous block), or the block's first valid
    sample, or 0 for a channel with no valid sample yet.
    """
    held = np.array(x, dtype=np.float64, copy=True)
    last = np.array(last_valid, dtype=np.float64, copy=True)
    for ch in range(held.shape[1]):
        col = held[:, ch]
        valid = np.isfinite(col)
        if valid.all():
            last[ch] = col[-1]
            continue
        seed = last[ch]
        if not np.isfinite(seed):
            seed = col[valid][0] if valid.any() else 0.0
        idx = np.where(valid, np.arange(len(col)), -1)
        np.maximum.accumulate(idx, out=idx)
        held[:, ch] = np.where(idx >= 0, col[np.maximum(idx, 0)], seed)
        if valid.any():
            last[ch] = col[valid][-1]
    return held, last


class CausalSosFilter:
    """Second-order-section IIR filter applied block by block, one state per channel."""

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        self.sos = np.asarray(sos, dtype=np.float64)
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        self._zi: np.ndarray | None = None
        self._last_valid = np.full(self.n_channels, np.nan)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter ``x`` ``(n, channels)``; non-finite samples are held at the last valid value first."""
        held, self._last_valid = hold_invalid(x, self._last_valid)
        if self._zi is None:
            # Start in steady state on the first sample to avoid the step transient of a DC offset.
            self._zi = sosfilt_zi(self.sos)[:, :, None] * held[0][None, None, :]
        out, self._zi = sosfilt(self.sos, held, axis=0, zi=self._zi)
        return out

    def state_dict(self) -> dict:
        return {
            "zi": None if self._zi is None else self._zi.tolist(),
            "last_valid": [None if not np.isfinite(v) else float(v) for v in self._last_valid],
        }

    def load_state_dict(self, state: dict) -> None:
        self._zi = None if state.get("zi") is None else np.asarray(state["zi"], dtype=np.float64)
        self._last_valid = np.array([np.nan if v is None else v for v in state["last_valid"]], dtype=np.float64)


class Windower:
    """Sliding windows aligned on multiples of ``step`` from sample 0 of the session."""

    def __init__(self, window: int = 500, step: int = 125, warmup: int = 0) -> None:
        self.window = window
        self.step = step
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        self._buf: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._buf_start = 0
        self._next = 0

    def feed(
        self, start: int, data: np.ndarray, mask: np.ndarray, restart: bool
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Append samples and yield ``(window_start, samples (window, channels), invalid (window,))``."""
        if restart or self._buf is None:
            self._buf = data[:0]
            self._mask = mask[:0]
            self._buf_start = start
            first = start + self.warmup
            self._next = -(-first // self.step) * self.step
        self._buf = np.concatenate([self._buf, data])
        self._mask = np.concatenate([self._mask, mask])
        end = self._buf_start + len(self._buf)
        while self._next + self.window <= end:
            lo = self._next - self._buf_start
            yield self._next, self._buf[lo : lo + self.window], self._mask[lo : lo + self.window]
            self._next += self.step
        drop = min(max(0, self._next - self._buf_start), len(self._buf))
        if drop:
            self._buf = self._buf[drop:]
            self._mask = self._mask[drop:]
            self._buf_start += drop
