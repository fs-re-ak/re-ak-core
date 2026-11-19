"""Floor-only per-channel baseline for 2.8.0 amplitude features.

Calibrate the quiet floor. Do not estimate a ceiling. Dead windows
(RMS at or below ``noise_floor``) are dropped. A short buffer falls back
to the population default. A long buffer uses the ``q``th percentile,
clamped to the population 5th–95th percentile of session baselines.

The feature is ``log(x) - log(baseline)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FLOOR_Q = 15.0
MIN_WINDOWS = 30
LOG_EPS = 1e-12
NOISE_P1_SCALE = 0.5
N_RMS_CHANNELS = 8

RMS_CHANNEL_NAMES = [
    "Supraorbital_L",
    "Supraorbital_R",
    "Zygomatic_L",
    "Zygomatic_R",
    "Temporal_EEG",
    "Glabella",
    "Temporal_R",
    "Nasolabial",
]


def floor_ratio(
    x: np.ndarray,
    baseline: np.ndarray | float,
    eps: float = LOG_EPS,
) -> np.ndarray:
    """``log(x) - log(baseline)``. ``eps`` guards zeros only."""
    xx = np.maximum(np.asarray(x, dtype=np.float64), float(eps))
    bb = np.maximum(np.asarray(baseline, dtype=np.float64), float(eps))
    return np.log(xx) - np.log(bb)


def _clamp_side(raw: float, clipped: float, lo: float, hi: float) -> str:
    if raw < lo:
        return "low"
    if raw > hi:
        return "high"
    return "none"


def session_floor(
    rms: np.ndarray,
    table: dict[str, Any],
) -> dict[str, Any]:
    """Per-channel floor from a recording buffer.

    Parameters
    ----------
    rms:
        Shape ``[n_windows, n_channels]``.
    table:
        ``floor_stats.json`` contents (noise floor, population clamp, ``q``,
        ``min_windows``).
    """
    rms = np.asarray(rms, dtype=np.float64)
    if rms.ndim != 2:
        raise ValueError(f"rms must be [n_windows, n_channels], got {rms.shape}")
    n_ch = int(rms.shape[1])
    noise = np.asarray(table["noise_floor"], dtype=np.float64)
    pop_default = np.asarray(table["pop_default"], dtype=np.float64)
    pop_lo = np.asarray(table["pop_lo"], dtype=np.float64)
    pop_hi = np.asarray(table["pop_hi"], dtype=np.float64)
    if noise.shape != (n_ch,) or pop_default.shape != (n_ch,):
        raise ValueError(
            f"floor table channel count {noise.shape} does not match rms {rms.shape}"
        )
    q = float(table.get("q", FLOOR_Q))
    min_windows = int(table.get("min_windows", MIN_WINDOWS))

    baselines = np.empty(n_ch, dtype=np.float64)
    sources: list[str] = []
    clamped: list[str] = []
    n_valid: list[int] = []
    n_dead: list[int] = []
    for ch in range(n_ch):
        col = rms[:, ch]
        dead = col <= noise[ch]
        valid = col[~dead]
        n_dead.append(int(dead.sum()))
        n_valid.append(int(valid.size))
        if valid.size < min_windows:
            baselines[ch] = float(pop_default[ch])
            sources.append("population")
            clamped.append("none")
            continue
        raw = float(np.percentile(valid, q))
        lo = float(pop_lo[ch])
        hi = float(pop_hi[ch])
        clipped = float(np.clip(raw, lo, hi))
        baselines[ch] = clipped
        sources.append("session")
        clamped.append(_clamp_side(raw, clipped, lo, hi))

    return {
        "recipe": "floor_baseline",
        "baseline": baselines,
        "source": sources,
        "clamped": clamped,
        "n_valid_windows": n_valid,
        "n_dead_windows": n_dead,
        "n_windows": int(rms.shape[0]),
        "q": q,
        "min_windows": min_windows,
    }


def fit_population_floor(
    session_rms: list[np.ndarray],
    q: float = FLOOR_Q,
    min_windows: int = MIN_WINDOWS,
    noise_p1_scale: float = NOISE_P1_SCALE,
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Population noise floor and clamp stats from per-session RMS stacks.

    ``noise_floor = noise_p1_scale * p1`` of pooled quality-kept window RMS.
    Session baselines are the ``q``th percentile of windows above that floor,
    kept only when a channel has at least ``min_windows`` valid windows.
    ``pop_default`` / ``pop_lo`` / ``pop_hi`` are the median and the 5th and
    95th percentiles of those session baselines.
    """
    if not session_rms:
        raise ValueError("no sessions for floor population stats")
    stacked = np.concatenate([np.asarray(s, dtype=np.float64) for s in session_rms], axis=0)
    if stacked.ndim != 2:
        raise ValueError(f"session RMS must be 2-D, got {stacked.shape}")
    n_ch = int(stacked.shape[1])
    names = list(channel_names) if channel_names is not None else [
        f"ch{i}" for i in range(n_ch)
    ]
    p1 = np.percentile(stacked, 1, axis=0)
    noise = np.asarray(noise_p1_scale * p1, dtype=np.float64)

    per_ch: list[list[float]] = [[] for _ in range(n_ch)]
    n_used = 0
    for rms in session_rms:
        block = np.asarray(rms, dtype=np.float64)
        used_here = False
        for ch in range(n_ch):
            valid = block[:, ch][block[:, ch] > noise[ch]]
            if int(valid.size) >= int(min_windows):
                per_ch[ch].append(float(np.percentile(valid, q)))
                used_here = True
        if used_here:
            n_used += 1

    pop_default = np.empty(n_ch, dtype=np.float64)
    pop_lo = np.empty(n_ch, dtype=np.float64)
    pop_hi = np.empty(n_ch, dtype=np.float64)
    n_sessions_ch: list[int] = []
    for ch in range(n_ch):
        arr = np.asarray(per_ch[ch], dtype=np.float64)
        n_sessions_ch.append(int(arr.size))
        if arr.size == 0:
            valid = stacked[:, ch][stacked[:, ch] > noise[ch]]
            fallback = float(np.percentile(valid, q)) if valid.size else float(np.median(stacked[:, ch]))
            pop_default[ch] = pop_lo[ch] = pop_hi[ch] = fallback
        else:
            pop_default[ch] = float(np.median(arr))
            pop_lo[ch] = float(np.percentile(arr, 5))
            pop_hi[ch] = float(np.percentile(arr, 95))
        if pop_hi[ch] < pop_lo[ch]:
            pop_lo[ch], pop_hi[ch] = pop_hi[ch], pop_lo[ch]

    return {
        "version": "2.8.0",
        "recipe": "floor_baseline",
        "q": float(q),
        "min_windows": int(min_windows),
        "noise_p1_scale": float(noise_p1_scale),
        "band_hz": [20.0, 110.0],
        "notch_hz": 60.0,
        "n_sources": int(n_used),
        "n_windows_pooled": int(stacked.shape[0]),
        "rms_channels": names,
        "noise_floor": noise.tolist(),
        "pop_default": pop_default.tolist(),
        "pop_lo": pop_lo.tolist(),
        "pop_hi": pop_hi.tolist(),
        "n_sessions_per_channel": n_sessions_ch,
        "notes": (
            "noise_floor = 0.5 * p1 of quality-kept field window RMS. "
            "pop_* are the median and the 5th/95th percentiles of per-session "
            "q=15 baselines on windows above that floor. Required at inference."
        ),
    }


def calibration_record(result: dict[str, Any], channel_names: list[str] | None = None) -> dict[str, Any]:
    """JSON-safe ``last_calibration`` payload."""
    names = channel_names or RMS_CHANNEL_NAMES
    baselines = np.asarray(result["baseline"], dtype=float)
    channels = []
    for i, name in enumerate(names):
        channels.append(
            {
                "channel": name,
                "baseline": float(baselines[i]),
                "source": result["source"][i],
                "clamped": result["clamped"][i],
                "n_valid_windows": int(result["n_valid_windows"][i]),
                "n_dead_windows": int(result["n_dead_windows"][i]),
            }
        )
    return {
        "recipe": "floor_baseline",
        "n_windows": int(result["n_windows"]),
        "q": result["q"],
        "min_windows": result["min_windows"],
        "channels": channels,
    }


def load_floor_table(path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("noise_floor", "pop_default", "pop_lo", "pop_hi"):
        if key not in data:
            raise ValueError(f"floor_stats.json missing {key}")
    return data
