# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Resampling onto regular time grids.

- :func:`resample_to_grid` projects irregular Shimmer samples onto their nominal 51.2 Hz grid,
  keeping long disconnects as NaN.
- :func:`resample_to_timebase` brings a result table onto the shared 2 Hz session timebase
  without interpolating through NaN gaps.
- :func:`snap_to_timebase` is the nearest-neighbour counterpart: each output row copies the
  nearest source row without blending.  Use this for one-hot / label tables (e.g. Emotions.csv)
  where linear blending would introduce artificial fractional values.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import logging
import os
import textwrap

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from reak_core.constants import OUTPUT_STEP_S, SHIMMER_SAMPLING_RATE

logger = logging.getLogger(__name__)

GAP_THRESHOLD_S = 5.0
"""Gaps at or above this duration are true disconnects and stay NaN."""


def resample_to_grid(
    timestamps,
    values,
    signal_name: str,
    out_dir: str,
    nominal_fs: float = SHIMMER_SAMPLING_RATE,
    gap_threshold_s: float = GAP_THRESHOLD_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project an irregularly sampled signal onto a strict ``nominal_fs`` grid.

    Writes ``{signal_name}_resampled.csv`` and ``{signal_name}_resample_report.txt`` to
    ``out_dir``. Returns ``(t_grid, resampled, gap_mask)``: grid in seconds, linearly
    interpolated values (NaN inside gaps of ``gap_threshold_s`` or more), and the gap mask.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    values = np.asarray(values, dtype=float)
    n_raw = len(timestamps)
    t_first, t_last = timestamps[0], timestamps[-1]
    duration = t_last - t_first
    effective_fs = (n_raw - 1) / duration if duration > 0 else 0.0

    dt = np.diff(timestamps)
    gaps = [(timestamps[i], timestamps[i + 1], dt[i]) for i in np.where(dt >= gap_threshold_s)[0]]
    if gaps:
        logger.warning(
            "%s: %d gap(s) >= %.0f s; those regions will be NaN", signal_name.upper(), len(gaps), gap_threshold_s
        )

    t_grid = np.arange(t_first, t_last, 1.0 / nominal_fs)
    interpolator = interp1d(timestamps, values, kind="linear", bounds_error=False, fill_value=np.nan)
    resampled = interpolator(t_grid)
    gap_mask = np.zeros(len(t_grid), dtype=bool)
    for t_start, t_end, _ in gaps:
        gap_mask |= (t_grid >= t_start) & (t_grid <= t_end)
    resampled[gap_mask] = np.nan
    n_nan = int(gap_mask.sum())
    n_added = max(0, len(t_grid) - n_raw - n_nan)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"Timestamp": t_grid, "Value": resampled}).to_csv(
        os.path.join(out_dir, f"{signal_name}_resampled.csv"), index=False
    )
    label = signal_name.upper()
    gap_lines = (
        "".join(
            f"  Gap {k:>2d}:  t = {t_s:8.2f} s  →  {t_e:8.2f} s   duration = {dur:.2f} s\n"
            for k, (t_s, t_e, dur) in enumerate(gaps, start=1)
        )
        or "  (none)\n"
    )
    report = (
        textwrap.dedent(f"""\
        Shimmer {label} Resample Report
        {"=" * (len(label) + 24)}
        Raw samples             : {n_raw}
        Duration                : {duration:.2f} s
        Effective sample rate   : {effective_fs:.2f} Hz  (nominal: {nominal_fs} Hz)
        Grid samples at {nominal_fs} Hz : {len(t_grid)}
        NaN samples (large gaps): {n_nan}  ({n_nan / nominal_fs:.2f} s)
        Added by interpolation  : {n_added}  (grid - raw - NaN)

        Large gaps detected (>= {gap_threshold_s:.1f} s)
    """)
        + gap_lines
    )
    with open(os.path.join(out_dir, f"{signal_name}_resample_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    logger.info(
        "%s: %d raw -> %d grid samples (%d added, %d NaN), effective %.2f Hz",
        label,
        n_raw,
        len(t_grid),
        n_added,
        n_nan,
        effective_fs,
    )
    return t_grid, resampled, gap_mask


def snap_to_timebase(df: pd.DataFrame, target_interval: float = OUTPUT_STEP_S) -> pd.DataFrame:
    """Resample a table with a ``Timestamp`` column onto ``0, step, 2*step, …`` using nearest-neighbour.

    Each output row takes the value of whichever source row has the closest timestamp.  No values
    are blended.  An output point whose nearest source point is more than one output step away
    is considered outside the covered range and stays NaN — this preserves true gaps (e.g. signal
    dropouts) without filling them.

    Use this for one-hot / label tables (e.g. Emotions.csv) where linear interpolation would
    introduce fractional values that drift with the headset clock, creating spurious low-frequency
    oscillations in the output.
    """
    if "Timestamp" not in df.columns:
        return df
    src_ts = df["Timestamp"].to_numpy(dtype=float)
    new_ts = np.arange(0, src_ts.max() + target_interval, target_interval)
    resampled: dict[str, np.ndarray] = {"Timestamp": new_ts}

    # For each output point, find the index of the nearest source row.
    # np.searchsorted gives the insertion point; we check both neighbours.
    right_idx = np.searchsorted(src_ts, new_ts, side="left")  # first src >= grid point
    right_idx = np.clip(right_idx, 0, len(src_ts) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(src_ts) - 1)
    dist_right = np.abs(src_ts[right_idx] - new_ts)
    dist_left = np.abs(src_ts[left_idx] - new_ts)
    nearest = np.where(dist_left <= dist_right, left_idx, right_idx)
    nearest_dist = np.minimum(dist_left, dist_right)

    # A grid point with no source row within one step is outside coverage → NaN.
    out_of_range = nearest_dist > target_interval

    for col in df.columns:
        if col == "Timestamp":
            continue
        y = df[col].to_numpy(dtype=float)
        values = y[nearest].copy()
        values[out_of_range] = np.nan
        # Also propagate NaN from the source row itself.
        values[np.isnan(y[nearest])] = np.nan
        resampled[col] = values
    return pd.DataFrame(resampled)


def resample_to_timebase(df: pd.DataFrame, target_interval: float = OUTPUT_STEP_S) -> pd.DataFrame:
    """Resample a table with a ``Timestamp`` column (seconds) onto ``0, step, 2*step, …``.

    Linear interpolation between valid source points only. An output point whose two enclosing
    valid source points are more than two output steps apart lies in a NaN gap and stays NaN.
    """
    if "Timestamp" not in df.columns:
        return df
    src_ts = df["Timestamp"].to_numpy(dtype=float)
    new_ts = np.arange(0, src_ts.max() + target_interval, target_interval)
    resampled = {"Timestamp": new_ts}
    for col in df.columns:
        if col == "Timestamp":
            continue
        y = df[col].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        if not valid.any():
            resampled[col] = np.full(len(new_ts), np.nan)
            continue
        values = np.interp(new_ts, src_ts[valid], y[valid], left=np.nan, right=np.nan)
        valid_ts = src_ts[valid]
        left = np.searchsorted(valid_ts, new_ts, side="right") - 1
        right = left + 1
        bracketed = (left >= 0) & (right < len(valid_ts))
        gap = valid_ts[np.clip(right, 0, len(valid_ts) - 1)] - valid_ts[np.clip(left, 0, len(valid_ts) - 1)]
        values[bracketed & (gap > target_interval * 2)] = np.nan
        resampled[col] = values
    return pd.DataFrame(resampled)
