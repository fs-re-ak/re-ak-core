# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
UWB and GPS raw files: loading and validation only.

The core reads ``rawData/uwb.csv`` and ``rawData/gps.csv`` to know their timing and quality.
Multilateration, zones and GPS resampling stay in the wrappers (open-core scope decision).

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from reak_core.io.timestamps import normalise_timestamps_to_seconds

logger = logging.getLogger(__name__)

UWB_FILENAMES: tuple[str, ...] = ("uwb.csv", "uwb_0.csv", "uwb.tmp")
GPS_FILENAMES: tuple[str, ...] = ("gps.csv", "gps.tmp")


@dataclass
class StreamCheck:
    """Validation summary of an auxiliary stream (no signal values)."""

    present: bool
    path: str | None = None
    rows: int = 0
    duration_s: float = 0.0
    non_monotonic_steps: int = 0
    gaps_over_5s: int = 0
    valid: bool = False
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "path"}


def _timestamps(path: str) -> np.ndarray:
    """First column of every row that starts with a number (header and log lines are skipped)."""
    values = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            first = line.split(",", 1)[0].strip()
            try:
                values.append(float(first))
            except ValueError:
                continue
    return np.asarray(values, dtype=float)


def check_stream(session_root: str, filenames: tuple[str, ...]) -> StreamCheck:
    for name in filenames:
        path = os.path.join(session_root, "rawData", name)
        if os.path.isfile(path):
            break
    else:
        return StreamCheck(present=False)
    check = StreamCheck(present=True, path=path)
    ts = _timestamps(path)
    check.rows = int(ts.size)
    if ts.size < 2:
        check.problems.append("fewer than 2 timestamped rows")
        return check
    t = normalise_timestamps_to_seconds(ts)
    steps = np.diff(t)
    check.duration_s = float(t[-1] - t[0])
    check.non_monotonic_steps = int(np.sum(steps < 0))
    check.gaps_over_5s = int(np.sum(steps >= 5.0))
    if check.non_monotonic_steps:
        check.problems.append("timestamps go backwards")
    if check.duration_s <= 0:
        check.problems.append("zero duration")
    check.valid = not check.problems
    return check


def check_uwb(session_root: str) -> StreamCheck:
    return check_stream(session_root, UWB_FILENAMES)


def check_gps(session_root: str) -> StreamCheck:
    return check_stream(session_root, GPS_FILENAMES)
