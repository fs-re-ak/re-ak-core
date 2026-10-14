# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Timestamp unit detection (decision D1: Prime behaviour).

Recording files store their time column with no unit information: Unix epoch in seconds (Vizia),
milliseconds (legacy files), microseconds or nanoseconds, or a relative clock in any of these.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import numpy as np


def normalise_timestamps_to_seconds(timestamps) -> np.ndarray:
    """Return timestamps in seconds from the first sample, detecting the unit.

    Absolute (Unix epoch) clocks are recognised by the magnitude of the first value:
    ~1.7e9 seconds, ~1.7e12 milliseconds, ~1.7e15 microseconds, ~1.7e18 nanoseconds.
    Relative clocks (first value below 1e8) are recognised by the median positive sample
    interval: every signal here runs at 1 Hz or more, so a median step above 1 cannot be
    seconds (milliseconds), above 1e3 is microseconds and above 1e6 is nanoseconds.
    """
    ts = np.asarray(timestamps, dtype=float)
    if ts.size == 0:
        return ts
    first = float(abs(ts[0]))
    if first > 1e17:
        ts = ts / 1e9
    elif first > 1e14:
        ts = ts / 1e6
    elif first > 1e11:
        ts = ts / 1e3
    elif first > 1e8:
        pass
    elif ts.size > 1:
        diffs = np.diff(ts)
        positive = diffs[diffs > 0]
        if positive.size:
            median_dt = float(np.median(positive))
            if median_dt > 1e6:
                ts = ts / 1e9
            elif median_dt > 1e3:
                ts = ts / 1e6
            elif median_dt > 1.0:
                ts = ts / 1e3
    return ts - ts[0]
