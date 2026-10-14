# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Recording metadata written to ``features/metainfo.json``.

The duration is computed from timestamps normalised with decision D1 (decision D25). Both
current systems divided raw timestamps by 1000, assuming milliseconds, and reported a duration
of 0 for Vizia sessions, whose timestamps are in seconds. ``uwbValid`` and ``gpsValid`` come
from the core's stream validation.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from reak_core.io.csvfiles import loadtxt_drop_last_if_incomplete
from reak_core.io.exg import load_exg_matrix
from reak_core.io.positioning import check_gps, check_uwb
from reak_core.io.session import Session
from reak_core.io.timestamps import normalise_timestamps_to_seconds

logger = logging.getLogger(__name__)

METAINFO = os.path.join("features", "metainfo.json")
_SHIMMER_FILES = ("shimmer.csv", "rawShimmer_0.csv", "gsr.tmp", "gsr.csv")


def _duration_s(column: np.ndarray) -> float:
    t = normalise_timestamps_to_seconds(column)
    return float(t[-1] - t[0]) if t.size else 0.0


def _round_half(x: float) -> float:
    return 0.5 * round(x / 0.5)


def recording_info(session: Session) -> dict:
    """Readability flags and duration (seconds, rounded to 0.5 s) of the session's raw streams."""
    shimmer_duration = 0.0
    shimmer_readable = False
    for name in _SHIMMER_FILES:
        path = os.path.join(session.raw_dir, name)
        if os.path.exists(path):
            try:
                shimmer_duration = _duration_s(loadtxt_drop_last_if_incomplete(path)[:, 0])
                shimmer_readable = True
            except (OSError, ValueError, IndexError):
                pass
            break
    eeg_duration = 0.0
    eeg_readable = False
    exg_path = session.modalities()["exg"]
    if exg_path:
        try:
            eeg_duration = _duration_s(load_exg_matrix(exg_path)[:, 0])
            eeg_readable = True
        except (OSError, ValueError, IndexError):
            pass
    return {
        "duration": _round_half(max(shimmer_duration, eeg_duration)),
        "eegReadable": eeg_readable,
        "shimmerReadable": shimmer_readable,
        "uwbValid": check_uwb(session.root).valid,
        "gpsValid": check_gps(session.root).valid,
    }


def write_metainfo(session: Session) -> dict:
    """Write ``features/metainfo.json`` (same keys and string values as NucleusKit) and return the info."""
    info = recording_info(session)
    data = {
        "basepath": session.root + os.sep,
        "duration": str(info["duration"]),
        "eegReadable": str(info["eegReadable"]),
        "shimmerReadable": str(info["shimmerReadable"]),
        "povCameraPresent": "False",
        "uwbValid": str(info["uwbValid"]),
        "gpsValid": str(info["gpsValid"]),
    }
    os.makedirs(session.path("features"), exist_ok=True)
    with open(session.path(METAINFO), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)
    logger.info("Recording duration %.1f s", info["duration"])
    return info
