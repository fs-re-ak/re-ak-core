# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""fEMG: signal preparation, windowing, emotion inference and post-processing."""

from reak_core.femg.emotions import EmotionRun, compute_emotions, outputs_are_current
from reak_core.femg.signal import Window, filter_exg, interpolate_invalid, prepare, windows

__all__ = [
    "EmotionRun",
    "Window",
    "compute_emotions",
    "filter_exg",
    "interpolate_invalid",
    "outputs_are_current",
    "prepare",
    "windows",
]
