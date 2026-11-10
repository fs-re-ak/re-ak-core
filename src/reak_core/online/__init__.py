# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Online (causal, streaming) processing of Hermes headset streams for live use.

One instance per headset holds that stream's state (filters, windows, calibration). Feed raw µV blocks
with :meth:`push` as they arrive; each call returns the windows it completed, on the 2 Hz timebase of
the offline outputs. See ``doc/pipeline_online.md``.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from reak_core.online.emotions import OnlineEmotionClassifier, OnlineEmotionResult
from reak_core.online.engagement import OnlineEngagement, OnlineEngagementResult
from reak_core.online.normalizer import PercentileMinMaxNormalizer
from reak_core.online.stream import CausalSosFilter, SampleClock, Windower

__all__ = [
    "CausalSosFilter",
    "OnlineEmotionClassifier",
    "OnlineEmotionResult",
    "OnlineEngagement",
    "OnlineEngagementResult",
    "PercentileMinMaxNormalizer",
    "SampleClock",
    "Windower",
]
