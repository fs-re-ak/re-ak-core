# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
What the core must reproduce, output by output (open-core decisions D1–D25).

``EXPECTATIONS`` maps a session output to ``(rule, decisions, models)``:

- rule ``"nk"`` / ``"prime"``: the core must match that system's reference
  (``tests/references/<system>/<model>/<case>/``) within the reference tolerance, for each model of
  ``models``, except for the cases listed in ``EXCEPTIONS``.
- rule ``"new"``: the core follows a new rule. The output is not compared; its difference with both
  systems is measured and recorded in the CHANGELOG.

``EXCEPTIONS`` maps an output to ``{case: decisions}``: cases where a decision changes the result of an
otherwise matching output.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

PRIMARY = ("classical-emotion-2.6.0",)
BOTH = ("classical-emotion-2.6.0", "classical-emotion-2.5.0")

EXPECTATIONS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "features/events/playback_annotations.json": ("nk", (), PRIMARY),
    "features/events/processedFeatureEvents.csv": ("prime", ("D3", "D16"), PRIMARY),
    "features/metainfo.json": ("new", ("D25",), PRIMARY),
    # D26: nearest-neighbour resampling for one-hot labels removes clock-drift triangles.
    # References regenerated from the core (Prime and NK still use linear interpolation there).
    "results/Emotions.csv": ("prime", ("D8", "D9", "D26"), BOTH),
    "results/emotion_model.json": ("prime", ("D10",), BOTH),
    "features/emotions/avgRms.csv": ("prime", ("D8", "D10"), BOTH),
    "features/emotions/rmsSignals.csv": ("nk", ("D10",), BOTH),
    "features/emotions/emotionClassifierInputs.csv": ("new", ("D10",), BOTH),
    "features/emotions/relabeled-samples.csv": ("new", ("D10",), BOTH),
    "results/Cognition.csv": ("nk", ("D4", "D5", "D23"), PRIMARY),
    "features/cognition/powerBands.csv": ("nk", ("D4",), PRIMARY),
    "features/cognition/artefactStats.csv": ("nk", (), PRIMARY),
    "features/cognition/epochMetrics.csv": ("nk", (), PRIMARY),
    "results/Arousal.csv": ("nk", (), PRIMARY),
    "features/eda/SCR_events.csv": ("nk", (), PRIMARY),
    "results/HeartDynamics.csv": ("nk", ("D11", "D12", "D13", "D14", "D15"), PRIMARY),
    "features/ppg/ppg_resampled.csv": ("nk", ("D11", "D12"), PRIMARY),
}

# Invalid-sample marking and windowing (D6, D7, D24) change which windows are classified.
_INVALID_SAMPLES = {
    "exg_disconnects": ("D6",),  # 187 500 µV is no longer marked
    "exg_near_zero": ("D6",),  # no near-zero rule
    "exg_single_invalid": ("D6", "D7"),  # one invalid sample voids the window
    "exg_dead_channel": ("D6",),  # the 187 500 µV channel is no longer marked
    "exg_inf_values": ("D7", "D24"),
    "exg_full_scale": ("D6",),  # full-scale samples are marked by the core only
}
# NucleusKit misreads µs and relative-ms clocks (D1); Prime reads named columns by position (D2).
_NK_TIME_AXIS = {"timestamps_epoch_us": ("D1",), "timestamps_relative_ms": ("D1",)}
_PRIME_NAMED_COLUMNS = {"eeg_named_columns": ("D2",)}

# Cognition is computed on samples marked with D6 / D24 (the old 187 500 µV value and inf now behave differently).
_COGNITION_INVALID = {
    "exg_disconnects": ("D6",),
    "exg_single_invalid": ("D6",),
    "exg_dead_channel": ("D6",),  # the saturated NOSE channel now enters the NLMS decorrelation
    "exg_inf_values": ("D24",),
    "exg_full_scale": ("D6",),
}

EXCEPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "results/Emotions.csv": {**_INVALID_SAMPLES, **_PRIME_NAMED_COLUMNS},
    "features/emotions/avgRms.csv": {**_INVALID_SAMPLES, **_PRIME_NAMED_COLUMNS},
    "results/Cognition.csv": {**_COGNITION_INVALID, **_NK_TIME_AXIS},
    "features/cognition/powerBands.csv": {**_COGNITION_INVALID, **_NK_TIME_AXIS},
    "features/cognition/epochMetrics.csv": {**_COGNITION_INVALID, **_NK_TIME_AXIS},
    "features/cognition/artefactStats.csv": {"exg_disconnects": ("D6",), "exg_inf_values": ("D24",)},
    "results/HeartDynamics.csv": dict(_NK_TIME_AXIS),
    "features/ppg/ppg_resampled.csv": dict(_NK_TIME_AXIS),
    # NucleusKit has no near-zero rule either, so that case matches.
    "features/emotions/rmsSignals.csv": {
        **{k: v for k, v in _INVALID_SAMPLES.items() if k != "exg_near_zero"},
        **_NK_TIME_AXIS,
    },
}

# Cases where the core must match another system than the designated one: arousal is the same code in
# both systems, and only Prime reads µs and relative-ms clocks correctly (D1).
ALTERNATES: dict[str, dict[str, str]] = {
    "results/Arousal.csv": {"timestamps_epoch_us": "prime", "timestamps_relative_ms": "prime"},
    "features/eda/SCR_events.csv": {"timestamps_epoch_us": "prime", "timestamps_relative_ms": "prime"},
}
