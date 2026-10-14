# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Hardware constants of the Hermès headset and the Shimmer wristband.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hermès headset (EEG / fEMG)
# ---------------------------------------------------------------------------
EXG_SAMPLING_RATE: int = 250
"""EXG acquisition rate in Hz."""

EXG_GAIN: int = 12
EXG_VREF_V: float = 4.5
EXG_LSB_UV: float = 2 * EXG_VREF_V * 1e6 / (EXG_GAIN * 2**24)
"""One ADC step in µV (24-bit ADC)."""

FULL_SCALE_UV: float = 375000.0
"""ADC full scale in µV at gain 12 (positive maximum 374 999.955, negative −375 000)."""

FULL_SCALE_TOLERANCE_UV: float = 0.1
"""A raw sample whose absolute value is within this distance of FULL_SCALE_UV is invalid (D6)."""

CHANNEL_NAMES: tuple[str, ...] = ("AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE")
"""Canonical electrode names in hardware order (ch1..ch8)."""

OUTPUT_RMS_COLUMNS: tuple[str, ...] = (
    "HEAD_R_RMS",
    "HEAD_L_RMS",
    "CHEEK_R_RMS",
    "CHEEK_L_RMS",
    "EAR_R_RMS",
    "FOREHEAD_L_RMS",
    "BROW_RMS",
    "NOSE_RMS",
)
"""Per-channel RMS column names written to the emotion feature files (D10), same order as CHANNEL_NAMES."""

N_CHANNELS: int = len(CHANNEL_NAMES)

EEG_SOURCE_CHANNELS: tuple[int, ...] = (0, 1, 4)
"""Hardware indexes the EEG derivation uses (ch1 AF8, ch2 AF7, ch5 EAR_R); cognition judges validity on them (D23)."""

EXG_FILENAMES: tuple[str, ...] = ("rawEEG_0.csv", "eeg.tmp", "eeg.csv", "eegRec_0.csv")
"""Raw EXG file names, in discovery order."""

_RAW_COLUMN_INDEX: dict[str, int] = {}
for _i, _electrode in enumerate(CHANNEL_NAMES):
    _RAW_COLUMN_INDEX[_electrode.lower()] = _i
    _RAW_COLUMN_INDEX[f"ch{_i + 1}"] = _i
    _RAW_COLUMN_INDEX[OUTPUT_RMS_COLUMNS[_i].lower()] = _i
_RAW_COLUMN_INDEX.update({"head_r": 0, "head_l": 1, "forehead_l": 5, "brow": 6})


def hardware_channel_index(column_name: str) -> int | None:
    """Map a raw or legacy column header to the hardware index 0..7.

    Accepts ``ch1``..``ch8``, electrode names (``AF8`` …), legacy labels (``HEAD_R`` …) and the
    output RMS names (``HEAD_R_RMS`` …). Returns ``None`` for anything else.
    """
    return _RAW_COLUMN_INDEX.get(str(column_name).strip().lower())


# ---------------------------------------------------------------------------
# Shimmer wristband (EDA / PPG)
# ---------------------------------------------------------------------------
SHIMMER_SAMPLING_RATE: float = 51.2
"""Nominal Shimmer sampling rate in Hz."""

SHIMMER_TIMESTAMP_COLUMN: int = 0

EDA_FILES: tuple[tuple[str, int], ...] = (
    ("shimmer.csv", 4),
    ("rawShimmer_0.csv", 4),
    ("gsr.tmp", 1),
    ("gsr.csv", 1),
)
"""Candidate files and the column holding the GSR value (kΩ for the multi-channel exports)."""

PPG_FILES: tuple[tuple[str, int], ...] = (
    ("shimmer.csv", 5),
    ("rawShimmer_0.csv", 5),
    ("ppg.tmp", 1),
    ("ppg.csv", 1),
)
"""Candidate files and the column holding the PPG value."""

# ---------------------------------------------------------------------------
# Session timebase
# ---------------------------------------------------------------------------
OUTPUT_STEP_S: float = 0.5
"""Shared 2 Hz timebase of every results/*.csv table."""
