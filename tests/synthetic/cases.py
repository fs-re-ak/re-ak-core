# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Catalogue of synthetic reference cases.

Each case starts from :func:`generator.base_session` and applies one modifier aimed at a
divergence or decision of the open-core Phase 1 audit (D1–D22). ``targets`` names them so
the divergence report can be read case by case.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .generator import (
    DISCONNECT_UV,
    ELECTRODES,
    EXG_FS,
    SHIMMER_FS,
    Session,
    band_noise,
    base_session,
    build_ppg,
)


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    targets: tuple[str, ...]
    description: str
    modify: Callable[[Session, np.random.Generator], None]


def _noop(session: Session, rng: np.random.Generator) -> None:
    return None


def _exg_rows(session: Session, start_s: float, end_s: float) -> slice:
    return slice(int(start_s * EXG_FS), int(end_s * EXG_FS))


def _disconnects(session: Session, rng: np.random.Generator) -> None:
    # 3 s disconnect burst on CHEEK_R, plus isolated saturated samples on AF7 and EAR_R
    session.exg[_exg_rows(session, 40.0, 43.0), 2] = DISCONNECT_UV
    for t in (60.0, 61.3, 95.7):
        session.exg[int(t * EXG_FS), 1] = -DISCONNECT_UV
        session.exg[int(t * EXG_FS) + 3, 4] = DISCONNECT_UV
    # values just inside / outside the two tolerances in use (0.1 for EEG, 2.875 for fEMG)
    session.exg[int(80.0 * EXG_FS), 0] = DISCONNECT_UV - 0.05
    session.exg[int(80.4 * EXG_FS), 0] = DISCONNECT_UV - 2.5
    session.exg[int(80.8 * EXG_FS), 0] = DISCONNECT_UV - 3.5


def _near_zero(session: Session, rng: np.random.Generator) -> None:
    # Remove the DC offset of CHEEK_L so the raw signal crosses 0 µV many times (D6)
    session.exg[:, 3] -= np.median(session.exg[:, 3])


def _single_invalid_sample(session: Session, rng: np.random.Generator) -> None:
    # One saturated sample inside each of several otherwise clean windows (D6 / D7)
    for t in (20.11, 50.37, 85.02, 110.66):
        session.exg[int(t * EXG_FS), 5] = DISCONNECT_UV


def _dead_channel(session: Session, rng: np.random.Generator) -> None:
    # NOSE saturated for the whole recording: an entirely non-finite channel after marking
    session.exg[:, 7] = DISCONNECT_UV


def _inf_values(session: Session, rng: np.random.Generator) -> None:
    for t, col in ((33.0, 0), (33.004, 0), (70.5, 6), (101.2, 3)):
        session.exg_text_overrides[(int(t * EXG_FS), col)] = "inf"
    session.exg_text_overrides[(int(90.0 * EXG_FS), 2)] = "-inf"


def _exg_time_gaps(session: Session, rng: np.random.Generator) -> None:
    # Drop packets: a 0.4 s gap and a 7 s gap (above the 5 s gap threshold)
    t = session.exg_t
    keep = ~(((t >= 45.0) & (t < 45.4)) | ((t >= 70.0) & (t < 77.0)))
    session.exg_t = t[keep]
    session.exg = session.exg[keep]


def _rms_threshold(session: Session, rng: np.random.Generator) -> None:
    # Two 6 s plateaus with channel RMS sum ≈ 190 µV and ≈ 210 µV, in a band (26–34 Hz) well inside what both
    # releases keep: 2.5.0 filters 15–40 Hz, 2.6.0 filters 20–110 Hz with a 60 Hz notch
    for start, total in ((30.0, 190.0), (90.0, 210.0)):
        rows = _exg_rows(session, start, start + 6.0)
        n = rows.stop - rows.start
        for ch in range(len(ELECTRODES)):
            session.exg[rows, ch] += band_noise(rng, n, EXG_FS, 26.0, 34.0, total / len(ELECTRODES))


def _truncated_last_line(session: Session, rng: np.random.Generator) -> None:
    session.truncate_eeg_last_line = True


def _named_columns(session: Session, rng: np.random.Generator) -> None:
    session.named_eeg_columns = ("AF7", "AF8", "CHEEK_L", "CHEEK_R", "AFz", "EAR_R", "NOSE", "BROW_L")


def _epoch_ms(session: Session, rng: np.random.Generator) -> None:
    session.timestamp_unit = "epoch_ms"


def _epoch_us(session: Session, rng: np.random.Generator) -> None:
    session.timestamp_unit = "epoch_us"


def _relative_ms(session: Session, rng: np.random.Generator) -> None:
    session.timestamp_unit = "relative_ms"


def _legacy_names(session: Session, rng: np.random.Generator) -> None:
    session.legacy_names = True
    session.timestamp_unit = "epoch_ms"


def _ppg_dicrotic(session: Session, rng: np.random.Generator) -> None:
    session.ppg_mv = build_ppg(rng, np.arange(len(session.ppg_mv)) / SHIMMER_FS, dicrotic_ratio=0.75)


def _ppg_missing_beats(session: Session, rng: np.random.Generator) -> None:
    session.ppg_mv = build_ppg(
        rng, np.arange(len(session.ppg_mv)) / SHIMMER_FS, skipped_beats=(40, 41, 80, 120, 121, 122)
    )


def _ppg_artifacts(session: Session, rng: np.random.Generator) -> None:
    ppg = session.ppg_mv
    sh = lambda s: int(s * SHIMMER_FS)  # noqa: E731
    ppg[sh(25.0) : sh(25.0) + 3] += 900.0  # spike (MAD stage)
    ppg[sh(48.0) : sh(50.0)] = 2600.0  # clipping plateau
    ppg[sh(60.0) : sh(72.0)] = np.nan  # mostly-NaN SQI windows (D11)
    ppg[sh(95.0) : sh(110.0)] = 1800.0 + rng.normal(0, 40.0, sh(110.0) - sh(95.0))  # noise, no pulse (SQI)


def _shimmer_gap(session: Session, rng: np.random.Generator) -> None:
    t = session.shimmer_t
    keep = ~((t >= 55.0) & (t < 75.0))
    for name in ("shimmer_t", "gsr_kohm", "ppg_mv"):
        setattr(session, name, getattr(session, name)[keep])
    session.accel = session.accel[keep]


CASES: tuple[Case, ...] = (
    Case("baseline", 1001, ("all",), "Clean session; reference for every step", _noop),
    Case(
        "exg_disconnects",
        1002,
        ("D6", "D7"),
        "Disconnect burst, isolated saturated samples, tolerance edges",
        _disconnects,
    ),
    Case("exg_near_zero", 1003, ("D6",), "CHEEK_L without DC offset: raw samples within 1 µV of 0", _near_zero),
    Case(
        "exg_single_invalid",
        1004,
        ("D6", "D7"),
        "One invalid sample inside otherwise clean windows",
        _single_invalid_sample,
    ),
    Case("exg_dead_channel", 1005, ("D6", "D7"), "NOSE saturated for the whole recording", _dead_channel),
    Case("exg_inf_values", 1006, ("D2", "D6", "D7"), "inf / -inf cells in the EXG file", _inf_values),
    Case("exg_time_gaps", 1007, ("D4", "D7"), "Missing EXG packets: 0.4 s and 7 s gaps", _exg_time_gaps),
    Case("emg_rms_threshold", 1008, ("D8", "D10"), "Plateaus with channel RMS sum ≈ 190 and ≈ 210 µV", _rms_threshold),
    Case("eeg_truncated_last_line", 1009, ("D2",), "Last EXG row cut in the middle", _truncated_last_line),
    Case("eeg_named_columns", 1010, ("D2",), "EXG header with electrode names in a shuffled order", _named_columns),
    Case("timestamps_epoch_ms", 1011, ("D1", "D17"), "All timestamps in epoch milliseconds", _epoch_ms),
    Case("timestamps_epoch_us", 1012, ("D1",), "All timestamps in epoch microseconds", _epoch_us),
    Case("timestamps_relative_ms", 1013, ("D1",), "Timestamps relative to start, in milliseconds", _relative_ms),
    Case(
        "legacy_file_names",
        1014,
        ("D3", "D17"),
        "rawEEG_0.csv / rawShimmer_0.csv / rawEvents.csv, no headers",
        _legacy_names,
    ),
    Case("ppg_dicrotic_notch", 1015, ("D13", "D14"), "Pronounced dicrotic notch", _ppg_dicrotic),
    Case("ppg_missing_beats", 1016, ("D14", "D15"), "Skipped beats", _ppg_missing_beats),
    Case(
        "ppg_artifacts", 1017, ("D11", "D12", "D13"), "Spike, clipping, 12 s of NaN, 15 s without pulse", _ppg_artifacts
    ),
    Case("shimmer_gap", 1018, ("D11", "D15"), "20 s of missing Shimmer rows", _shimmer_gap),
)


def build(case: Case) -> Session:
    session = base_session(case.name, case.seed)
    case.modify(session, np.random.default_rng(case.seed + 50_000))
    return session


def by_name(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)
