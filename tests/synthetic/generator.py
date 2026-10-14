# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Deterministic synthetic session generator.

Builds sessions in the on-disk layout written by Vizia Mobile (``rawData/eeg.csv``,
``rawData/shimmer.csv``, ``rawData/event.csv``) so that the current NucleusKit and Prime
pipelines, and later ``reak_core``, can be run on identical inputs. No real participant
data is involved: every signal is computed from a seeded random generator.

Signals are plausible, not physiological ground truth. They exist to exercise code paths
(filters, artefact rejection, windowing, heuristics) reproducibly, not to validate the
algorithms' accuracy.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Hardware conventions (match Vizia Mobile and the Hermès / Shimmer devices)
# ---------------------------------------------------------------------------
EXG_FS = 250.0  # Hermès sampling rate (Hz)
EXG_CHANNELS = 8  # ch1..ch8 = AF8, AF7, CHEEK_R, CHEEK_L, EAR_R, AFz, BROW_L, NOSE
EXG_PACKET_SAMPLES = 10  # samples per BLE packet; Vizia stamps a whole packet with its receive time
DISCONNECT_UV = 187500.0  # ADC saturation value written on electrode disconnect
SHIMMER_FS = 51.2  # Shimmer sampling rate (Hz)
START_EPOCH_S = 1791000000.0  # 2026-10-03, arbitrary but fixed

EEG_HEADER = "timestamp,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8"
SHIMMER_HEADER = "timestamp,ax,ay,az,gsrKOhm,ppgMv"
EVENT_HEADER = "timestamp,eventName,data"
ELECTRODES = ("AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE")

# Per-channel DC offsets (µV). Large enough that the raw signal never crosses 0 unless a
# case asks for it (near-zero sample marking, decision D6).
_DC_OFFSETS_UV = np.array([820.0, -650.0, 430.0, -910.0, 310.0, 720.0, -540.0, 610.0])

# Channel weights of a few facial-expression-like EMG patterns (order of ELECTRODES).
_EXPRESSION_PATTERNS = np.array(
    [
        [0.2, 0.2, 1.0, 1.0, 0.1, 0.2, 0.1, 0.6],  # cheeks / nose (smile-like)
        [0.6, 0.6, 0.1, 0.1, 0.1, 0.9, 1.0, 0.3],  # brow / forehead (frown-like)
        [0.3, 0.3, 0.2, 0.2, 0.1, 0.4, 0.3, 1.0],  # nose (disgust-like)
        [0.9, 0.9, 0.3, 0.3, 0.2, 1.0, 0.5, 0.2],  # forehead (surprise-like)
        [0.2, 0.2, 1.0, 0.2, 0.1, 0.1, 0.1, 0.2],  # one cheek (asymmetric)
    ]
)


@dataclass
class Session:
    """In-memory synthetic session before it is written to disk."""

    case: str
    exg_t: np.ndarray  # sample times, seconds from start
    exg: np.ndarray  # (n, 8) µV
    shimmer_t: np.ndarray  # sample times, seconds from start
    gsr_kohm: np.ndarray
    ppg_mv: np.ndarray
    accel: np.ndarray  # (n, 3) in g
    events: list = field(default_factory=list)  # (seconds from start, name, data)
    # Writing options set by case modifiers
    timestamp_unit: str = "epoch_s"  # epoch_s (Vizia), epoch_ms, epoch_us, relative_ms
    legacy_names: bool = False  # rawEEG_0.csv / rawShimmer_0.csv / rawEvents.csv, no header
    named_eeg_columns: tuple | None = None
    truncate_eeg_last_line: bool = False
    exg_text_overrides: dict = field(default_factory=dict)  # {(row, col): "inf"}


# ---------------------------------------------------------------------------
# Signal building blocks
# ---------------------------------------------------------------------------


def band_noise(rng: np.random.Generator, n: int, fs: float, lo: float, hi: float, rms: float) -> np.ndarray:
    """White noise band-limited to [lo, hi] Hz with an FFT mask, scaled to the requested RMS."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    x = np.fft.irfft(spec, n)
    current = np.sqrt(np.mean(x**2))
    return x * (rms / current) if current > 0 else x


def smooth_envelope(n: int, fs: float, starts: list, durations: list, ramp_s: float = 0.3) -> np.ndarray:
    """Sum of trapezoid envelopes (0..1) for the given episodes."""
    env = np.zeros(n)
    t = np.arange(n) / fs
    for start, dur in zip(starts, durations):
        rise = np.clip((t - start) / ramp_s, 0.0, 1.0)
        fall = np.clip((start + dur - t) / ramp_s, 0.0, 1.0)
        env = np.maximum(env, np.minimum(rise, fall))
    return env


def build_exg(rng: np.random.Generator, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Eight-channel Hermès EXG in µV: DC offsets, drift, EEG-like background, line noise, EMG episodes."""
    n = int(round(duration_s * EXG_FS))
    t = np.arange(n) / EXG_FS
    exg = np.empty((n, EXG_CHANNELS))
    for ch in range(EXG_CHANNELS):
        background = (
            band_noise(rng, n, EXG_FS, 0.5, 4.0, 6.0)  # delta
            + band_noise(rng, n, EXG_FS, 4.0, 8.0, 3.0)  # theta
            + band_noise(rng, n, EXG_FS, 13.0, 22.0, 2.0)  # beta
            + band_noise(rng, n, EXG_FS, 1.0, 120.0, 1.5)
        )  # broadband
        alpha_gain = 6.0 if ch == 4 else 2.0  # stronger alpha on EAR_R (T9/T10 source)
        alpha = (
            alpha_gain
            * (1.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
            * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        )
        drift = 20.0 * np.sin(2 * np.pi * 0.02 * t + rng.uniform(0, 2 * np.pi))
        line = 2.0 * np.sin(2 * np.pi * 60.0 * t + rng.uniform(0, 2 * np.pi))
        exg[:, ch] = _DC_OFFSETS_UV[ch] + drift + background + alpha + line

    # Expression-like EMG episodes: every ~6 s, 1.5–3.5 s long, random pattern and intensity.
    starts, durations, patterns, levels = [], [], [], []
    start = 3.0
    while start < duration_s - 4.0:
        starts.append(start)
        durations.append(rng.uniform(1.5, 3.5))
        patterns.append(rng.integers(len(_EXPRESSION_PATTERNS)))
        levels.append(rng.uniform(4.0, 22.0))
        start += rng.uniform(4.5, 8.0)
    for s, d, p, lvl in zip(starts, durations, patterns, levels):
        env = smooth_envelope(n, EXG_FS, [s], [d])
        for ch in range(EXG_CHANNELS):
            weight = _EXPRESSION_PATTERNS[p][ch]
            if weight > 0:
                exg[:, ch] += env * band_noise(rng, n, EXG_FS, 20.0, 120.0, lvl * weight)
    return t, exg


def build_ppg(
    rng: np.random.Generator,
    t: np.ndarray,
    hr_bpm: float = 72.0,
    dicrotic_ratio: float = 0.25,
    skipped_beats: tuple = (),
) -> np.ndarray:
    """PPG in mV: systolic + dicrotic Gaussian pulses on a baseline, with respiratory HRV."""
    duration = t[-1] + 1.0
    beats, b = [], 0.3
    while b < duration:
        beats.append(b)
        rr = (
            60.0 / hr_bpm
            + 0.04 * np.sin(2 * np.pi * 0.25 * b)
            + 0.03 * np.sin(2 * np.pi * 0.1 * b)
            + rng.normal(0, 0.01)
        )
        b += rr
    beats = np.asarray(beats)
    keep = np.ones(len(beats), dtype=bool)
    for idx in skipped_beats:
        if idx < len(beats):
            keep[idx] = False
    pulse = np.zeros_like(t)
    for bt in beats[keep]:
        pulse += np.exp(-0.5 * ((t - bt) / 0.08) ** 2)
        pulse += dicrotic_ratio * np.exp(-0.5 * ((t - bt - 0.30) / 0.07) ** 2)
    baseline = 1800.0 + 25.0 * np.sin(2 * np.pi * 0.03 * t)
    return baseline + 120.0 * pulse + rng.normal(0, 1.5, size=t.shape)


def build_eda_kohm(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """Skin resistance in kΩ from a tonic level plus Bateman-shaped SCRs (in µS)."""
    tonic = 4.0 + 0.6 * np.sin(2 * np.pi * t / 240.0)
    phasic = np.zeros_like(t)
    onset = 5.0
    while onset < t[-1]:
        amp = rng.uniform(0.15, 0.8)
        dt = np.clip(t - onset, 0.0, None)
        shape = np.exp(-dt / 2.0) - np.exp(-dt / 0.75)
        phasic += amp * shape / 0.42 * (t >= onset)
        onset += rng.uniform(8.0, 22.0)
    micro_s = tonic + phasic + rng.normal(0, 0.004, size=t.shape)
    return 1000.0 / micro_s


def packet_timestamps(rng: np.random.Generator, t: np.ndarray, packet: int) -> np.ndarray:
    """Vizia stamps every sample of a BLE packet with the packet's receive time (s from start)."""
    out = np.empty_like(t)
    for i in range(0, len(t), packet):
        j = min(i + packet, len(t))
        out[i:j] = t[j - 1] + rng.uniform(0.005, 0.025)
    return np.maximum.accumulate(out)


# ---------------------------------------------------------------------------
# Session assembly and writing
# ---------------------------------------------------------------------------


def base_session(case: str, seed: int, duration_s: float = 150.0) -> Session:
    """Clean session: EXG with expression episodes, Shimmer PPG / EDA, Vizia bootstrap events."""
    rng = np.random.default_rng(seed)
    exg_t, exg = build_exg(rng, duration_s)
    exg_t = packet_timestamps(rng, exg_t, EXG_PACKET_SAMPLES)

    n_sh = int(duration_s * SHIMMER_FS)
    sh_t = np.arange(n_sh) / SHIMMER_FS
    ppg = build_ppg(rng, sh_t)
    gsr = build_eda_kohm(rng, sh_t)
    accel = np.column_stack([rng.normal(0, 0.01, n_sh), rng.normal(0, 0.01, n_sh), 1.0 + rng.normal(0, 0.01, n_sh)])
    sh_t = sh_t + rng.uniform(0.0, 0.004, size=n_sh)

    events = [
        (
            0.0,
            "RECORDING_ONSET",
            json.dumps({"epochMs": int(START_EPOCH_S * 1000), "sessionId": f"synthetic-{case}"}, separators=(",", ":")),
        ),
        (
            0.0,
            "RECIPE_CONFIG",
            json.dumps({"model": "standard-latest", "process": "hermes", "update": None}, separators=(",", ":")),
        ),
        (30.0, "Button pressed", "[]"),
        (75.0, "Button pressed", "[]"),
        (120.0, "Button pressed", "[]"),
    ]
    return Session(
        case=case, exg_t=exg_t, exg=exg, shimmer_t=sh_t, gsr_kohm=gsr, ppg_mv=ppg, accel=accel, events=events
    )


def _format_time(rel_s: np.ndarray, unit: str) -> list[str]:
    if unit == "epoch_s":
        return [f"{START_EPOCH_S + x:.3f}" for x in rel_s]
    if unit == "epoch_ms":
        return [f"{round((START_EPOCH_S + x) * 1e3):d}" for x in rel_s]
    if unit == "epoch_us":
        return [f"{round((START_EPOCH_S + x) * 1e6):d}" for x in rel_s]
    if unit == "relative_ms":
        return [f"{x * 1e3:.3f}" for x in rel_s]
    raise ValueError(f"unknown timestamp unit {unit}")


def _fmt(v: float) -> str:
    if np.isnan(v):
        return "NaN"
    return f"{v:.4f}"


def write_session(session: Session, root: str) -> dict:
    """Write the session under ``root/rawData`` and return {relative path: sha256}."""
    import hashlib

    raw = os.path.join(root, "rawData")
    os.makedirs(raw, exist_ok=True)
    legacy = session.legacy_names
    written = {}

    def _write(name: str, text: str) -> None:
        path = os.path.join(raw, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        written[f"rawData/{name}"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # EXG
    times = _format_time(session.exg_t, session.timestamp_unit)
    columns = list(range(EXG_CHANNELS))
    header = EEG_HEADER
    if session.named_eeg_columns:
        columns = [ELECTRODES.index(name) for name in session.named_eeg_columns]
        header = "timestamp," + ",".join(session.named_eeg_columns)
    lines = [] if legacy else [header]
    for i, ts in enumerate(times):
        cells = []
        for col in columns:
            cells.append(session.exg_text_overrides.get((i, col), _fmt(session.exg[i, col])))
        lines.append(ts + "," + ",".join(cells))
    text = "\n".join(lines) + "\n"
    if session.truncate_eeg_last_line:
        text = text[: text.rstrip("\n").rfind(",") - 3]  # cut the last row in the middle of a value
    _write("rawEEG_0.csv" if legacy else "eeg.csv", text)

    # Shimmer
    times = _format_time(session.shimmer_t, session.timestamp_unit)
    lines = [] if legacy else [SHIMMER_HEADER]
    for i, ts in enumerate(times):
        ax, ay, az = session.accel[i]
        lines.append(f"{ts},{ax:.4f},{ay:.4f},{az:.4f},{_fmt(session.gsr_kohm[i])},{_fmt(session.ppg_mv[i])}")
    _write("rawShimmer_0.csv" if legacy else "shimmer.csv", "\n".join(lines) + "\n")

    # Events (Vizia writes the JSON payload unquoted, after the second comma)
    times = _format_time(np.array([e[0] for e in session.events]), session.timestamp_unit)
    lines = [] if legacy else [EVENT_HEADER]
    for ts, (_, name, data) in zip(times, session.events):
        lines.append(f"{ts},{name},{data}")
    _write("rawEvents.csv" if legacy else "event.csv", "\n".join(lines) + "\n")
    return written
