# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Channel fixer (decision D20): repair one bad RMS channel of ``features/emotions/rmsSignals.csv`` from the
other seven channels with a trained linear model.

The models are NucleusKit's channel-fix linear regressions, stored as coefficients in
``reak_core/tools/channel_fix_models/chan<i>_fixer.json`` instead of scikit-learn pickles (they were
saved with scikit-learn 1.5.2; their predictions are identical).

Paths under the session root:

- working file: ``features/emotions/rmsSignals.csv``;
- frozen baseline (written on first use): ``features/emotions/original/rmsSignals.csv``;
- operations log: ``features/emotions/original/operations.log``;
- report: ``features/emotions/chanFixReport/report.html``.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reak_core.constants import CHANNEL_NAMES
from reak_core.tools.channel_fix_report import write_comparison_report
from reak_core.tools.rms_columns import as_output_rms_frame, normalize_rms_dataframe
from reak_core.tools.rms_ops import append_operation, emotions_dir, ensure_baseline_snapshot, working_rms_csv

MODELS_DIR = Path(__file__).parent / "channel_fix_models"


@dataclass(frozen=True)
class ChannelFixModel:
    target_channel: int
    input_channels: tuple[int, ...]
    coef: np.ndarray
    intercept: float

    @classmethod
    def load(cls, target_channel: int, models_dir: str | Path | None = None) -> ChannelFixModel:
        path = Path(models_dir or MODELS_DIR) / f"chan{target_channel}_fixer.json"
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            int(doc["target_channel"]), tuple(doc["input_channels"]), np.asarray(doc["coef"]), float(doc["intercept"])
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.asarray(inputs, dtype=float) @ self.coef + self.intercept


def report_html(session_root: str | Path) -> Path:
    return emotions_dir(session_root) / "chanFixReport" / "report.html"


def fix_session(
    session_root: str | Path,
    target_channel_idx: int,
    *,
    models_dir: str | Path | None = None,
    ignored_channels: Sequence[int] | None = None,
    clip_low: float = 0.0,
    clip_high: float = 30.0,
) -> Path:
    """Repair ``target_channel_idx`` in the working ``rmsSignals.csv``; returns the report path.

    Predictions outside ``[clip_low, clip_high]`` µV become NaN and are interpolated. The report compares
    the working file just before this repair with the result; re-runs operate on the current working file
    (use :func:`reak_core.tools.rms_ops.revert_working_to_original` to start again from the baseline).
    """
    session_root = Path(session_root).resolve()
    work = working_rms_csv(session_root)
    if not work.is_file():
        raise FileNotFoundError(f"Missing RMS file (nothing to repair): {work}")
    n_chan = len(CHANNEL_NAMES)
    if not 0 <= target_channel_idx < n_chan:
        raise IndexError(f"target_channel_idx={target_channel_idx} out of range for {n_chan} channels")
    model = ChannelFixModel.load(target_channel_idx, models_dir)
    inputs = [i for i in range(n_chan) if i != target_channel_idx and i not in set(ignored_channels or ())]
    if tuple(inputs) != model.input_channels:
        raise ValueError(f"The model for channel {target_channel_idx} needs channels {list(model.input_channels)}")

    ensure_baseline_snapshot(session_root)
    before = normalize_rms_dataframe(pd.read_csv(work))
    if before.shape[1] < 2:
        raise ValueError("Expected a Timestamp column and the channel columns")
    report_dir = report_html(session_root).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    for old in report_dir.iterdir():
        if old.is_file():
            old.unlink()

    values = before.values
    data = values[:, 1:].astype(float)
    repaired_channel = model.predict(data[:, inputs])
    repaired_channel[(repaired_channel < clip_low) | (repaired_channel > clip_high)] = np.nan
    data[:, target_channel_idx] = repaired_channel
    repaired = pd.DataFrame(np.column_stack([values[:, 0], data]), columns=list(before.columns)).interpolate()
    as_output_rms_frame(repaired).to_csv(work, index=False)

    html = report_html(session_root)
    write_comparison_report(html, before, repaired, target_channel_idx, list(CHANNEL_NAMES))
    append_operation(session_root, f"channel_fix target={CHANNEL_NAMES[target_channel_idx]}")
    return html
