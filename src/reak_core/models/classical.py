# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Classical-emotion releases (2.5.0, 2.6.0) as :class:`EmotionModel`.

Each release ships its own ``exg_emotion`` runtime, and its ``pipeline.joblib`` references that
package by name. Loading a release therefore puts its folder first on ``sys.path`` and evicts any
``exg_emotion`` module loaded from another release, exactly as NucleusKit and Prime do, so the
release code that runs is the code the model was exported with.

The model performs per-subject normalisation: :meth:`set_subject_context` must be called with all
valid windows of a recording before the first :meth:`predict`, and :meth:`reset` between recordings.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from reak_core.constants import CHANNEL_NAMES
from reak_core.models.base import EMOTION_COLUMNS, EmotionModel, EmotionWindowResult, ModelInfo
from reak_core.models.registry import Release

_LABEL_MAP: dict[str, str] = {col.upper(): col for col in EMOTION_COLUMNS}
_TEMPORAL_EEG_IDX = 4  # EAR_R carries the temporal EEG and is excluded from WL and MDF
_EMG_CHANNELS = tuple(c for i, c in enumerate(CHANNEL_NAMES) if i != _TEMPORAL_EEG_IDX)

FEATURE_COLUMNS: list[str] = (
    [f"rms_{c}" for c in CHANNEL_NAMES]
    + [f"wl_{c}" for c in _EMG_CHANNELS]
    + [f"mdf_{c}" for c in _EMG_CHANNELS]
    + ["zyg_asymmetry", "ant_corr", "nm_ratio_a"]
)
"""Names of the 25 ``extended_nm_a`` features written to ``emotionClassifierInputs.csv``."""


def _is_release_path(path: str) -> bool:
    try:
        folder = Path(path)
    except (OSError, TypeError, ValueError):
        return False
    return (folder / "exg_emotion").is_dir() and (folder / "manifest.json").is_file()


def _activate(release_dir: Path) -> None:
    """Make ``release_dir``'s ``exg_emotion`` the one Python imports."""
    for name in list(sys.modules):
        if name == "exg_emotion" or name.startswith("exg_emotion."):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not _is_release_path(p)]
    sys.path.insert(0, str(release_dir.resolve()))


class ClassicalReleaseModel(EmotionModel):
    """A classical-emotion release loaded from its folder."""

    def __init__(self, predictor, release: Release) -> None:  # noqa: ANN001
        self._predictor = predictor
        self.release = release
        self.info = ModelInfo(
            key=release.key,
            name=release.name,
            version=release.version,
            feature_set=str(release.config().get("feature_set", "")),
            kind="classical",
            release_dir=release.path,
        )
        self.preprocess = release.preprocess()
        self.postprocess = release.postprocess()
        self.feature_columns = list(FEATURE_COLUMNS)

    @classmethod
    def load(cls, release: Release) -> ClassicalReleaseModel:
        _activate(release.path)
        from exg_emotion.runtime.predictor import Predictor  # type: ignore[import-not-found]

        return cls(Predictor.from_release_dir(release.path), release)

    def reset(self) -> None:
        inner = self._predictor._model
        inner._subject_mean = None
        inner._subject_std = None
        inner._subject_wl_mean = None
        inner._subject_wl_std = None

    def set_subject_context(self, windows: np.ndarray) -> None:
        self._predictor.set_subject_context(windows)

    def predict(self, samples: np.ndarray) -> EmotionWindowResult:
        prediction = self._predictor.predict(samples)
        probabilities = {_LABEL_MAP.get(k, k.title()): float(v) for k, v in prediction.probabilities.items()}
        label = _LABEL_MAP.get(prediction.label, prediction.label.title())
        channel_rms = np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2, axis=1))
        try:
            features = np.asarray(self._predictor._model._window_features(samples)[0], dtype=np.float64)
        except Exception:  # noqa: BLE001 — features are diagnostic only
            features = np.full(len(FEATURE_COLUMNS), np.nan)
        return EmotionWindowResult(
            probabilities=probabilities,
            label=label,
            confidence=float(probabilities.get(label, 0.0)),
            channel_rms=channel_rms,
            model_features=features,
        )
