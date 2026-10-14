# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Emotion model contract.

Every emotion model, built in or provided by an extension through the ``reak_core.models`` entry
point, implements :class:`EmotionModel`. The fEMG pipeline depends only on this interface.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMOTION_COLUMNS: list[str] = ["Neutral", "Happiness", "Anger", "Surprise", "Contempt", "Disgust", "Fear", "Sadness"]
"""Emotion column order of ``results/Emotions.csv`` (downstream consumers depend on it)."""


@dataclass(frozen=True)
class PreprocessSpec:
    """Filters a release expects on the whole recording before windowing.

    ``used_fallback`` is True when the release does not declare ``bandpass_hz`` (2.5.0 contract:
    15–40 Hz, no notch).
    """

    bandpass_low: float = 15.0
    bandpass_high: float = 40.0
    notch_hz: float | None = None
    used_fallback: bool = False


@dataclass(frozen=True)
class PostprocessSpec:
    """Post-classifier rules of a release (decision D8); overridable per pipeline run."""

    one_hot: bool = False
    high_rms_enabled: bool = True
    high_rms_threshold_uv: float = 200.0
    high_rms_max_gap_windows: int = 6
    low_confidence_enabled: bool = False
    low_confidence_threshold: float = 0.3

    def with_overrides(self, **overrides) -> PostprocessSpec:
        values = {**self.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        return PostprocessSpec(**values)


@dataclass(frozen=True)
class EmotionWindowResult:
    """Output of one classified window."""

    probabilities: dict[str, float]
    """Probabilities keyed by EMOTION_COLUMNS labels."""
    label: str
    """Emitted label (title case, one of EMOTION_COLUMNS)."""
    confidence: float
    """Probability of the emitted label."""
    channel_rms: np.ndarray
    """Raw per-channel RMS of the filtered window (8 values, hardware order)."""
    model_features: np.ndarray
    """Feature vector passed to the classifier."""


@dataclass(frozen=True)
class ModelInfo:
    """Identity of a model, written to ``results/emotion_model.json``."""

    key: str
    name: str
    version: str
    feature_set: str = ""
    kind: str = "classical"
    release_dir: Path | None = None
    extra: dict = field(default_factory=dict)


class EmotionModel(ABC):
    """Stateful emotion classifier working on filtered windows of shape (8, 500)."""

    info: ModelInfo
    preprocess: PreprocessSpec
    postprocess: PostprocessSpec
    feature_columns: list[str]

    @abstractmethod
    def reset(self) -> None:
        """Forget the calibration of the previous recording."""

    def set_subject_context(self, windows: np.ndarray) -> None:  # noqa: B027 — optional hook
        """Calibrate per-subject statistics on all valid windows ``(n_windows, 8, 500)`` of a recording."""

    @abstractmethod
    def predict(self, samples: np.ndarray) -> EmotionWindowResult:
        """Classify one filtered window of shape (8, 500)."""
