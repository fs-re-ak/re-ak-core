# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Emotion models: contract, built-in classical-emotion releases (2.5.0, 2.6.0) and registry."""

from reak_core.models.base import (
    EMOTION_COLUMNS,
    EmotionModel,
    EmotionWindowResult,
    ModelInfo,
    PostprocessSpec,
    PreprocessSpec,
)
from reak_core.models.registry import (
    ALIASES,
    DEFAULT_MODEL,
    Release,
    ReleaseIntegrityError,
    available_models,
    builtin_releases,
    load_model,
    resolve_key,
)

__all__ = [
    "ALIASES",
    "DEFAULT_MODEL",
    "EMOTION_COLUMNS",
    "EmotionModel",
    "EmotionWindowResult",
    "ModelInfo",
    "PostprocessSpec",
    "PreprocessSpec",
    "Release",
    "ReleaseIntegrityError",
    "available_models",
    "builtin_releases",
    "load_model",
    "resolve_key",
]
