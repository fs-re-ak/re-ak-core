# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Feature-set descriptor of an emotion model.

A feature set is identified by how its vectors are computed, not by its name: the same
``feature_set`` name has meant different values across releases (filters, amplitude calibration,
feature code). The descriptor gathers everything that defines the values of a vector, and its
``signature`` is the SHA-256 of that content. Two models with the same signature produce
interchangeable vectors, so stored features can be re-scored by any model sharing it.

Written next to ``emotionClassifierInputs.csv`` as ``features/emotions/feature_set.json`` and
uploaded with the vectors by Prime.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reak_core import __version__
from reak_core.constants import EXG_SAMPLING_RATE
from reak_core.femg.signal import STEP_SAMPLES, WINDOW_SAMPLES
from reak_core.models.base import EmotionModel

MODALITY = "femg"

_FEATURE_CODE_PREFIXES = ("exg_emotion/models/", "exg_emotion/analysis/")
"""Release files whose code defines the feature values (feature functions and calibrations)."""


def _code_checksums(release_dir: Path | None) -> dict[str, str] | None:
    """Checksums of the release's feature code, from its manifest; ``None`` when unknown."""
    if release_dir is None:
        return None
    manifest = Path(release_dir) / "manifest.json"
    try:
        checksums = json.loads(manifest.read_text(encoding="utf-8"))["reak_core"]["checksums"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    code = {k: v for k, v in sorted(checksums.items()) if k.startswith(_FEATURE_CODE_PREFIXES)}
    return code or None


def _release_config(release_dir: Path | None) -> dict:
    if release_dir is None:
        return {}
    try:
        return json.loads((Path(release_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def feature_set_descriptor(model: EmotionModel) -> dict:
    """Describe the feature vectors ``model`` computes.

    Returns ``{signature, modality, name, columns, dimension, window_ms, hop_ms, extraction,
    producer}``. ``signature`` covers ``modality``, ``columns``, ``window_ms``, ``hop_ms`` and
    ``extraction``; ``name`` is a display label and ``producer`` (re-ak-core version, model key)
    is provenance, neither changes the values. When the feature code of a model is unknown (no
    release manifest), the model key is part of the extraction, so its vectors are never mixed
    with another model's.
    """
    info = model.info
    config = _release_config(info.release_dir)
    pre = model.preprocess
    extraction: dict = {
        "sample_rate_hz": EXG_SAMPLING_RATE,
        "bandpass_hz": [pre.bandpass_low, pre.bandpass_high],
        "notch_hz": pre.notch_hz,
        "amplitude_calibration": config.get("amplitude_calibration") or "none",
        "input_channels": config.get("input_channels"),
        "drop_channels": config.get("drop_channels") or [],
        "apply_nlms": bool(config.get("apply_nlms")),
    }
    code = _code_checksums(info.release_dir)
    if code is None:
        extraction["model_key"] = info.key
    else:
        extraction["code_checksums"] = code

    columns = list(model.feature_columns)
    identity = {
        "modality": MODALITY,
        "columns": columns,
        "window_ms": round(WINDOW_SAMPLES * 1000 / EXG_SAMPLING_RATE),
        "hop_ms": round(STEP_SAMPLES * 1000 / EXG_SAMPLING_RATE),
        "extraction": extraction,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "signature": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "name": info.feature_set,
        "dimension": len(columns),
        **identity,
        "producer": {"reak_core_version": __version__, "model": info.key},
    }
