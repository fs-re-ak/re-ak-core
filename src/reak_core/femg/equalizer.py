# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Per-channel power equalization (RMS equalizer) of stored emotion feature vectors.

The equalizer scales each channel of each window by a positive constant
``k_c = max(rms_c * 10 ** (gain_db_c / 20) + offset_uv_c, epsilon) / rms_c``, which is what scaling
the filtered samples of that channel would do. For feature sets where every feature is either a
function of the per-channel RMS or invariant to a per-channel scale, the equalized vector is
computed exactly from the stored vector and the stored channel RMS, without the raw signal.

:func:`scoring_spec` describes such a model in plain arrays: the rescale recipe of each feature
column, the classifier (``StandardScaler`` + multinomial ``LogisticRegression``), the label mapping,
the post-processing and the training Neutral channel profile. It is written next to the features as
``features/emotions/scoring_model.json`` and uploaded with them, so the CAS server can preview,
apply and revert an equalization without Python. :func:`apply_channel_eq` and :func:`score` are the
reference implementation of that spec; the C# implementation is pinned to them by a parity fixture.

Only ``extended_nm_a_rmsunit`` (2.11.0 through 2.14.0) is supported: its 19 features are the RMS unit
vector, the mean RMS, the zygomatic asymmetry and NM-A ratio (functions of the RMS), and the median
frequencies and antagonist envelope correlation (invariant to a positive scale).

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

from reak_core.femg.feature_set import feature_set_descriptor
from reak_core.models.base import EMOTION_COLUMNS, EmotionModel

SPEC_FORMAT = 1
EPSILON_UV = 1e-3
"""Floor of an equalized channel RMS (µV). Well above the 1e-6 silence gates of the features."""

SUPPORTED_FEATURE_SETS = ("extended_nm_a_rmsunit",)

_LABEL_MAP: dict[str, str] = {col.upper(): col for col in EMOTION_COLUMNS}
_CONTEMPT_VARIANTS = ("CONTEMPT_LEFT", "CONTEMPT_RIGHT")

# Ratio features of extended_nm_a_rmsunit: (a - b) / (a + b + eps), 0 when a + b < gate.
_RATIOS: dict[str, dict] = {
    "asym_zyg": {"a": "Zygomatic_L", "b": "Zygomatic_R", "eps": 1e-6, "gate": None},
    "nm_ratio_a": {"a": "Glabella", "b": "Nasolabial", "eps": 1e-6, "gate": 1e-6},
}
_INVARIANT_PREFIXES = ("mdf_", "ant_corr_")
_UNIT_EPS = 1e-12


def _recipe(columns: list[str], channels: list[str]) -> list[dict] | None:
    """Rescale recipe of each feature column, or ``None`` if a column is not supported."""
    recipe: list[dict] = []
    for column in columns:
        if column.startswith("rms_unit_") and column[len("rms_unit_") :] in channels:
            recipe.append({"kind": "rms_unit", "channel": channels.index(column[len("rms_unit_") :]), "eps": _UNIT_EPS})
        elif column == "rms_mean":
            recipe.append({"kind": "rms_mean"})
        elif column.startswith(_INVARIANT_PREFIXES):
            recipe.append({"kind": "invariant"})
        elif column in _RATIOS:
            r = _RATIOS[column]
            recipe.append(
                {
                    "kind": "ratio",
                    "a": channels.index(r["a"]),
                    "b": channels.index(r["b"]),
                    "eps": r["eps"],
                    "gate": r["gate"],
                }
            )
        else:
            return None
    return recipe


def _pipeline(model: EmotionModel):  # noqa: ANN202
    """The release's inner classifier model, or ``None`` when it is not a plain classical release."""
    predictor = getattr(model, "_predictor", None)
    inner = getattr(predictor, "_model", None)
    if inner is None or getattr(model, "_feature_fn", None) is not None:
        return None
    if getattr(inner, "_specialist_pipeline", None) is not None or getattr(inner, "_suppress_fear", False):
        return None
    return inner


def neutral_reference(model_key: str) -> dict | None:
    """Training Neutral channel profile of a model (median channel RMS and unit vector), if shipped."""
    try:
        text = (
            resources.files("reak_core.femg").joinpath("eq_reference", f"{model_key}.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None
    data = json.loads(text)
    return {
        "source": data.get("source"),
        "n_windows": data.get("n_neutral"),
        "channel_rms_median": [float(v) for v in data["channel_rms_median"]],
        "rms_unit_median": [float(v) for v in data["rms_unit_median"]],
    }


def scoring_spec(model: EmotionModel) -> dict | None:
    """Plain-array description of ``model`` for rescaling and scoring stored vectors.

    Returns ``None`` when the model cannot be equalized from stored features: an unsupported
    feature set, a classifier other than ``StandardScaler`` + ``LogisticRegression``, a specialist
    stage, Fear suppression, per-subject calibration, or the low-confidence relabel enabled.
    """
    info = model.info
    if info.feature_set not in SUPPORTED_FEATURE_SETS:
        return None
    inner = _pipeline(model)
    if inner is None:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # pragma: no cover - sklearn is a dependency of every release
        return None
    steps = getattr(inner._pipeline, "named_steps", {})
    scaler, clf = steps.get("scaler"), steps.get("classifier")
    if not isinstance(scaler, StandardScaler) or not isinstance(clf, LogisticRegression):
        return None
    classes = [str(c) for c in clf.classes_]
    if clf.coef_.shape[0] != len(classes) or len(classes) < 3:
        return None  # binary or one-vs-rest layouts are not supported

    channels = list(model.release.manifest.get("input", {}).get("channels") or [])
    columns = list(model.feature_columns)
    recipe = _recipe(columns, channels) if len(channels) == 8 else None
    if recipe is None or clf.coef_.shape[1] != len(columns):
        return None

    merge = bool(getattr(inner, "_merge_contempt", False))
    labels = {c: ("Contempt" if merge and c in _CONTEMPT_VARIANTS else _LABEL_MAP.get(c, c.title())) for c in classes}
    post = model.postprocess
    if post.low_confidence_enabled:
        return None  # the low-confidence relabel needs probabilities the server does not rebuild
    descriptor = feature_set_descriptor(model)
    return {
        "format": SPEC_FORMAT,
        "model_key": info.key,
        "model_version": info.version,
        "signature": descriptor["signature"],
        "columns": columns,
        "channels": channels,
        "classifier": {
            "kind": "standard_scaler+multinomial_logistic_regression",
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
            "coef": [[float(v) for v in row] for row in clf.coef_],
            "intercept": [float(v) for v in clf.intercept_],
            "classes": classes,
        },
        "labels": labels,
        "postprocessing": {
            "one_hot": bool(post.one_hot),
            "high_rms": {
                "enabled": bool(post.high_rms_enabled),
                "threshold_uv": float(post.high_rms_threshold_uv),
                "max_gap_windows": int(post.high_rms_max_gap_windows),
            },
        },
        "eq": {"epsilon_uv": EPSILON_UV, "recipe": recipe},
        "neutral_reference": neutral_reference(info.key),
    }


def channel_scales(channel_rms: np.ndarray, gains_db, offsets_uv, epsilon_uv: float = EPSILON_UV) -> np.ndarray:
    """Per-window, per-channel amplitude scale ``k`` (``(n, 8)``); 1 where the channel RMS is 0."""
    rms = np.asarray(channel_rms, dtype=np.float64)
    gains = 10.0 ** (np.asarray(gains_db, dtype=np.float64) / 20.0)
    target = np.maximum(rms * gains + np.asarray(offsets_uv, dtype=np.float64), epsilon_uv)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(rms > 0.0, target / rms, 1.0)
    return k


def apply_channel_eq(
    vectors: np.ndarray, channel_rms: np.ndarray, gains_db, offsets_uv, spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Equalized feature vectors and channel RMS, from stored vectors ``(n, d)`` and RMS ``(n, 8)``."""
    vectors = np.asarray(vectors, dtype=np.float64)
    rms = np.asarray(channel_rms, dtype=np.float64)
    new_rms = rms * channel_scales(rms, gains_db, offsets_uv, spec["eq"]["epsilon_uv"])
    out = vectors.copy()
    norms = np.linalg.norm(new_rms, axis=1)
    for j, step in enumerate(spec["eq"]["recipe"]):
        kind = step["kind"]
        if kind == "rms_unit":
            out[:, j] = new_rms[:, step["channel"]] / (norms + step["eps"])
        elif kind == "rms_mean":
            out[:, j] = new_rms.mean(axis=1)
        elif kind == "ratio":
            a, b = new_rms[:, step["a"]], new_rms[:, step["b"]]
            ratio = (a - b) / (a + b + step["eps"])
            if step["gate"] is not None:
                ratio = np.where(a + b < step["gate"], 0.0, ratio)
            out[:, j] = ratio
        elif kind != "invariant":
            raise ValueError(f"unknown recipe kind: {kind}")
    return out, new_rms


def score(vectors: np.ndarray, spec: dict) -> tuple[list[str], np.ndarray]:
    """Public labels and their confidence for feature vectors ``(n, d)``.

    Argmax over the raw classes, then the contempt merge (the label maps to ``Contempt`` and its
    confidence is the summed probability of the merged classes), as the release's ``predict``.
    """
    clf = spec["classifier"]
    z = (np.asarray(vectors, dtype=np.float64) - np.asarray(clf["mean"])) / np.asarray(clf["scale"])
    logits = z @ np.asarray(clf["coef"]).T + np.asarray(clf["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    public = [spec["labels"][c] for c in clf["classes"]]
    labels: list[str] = []
    confidence = np.empty(len(proba))
    for i, row in enumerate(proba):
        label = public[int(np.argmax(row))]
        labels.append(label)
        confidence[i] = sum(p for p, name in zip(row, public) if name == label)
    return labels, confidence
