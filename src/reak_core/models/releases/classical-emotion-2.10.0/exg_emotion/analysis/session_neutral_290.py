"""classical-emotion 2.9.0 features and decision layer.

This module is not wired into the production predictor. The production
pipeline and the 2.6.0, 2.7.0, and 2.8.0 checkpoints stay untouched.

Amplitude is ``log(RMS) - N`` per channel. Waveform length uses the same
shrinkage. The remaining features (median frequency, zygomatic asymmetry,
antagonist correlation, NM-A) are scale-free and are not session-scaled.
There is no soft-direction feature and no fear suppression.
"""

from __future__ import annotations

import numpy as np

from exg_emotion.models.classical_v1.features import (
    _DEFAULT_ANT_A,
    _DEFAULT_ANT_B,
    _DEFAULT_EMG_INDICES,
    _DEFAULT_NM_A_M,
    _DEFAULT_NM_A_N,
    compute_antagonist_corr_batch,
    compute_asymmetry_batch,
    compute_mdf_batch,
    compute_nm_ratio_batch,
    compute_rms_batch,
    compute_wl_batch,
)

PUBLIC = [
    "NEUTRAL",
    "HAPPINESS",
    "ANGER",
    "DISGUST",
    "FEAR",
    "SURPRISE",
    "SADNESS",
    "CONTEMPT",
]
EMOTIONS = [c for c in PUBLIC if c != "NEUTRAL"]
N_FEATURES = 25
WL_CHANNELS = list(_DEFAULT_EMG_INDICES)


def valid_log_rms(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``window`` is ``[8, samples]``. Returns log RMS and coverage."""
    finite = np.isfinite(window)
    coverage = finite.mean(axis=1)
    filled = np.where(finite, window, 0.0)
    rms = np.sqrt((filled ** 2).sum(axis=1) / np.maximum(finite.sum(axis=1), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rms = np.log(rms)
    log_rms[~np.isfinite(log_rms)] = np.nan
    return log_rms, coverage


def quality_keep(X: np.ndarray, rail: float = 10.5, min_coverage: float = 0.80) -> np.ndarray:
    """Drop windows below the coverage floor or above the rail on any channel."""
    keep = np.ones(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        log_rms, coverage = valid_log_rms(X[i])
        if np.any(coverage < min_coverage) or np.any(~np.isfinite(log_rms)) or np.any(log_rms > rail):
            keep[i] = False
    return keep


def clip_amplitude(X: np.ndarray, ood_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cap each channel's RMS at ``exp(ood_log)`` by a positive gain.

    The within-channel shape is unchanged. A window is never dropped.
    Returns the clipped windows and a boolean ``[n, 8]`` of clipped channels.
    """
    rms = compute_rms_batch(X)
    cap = np.exp(np.asarray(ood_log, dtype=np.float64)).reshape(1, -1)
    over = rms > cap
    scale = np.ones_like(rms)
    scale[over] = cap.repeat(len(rms), axis=0)[over] / rms[over]
    return X * scale[:, :, None].astype(X.dtype), over


def blend(observed: np.ndarray, prior: np.ndarray, n: np.ndarray, n0: float) -> np.ndarray:
    """``N = w * observed + (1 - w) * prior``, ``w = n / (n + n0)``."""
    observed = np.asarray(observed, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    w = n / (n + float(n0))
    w = np.where((n < 8.0) | ~np.isfinite(observed), 0.0, w)
    blended = w * np.where(np.isfinite(observed), observed, prior) + (1.0 - w) * prior
    return np.where(np.isfinite(blended), blended, prior)


def scale_free_block(X: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """MDF (7), zygomatic asymmetry, antagonist correlation, NM-A. Shape ``[n, 10]``.

    These four blocks are the scale-free part of ``extended_nm_a``. They are
    computed with the same functions as 2.6.0 and are not session-scaled.
    """
    mdf = compute_mdf_batch(X, emg_indices=WL_CHANNELS, fs=fs)
    asym = compute_asymmetry_batch(X)
    corr = compute_antagonist_corr_batch(X, _DEFAULT_ANT_A, _DEFAULT_ANT_B)
    nm = compute_nm_ratio_batch(X, _DEFAULT_NM_A_N, list(_DEFAULT_NM_A_M))
    return np.column_stack([mdf, asym, corr, nm])


def amplitude_block(X: np.ndarray, n_rms: np.ndarray, n_wl: np.ndarray) -> np.ndarray:
    """``log(RMS) - N`` on 8 channels and ``log(WL) - N`` on 7 EMG channels."""
    rms = compute_rms_batch(X)
    wl = compute_wl_batch(X)[:, WL_CHANNELS]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rms = np.log(np.maximum(rms, 1e-12))
        log_wl = np.log(np.maximum(wl, 1e-12))
    return np.column_stack([log_rms - n_rms, log_wl - n_wl])


def assemble(amplitude: np.ndarray, scale_free: np.ndarray) -> np.ndarray:
    return np.column_stack([amplitude, scale_free]).astype(np.float64)


def merge_public(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    """Sum CONTEMPT laterals into one public class. Rows are renormalised."""
    out = np.zeros((len(proba), len(PUBLIC)), dtype=np.float64)
    for j, name in enumerate(classes):
        public = "CONTEMPT" if str(name).startswith("CONTEMPT") else str(name)
        if public not in PUBLIC:
            continue
        out[:, PUBLIC.index(public)] += proba[:, j]
    total = out.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return out / total


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-3)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_logits(logits: np.ndarray, classes: list[str], all_classes: list[str]) -> np.ndarray:
    out = np.full((len(logits), len(all_classes)), -30.0, dtype=np.float64)
    index = {name: i for i, name in enumerate(all_classes)}
    for j, name in enumerate(classes):
        if name in index:
            out[:, index[name]] = logits[:, j]
    return out


def decide(proba_public: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Argmax, then NEUTRAL when the winning emotion is below its threshold.

    No class is dropped and no window is abstained.
    """
    k = np.argmax(proba_public, axis=1)
    labels = np.array([PUBLIC[i] for i in k], dtype=object)
    for i, name in enumerate(labels):
        if name == "NEUTRAL":
            continue
        if float(proba_public[i, PUBLIC.index(name)]) < float(thresholds.get(name, 0.0)):
            labels[i] = "NEUTRAL"
    return labels.astype(str)


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return 0.0
    b2 = beta * beta
    den = b2 * precision + recall
    if den <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / den


def pr_at_threshold(y: np.ndarray, proba: np.ndarray, class_name: str, threshold: float) -> tuple[float, float]:
    """Precision and recall when ``class_name`` is kept only above ``threshold``."""
    k = int(PUBLIC.index(class_name))
    true = y == class_name
    winner = np.argmax(proba, axis=1) == k
    pred = winner & (proba[:, k] >= threshold)
    tp = float(np.sum(pred & true))
    fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def best_fbeta_threshold(
    y: np.ndarray,
    proba: np.ndarray,
    class_name: str,
    beta: float = 0.5,
    grid: np.ndarray | None = None,
) -> dict:
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    rows = []
    best = {"threshold": 0.5, "fbeta": -1.0, "precision": float("nan"), "recall": float("nan")}
    for t in grid:
        precision, recall = pr_at_threshold(y, proba, class_name, float(t))
        score = fbeta(precision, recall, beta)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "fbeta": score})
        if score > best["fbeta"]:
            best = {"threshold": float(t), "fbeta": score, "precision": precision, "recall": recall}
    return {"best": best, "curve": rows}


def precision_target_threshold(curve: list[dict], target: float) -> float:
    """Lowest threshold whose precision reaches ``target``, else NaN.

    Among those, the one with the highest recall is kept so a later
    adjustment can trade specificity for coverage without a new search.
    """
    ok = [r for r in curve if np.isfinite(r["precision"]) and r["precision"] + 1e-12 >= target]
    if not ok:
        return float("nan")
    ok.sort(key=lambda r: (r["recall"] if np.isfinite(r["recall"]) else -1.0), reverse=True)
    return float(ok[0]["threshold"])
