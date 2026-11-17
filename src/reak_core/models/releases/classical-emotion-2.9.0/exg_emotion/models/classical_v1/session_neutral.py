"""Inference for the released classical-emotion 2.9.0 package.

This is not the production predictor. ``releases/aliases.json`` stays on
2.6.0. Windows must already be notch-filtered at 60 Hz and bandpassed at
20–110 Hz. This module does not filter.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from exg_emotion.analysis.session_neutral_290 import (
    PUBLIC,
    WL_CHANNELS,
    align_logits,
    amplitude_block,
    assemble,
    blend,
    clip_amplitude,
    decide,
    merge_public,
    scale_free_block,
    softmax_temperature,
    valid_log_rms,
)
from exg_emotion.core.types import EmotionPrediction
from exg_emotion.models.classical_v1.features import compute_wl_batch

CHANNELS = [
    "Supraorbital_L",
    "Supraorbital_R",
    "Zygomatic_L",
    "Zygomatic_R",
    "Temporal_EEG",
    "Glabella",
    "Temporal_R",
    "Nasolabial",
]
LABELS_9 = [
    "NEUTRAL",
    "HAPPINESS",
    "ANGER",
    "DISGUST",
    "FEAR",
    "SURPRISE",
    "SADNESS",
    "CONTEMPT_LEFT",
    "CONTEMPT_RIGHT",
]


def _neutral(values: np.ndarray, in_domain: np.ndarray, prior: np.ndarray, n0: float, quantile: float) -> np.ndarray:
    """p20 on in-domain rows, shrunk toward ``prior``. Fewer than 8 rows stays at the prior."""
    p20 = np.full(values.shape[1], np.nan, dtype=np.float64)
    counts = np.zeros(values.shape[1], dtype=np.float64)
    for c in range(values.shape[1]):
        chosen = values[in_domain[:, c], c]
        chosen = chosen[np.isfinite(chosen)]
        counts[c] = float(chosen.size)
        if chosen.size >= 8:
            p20[c] = float(np.quantile(chosen, quantile))
    return blend(p20, prior, counts, n0)


class SessionNeutralModel:
    """2.9.0 session-neutral logistic regression with temperature and F0.5 thresholds.

    Call :meth:`set_subject_context` once on the recording for the offline
    neutral. For a causal stream, call :meth:`reset_causal`, :meth:`push` each
    finished window, and :meth:`advance_to` with the start time of the window
    about to be scored. A window starting at ``t`` uses the update at
    ``floor(t / 10)``.
    """

    name = "classical-emotion"
    version = "2.9.0"

    def __init__(
        self,
        pipeline,
        *,
        temperature: float,
        thresholds: dict[str, float],
        prior_rms: np.ndarray,
        prior_wl: np.ndarray,
        ood_log_rms: np.ndarray,
        n0: float = 60.0,
        quantile: float = 0.20,
        rail: float = 10.5,
        min_coverage: float = 0.80,
        update_s: float = 10.0,
        rate_limit: float = 0.05,
    ) -> None:
        self._pipeline = pipeline
        self.temperature = float(temperature)
        self.thresholds = {str(k): float(v) for k, v in thresholds.items()}
        self.prior_rms = np.asarray(prior_rms, dtype=np.float64).copy()
        self.prior_wl = np.asarray(prior_wl, dtype=np.float64).copy()
        self.ood_log_rms = np.asarray(ood_log_rms, dtype=np.float64).copy()
        self.n0 = float(n0)
        self.quantile = float(quantile)
        self.rail = float(rail)
        self.min_coverage = float(min_coverage)
        self.update_s = float(update_s)
        self.rate_limit = float(rate_limit)
        self.n_rms = self.prior_rms.copy()
        self.n_wl = self.prior_wl.copy()
        self.last_ood_channels: list[str] = []
        self._hist_rms: list[np.ndarray] = []
        self._hist_wl: list[np.ndarray] = []
        self._hist_end: list[float] = []
        self._bucket = 0

    @classmethod
    def from_release_dir(cls, release_dir: str | Path) -> "SessionNeutralModel":
        release_dir = Path(release_dir)
        config = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
        temperature = json.loads((release_dir / "temperature.json").read_text(encoding="utf-8"))
        raw_thresholds = json.loads((release_dir / "thresholds.json").read_text(encoding="utf-8"))
        wl_prior = json.loads((release_dir / "wl_prior.json").read_text(encoding="utf-8"))
        thresholds = {
            name: float(pack["threshold"])
            for name, pack in raw_thresholds.items()
        }
        prior_rms = np.array([float(config["prior_global"][name]) for name in CHANNELS], dtype=np.float64)
        ood = np.array([float(config["ood_clip_log_rms"][name]) for name in CHANNELS], dtype=np.float64)
        prior_wl = np.array(wl_prior["prior"], dtype=np.float64)
        pipeline = joblib.load(release_dir / "pipeline.joblib")
        return cls(
            pipeline,
            temperature=float(temperature["temperature"]),
            thresholds=thresholds,
            prior_rms=prior_rms,
            prior_wl=prior_wl,
            ood_log_rms=ood,
            n0=float(config.get("n0", 60.0)),
            quantile=float(config.get("quantile", 0.20)),
        )

    def set_subject_context(self, windows: np.ndarray) -> None:
        """Offline neutral from these windows. Out-of-domain channels are left out of the p20."""
        log_rms, log_wl = _measurements(windows)
        self.n_rms = _neutral(
            log_rms,
            np.isfinite(log_rms) & (log_rms <= self.ood_log_rms.reshape(1, -1)),
            self.prior_rms,
            self.n0,
            self.quantile,
        )
        rms_wl = log_rms[:, WL_CHANNELS]
        ood_wl = self.ood_log_rms[WL_CHANNELS]
        self.n_wl = _neutral(
            log_wl,
            np.isfinite(rms_wl) & (rms_wl <= ood_wl.reshape(1, -1)) & np.isfinite(log_wl),
            self.prior_wl,
            self.n0,
            self.quantile,
        )

    def reset_causal(self) -> None:
        """Start a stream at the prior. Later :meth:`advance_to` steps are at most 0.05 log."""
        self.n_rms = self.prior_rms.copy()
        self.n_wl = self.prior_wl.copy()
        self._hist_rms = []
        self._hist_wl = []
        self._hist_end = []
        self._bucket = 0

    def push(self, window: np.ndarray, t_end_s: float) -> None:
        """Store one finished window. ``t_end_s`` is the time its last sample is available."""
        log_rms, log_wl = _measurements(np.asarray(window, dtype=np.float64)[None, ...])
        self._hist_rms.append(log_rms[0])
        self._hist_wl.append(log_wl[0])
        self._hist_end.append(float(t_end_s))

    def advance_to(self, t_s: float) -> None:
        """Move the neutral up to ``floor(t_s / 10)`` using only windows that have already ended."""
        target = int(np.floor(float(t_s) / self.update_s))
        if not self._hist_end:
            self._bucket = max(self._bucket, target)
            return
        ends = np.asarray(self._hist_end, dtype=np.float64)
        log_rms = np.vstack(self._hist_rms)
        log_wl = np.vstack(self._hist_wl)
        while self._bucket < target:
            self._bucket += 1
            ready = ends <= (self._bucket * self.update_s)
            if not np.any(ready):
                continue
            rms_now = log_rms[ready]
            wl_now = log_wl[ready]
            proposed_rms = _neutral(
                rms_now,
                np.isfinite(rms_now) & (rms_now <= self.ood_log_rms.reshape(1, -1)),
                self.prior_rms,
                self.n0,
                self.quantile,
            )
            rms_wl = rms_now[:, WL_CHANNELS]
            ood_wl = self.ood_log_rms[WL_CHANNELS]
            proposed_wl = _neutral(
                wl_now,
                np.isfinite(rms_wl) & (rms_wl <= ood_wl.reshape(1, -1)) & np.isfinite(wl_now),
                self.prior_wl,
                self.n0,
                self.quantile,
            )
            step = self.rate_limit
            self.n_rms = self.n_rms + np.clip(proposed_rms - self.n_rms, -step, step)
            self.n_wl = self.n_wl + np.clip(proposed_wl - self.n_wl, -step, step)

    def predict(self, window: np.ndarray) -> EmotionPrediction:
        """Classify one window with the neutral frozen by the last context or causal update."""
        samples = np.asarray(window, dtype=np.float64)
        if samples.shape[0] != 8:
            raise ValueError(f"expected 8 channels, got shape {samples.shape}")
        log_rms, coverage = valid_log_rms(samples)
        rejected = bool(
            np.any(coverage < self.min_coverage)
            or np.any(~np.isfinite(log_rms))
            or np.any(log_rms > self.rail)
        )
        clipped, over = clip_amplitude(samples.reshape(1, *samples.shape), self.ood_log_rms)
        self.last_ood_channels = [CHANNELS[i] for i, flag in enumerate(over[0]) if flag]
        features = assemble(
            amplitude_block(clipped, self.n_rms.reshape(1, -1), self.n_wl.reshape(1, -1)),
            scale_free_block(clipped),
        )
        proba = _public_probability(self._pipeline, features, self.temperature)[0]
        label = str(decide(proba.reshape(1, -1), self.thresholds)[0])
        raw = PUBLIC[int(np.argmax(proba))]
        reasons = []
        if raw != label:
            reasons.append(f"{raw} probability below its F0.5 threshold; emitted NEUTRAL")
        if self.last_ood_channels:
            reasons.append("clipped " + ", ".join(self.last_ood_channels))
        if rejected:
            reasons.append("coverage below 0.80 or log RMS above 10.5")
        return EmotionPrediction(
            label=label,
            probabilities={name: float(proba[i]) for i, name in enumerate(PUBLIC)},
            model_name=self.name,
            model_version=self.version,
            usable=not rejected,
            reason="; ".join(reasons) or None,
            low_confidence=raw != label,
            fear_suppressed=False,
        )


def _measurements(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[1] != 8:
        raise ValueError(f"expected [n, 8, samples], got {windows.shape}")
    log_rms = np.vstack([valid_log_rms(window)[0] for window in windows])
    wl = compute_wl_batch(windows)[:, WL_CHANNELS]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_wl = np.log(np.maximum(wl, 1e-12))
    return log_rms, log_wl


def _public_probability(pipeline, features: np.ndarray, temperature: float) -> np.ndarray:
    classes = [str(c) for c in pipeline.named_steps["clf"].classes_]
    logits = align_logits(pipeline.decision_function(features), classes, LABELS_9)
    return merge_public(softmax_temperature(logits, temperature), LABELS_9)
