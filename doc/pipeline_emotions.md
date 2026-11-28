# Emotions — Hermes facial EMG

**Module:** `reak_core.femg.emotions`  
**Entry point:** `compute_emotions(session_root, model=None)`  
**Primary output:** `results/Emotions.csv`

Moment-to-moment emotion intensities from the 8-channel Hermes EXG recording. The default model is the alias `standard-latest`, which resolves to **classical-emotion 2.14.0**. `classical-emotion` remains the alias for 2.5.0. Selectable releases also include **2.5.0–2.13.0**. All are loaded from `reak_core.models.releases` after their manifest checksums match. Their README and INTEGRATION files still describe copying the release folder into another project; that text is part of the checksummed release and is not the install path.

## Input and invalid samples

The same EXG file as cognition. See [overview](overview.md) for discovery order and the invalid-sample rule (non-finite, or within 0.1 µV of ±375 000 µV).

Preparation, in `reak_core.femg.signal`:

1. Mark invalid samples.
2. Interpolate them per channel, only so the filter does not spread gaps. A channel with no valid sample is left as it is.
3. Apply the release filter to the whole recording, zero-phase. 2.6.0–2.14.0 declare a 60 Hz notch and a 20–110 Hz bandpass. 2.5.0 does not declare a band, so the fallback is 15–40 Hz and no notch.
4. Put the invalid samples back to missing.
5. Slide a 2.0 s window (500 samples) every 0.5 s (125 samples). **A window that contains any non-finite sample is null.** This replaces the 1.1 rule that nullified a window only when more than 10% of its samples were missing.

## Models

| Release | Filter | `Emotions.csv` cells | Notes |
|---|---|---|---|
| classical-emotion 2.14.0 (default) | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Same RMS L2 unit vector as 2.13.0; ANGER ≥ Neutral Q2, other non-NEUTRAL ≥ Neutral Q3; MB001 included. Fear is emitted. Do not apply NLMS |
| classical-emotion 2.13.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Same as 2.12.0, with non-NEUTRAL training windows below Neutral Q3 mean RMS dropped. Fear is emitted. Do not apply NLMS |
| classical-emotion 2.12.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Calibration-free RMS L2 unit vector (19 features) on the 2.10.0 sample policy. Fear is emitted. Do not apply NLMS |
| classical-emotion 2.11.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Same calibration-free RMS L2 unit vector as 2.7.0 (19 features), with fear emitted and FS002 sessions in training. Do not apply NLMS |
| classical-emotion 2.10.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Same session-neutral recipe as 2.9.0 with an adjusted training set (FS002 sessions added; extra subject exclusions). Fear is emitted. Do not apply NLMS |
| classical-emotion 2.9.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Session-neutral log RMS (25 features): `N = w * p20 + (1 - w) * prior`. Fear is emitted. Loaded with `SessionNeutralModel`, not `Predictor`. Do not apply NLMS |
| classical-emotion 2.8.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Floor-only log-ratio amplitude (27 features). Fear is not emitted. Do not apply NLMS |
| classical-emotion 2.7.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Calibration-free RMS L2 unit vector (19 features). Fear is not emitted. Do not apply NLMS |
| classical-emotion 2.6.0 | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Gated p95 amplitude with a field prior. Fear is not emitted as the winning label. Do not apply NLMS to these windows |
| classical-emotion 2.5.0 | 15–40 Hz, no notch | Class probabilities | Confidence used by the heuristics is the highest merged probability |

Columns, in order: `Neutral`, `Happiness`, `Anger`, `Surprise`, `Contempt`, `Disgust`, `Fear`, `Sadness`.

High-RMS and low-confidence relabeling run on every wrapper (decision D8). Defaults are in each release manifest and can be overridden on `compute_emotions`. For 2.6.0–2.14.0 the high-RMS gate is on, at 200 µV, filling gaps of up to 6 windows, and low-confidence relabeling is off. 2.9.0 and 2.10.0 already turn a below-threshold emotion into NEUTRAL inside the release; the wrapper does not apply a second low-confidence pass. Low-confidence relabeling is on for 2.5.0 (threshold 0.3).

## Output timebase

Classifier rows sit on the window's median timestamp, about 0.5 s apart and not exactly on the 2 Hz grid, because the headset writes timestamps in bursts. `Emotions.csv` is then written with `snap_to_timebase`: each output row copies the nearest window and does not blend neighbours. A linear resample of a 2.6.0 one-hot row produced a slow triangular mix (for example Sadness 0.905 / Neutral 0.095) whose period followed the headset clock. That mix is what 1.0.2 removed. Null windows are written as `NULL`. Continuous result tables still use linear interpolation.

## Files

| File | Contents |
|---|---|
| `results/Emotions.csv` | Intensities on the 2 Hz grid |
| `results/emotion_model.json` | Model key, name, version, feature set and kind. This is the only sidecar in `results/`. `features/emotions/model_info.json` is removed if a previous run left one |
| `features/emotions/rmsSignals.csv` | Raw per-channel RMS of every window, µV. Null windows are missing. Always written |
| `features/emotions/emotionClassifierInputs.csv` | Features, raw label, confidence, raw `HEAD_*_RMS` columns, `HighRmsArtifact` |
| `features/emotions/feature_set.json` | Descriptor of the feature columns of `emotionClassifierInputs.csv`: `signature`, `modality`, `name`, `columns`, `dimension`, `window_ms`, `hop_ms`, `extraction`, `producer`. See below |
| `features/emotions/scoring_model.json` | Equalizer scoring spec (rescale recipe, classifier arrays, label mapping, post-processing, training Neutral profile), only for models that support it. See below |
| `features/emotions/avgRms.csv`, `relabeled-samples.csv` | Heuristic diagnostics |
| `features/emotions/report_*.png` | Emotions report, from the pipeline exporter |

The step is skipped when those outputs already exist and `emotion_model.json` names the same model. Pass `force=True` to recompute. A skipped step still writes `feature_set.json` and `scoring_model.json` if they are missing.

### Feature-set signature

`feature_set.json` identifies the feature vectors by how they are computed, not by the `feature_set` name, which has meant different values across releases. `extraction` holds the sample rate, band-pass and notch, the amplitude calibration, the input and dropped channels, NLMS, and the checksums of the release's feature code (`exg_emotion/models/`, `exg_emotion/analysis/`, from its manifest). `signature` is the SHA-256 of the canonical JSON of `modality`, `columns`, `window_ms`, `hop_ms` and `extraction`. Models with the same signature produce interchangeable vectors: 2.11.0 to 2.14.0 share one. `name` and `producer` (re-ak-core version, model key) do not enter the signature. A model without a release manifest has its key in `extraction`, so its vectors are never mixed with another model's. Prime uploads the vectors with this descriptor to the RE-AK CAS feature store.

### Equalizer scoring spec

`scoring_model.json` lets a server apply a per-channel power equalization (RMS equalizer) to the stored vectors and score them again, without the raw signal and without Python (`reak_core.femg.equalizer`). It is written only for models that support it: `extended_nm_a_rmsunit` (2.11.0 to 2.14.0), a `StandardScaler` + multinomial `LogisticRegression` pipeline, no specialist, no Fear suppression and no low-confidence relabel.

- The equalizer scales channel `c` of each window by `k = max(rms_c * 10^(gain_db_c / 20) + offset_uv_c, epsilon) / rms_c` (`epsilon` = 1e-3 µV; a channel with RMS 0 stays 0). This is what scaling the filtered samples would do.
- `eq.recipe` gives, per feature column, how it changes: `rms_unit` (channel RMS over the L2 norm of the 8 RMS, `+ eps`), `rms_mean`, `ratio` (`(a - b) / (a + b + eps)`, 0 below `gate`), or `invariant` (median frequencies and the antagonist envelope correlation do not change under a positive scale). The rescaled vector is exactly the vector of the scaled samples.
- `classifier` holds the scaler mean and scale, the coefficients and intercepts, and the raw classes; `labels` maps each raw class to the emitted label (both contempt sides to `Contempt`). The label is the argmax over the raw classes; its confidence is the summed probability of the classes with that label.
- `postprocessing` repeats the release's one-hot and high-RMS rules, and `neutral_reference` holds the median channel RMS and unit vector of the training Neutral windows (`femg/eq_reference/<model key>.json`, aggregate values only), when shipped.

`apply_channel_eq` and `score` are the reference implementation. `tests/fixtures/eq_parity.json` pins the RE-AK CAS (C#) implementation to it.
