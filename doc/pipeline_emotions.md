# Emotions — Hermes facial EMG

**Module:** `reak_core.femg.emotions`  
**Entry point:** `compute_emotions(session_root, model=None)`  
**Primary output:** `results/Emotions.csv`

Moment-to-moment emotion intensities from the 8-channel Hermes EXG recording. The default model is the alias `standard-latest`, which resolves to **classical-emotion 2.6.0** (decision D9). `classical-emotion` remains the alias for 2.5.0. Both releases are loaded from `reak_core.models.releases` after their manifest checksums match. Their README and INTEGRATION files still describe copying the release folder into another project; that text is part of the checksummed release and is not the install path.

## Input and invalid samples

The same EXG file as cognition. See [overview](overview.md) for discovery order and the invalid-sample rule (non-finite, or within 0.1 µV of ±375 000 µV).

Preparation, in `reak_core.femg.signal`:

1. Mark invalid samples.
2. Interpolate them per channel, only so the filter does not spread gaps. A channel with no valid sample is left as it is.
3. Apply the release filter to the whole recording, zero-phase. 2.6.0 declares a 60 Hz notch and a 20–110 Hz bandpass. 2.5.0 does not declare a band, so the fallback is 15–40 Hz and no notch.
4. Put the invalid samples back to missing.
5. Slide a 2.0 s window (500 samples) every 0.5 s (125 samples). **A window that contains any non-finite sample is null.** This replaces the 1.1 rule that nullified a window only when more than 10% of its samples were missing.

## Models

| Release | Filter | `Emotions.csv` cells | Notes |
|---|---|---|---|
| classical-emotion 2.6.0 (default) | 60 Hz notch, then 20–110 Hz | One-hot of the emitted label: 1 and 0 | Gated p95 amplitude with a field prior. Fear is not emitted as the winning label. Do not apply NLMS to these windows |
| classical-emotion 2.5.0 | 15–40 Hz, no notch | Class probabilities | Confidence used by the heuristics is the highest merged probability |

Columns, in order: `Neutral`, `Happiness`, `Anger`, `Surprise`, `Contempt`, `Disgust`, `Fear`, `Sadness`.

High-RMS and low-confidence relabeling run on every wrapper (decision D8). Defaults are in each release manifest and can be overridden on `compute_emotions`. For both releases the high-RMS gate is on, at 200 µV, filling gaps of up to 6 windows. Low-confidence relabeling is on for 2.5.0 (threshold 0.3) and off for 2.6.0.

## Output timebase

Classifier rows sit on the window's median timestamp, about 0.5 s apart and not exactly on the 2 Hz grid, because the headset writes timestamps in bursts. `Emotions.csv` is then written with `snap_to_timebase`: each output row copies the nearest window and does not blend neighbours. A linear resample of a 2.6.0 one-hot row produced a slow triangular mix (for example Sadness 0.905 / Neutral 0.095) whose period followed the headset clock. That mix is what 1.0.2 removed. Null windows are written as `NULL`. Continuous result tables still use linear interpolation.

## Files

| File | Contents |
|---|---|
| `results/Emotions.csv` | Intensities on the 2 Hz grid |
| `results/emotion_model.json` | Model key, name, version, feature set and kind. This is the only sidecar. `features/emotions/model_info.json` is removed if a previous run left one |
| `features/emotions/rmsSignals.csv` | Raw per-channel RMS of every window, µV. Null windows are missing. Always written |
| `features/emotions/emotionClassifierInputs.csv` | Features, raw label, confidence, raw `HEAD_*_RMS` columns, `HighRmsArtifact` |
| `features/emotions/avgRms.csv`, `relabeled-samples.csv` | Heuristic diagnostics |
| `features/emotions/report_*.png` | Emotions report, from the pipeline exporter |

The step is skipped when those outputs already exist and `emotion_model.json` names the same model. Pass `force=True` to recompute.
