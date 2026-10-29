# Changelog

All notable changes of `re-ak-core` (import package `reak_core`). Versions follow semantic versioning: a change
that is not backward compatible in the public API or in the output format (files, columns, units) requires a new
major version. A bug fix that corrects output values without changing their format is a patch release, and is
listed under "Fixed" with the affected files.

## 1.0.2

### Fixed

- `results/Emotions.csv` no longer shows a slow triangular oscillation whose period (typically
  ~6–7 minutes) was set by the headset clock drift, not by any physiological signal.
  **Root cause**: the Hermes headset writes one timestamp per 10 samples (~40 ms bursts with zeros
  in between), so the classifier window times step by ~0.5015 s rather than exactly 0.5 s.
  Resampling those one-hot labels onto the strict 2 Hz output grid with linear interpolation blended
  adjacent labels — e.g. Sadness 0.905 / Neutral 0.095 — and the blend weight drifted at a rate
  determined by the fractional mismatch, producing the triangle.
  **Fix**: `Emotions.csv` is now written via the new `snap_to_timebase` (nearest-neighbour), which
  copies the label of the closest classifier window without blending. Continuous signals (Arousal,
  Cognition, HeartDynamics) still use linear interpolation through `resample_to_timebase`.
  This changes the values in `results/Emotions.csv` (same columns): sessions reprocessed with 1.0.2 differ from
  1.0.1 outputs. The stored NucleusKit and Prime references have been updated accordingly.

### Added

- `reak_core.sync.snap_to_timebase`: nearest-neighbour resampling onto an output timebase, for categorical or
  one-hot signals.

## 1.0.1

First release published on PyPI (`pip install re-ak-core`); no 1.0.0 was released. It contains the
1.0.0a1 pre-release below, with these packaging changes:

- Dependency minimums raised to the validated versions: numpy 2.5, pandas 2.3, joblib 1.6, matplotlib 3.11.
- scipy kept below 1.18: its `filtfilt` results differ by about 1e-6, enough to change values rounded to 3
  decimals in `features/cognition/epochMetrics.csv`.
- pandas stays on 2.x (neurokit2 0.2.13 requires `pandas<3`); scikit-learn stays on 1.6 for the model releases.
- SPDX license metadata, project URLs and classifiers.

## 1.0.0a1 (pre-release)

First version of the open processing core, extracted from NucleusKit-ProcessingPipeline 1.1 and Prime 3.1.0.
`reak_core` 1.0 is the new reference: where the two systems disagreed, the open-core decisions (D1–D25) chose the
behaviour, and every resulting change is measured below. 1.0.0 will be tagged after NucleusKit and Prime consume
the core and after a validation on real recordings.

### Added

- Session I/O: layout, raw-stream discovery, robust CSV loading, timestamp unit detection, UWB / GPS validation.
- Processing: EEG cognition (Engagement), fEMG emotions with the classical-emotion 2.5.0 and 2.6.0 releases, EDA
  arousal, PPG heart dynamics, feature events, playback annotations.
- `Pipeline` with `Step` / `Exporter` extension points (entry points `reak_core.steps`, `reak_core.exporters`,
  `reak_core.models`), `results/processing.json` with a per-session quality summary, a command line (`reak-core`).
- Tools: channel fixer (models stored as JSON coefficients), PPG fixer, RMS baseline and revert; per-session emotions
  report.

### Behaviour compared with NucleusKit 1.1 and Prime 3.1.0

- **D1** Timestamps in epoch µs / ns and relative clocks are read correctly (NucleusKit read them 1000 times too long).
- **D2** EXG files with named columns are reordered into hardware order and a truncated last line is ignored (Prime read
  named columns by position and skipped cognition and emotions on truncated files).
- **D3, D16** Raw events are also read from `event.csv`; administrative events are excluded; processed events go to
  `features/events/`.
- **D4** No 60 Hz notch after the 0.5–30 Hz cognition bandpass (Prime applied one).
- **D5** Only Engagement is computed.
- **D6** A raw EXG sample is invalid when it is non-finite or at ADC full scale (±375 000 µV, gain 12). 187 500 µV
  (the full scale before the gain change) and near-zero samples are no longer invalid.
- **D7** Invalid samples are interpolated only to filter, then masked again; a 2 s emotion window containing any
  invalid sample is null (instead of "more than 10 %" with zero filling).
- **D8** The high-RMS and low-confidence heuristics apply on every wrapper, with per-release defaults and overridable
  parameters (NucleusKit had none). The confidence of 2.5.0 is the highest merged probability, as in Prime.
- **D9** The default emotion model is classical-emotion 2.6.0 (NucleusKit's was 2.5.0).
- **D10** One output schema: `rmsSignals.csv` always written, raw `HEAD_*_RMS` columns and `HighRmsArtifact` in
  `emotionClassifierInputs.csv`, `avgRms.csv` and `relabeled-samples.csv`, a single sidecar `results/emotion_model.json`
  (`features/emotions/model_info.json` is no longer written).
- **D11–D15** PPG artifact rejection, normalisation, missed-beat filling and HRV follow NucleusKit (Prime differed).
- **D19, D20** The emotions report and the channel fixer are part of the core.
- **D23** Cognition judges sample validity on the channels it uses (a dead cheek, brow or nose electrode no longer
  blanks Engagement in Prime).
- **D24** `inf` / `-inf` raw values are invalid samples, so only the windows containing them are null (NucleusKit's
  emotion step failed on such files and Prime's emotion output became entirely null).
- **D25** The recording duration in `features/metainfo.json` uses normalised timestamps (both systems reported 0 for
  Vizia sessions, whose timestamps are in seconds).

### Measured differences

<!-- GENERATED:BEGIN -->

Measured by `tools/write_changelog.py` with reak_core 1.0.0a1 on the 19 synthetic cases of `tests/synthetic` (models 2.6.0 and 2.5.0), against the references of NucleusKit `addaf90, d4a4026, fad3728` and Prime `50e162d, 72f8c54` (same processing code as NucleusKit 1.1 and Prime 3.1.0). Do not edit this block by hand.

- 249 output / model / case combinations are identical to the system the decisions designate (within rtol 1e-6, atol 1e-9).
- Every other difference is listed below under the decision that causes it. Differences with the system a decision did not retain are not repeated: they follow from the decision itself (see `tests/references/DIVERGENCES.md`).

### Null emotion rows (2 Hz grid) where the systems disagree

| Case | Model | NucleusKit | Prime | Core |
|---|---|---|---|---|
| exg_disconnects | 2.6.0 | 13 | 13 | 3 |
| exg_disconnects | 2.5.0 | 13 | 13 | 3 |
| exg_near_zero | 2.6.0 | 3 | 15 | 3 |
| exg_near_zero | 2.5.0 | 3 | 15 | 3 |
| exg_dead_channel | 2.6.0 | 299 | 299 | 3 |
| exg_dead_channel | 2.5.0 | 299 | 299 | 3 |
| exg_inf_values | 2.6.0 | n/a | 299 | 23 |
| exg_inf_values | 2.5.0 | n/a | 299 | 23 |
| exg_full_scale | 2.6.0 | 3 | 3 | 35 |
| exg_full_scale | 2.5.0 | 3 | 3 | 35 |

### D1

- `features/cognition/epochMetrics.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: nan_frac: 150 of 150 rows (max abs diff 0.1)
- `features/cognition/epochMetrics.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: nan_frac: 150 of 150 rows (max abs diff 0.1)
- `features/cognition/powerBands.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: Timestamp: 2376 of 2376 rows (max abs diff 1.488e+05)
- `features/cognition/powerBands.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: Timestamp: 2376 of 2376 rows (max abs diff 1.488e+05)
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `timestamps_epoch_us`, against NucleusKit: Timestamp: 297 of 297 rows (max abs diff 1.488e+05)
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `timestamps_relative_ms`, against NucleusKit: Timestamp: 297 of 297 rows (max abs diff 1.488e+05)
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: Timestamp: 297 of 297 rows (max abs diff 1.488e+05)
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: Timestamp: 297 of 297 rows (max abs diff 1.488e+05)
- `features/ppg/ppg_resampled.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: summary: 7678921 rows against 7679 rows or columns differ
- `features/ppg/ppg_resampled.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: summary: 7679052 rows against 7680 rows or columns differ
- `results/Cognition.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: summary: 297981 rows against 299 rows or columns differ
- `results/Cognition.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: summary: 297954 rows against 299 rows or columns differ
- `results/HeartDynamics.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: only in core
- `results/HeartDynamics.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: only in core

### D2

- `features/emotions/avgRms.csv`, model 2.5.0, case `eeg_named_columns`, against Prime: PredictedLabel: 91 of 297 rows; PredictedConfidence: 297 of 297 rows (max abs diff 0.5416)
- `features/emotions/avgRms.csv`, model 2.6.0, case `eeg_named_columns`, against Prime: PredictedLabel: 127 of 297 rows; PredictedConfidence: 297 of 297 rows (max abs diff 0.7562)
- `results/Emotions.csv`, model 2.5.0, case `eeg_named_columns`, against Prime: Neutral: 290 of 299 rows (max abs diff 0.7634); Happiness: 289 of 299 rows (max abs diff 0.4345); Anger: 294 of 299 rows (max abs diff 0.7274); Surprise: 296 of 299 rows (max abs diff 0.6692); Contempt: 291 of 299 rows (max abs diff 0.5378); Disgust: 293 of 299 rows (max abs diff 0.9436); and 2 more
- `results/Emotions.csv`, model 2.6.0, case `eeg_named_columns`, against Prime: Neutral: 63 of 299 rows (max abs diff 1); Happiness: 10 of 299 rows (max abs diff 1); Anger: 58 of 299 rows (max abs diff 1); Surprise: 16 of 299 rows (max abs diff 1); Contempt: 99 of 299 rows (max abs diff 1); Disgust: 68 of 299 rows (max abs diff 1); and 1 more

### D3

- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `baseline`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `eeg_named_columns`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `emg_rms_threshold`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_dead_channel`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_disconnects`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_full_scale`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_near_zero`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_time_gaps`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `ppg_artifacts`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `ppg_dicrotic_notch`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `ppg_missing_beats`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `shimmer_gap`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `timestamps_epoch_ms`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `timestamps_epoch_us`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `timestamps_relative_ms`, against Prime: only in core

### D6

- `features/cognition/artefactStats.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: n_epochs_rejected: 1 of 1 rows (max abs diff 4); pct_epochs_rejected: 1 of 1 rows (max abs diff 2.7); n_rejected_by_peak_amp: 1 of 1 rows (max abs diff 4); n_samples_flagged: 1 of 1 rows (max abs diff 1000); pct_samples_flagged: 1 of 1 rows (max abs diff 2.7)
- `features/cognition/epochMetrics.csv`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: T9_peak_amp: 150 of 150 rows (max abs diff 1.752); T10_peak_amp: 150 of 150 rows (max abs diff 1.752); T9_ptp: 150 of 150 rows (max abs diff 1.84); T10_ptp: 150 of 150 rows (max abs diff 1.84)
- `features/cognition/epochMetrics.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: nan_frac: 4 of 150 rows (max abs diff 0.008); T9_peak_amp: 115 of 150 rows (max abs diff 2.25e+04); T10_peak_amp: 115 of 150 rows (max abs diff 2.25e+04); T9_ptp: 110 of 150 rows (max abs diff 2.62e+04); T10_ptp: 110 of 150 rows (max abs diff 2.62e+04); rejected: 4 of 150 rows (max abs diff 1); and 2 more
- `features/cognition/epochMetrics.csv`, model 2.6.0, case `exg_full_scale`, against NucleusKit: nan_frac: 4 of 150 rows (max abs diff 0.004); T9_peak_amp: 102 of 150 rows (max abs diff 61.22); T10_peak_amp: 102 of 150 rows (max abs diff 61.22); T9_ptp: 105 of 150 rows (max abs diff 72.5); T10_ptp: 105 of 150 rows (max abs diff 72.5)
- `features/cognition/epochMetrics.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: T9_peak_amp: 110 of 150 rows (max abs diff 2.012); T10_peak_amp: 110 of 150 rows (max abs diff 2.012); T9_ptp: 107 of 150 rows (max abs diff 2.432); T10_ptp: 107 of 150 rows (max abs diff 2.432)
- `features/cognition/powerBands.csv`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: power: 2376 of 2376 rows (max abs diff 0.981)
- `features/cognition/powerBands.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: power: 1752 of 2376 rows (max abs diff 3.709e+05)
- `features/cognition/powerBands.csv`, model 2.6.0, case `exg_full_scale`, against NucleusKit: power: 1832 of 2376 rows (max abs diff 30.43)
- `features/cognition/powerBands.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: power: 2120 of 2376 rows (max abs diff 0.0859)
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_dead_channel`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_disconnects`, against Prime: 288 rows against 297 rows; Timestamp: 211 of 288 rows (max abs diff 4.516); AvgRMS: 213 of 288 rows (max abs diff 1255); PredictedLabel: 194 of 288 rows; PredictedConfidence: 288 of 288 rows (max abs diff 0.6859)
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_full_scale`, against Prime: 297 rows against 270 rows; Timestamp: 193 of 270 rows (max abs diff 13.51); AvgRMS: 195 of 270 rows (max abs diff 1467); PredictedLabel: 163 of 270 rows; PredictedConfidence: 270 of 270 rows (max abs diff 0.6803)
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_near_zero`, against Prime: 287 rows against 297 rows; Timestamp: 241 of 287 rows (max abs diff 5.017); AvgRMS: 282 of 287 rows (max abs diff 2.779); PredictedLabel: 149 of 287 rows; PredictedConfidence: 287 of 287 rows (max abs diff 0.64)
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_single_invalid`, against Prime: AvgRMS: 29 of 297 rows (max abs diff 441.9); PredictedLabel: 14 of 297 rows; PredictedConfidence: 297 of 297 rows (max abs diff 0.415)
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_dead_channel`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_disconnects`, against Prime: 288 rows against 297 rows; Timestamp: 211 of 288 rows (max abs diff 4.516); AvgRMS: 215 of 288 rows (max abs diff 2382); PredictedLabel: 146 of 288 rows; PredictedConfidence: 288 of 288 rows (max abs diff 0.7155)
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_full_scale`, against Prime: 297 rows against 270 rows; Timestamp: 193 of 270 rows (max abs diff 13.51); AvgRMS: 198 of 270 rows (max abs diff 2393); PredictedLabel: 185 of 270 rows; PredictedConfidence: 270 of 270 rows (max abs diff 0.7024)
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_near_zero`, against Prime: 287 rows against 297 rows; Timestamp: 241 of 287 rows (max abs diff 5.017); AvgRMS: 282 of 287 rows (max abs diff 7.182); PredictedLabel: 168 of 287 rows; PredictedConfidence: 287 of 287 rows (max abs diff 0.8284)
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: AvgRMS: 50 of 297 rows (max abs diff 839.8); PredictedLabel: 22 of 297 rows; PredictedConfidence: 297 of 297 rows (max abs diff 0.6107)
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_dead_channel`, against NucleusKit: HEAD_R_RMS: 0 of 297 rows; HEAD_L_RMS: 0 of 297 rows; CHEEK_R_RMS: 0 of 297 rows; CHEEK_L_RMS: 0 of 297 rows; EAR_R_RMS: 0 of 297 rows; FOREHEAD_L_RMS: 0 of 297 rows; and 2 more
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_disconnects`, against NucleusKit: HEAD_R_RMS: 8 of 297 rows (max abs diff 4986); HEAD_L_RMS: 19 of 297 rows (max abs diff 4999); CHEEK_R_RMS: 4 of 297 rows (max abs diff 4128); CHEEK_L_RMS: 0 of 297 rows; EAR_R_RMS: 18 of 297 rows (max abs diff 5018); FOREHEAD_L_RMS: 0 of 297 rows; and 2 more
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_full_scale`, against NucleusKit: HEAD_R_RMS: 4 of 297 rows (max abs diff 4424); HEAD_L_RMS: 8 of 297 rows (max abs diff 4407); CHEEK_R_RMS: 6 of 297 rows (max abs diff 8282); CHEEK_L_RMS: 0 of 297 rows; EAR_R_RMS: 0 of 297 rows; FOREHEAD_L_RMS: 4 of 297 rows (max abs diff 71.16); and 2 more
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_single_invalid`, against NucleusKit: FOREHEAD_L_RMS: 29 of 297 rows (max abs diff 3533)
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: HEAD_R_RMS: 0 of 297 rows; HEAD_L_RMS: 0 of 297 rows; CHEEK_R_RMS: 0 of 297 rows; CHEEK_L_RMS: 0 of 297 rows; EAR_R_RMS: 0 of 297 rows; FOREHEAD_L_RMS: 0 of 297 rows; and 2 more
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: HEAD_R_RMS: 15 of 297 rows (max abs diff 9387); HEAD_L_RMS: 29 of 297 rows (max abs diff 9486); CHEEK_R_RMS: 9 of 297 rows (max abs diff 4213); CHEEK_L_RMS: 0 of 297 rows; EAR_R_RMS: 31 of 297 rows (max abs diff 9522); FOREHEAD_L_RMS: 0 of 297 rows; and 2 more
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_full_scale`, against NucleusKit: HEAD_R_RMS: 10 of 297 rows (max abs diff 4820); HEAD_L_RMS: 22 of 297 rows (max abs diff 4891); CHEEK_R_RMS: 11 of 297 rows (max abs diff 8453); CHEEK_L_RMS: 0 of 297 rows; EAR_R_RMS: 0 of 297 rows; FOREHEAD_L_RMS: 10 of 297 rows (max abs diff 468.8); and 2 more
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: FOREHEAD_L_RMS: 55 of 297 rows (max abs diff 6706)
- `results/Cognition.csv`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: Engagement: 296 of 299 rows (max abs diff 0.5913)
- `results/Cognition.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: Engagement: 216 of 299 rows (max abs diff 1.165)
- `results/Cognition.csv`, model 2.6.0, case `exg_full_scale`, against NucleusKit: Engagement: 228 of 299 rows (max abs diff 0.4411)
- `results/Cognition.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: Engagement: 262 of 299 rows (max abs diff 0.3598)
- `results/Emotions.csv`, model 2.5.0, case `exg_dead_channel`, against Prime: Neutral: 0 of 299 rows; Happiness: 0 of 299 rows; Anger: 0 of 299 rows; Surprise: 0 of 299 rows; Contempt: 0 of 299 rows; Disgust: 0 of 299 rows; and 2 more
- `results/Emotions.csv`, model 2.5.0, case `exg_disconnects`, against Prime: Neutral: 285 of 299 rows (max abs diff 0.7509); Happiness: 286 of 299 rows (max abs diff 0.9864); Anger: 286 of 299 rows (max abs diff 0.3008); Surprise: 286 of 299 rows (max abs diff 0.5895); Contempt: 286 of 299 rows (max abs diff 0.9864); Disgust: 286 of 299 rows (max abs diff 0.4592); and 2 more
- `results/Emotions.csv`, model 2.5.0, case `exg_full_scale`, against Prime: Neutral: 261 of 299 rows (max abs diff 0.7234); Happiness: 261 of 299 rows (max abs diff 0.995); Anger: 262 of 299 rows (max abs diff 0.6697); Surprise: 264 of 299 rows (max abs diff 0.3939); Contempt: 261 of 299 rows (max abs diff 0.9949); Disgust: 261 of 299 rows (max abs diff 0.3018); and 2 more
- `results/Emotions.csv`, model 2.5.0, case `exg_near_zero`, against Prime: Neutral: 275 of 299 rows (max abs diff 0.3529); Happiness: 274 of 299 rows (max abs diff 0.245); Anger: 281 of 299 rows (max abs diff 0.0379); Surprise: 283 of 299 rows (max abs diff 0.182); Contempt: 281 of 299 rows (max abs diff 0.1593); Disgust: 282 of 299 rows (max abs diff 0.01743); and 2 more
- `results/Emotions.csv`, model 2.5.0, case `exg_single_invalid`, against Prime: Neutral: 287 of 299 rows (max abs diff 0.808); Happiness: 282 of 299 rows (max abs diff 0.2363); Anger: 287 of 299 rows (max abs diff 0.5091); Surprise: 296 of 299 rows (max abs diff 0.4969); Contempt: 284 of 299 rows (max abs diff 0.958); Disgust: 288 of 299 rows (max abs diff 0.1908); and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_dead_channel`, against Prime: Neutral: 0 of 299 rows; Happiness: 0 of 299 rows; Anger: 0 of 299 rows; Surprise: 0 of 299 rows; Contempt: 0 of 299 rows; Disgust: 0 of 299 rows; and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_disconnects`, against Prime: Neutral: 57 of 299 rows (max abs diff 1); Happiness: 35 of 299 rows (max abs diff 1); Anger: 8 of 299 rows (max abs diff 1); Surprise: 2 of 299 rows (max abs diff 0.9393); Contempt: 18 of 299 rows (max abs diff 1); Disgust: 6 of 299 rows (max abs diff 1); and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_full_scale`, against Prime: Neutral: 57 of 299 rows (max abs diff 1); Happiness: 12 of 299 rows (max abs diff 1); Anger: 23 of 299 rows (max abs diff 1); Surprise: 10 of 299 rows (max abs diff 1); Contempt: 40 of 299 rows (max abs diff 1); Disgust: 23 of 299 rows (max abs diff 1); and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_near_zero`, against Prime: Neutral: 6 of 299 rows (max abs diff 0.9614); Happiness: 2 of 299 rows (max abs diff 0.4754); Anger: 6 of 299 rows (max abs diff 0.9562); Surprise: 0 of 299 rows; Contempt: 2 of 299 rows (max abs diff 0.951); Disgust: 0 of 299 rows; and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: Neutral: 15 of 299 rows (max abs diff 1); Anger: 16 of 299 rows (max abs diff 1); Surprise: 6 of 299 rows (max abs diff 0.9739); Contempt: 4 of 299 rows (max abs diff 0.9581); Disgust: 10 of 299 rows (max abs diff 1); Sadness: 19 of 299 rows (max abs diff 1)

### D7

- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_single_invalid`, against Prime: AvgRMS: 29 of 297 rows (max abs diff 441.9); PredictedLabel: 14 of 297 rows; PredictedConfidence: 297 of 297 rows (max abs diff 0.415)
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: AvgRMS: 50 of 297 rows (max abs diff 839.8); PredictedLabel: 22 of 297 rows; PredictedConfidence: 297 of 297 rows (max abs diff 0.6107)
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_single_invalid`, against NucleusKit: FOREHEAD_L_RMS: 29 of 297 rows (max abs diff 3533)
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: FOREHEAD_L_RMS: 55 of 297 rows (max abs diff 6706)
- `results/Emotions.csv`, model 2.5.0, case `exg_inf_values`, against Prime: Neutral: 0 of 299 rows; Happiness: 0 of 299 rows; Anger: 0 of 299 rows; Surprise: 0 of 299 rows; Contempt: 0 of 299 rows; Disgust: 0 of 299 rows; and 2 more
- `results/Emotions.csv`, model 2.5.0, case `exg_single_invalid`, against Prime: Neutral: 287 of 299 rows (max abs diff 0.808); Happiness: 282 of 299 rows (max abs diff 0.2363); Anger: 287 of 299 rows (max abs diff 0.5091); Surprise: 296 of 299 rows (max abs diff 0.4969); Contempt: 284 of 299 rows (max abs diff 0.958); Disgust: 288 of 299 rows (max abs diff 0.1908); and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_inf_values`, against Prime: Neutral: 0 of 299 rows; Happiness: 0 of 299 rows; Anger: 0 of 299 rows; Surprise: 0 of 299 rows; Contempt: 0 of 299 rows; Disgust: 0 of 299 rows; and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: Neutral: 15 of 299 rows (max abs diff 1); Anger: 16 of 299 rows (max abs diff 1); Surprise: 6 of 299 rows (max abs diff 0.9739); Contempt: 4 of 299 rows (max abs diff 0.9581); Disgust: 10 of 299 rows (max abs diff 1); Sadness: 19 of 299 rows (max abs diff 1)

### D8

- `features/emotions/avgRms.csv`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `results/Emotions.csv`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `results/Emotions.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core

### D9

- `results/Emotions.csv`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `results/Emotions.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core

### D10

- `features/emotions/avgRms.csv`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `baseline`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 2 of 297 rows (max abs diff 0.01521); HEAD_R_RMS: 297 of 297 rows (max abs diff 4.472); HEAD_L_RMS: 297 of 297 rows (max abs diff 4.941); CHEEK_R_RMS: 297 of 297 rows (max abs diff 5.876); CHEEK_L_RMS: 297 of 297 rows (max abs diff 6.013); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `baseline`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `eeg_named_columns`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 2 of 297 rows (max abs diff 0.2235); HEAD_R_RMS: 297 of 297 rows (max abs diff 7.284); HEAD_L_RMS: 297 of 297 rows (max abs diff 7.689); CHEEK_R_RMS: 297 of 297 rows (max abs diff 7.841); CHEEK_L_RMS: 297 of 297 rows (max abs diff 6.298); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `eeg_named_columns`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); rms_AF8: 297 of 297 rows (max abs diff 1.261); rms_AF7: 297 of 297 rows (max abs diff 1.261); rms_CHEEK_R: 297 of 297 rows (max abs diff 3.97); rms_CHEEK_L: 297 of 297 rows (max abs diff 3.97); rms_EAR_R: 297 of 297 rows (max abs diff 3.463); and 22 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `eeg_truncated_last_line`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 1 of 296 rows (max abs diff 0.01224); HEAD_R_RMS: 296 of 296 rows (max abs diff 5.058); HEAD_L_RMS: 296 of 296 rows (max abs diff 5.294); CHEEK_R_RMS: 296 of 296 rows (max abs diff 8.931); CHEEK_L_RMS: 296 of 296 rows (max abs diff 8.413); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `emg_rms_threshold`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 5 of 297 rows (max abs diff 0.04538); HEAD_R_RMS: 297 of 297 rows (max abs diff 29.22); HEAD_L_RMS: 297 of 297 rows (max abs diff 28.77); CHEEK_R_RMS: 297 of 297 rows (max abs diff 28.36); CHEEK_L_RMS: 297 of 297 rows (max abs diff 27.35); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `emg_rms_threshold`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_dead_channel`, against NucleusKit: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_dead_channel`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_disconnects`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; 288 rows against 297 rows; Timestamp: 211 of 288 rows (max abs diff 4.516); rms_AF8: 288 of 288 rows (max abs diff 8.68); rms_AF7: 288 of 288 rows (max abs diff 7.297); rms_CHEEK_R: 288 of 288 rows (max abs diff 6.7); and 32 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_disconnects`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 288 rows against 297 rows; Timestamp: 211 of 288 rows (max abs diff 4.516); rms_AF8: 288 of 288 rows (max abs diff 8.68); rms_AF7: 288 of 288 rows (max abs diff 7.31); rms_CHEEK_R: 288 of 288 rows (max abs diff 6.7); and 25 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_full_scale`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; 297 rows against 270 rows; Timestamp: 193 of 270 rows (max abs diff 13.51); rms_AF8: 270 of 270 rows (max abs diff 16.5); rms_AF7: 270 of 270 rows (max abs diff 7.661); rms_CHEEK_R: 270 of 270 rows (max abs diff 6.61); and 32 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_full_scale`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 297 rows against 270 rows; Timestamp: 193 of 270 rows (max abs diff 13.51); rms_AF8: 270 of 270 rows (max abs diff 16.5); rms_AF7: 270 of 270 rows (max abs diff 7.661); rms_CHEEK_R: 270 of 270 rows (max abs diff 6.61); and 25 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_near_zero`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 3 of 297 rows (max abs diff 0.06262); HEAD_R_RMS: 297 of 297 rows (max abs diff 8.525); HEAD_L_RMS: 297 of 297 rows (max abs diff 7.412); CHEEK_R_RMS: 297 of 297 rows (max abs diff 9.027); CHEEK_L_RMS: 297 of 297 rows (max abs diff 8.928); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_near_zero`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 287 rows against 297 rows; Timestamp: 241 of 287 rows (max abs diff 5.017); rms_AF8: 287 of 287 rows (max abs diff 4.795); rms_AF7: 287 of 287 rows (max abs diff 4.164); rms_CHEEK_R: 287 of 287 rows (max abs diff 4.778); and 24 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_single_invalid`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; rms_AFz: 297 of 297 rows (max abs diff 1.649); wl_AFz: 297 of 297 rows (max abs diff 4.691); mdf_AFz: 25 of 297 rows (max abs diff 13.35); ant_corr: 32 of 297 rows (max abs diff 0.5759); nm_ratio_a: 30 of 297 rows (max abs diff 1.041); and 10 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_single_invalid`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); rms_AFz: 297 of 297 rows (max abs diff 1.89); wl_AFz: 297 of 297 rows (max abs diff 4.405); mdf_AFz: 25 of 297 rows (max abs diff 13.35); ant_corr: 32 of 297 rows (max abs diff 0.5759); nm_ratio_a: 30 of 297 rows (max abs diff 1.041); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_time_gaps`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 4 of 282 rows (max abs diff 0.1127); HEAD_R_RMS: 282 of 282 rows (max abs diff 6.381); HEAD_L_RMS: 282 of 282 rows (max abs diff 6.778); CHEEK_R_RMS: 282 of 282 rows (max abs diff 8.952); CHEEK_L_RMS: 282 of 282 rows (max abs diff 8.339); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `exg_time_gaps`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `legacy_file_names`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 3 of 297 rows (max abs diff 0.08373); HEAD_R_RMS: 297 of 297 rows (max abs diff 8.594); HEAD_L_RMS: 297 of 297 rows (max abs diff 7.691); CHEEK_R_RMS: 297 of 297 rows (max abs diff 8.505); CHEEK_L_RMS: 297 of 297 rows (max abs diff 8.2); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `legacy_file_names`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `ppg_artifacts`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 2 of 297 rows (max abs diff 0.06731); HEAD_R_RMS: 297 of 297 rows (max abs diff 4.967); HEAD_L_RMS: 297 of 297 rows (max abs diff 5.56); CHEEK_R_RMS: 297 of 297 rows (max abs diff 7.361); CHEEK_L_RMS: 297 of 297 rows (max abs diff 6.747); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `ppg_artifacts`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `ppg_dicrotic_notch`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 4 of 297 rows (max abs diff 0.1009); HEAD_R_RMS: 297 of 297 rows (max abs diff 5.443); HEAD_L_RMS: 297 of 297 rows (max abs diff 6.318); CHEEK_R_RMS: 297 of 297 rows (max abs diff 8.395); CHEEK_L_RMS: 297 of 297 rows (max abs diff 8.25); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `ppg_dicrotic_notch`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `ppg_missing_beats`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 3 of 297 rows (max abs diff 0.02513); HEAD_R_RMS: 297 of 297 rows (max abs diff 7.482); HEAD_L_RMS: 297 of 297 rows (max abs diff 6.959); CHEEK_R_RMS: 297 of 297 rows (max abs diff 7.975); CHEEK_L_RMS: 297 of 297 rows (max abs diff 7.245); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `ppg_missing_beats`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `shimmer_gap`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 2 of 297 rows (max abs diff 0.05405); HEAD_R_RMS: 297 of 297 rows (max abs diff 7.256); HEAD_L_RMS: 297 of 297 rows (max abs diff 7.884); CHEEK_R_RMS: 297 of 297 rows (max abs diff 7.768); CHEEK_L_RMS: 297 of 297 rows (max abs diff 7.531); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `shimmer_gap`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `timestamps_epoch_ms`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; PredictedConfidence: 1 of 297 rows (max abs diff 0.04913); HEAD_R_RMS: 297 of 297 rows (max abs diff 8.227); HEAD_L_RMS: 297 of 297 rows (max abs diff 7.088); CHEEK_R_RMS: 297 of 297 rows (max abs diff 7.311); CHEEK_L_RMS: 297 of 297 rows (max abs diff 6.505); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `timestamps_epoch_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `timestamps_epoch_us`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; Timestamp: 297 of 297 rows (max abs diff 1.488e+05); PredictedConfidence: 2 of 297 rows (max abs diff 0.02162); HEAD_R_RMS: 297 of 297 rows (max abs diff 4.717); HEAD_L_RMS: 297 of 297 rows (max abs diff 5.867); CHEEK_R_RMS: 297 of 297 rows (max abs diff 6.056); and 5 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `timestamps_epoch_us`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `timestamps_relative_ms`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; Timestamp: 297 of 297 rows (max abs diff 1.488e+05); PredictedConfidence: 1 of 297 rows (max abs diff 0.01854); HEAD_R_RMS: 297 of 297 rows (max abs diff 6.775); HEAD_L_RMS: 297 of 297 rows (max abs diff 6.694); CHEEK_R_RMS: 297 of 297 rows (max abs diff 7.304); and 5 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.5.0, case `timestamps_relative_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `baseline`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 10.31); HEAD_L_RMS: 297 of 297 rows (max abs diff 10.73); CHEEK_R_RMS: 297 of 297 rows (max abs diff 13.14); CHEEK_L_RMS: 297 of 297 rows (max abs diff 12.62); EAR_R_RMS: 297 of 297 rows (max abs diff 2.493); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `baseline`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `eeg_named_columns`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 15.48); HEAD_L_RMS: 297 of 297 rows (max abs diff 15.5); CHEEK_R_RMS: 297 of 297 rows (max abs diff 16.38); CHEEK_L_RMS: 297 of 297 rows (max abs diff 14.35); EAR_R_RMS: 297 of 297 rows (max abs diff 3.707); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `eeg_named_columns`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); rms_AF8: 297 of 297 rows (max abs diff 0.09481); rms_AF7: 297 of 297 rows (max abs diff 0.09402); rms_CHEEK_R: 297 of 297 rows (max abs diff 1.085); rms_CHEEK_L: 297 of 297 rows (max abs diff 1.097); rms_EAR_R: 297 of 297 rows (max abs diff 0.97); and 22 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `eeg_truncated_last_line`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 296 of 296 rows (max abs diff 10.77); HEAD_L_RMS: 296 of 296 rows (max abs diff 11.22); CHEEK_R_RMS: 296 of 296 rows (max abs diff 17.14); CHEEK_L_RMS: 296 of 296 rows (max abs diff 17.02); EAR_R_RMS: 296 of 296 rows (max abs diff 2.25); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `emg_rms_threshold`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 28.72); HEAD_L_RMS: 297 of 297 rows (max abs diff 28.05); CHEEK_R_RMS: 297 of 297 rows (max abs diff 27.96); CHEEK_L_RMS: 297 of 297 rows (max abs diff 26.86); EAR_R_RMS: 297 of 297 rows (max abs diff 29.1); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `emg_rms_threshold`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_dead_channel`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; 288 rows against 297 rows; Timestamp: 211 of 288 rows (max abs diff 4.516); rms_AF8: 288 of 288 rows (max abs diff 1.463); rms_AF7: 288 of 288 rows (max abs diff 1.665); rms_CHEEK_R: 288 of 288 rows (max abs diff 1.181); and 32 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_disconnects`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 288 rows against 297 rows; Timestamp: 211 of 288 rows (max abs diff 4.516); rms_AF8: 288 of 288 rows (max abs diff 1.463); rms_AF7: 288 of 288 rows (max abs diff 1.693); rms_CHEEK_R: 288 of 288 rows (max abs diff 1.181); and 25 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_full_scale`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; 297 rows against 270 rows; Timestamp: 193 of 270 rows (max abs diff 13.51); rms_AF8: 270 of 270 rows (max abs diff 48); rms_AF7: 270 of 270 rows (max abs diff 1.555); rms_CHEEK_R: 270 of 270 rows (max abs diff 1.195); and 32 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_full_scale`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 297 rows against 270 rows; Timestamp: 193 of 270 rows (max abs diff 13.51); rms_AF8: 270 of 270 rows (max abs diff 48); rms_AF7: 270 of 270 rows (max abs diff 1.555); rms_CHEEK_R: 270 of 270 rows (max abs diff 1.195); and 25 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_near_zero`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 16.62); HEAD_L_RMS: 297 of 297 rows (max abs diff 16.88); CHEEK_R_RMS: 297 of 297 rows (max abs diff 17.05); CHEEK_L_RMS: 297 of 297 rows (max abs diff 18.22); EAR_R_RMS: 297 of 297 rows (max abs diff 4.109); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_near_zero`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 287 rows against 297 rows; Timestamp: 241 of 287 rows (max abs diff 5.017); rms_AF8: 287 of 287 rows (max abs diff 1.086); rms_AF7: 287 of 287 rows (max abs diff 1.077); rms_CHEEK_R: 287 of 287 rows (max abs diff 1.202); and 24 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; rms_AFz: 297 of 297 rows (max abs diff 0.5394); wl_AFz: 297 of 297 rows (max abs diff 1.344); mdf_AFz: 46 of 297 rows (max abs diff 15.65); ant_corr: 62 of 297 rows (max abs diff 1.489); nm_ratio_a: 57 of 297 rows (max abs diff 1.102); and 10 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); rms_AFz: 297 of 297 rows (max abs diff 1.028); wl_AFz: 297 of 297 rows (max abs diff 1.344); mdf_AFz: 46 of 297 rows (max abs diff 21.31); ant_corr: 62 of 297 rows (max abs diff 1.489); nm_ratio_a: 57 of 297 rows (max abs diff 1.102); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_time_gaps`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 282 of 282 rows (max abs diff 13.04); HEAD_L_RMS: 282 of 282 rows (max abs diff 13.65); CHEEK_R_RMS: 282 of 282 rows (max abs diff 18.63); CHEEK_L_RMS: 282 of 282 rows (max abs diff 18.38); EAR_R_RMS: 282 of 282 rows (max abs diff 3.078); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `exg_time_gaps`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `legacy_file_names`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 17.71); HEAD_L_RMS: 297 of 297 rows (max abs diff 16.8); CHEEK_R_RMS: 297 of 297 rows (max abs diff 16.28); CHEEK_L_RMS: 297 of 297 rows (max abs diff 15.52); EAR_R_RMS: 297 of 297 rows (max abs diff 4.049); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `legacy_file_names`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `ppg_artifacts`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 11.32); HEAD_L_RMS: 297 of 297 rows (max abs diff 11.34); CHEEK_R_RMS: 297 of 297 rows (max abs diff 15.77); CHEEK_L_RMS: 297 of 297 rows (max abs diff 14.7); EAR_R_RMS: 297 of 297 rows (max abs diff 2.856); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `ppg_artifacts`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `ppg_dicrotic_notch`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 13.7); HEAD_L_RMS: 297 of 297 rows (max abs diff 13.46); CHEEK_R_RMS: 297 of 297 rows (max abs diff 16.45); CHEEK_L_RMS: 297 of 297 rows (max abs diff 16.69); EAR_R_RMS: 297 of 297 rows (max abs diff 3.255); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `ppg_dicrotic_notch`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `ppg_missing_beats`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 15.92); HEAD_L_RMS: 297 of 297 rows (max abs diff 14.87); CHEEK_R_RMS: 297 of 297 rows (max abs diff 14.34); CHEEK_L_RMS: 297 of 297 rows (max abs diff 14.13); EAR_R_RMS: 297 of 297 rows (max abs diff 3.5); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `ppg_missing_beats`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `shimmer_gap`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 15.78); HEAD_L_RMS: 297 of 297 rows (max abs diff 15.55); CHEEK_R_RMS: 297 of 297 rows (max abs diff 17.1); CHEEK_L_RMS: 297 of 297 rows (max abs diff 17.15); EAR_R_RMS: 297 of 297 rows (max abs diff 3.62); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `shimmer_gap`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `timestamps_epoch_ms`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; HEAD_R_RMS: 297 of 297 rows (max abs diff 15.95); HEAD_L_RMS: 297 of 297 rows (max abs diff 16.49); CHEEK_R_RMS: 297 of 297 rows (max abs diff 15.58); CHEEK_L_RMS: 297 of 297 rows (max abs diff 15.22); EAR_R_RMS: 297 of 297 rows (max abs diff 4.094); and 3 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `timestamps_epoch_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; Timestamp: 297 of 297 rows (max abs diff 1.488e+05); HEAD_R_RMS: 297 of 297 rows (max abs diff 10.82); HEAD_L_RMS: 297 of 297 rows (max abs diff 10.46); CHEEK_R_RMS: 297 of 297 rows (max abs diff 13.11); CHEEK_L_RMS: 297 of 297 rows (max abs diff 4.683); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `timestamps_epoch_us`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: columns only in the reference: none; only in the core: HighRmsArtifact; Timestamp: 297 of 297 rows (max abs diff 1.488e+05); HEAD_R_RMS: 297 of 297 rows (max abs diff 14.53); HEAD_L_RMS: 297 of 297 rows (max abs diff 15.1); CHEEK_R_RMS: 297 of 297 rows (max abs diff 16.64); CHEEK_L_RMS: 297 of 297 rows (max abs diff 9.443); and 4 more
- `features/emotions/emotionClassifierInputs.csv`, model 2.6.0, case `timestamps_relative_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `baseline`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `baseline`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `eeg_named_columns`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `eeg_named_columns`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 21 rows against 26 rows; Timestamp: 21 of 21 rows (max abs diff 50.5); PredictedConfidence: 21 of 21 rows (max abs diff 0.09245); OriginalLabel: 13 of 21 rows; OriginalConfidence: 21 of 21 rows (max abs diff 0.09245)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `eeg_truncated_last_line`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `emg_rms_threshold`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `emg_rms_threshold`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_dead_channel`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_dead_channel`, against Prime: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_disconnects`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_disconnects`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 22 rows against 70 rows; Timestamp: 22 of 22 rows (max abs diff 66); PredictedLabel: 5 of 22 rows; PredictedConfidence: 22 of 22 rows (max abs diff 0.4646); OriginalLabel: 16 of 22 rows; and 2 more
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_full_scale`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_full_scale`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 92 rows against 19 rows; Timestamp: 19 of 19 rows (max abs diff 51.01); PredictedLabel: 11 of 19 rows; PredictedConfidence: 19 of 19 rows (max abs diff 0.5149); OriginalLabel: 17 of 19 rows; and 2 more
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_near_zero`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_near_zero`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 16 rows against 15 rows; Timestamp: 13 of 15 rows (max abs diff 18.49); PredictedConfidence: 15 of 15 rows (max abs diff 0.06181); OriginalLabel: 9 of 15 rows; OriginalConfidence: 15 of 15 rows (max abs diff 0.06181)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_single_invalid`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_single_invalid`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 32 rows against 47 rows; Timestamp: 32 of 32 rows (max abs diff 98.49); PredictedLabel: 8 of 32 rows; PredictedConfidence: 32 of 32 rows (max abs diff 0.4382); OriginalLabel: 27 of 32 rows; and 2 more
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_time_gaps`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `exg_time_gaps`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `legacy_file_names`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `legacy_file_names`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `ppg_artifacts`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `ppg_artifacts`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `ppg_dicrotic_notch`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `ppg_dicrotic_notch`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `ppg_missing_beats`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `ppg_missing_beats`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `shimmer_gap`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `shimmer_gap`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `timestamps_epoch_ms`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `timestamps_epoch_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `timestamps_epoch_us`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `timestamps_epoch_us`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `timestamps_relative_ms`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.5.0, case `timestamps_relative_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `baseline`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `baseline`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `eeg_named_columns`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `eeg_named_columns`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `eeg_truncated_last_line`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `emg_rms_threshold`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `emg_rms_threshold`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_dead_channel`, against Prime: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_disconnects`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_disconnects`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 5 rows against 31 rows; Timestamp: 5 of 5 rows (max abs diff 41.01); PredictedLabel: 5 of 5 rows; PredictedConfidence: 5 of 5 rows (max abs diff 0.388); OriginalLabel: 5 of 5 rows; and 1 more
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_full_scale`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_full_scale`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 35 rows against 1 rows; Timestamp: 1 of 1 rows (max abs diff 42.49); PredictedLabel: 1 of 1 rows; PredictedConfidence: 1 of 1 rows (max abs diff 0.7585); OriginalLabel: 1 of 1 rows; and 1 more
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_near_zero`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_near_zero`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …); 0 rows against 20 rows
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_time_gaps`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `exg_time_gaps`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `legacy_file_names`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `legacy_file_names`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `ppg_artifacts`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `ppg_artifacts`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `ppg_dicrotic_notch`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `ppg_dicrotic_notch`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `ppg_missing_beats`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `ppg_missing_beats`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `shimmer_gap`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `shimmer_gap`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `timestamps_epoch_ms`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `timestamps_epoch_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `timestamps_epoch_us`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `timestamps_relative_ms`, against NucleusKit: only in core
- `features/emotions/relabeled-samples.csv`, model 2.6.0, case `timestamps_relative_ms`, against Prime: columns only in the reference: 8 columns (Supraorbital_L_RMS, …); only in the core: 8 columns (HEAD_R_RMS, …)
- `results/emotion_model.json`, model 2.5.0, case `eeg_truncated_last_line`, against Prime: only in core
- `results/emotion_model.json`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core

### D16

- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `baseline`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `eeg_named_columns`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `emg_rms_threshold`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_dead_channel`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_disconnects`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_full_scale`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_near_zero`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_single_invalid`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `exg_time_gaps`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `ppg_artifacts`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `ppg_dicrotic_notch`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `ppg_missing_beats`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `shimmer_gap`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `timestamps_epoch_ms`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `timestamps_epoch_us`, against Prime: only in core
- `features/events/processedFeatureEvents.csv`, model 2.6.0, case `timestamps_relative_ms`, against Prime: only in core

### D24

- `features/cognition/artefactStats.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: n_epochs_rejected: 1 of 1 rows (max abs diff 117); pct_epochs_rejected: 1 of 1 rows (max abs diff 78); n_rejected_by_nan: 1 of 1 rows (max abs diff 117); n_samples_flagged: 1 of 1 rows (max abs diff 2.925e+04); pct_samples_flagged: 1 of 1 rows (max abs diff 78)
- `features/cognition/epochMetrics.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: nan_frac: 117 of 150 rows (max abs diff 1); T9_peak_amp: 122 of 150 rows (max abs diff 13.43); T10_peak_amp: 122 of 150 rows (max abs diff 13.43); T9_ptp: 120 of 150 rows (max abs diff 15.81); T10_ptp: 120 of 150 rows (max abs diff 15.81); rejected: 117 of 150 rows (max abs diff 1); and 1 more
- `features/cognition/powerBands.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: power: 76 of 2376 rows (max abs diff 0.01727)
- `features/emotions/avgRms.csv`, model 2.5.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/avgRms.csv`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/emotions/rmsSignals.csv`, model 2.5.0, case `exg_inf_values`, against NucleusKit: only in core
- `features/emotions/rmsSignals.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: only in core
- `results/Cognition.csv`, model 2.6.0, case `exg_inf_values`, against NucleusKit: 66 rows against 299 rows; Engagement: 7 of 66 rows (max abs diff 2.31e-05)
- `results/Emotions.csv`, model 2.5.0, case `exg_inf_values`, against Prime: Neutral: 0 of 299 rows; Happiness: 0 of 299 rows; Anger: 0 of 299 rows; Surprise: 0 of 299 rows; Contempt: 0 of 299 rows; Disgust: 0 of 299 rows; and 2 more
- `results/Emotions.csv`, model 2.6.0, case `exg_inf_values`, against Prime: Neutral: 0 of 299 rows; Happiness: 0 of 299 rows; Anger: 0 of 299 rows; Surprise: 0 of 299 rows; Contempt: 0 of 299 rows; Disgust: 0 of 299 rows; and 2 more

### D25

- `features/metainfo.json`, model 2.6.0, case `baseline`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `baseline`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `eeg_named_columns`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `eeg_named_columns`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `eeg_truncated_last_line`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `eeg_truncated_last_line`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `emg_rms_threshold`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `emg_rms_threshold`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_dead_channel`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_dead_channel`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_disconnects`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_disconnects`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_full_scale`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_full_scale`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_inf_values`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_inf_values`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_near_zero`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_near_zero`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_single_invalid`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_single_invalid`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `exg_time_gaps`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `exg_time_gaps`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `legacy_file_names`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `ppg_artifacts`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `ppg_artifacts`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `ppg_dicrotic_notch`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `ppg_dicrotic_notch`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `ppg_missing_beats`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `ppg_missing_beats`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `shimmer_gap`, against NucleusKit: duration: '0.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `shimmer_gap`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `timestamps_epoch_ms`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `timestamps_epoch_us`, against NucleusKit: duration: '149979.0' -> '150.0'
- `features/metainfo.json`, model 2.6.0, case `timestamps_epoch_us`, against Prime: only in core
- `features/metainfo.json`, model 2.6.0, case `timestamps_relative_ms`, against Prime: only in core

<!-- GENERATED:END -->
