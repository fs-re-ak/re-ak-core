# re-ak-core processing overview

`reak_core` turns one session folder into engagement, emotions, arousal and heart dynamics on a shared 2 Hz timebase. It is the processing used by the Nucleus-Kit toolkit since 2.0.0 and by the re-akcas cloud processor since Prime 4.0.0.

Install, the `reak-core` command and extension entry points are in the [README](../README.md). Every behaviour change relative to Nucleus-Kit 1.1 and Prime 3.1.0 is in [CHANGELOG.md](../CHANGELOG.md).

## Default pipeline

`Pipeline.default()` runs these steps, then the emotions report. A failing step is recorded and the next step still runs. A step that lists outputs is skipped when those files already exist, unless `force=True` (`reak-core --force`).

| Step | Module | Primary output |
|---|---|---|
| `prepare` | `reak_core.io.session` | Output folders and `results/processingLog.txt` |
| `metainfo` | `reak_core.io.meta` | `features/metainfo.json` |
| `heart_dynamics` | `reak_core.ppg` | `results/HeartDynamics.csv` |
| `arousal` | `reak_core.eda` | `results/Arousal.csv` |
| `cognition` | `reak_core.eeg` | `results/Cognition.csv` |
| `emotions` | `reak_core.femg` | `results/Emotions.csv` |
| `feature_events` | `reak_core.sync.events` | `features/events/processedFeatureEvents.csv` and `processedWebFeatureEvents.csv` |
| `playback_annotations` | `reak_core.sync.events` | `features/events/playback_annotations.json`, written once |
| `emotions_report` | `reak_core.exporters` | `features/emotions/report_jitter_<emotion>.png`, `report_emotion_distribution.png` |

`results/processing.json` is written at the end of every run: core version, resolved emotion model, NLMS flag, each step's status, and a signal-quality summary.

Nucleus-Kit adds POV conversion, video rotation and GPS around this list. GPS and video are not part of the core. The core reads `rawData/gps.csv` and `rawData/uwb.csv` only to report timing and quality. It does not write `gpsDf.csv`, `positions.csv` or `zones.csv`.

## Session layout

A session root holds:

| Path | Role |
|---|---|
| `rawData/` | Inputs. When this folder is missing, `Session.open` creates it and moves every other top-level entry into it, except `features`, `results`, `meta` and `experimentConfigs` |
| `features/` | Intermediate and diagnostic files. `features/metainfo.json` is the session summary |
| `results/` | Tables on the shared timebase, plus `emotion_model.json` and `processing.json` |
| `meta/`, `experimentConfigs/` | Created by `prepare`. The core does not write `meta/metainfo.json` |

Raw files, first match wins:

| Stream | Names under `rawData/` |
|---|---|
| EXG (EEG and fEMG) | `rawEEG_0.csv`, `eeg.tmp`, `eeg.csv`, `eegRec_0.csv` |
| EDA | `shimmer.csv` column 4, `rawShimmer_0.csv` column 4, `gsr.tmp`, `gsr.csv` |
| PPG | `shimmer.csv` column 5, `rawShimmer_0.csv` column 5, `ppg.tmp`, `ppg.csv` |
| Events | `rawEvents.csv`, `events_0.csv`, `event.csv` |

A header row is optional. Named EXG columns are reordered into hardware order. A truncated last line is ignored. Timestamps in epoch microseconds, epoch nanoseconds and relative clocks are normalised to seconds from the first sample.

## Timebase

`results/*.csv` uses a 0.5 s grid (`Timestamp` in seconds from the start of the recording). Continuous signals (heart, arousal, cognition) are placed on that grid with linear interpolation that does not cross a gap. `Emotions.csv` is placed with nearest neighbour, so one-hot labels are copied and not blended. Hardware gaps of 5 s or more stay missing.

## Invalid EXG samples

A raw sample is invalid when it is non-finite (`NaN`, `inf`, `-inf`) or at ADC full scale: absolute value within 0.1 µV of 375 000 µV. Samples near zero, and the older full-scale value 187 500 µV, are valid.

Invalid samples are interpolated only so the filters can run, then set back to missing. For emotions, a 2.0 s window that contains any missing sample is null and its row in `Emotions.csv` is `NULL`. Cognition judges sample validity on the channels it uses: a dead cheek, brow or nose electrode does not by itself blank Engagement.

## Pages

| File | Topic |
|---|---|
| [pipeline_heart.md](pipeline_heart.md) | PPG and heart-rate variability |
| [pipeline_arousal.md](pipeline_arousal.md) | EDA, cvxEDA, SCR events |
| [pipeline_cognition.md](pipeline_cognition.md) | EEG Engagement |
| [pipeline_emotions.md](pipeline_emotions.md) | fEMG classifier, default model 2.6.0 |
| [pipeline_events.md](pipeline_events.md) | Feature events and playback annotations |
