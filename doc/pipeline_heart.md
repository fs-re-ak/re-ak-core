# Heart dynamics — Shimmer PPG

**Module:** `reak_core.ppg.heart`  
**Entry point:** `compute_heart_dynamics(session_root)`  
**Primary output:** `results/HeartDynamics.csv`

Time-domain heart-rate variability from the Shimmer PPG channel, on the shared 2 Hz timebase. Decisions D11–D15 keep Nucleus-Kit's artifact rejection, normalisation, missed-beat filling and HRV. Timestamp units follow decision D1: epoch microseconds, epoch nanoseconds and relative clocks are read as seconds, not as a value 1000 times too long.

## Input

First file that exists under `rawData/`:

| File | PPG column |
|---|---|
| `shimmer.csv` | 5 |
| `rawShimmer_0.csv` | 5 |
| `ppg.tmp` | 1 |
| `ppg.csv` | 1 |

Column 0 is the hardware timestamp. A header row is optional. Nominal rate: 51.2 Hz.

## Processing

The procedure and its constants live in `reak_core.ppg.heart`.

1. Resample onto a strict 51.2 Hz grid from the hardware timestamps. Gaps of 5 s or more stay missing.
2. Stage 1, spectral quality index: a window whose power is not in the cardiac band is blanked. This gate does not depend on amplitude.
3. Stage 2, local MAD: spikes and clipping that remain are blanked. Valid PPG at another amplitude is judged against its own neighbourhood.
4. Upsample the cleaned signal to 256 Hz and detect peaks with NeuroKit2 (Elgendi). Peaks inside a gap or a rejected region are discarded.
5. Sliding 30 s HRV windows at 2 Hz. A window that overlaps a gap or a rejected region has every metric set to missing, so an interval that jumps the gap is not reported as a beat.

Diagnostic files under `features/ppg/`: `ppg_resampled.csv`, `ppg_normalized.csv` (the rolling z-score passed to peak detection), `ppg_resample_report.txt`, `ppg_rejected.png`, `ppg_overview.png`, `ppg_spectrogram.png`.

## Output

`results/HeartDynamics.csv`, 0.5 s steps:

`Timestamp`, `mean_hr`, `mean_nn`, `sdnn`, `rmssd`, `pnn50`, `cvsd`, `cvnn`, `n_beats`

The step is skipped when `results/HeartDynamics.csv` already exists.
