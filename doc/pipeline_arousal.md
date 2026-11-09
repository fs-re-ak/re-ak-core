# Arousal — Shimmer EDA

**Module:** `reak_core.eda.arousal` (`cvxEDA` is vendored in `reak_core.eda.cvxEDA`)  
**Entry point:** `compute_arousal(session_root)`  
**Primary output:** `results/Arousal.csv`

Tonic and phasic electrodermal activity from the Shimmer GSR channel, using cvxEDA (Greco et al., IEEE TBME, 2015).

## Input

First file that exists under `rawData/`:

| File | EDA column |
|---|---|
| `shimmer.csv` | 4, resistance in kΩ |
| `rawShimmer_0.csv` | 4, resistance in kΩ |
| `gsr.tmp` | 1 |
| `gsr.csv` | 1 |

Column 0 is the hardware timestamp. A header row is optional. Nominal rate: 51.2 Hz. Timestamp units follow the same reading rules as the other streams (decision D1).

## Processing

The procedure and its constants live in `reak_core.eda.arousal`.

1. Resample onto a strict 51.2 Hz grid. Gaps of 5 s or more stay missing. The resampled resistance is saved in kΩ.
2. Convert resistance to conductance in µS (`1000 / resistance`). Later steps use conductance.
3. Reject artifacts with a rolling MAD, leaving long gaps missing.
4. Downsample to 4 Hz and decompose with cvxEDA. Gap regions are restored to missing after the decomposition.
5. Detect SCR events on the 4 Hz phasic component (minimum height 0.05, minimum prominence 0.05, at least 1 s apart, in z-score units).
6. Average the tonic and phasic signals onto the shared 2 Hz grid.

## Output

`results/Arousal.csv`, 0.5 s steps: `Timestamp`, `TonicEDA`, `PhasicEDA`.

Under `features/eda/`: `eda_resampled.csv` (kΩ), `eda_resample_report.txt`, `eda_overview.png` (µS), `SCR_events.csv`.

The step is skipped when `results/Arousal.csv` already exists.
