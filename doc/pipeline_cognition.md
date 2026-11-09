# Cognition — Hermes EEG

**Module:** `reak_core.eeg.cognition`  
**Entry point:** `compute_cognition(session_root, apply_nlms=True)`  
**Primary output:** `results/Cognition.csv`

Engagement from the temporal EEG electrodes. Only Engagement is emitted (decision D5). There is no 60 Hz notch after the 0.5–30 Hz bandpass (decision D4): a 60 Hz notch sits above that passband and is not applied.

## Input

First EXG file under `rawData/`: `rawEEG_0.csv`, `eeg.tmp`, `eeg.csv`, `eegRec_0.csv`. Eight channels at 250 Hz. Column 0 is the timestamp. A header is optional; named columns are reordered into hardware order, and a truncated last line is ignored.

EEG is derived from hardware channels AF8, AF7 and EAR_R (midpoint re-reference against the T9/T10 differential). Validity for this step is judged on those channels (decision D23). A dead cheek, brow or nose electrode does not by itself blank Engagement. Invalid raw samples are non-finite values and ADC full scale (±375 000 µV). See [overview](overview.md).

## Processing

The procedure lives in `reak_core.eeg.cognition`.

1. Optional causal NLMS decorrelation before filtering. The default pipeline leaves it on. The emotion model does not use this filter.
2. Hardware gaps of 5 s or more. Windows that straddle a gap are missing.
3. Bandpass 0.5–30 Hz, 4th-order Butterworth, zero-phase, on the whole recording.
4. Artefact rejection on T9/T10, 1 s epochs, three criteria: NaN fraction above 20%, peak amplitude above 500 µV, peak-to-peak above 500 µV. When more than 25% of a 2 s analysis window is flagged, that row is missing.
5. Welch power on T9 and T10 only, 2 s windows, 75% overlap. Bands: delta 0–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–22 Hz.
6. Engagement = bilateral (T9+T10) beta / (alpha + theta).
7. Linear resample onto the 0.5 s grid, without interpolating across gaps.

`features/cognition/temporalBandPowers.csv` keeps the bilateral band averages. Focus, cognitive load, frontal and lateralization formulas are not written to `Cognition.csv`.

## Output

`results/Cognition.csv`: `Timestamp`, `Engagement`.

Also under `features/cognition/`: `powerBands.csv`, `artefactStats.csv`, `epochMetrics.csv`, `temporalBandPowers.csv`, `eegArtefactPlot.png`.

The step is skipped when `results/Cognition.csv` already exists. If `powerBands.csv` is already present and contains T9 and T10, band powers are loaded from that file and the earlier EEG steps are not repeated.
