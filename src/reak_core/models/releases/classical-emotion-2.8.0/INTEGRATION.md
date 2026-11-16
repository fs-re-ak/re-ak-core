# classical-emotion v2.8.0 — Integration guide

27-D `extended_nm_a_floor` logreg on **20–110 Hz** windows (60 Hz notch).
Amplitude above rest is a **floor-only** log ratio. There is no ceiling,
no gate, and no `prior_ratios.json`.

| | |
|---|---|
| Feature set | `extended_nm_a_floor` (27 features) |
| Calibration | per-channel q=15 floor, clamped to the field population |
| Artifact | `floor_stats.json` (required) |
| Kept from 2.7.0 | RMS L2 unit vector, MDF, asymmetry, antagonist corr, NM-A |
| Dropped | Waveform length; raw mean RMS; gated-p95 ceiling |
| Classifier | logistic regression (`C=1.0`) + `StandardScaler` |
| Train exclusion | P009, P008, P004 |
| Do **not** apply | NLMS; 2.6.0 `set_subject_context` recipe |

Copy this **entire directory** (or unzip `classical-emotion-2.8.0.zip`).

---

## Honest caveats

1. LOSO numbers are posed HOLD under the three-subject exclusion. Compare
   them to the paired re-fit in `reports/floor_280/`, not to the published
   2.6.0 47.5% (that run trained with P009/P008/P004 included).
2. **Fear is not emitted as a label.** If argmax is FEAR, the next-best
   class is returned, `fear_suppressed=True`, and FEAR stays in
   `probabilities`.
3. Production alias stays **2.6.0** unless you retarget it.

---

## Pipeline position

```
raw 8-ch EXG
    → 60 Hz notch, then 20–110 Hz bandpass
    → 2.0 s windows (500 samples at 250 Hz)
    → set_subject_context once per recording
    → predict each window
```

This package does not filter or window. Do not run NLMS.

---

## Calibration

`set_subject_context(windows)` once, then freeze. Do not update it on a
short buffer that follows the current expression.

Per channel:

1. Drop windows with RMS `<= noise_floor` (dead electrode).
2. If fewer than `min_windows` (30, about 60 s) remain, use `pop_default`.
3. Otherwise baseline = clip(percentile(valid, 15), pop_lo, pop_hi).
4. Feature = `log(rms) - log(baseline)`.

`pred.last_calibration["channels"]` records, per channel, `baseline`,
`source` (`population` or `session`), `clamped` (`low` / `high` / `none`),
`n_valid_windows`, and `n_dead_windows`.

Calling `predict` before `set_subject_context` raises `RuntimeError`.

---

## Minimal integration

```python
import sys
from pathlib import Path
import numpy as np

release = Path("path/to/classical-emotion/2.8.0")
sys.path.insert(0, str(release))
from exg_emotion.runtime.predictor import Predictor

pred = Predictor.from_release_dir(release)
# windows: [n, 8, 500], already 20-110 Hz + notch, one recording
pred.set_subject_context(windows)
out = pred.predict(windows[0])
print(out.label, out.fear_suppressed)
```

Channel order: Supraorbital_L, Supraorbital_R, Zygomatic_L, Zygomatic_R,
Temporal_EEG, Glabella, Temporal_R, Nasolabial.

## Features (27-D)

| Offset | Count | What |
|---|---:|---|
| 0–7 | 8 | RMS L2 unit vector |
| 8–15 | 8 | log RMS minus log floor, per channel |
| 16 | 1 | Mean of those 8 log ratios |
| 17–23 | 7 | Median frequency 20–120 Hz, EMG only |
| 24 | 1 | Zygomatic `(L−R)/(L+R)` |
| 25 | 1 | Envelope corr Glabella × Supraorbital_L |
| 26 | 1 | NM-A `(Glabella−Nasolabial)/(Glabella+Nasolabial)` |
