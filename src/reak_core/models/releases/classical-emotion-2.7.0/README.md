# classical-emotion v2.7.0 — Integration guide

19-D `extended_nm_a_rmsunit` logreg on **20–110 Hz** windows (60 Hz notch).
**No session calibration.** No `prior_ratios.json`. `set_subject_context` is
not required and is a no-op for amplitude.

| | |
|---|---|
| Feature set | `extended_nm_a_rmsunit` (19 features) |
| Amplitude | RMS L2 unit vector (8) + raw mean RMS (1); no gated prior |
| Scale-free | MDF (7) + zygomatic asymmetry + Glabella×Supraorbital_L corr + NM-A |
| Dropped vs 2.6.0 | Waveform length; subject gated-p95 calibration |
| Classifier | logistic regression (`C=1.0`) + fold `StandardScaler` |
| Train exclusion | P009, P008, P004 out of the shipped fit |
| LOSO (paired re-fit) | see `metrics.json` / `reports/rmsunit_270/` |
| Do **not** apply | NLMS decorrelation; gated-prior `set_subject_context` |

Copy this **entire directory** (or unzip `classical-emotion-2.7.0.zip`) into the
consuming project. It is self-contained.

---

## Honest caveats

1. **Paired LOSO under this exclusion is below the re-fit 2.6.0 baseline**
   (~43.9% vs ~48.5% over 24 folds; ~46.4% vs ~51.8% over the 21
   train-eligible subjects). This package exists so integrators can try a
   calibration-free amplitude path. Production alias stays **2.6.0** unless
   you retarget it deliberately.
2. **Fear is still weak.** This release **does not emit FEAR as the predicted
   label**. If argmax is FEAR, the next-best class is returned,
   `fear_suppressed=True`, `low_confidence=True`, and FEAR stays in
   `probabilities`.
3. Numbers are posed HOLD LOSO, not labeled field false-positive rates.

---

## 1. Install

```bash
pip install -r requirements.txt
```

`numpy>=1.24`, `scikit-learn>=1.3`, `joblib>=1.3`.

---

## 2. Pipeline position

```
raw 8-ch EXG
    → hardware invalidation / NaN handling
    → 60 Hz notch, then 20–110 Hz bandpass (4th-order Butterworth, filtfilt)
      on the full recording
    → slice 2.0 s windows (500 samples at 250 Hz)
    → this model
```

This package **does not** filter, resample, or window.

**Do not run NLMS.** Do not use 15–40 Hz windows with this checkpoint.

---

## 3. Input contract

| Property | Value |
|---|---|
| Shape | `[8, 500]` per window — **channels × samples** |
| Dtype | finite `float32` or `float64` (no NaN/Inf) |
| Sample rate | 250 Hz |
| Duration | 2.0 s |
| Band | already 20–110 Hz after 60 Hz notch |

```python
CHANNELS = [
    "Supraorbital_L",  # ch1
    "Supraorbital_R",  # ch2
    "Zygomatic_L",     # ch3
    "Zygomatic_R",     # ch4
    "Temporal_EEG",    # ch5
    "Glabella",        # ch6
    "Temporal_R",      # ch7
    "Nasolabial",      # ch8
]
```

---

## 4. No subject calibration

Call `predict()` on each window. Do **not** call `set_subject_context`.
There is no rest median, gated p95, or field prior.

---

## 5. Minimal integration

```python
import sys
from pathlib import Path

import numpy as np

release = Path("path/to/classical-emotion/2.7.0")  # this folder
sys.path.insert(0, str(release))

from exg_emotion.runtime.predictor import Predictor

pred = Predictor.from_release_dir(release)

# window: float array [8, 500], 20-110 Hz + notch
out = pred.predict(window)
if out.fear_suppressed:
    pass  # do not write a fear claim; FEAR mass is still in out.probabilities
print(out.label, out.probabilities[out.label])
```

| Field | Meaning |
|---|---|
| `label` | Winning class (8-way after contempt merge; never FEAR) |
| `probabilities` | `dict[str, float]`, sums to 1.0 (includes FEAR) |
| `fear_suppressed` | argmax was FEAR; label is the next-best class |
| `low_confidence` | True when fear was suppressed |
| `model_name` | `"classical-emotion"` |
| `model_version` | `"2.7.0"` |

---

## 6. Features (19-D)

| Offset | Count | What | Norm |
|---|---:|---|---|
| 0–7 | 8 | RMS L2 unit vector, all 8 channels | per-window unit sphere |
| 8 | 1 | Mean of the 8 raw RMS values | none (population StandardScaler in the pipeline) |
| 9–15 | 7 | Median frequency 20–120 Hz, EMG only | none |
| 16 | 1 | Zygomatic `(L−R)/(L+R)` | scale-free |
| 17 | 1 | Envelope corr Glabella × Supraorbital_L | scale-free |
| 18 | 1 | NM-A `(Glabella − Nasolabial)/(Glabella + Nasolabial)` | scale-free |

---

## 7. Checklist

- [ ] Notch 60 Hz, then bandpass **20–110 Hz**; slice 2 s.
- [ ] Shape `[8, 500]`, channel order as above.
- [ ] No NLMS / no gated-prior calibration / no `prior_ratios.json`.
- [ ] Do not emit FEAR as a datastory (`fear_suppressed`).
- [ ] Handle 8-class output (`CONTEMPT`, not left/right).
