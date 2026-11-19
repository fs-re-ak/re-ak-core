# classical-emotion v2.12.0 — Integration guide

**19-D `extended_nm_a_rmsunit`** (same features as 2.7.0) on the **2.10.0
sample set**: HERMES_FS002_S02–S07 included, SADNESS/FEAR mean-RMS floor,
CONTEMPT half-sampled, and P004/P007/P008/P009/P013/P018 excluded from the
fit. **No session calibration.** FEAR is **emitted**.

| | |
|---|---|
| Feature set | `extended_nm_a_rmsunit` (19 features) |
| Amplitude | RMS L2 unit vector (8) + raw mean RMS (1) |
| Scale-free | MDF (7) + zygomatic asymmetry + Glabella×Supraorbital_L corr + NM-A |
| Sample policy | Same windows as classical-emotion 2.10.0 |
| Fear | **emitted** |
| Train exclusion | P004, P007, P008, P009, P013, P018 |
| Do **not** apply | NLMS; gated-prior `set_subject_context` |

Copy this **entire directory** (or unzip `classical-emotion-2.12.0.zip`).

The production alias in `releases/aliases.json` stays **classical-emotion:2.6.0**.

---

## Honest caveats

1. LOSO metrics: `reports/classical-emotion-2.12.0/`.
2. FS002 labels are instruction-bootstrap until a face review exists.
3. Calibration-free amplitude path — not session-neutral.

---

## Pipeline position

```
raw 8-ch EXG
    → 60 Hz notch, then 20–110 Hz bandpass
    → 2.0 s windows (500 samples at 250 Hz)
    → this model
```

Do not run NLMS. Do not call `set_subject_context`.

Channel order: Supraorbital_L, Supraorbital_R, Zygomatic_L, Zygomatic_R,
Temporal_EEG, Glabella, Temporal_R, Nasolabial.

---

## Minimal integration

```python
import sys
from pathlib import Path
import numpy as np

release = Path("path/to/classical-emotion/2.12.0")
sys.path.insert(0, str(release))
from exg_emotion.runtime.predictor import Predictor

pred = Predictor.from_release_dir(release)
out = pred.predict(window)  # [8, 500], 20-110 Hz
print(out.label, out.probabilities[out.label], out.fear_suppressed)
```

FEAR may be the predicted label. Output is 8-way after contempt merge.
