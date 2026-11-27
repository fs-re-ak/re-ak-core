# classical-emotion v2.14.0 — Integration guide

**19-D `extended_nm_a_rmsunit`** (same features as 2.7.0 / 2.13.0). No session
calibration. FEAR is **emitted**.

Training discards the same recordings as 2.13.0 / 2.12.0 / 2.10.0
(P004, P007, P008, P009, P013, P018). Sample rule on HOLD windows after
disconnect / coverage / rail gates (quantiles from included NEUTRAL):

- **All NEUTRAL** kept
- **ANGER** kept if mean channel RMS ≥ Neutral **Q2**
- **Other non-NEUTRAL** kept if mean channel RMS ≥ Neutral **Q3**

FS002_S02–S07 and MB001 are included.

| | |
|---|---|
| Feature set | `extended_nm_a_rmsunit` (19 features) |
| Amplitude | RMS L2 unit vector (8) + raw mean RMS (1) |
| Sample rule | ANGER ≥ Neutral Q2; other non-NEUTRAL ≥ Neutral Q3; NEUTRAL kept |
| Fear | **emitted** |
| Train exclusion | P004, P007, P008, P009, P013, P018 |
| Do **not** apply | NLMS; gated-prior `set_subject_context` |

Copy this **entire directory** (or unzip `classical-emotion-2.14.0.zip`).

The production alias stays **classical-emotion:2.6.0**.

---

## Pipeline position

```
raw 8-ch EXG
    → 60 Hz notch, then 20–110 Hz bandpass
    → 2.0 s windows (500 samples at 250 Hz)
    → this model
```

Do not run NLMS. Do not call `set_subject_context`.

---

## Minimal integration

```python
import sys
from pathlib import Path
import numpy as np

release = Path("path/to/classical-emotion/2.14.0")
sys.path.insert(0, str(release))
from exg_emotion.runtime.predictor import Predictor

pred = Predictor.from_release_dir(release)
out = pred.predict(window)  # [8, 500]
print(out.label, out.probabilities[out.label], out.fear_suppressed)
```

FEAR may be the predicted label. Output is 8-way after contempt merge.
