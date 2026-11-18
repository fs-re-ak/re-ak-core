# classical-emotion v2.10.0 — Integration guide

Same session-neutral recipe as 2.9.0 on **20–110 Hz** windows (60 Hz notch, Q=30),
with an adjusted training set: HERMES_FS002_S02–S07 added (each recording is its own
LOSO unit), P007/P013/P018 excluded in addition to P004/P008/P009, SADNESS/FEAR windows
with mean RMS below 6 dropped, and CONTEMPT half-sampled (LEFT + RIGHT).

| | |
|---|---|
| Feature set | `session_neutral_log_rms` (25 features) |
| Neutral | `N = w * p20 + (1 - w) * prior`, `w = n / (n + 60)` |
| Artifacts | `pipeline.joblib`, `temperature.json`, `thresholds.json`, `wl_prior.json` |
| Decision | temperature, then per-class F0.5 threshold; below threshold the label is NEUTRAL |
| Fear | **emitted**. This release does not suppress FEAR |
| Train exclusion | P004, P007, P008, P009, P013, P018 |
| Do **not** apply | NLMS; the 2.6.0 or 2.8.0 `Predictor` feature path |

Copy this **entire directory** (or unzip `classical-emotion-2.10.0.zip`).

The production alias in `releases/aliases.json` stays **classical-emotion:2.6.0**.
Loading `Predictor.from_registry(..., model="production")` still returns 2.6.0.
This folder is loaded with `SessionNeutralModel.from_release_dir`.

---

## Honest caveats

1. Nested LOSO accuracy on the included units is 0.476 [0.409, 0.545].
   Each FS002_S0x is a separate fold. Excluded subjects are scored with this
   shipped model, which never trained on them.
2. On the field, 2.10.0 is less often NEUTRAL than 2.6.0 (Malefycia about 0.301
   versus 0.364). It was not tuned toward a 50% NEUTRAL share.
3. There is no posterior smoother in this package.
4. FS002 labels are instruction-bootstrap until a face review exists.
5. The causal helper below uses the windows you push. The field report's RMS
   neutral used a separate 1-second cache of the full session. Those two
   trajectories are not the same object.

---

## Pipeline position

```
raw 8-ch EXG
    → 60 Hz notch (Q=30), then 20–110 Hz bandpass
    → 2.0 s windows (500 samples at 250 Hz)
    → set_subject_context once per recording, or the causal updates below
    → predict each window
```

This package does not filter or window. Do not run NLMS.

Channel order: Supraorbital_L, Supraorbital_R, Zygomatic_L, Zygomatic_R,
Temporal_EEG, Glabella, Temporal_R, Nasolabial.

## Minimal integration

```python
import sys
from pathlib import Path
import numpy as np

release = Path("path/to/classical-emotion/2.10.0")
sys.path.insert(0, str(release))
from exg_emotion.models.classical_v1.session_neutral import SessionNeutralModel

model = SessionNeutralModel.from_release_dir(release)
# windows: [n, 8, 500], already 20-110 Hz + notch, one recording
model.set_subject_context(windows)
out = model.predict(windows[0])
print(out.label, out.probabilities["HAPPINESS"], out.usable)
print(model.last_ood_channels)
```

`out.usable` is false when any channel has coverage below 80% or log(RMS) above 10.5.
The label is still filled. `out.low_confidence` is true when an emotion lost to its
F0.5 threshold and the emitted label is NEUTRAL. `out.fear_suppressed` is always false.

## Causal stream

```python
model.reset_causal()          # N starts at the prior
model.push(window, t_end_s)   # after the window has ended
model.advance_to(t_start)     # floor(t_start / 10); each step moves at most 0.05 log
out = model.predict(window)
```

A window starting at `t` must be advanced before it is pushed, so its own samples
are not inside that update.
