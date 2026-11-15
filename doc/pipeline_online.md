# Online (live) processing

`reak_core.online` processes a Hermes stream as it arrives, for live displays. One instance per headset keeps
that stream's state. Results are on the same 2 Hz timebase as the offline outputs: 2.0 s windows starting every
0.5 s, at multiples of 125 samples from the start of the session.

```python
from reak_core.models import load_model
from reak_core.online import OnlineEmotionClassifier, OnlineEngagement

model = load_model()                      # one model shared by every headset
emotions = OnlineEmotionClassifier(model)  # one per headset
engagement = OnlineEngagement()            # one per headset

for packet in stream:                      # raw µV, shape (n, 8), first_sample counted from session start
    for r in emotions.push(packet.samples, packet.first_sample):
        ...                                # r.status, r.label, r.intensities, r.time_s
    for r in engagement.push(packet.samples, packet.first_sample):
        ...                                # r.engagement, r.engagement_calibrated
```

## Stream handling

- **Placement.** `first_sample` places each block on the session's sample index.
  - Repeated samples are dropped.
  - A gap of up to 2 s (`max_fill_samples`) becomes invalid samples, so the windows over it are null.
  - A longer gap restarts the filters and windows, but keeps the calibration.
- **Reset.** `reset()` clears everything, calibration included. Call it for a new session or a new wearer.
- **Checkpoints.** `state_dict()` / `load_state_dict()` save and restore the calibration history (JSON-serialisable). Filters restart after a reload.

## Emotions (`OnlineEmotionClassifier`)

**Same as offline** (`pipeline_emotions.md`):
- release model;
- features;
- one-hot output of 2.6.0;
- post-processing thresholds.

**Differences forced by causality:**

| Step | Offline | Online |
|---|---|---|
| Filters | Notch and bandpass, zero-phase over the whole recording | The same cascade run twice forward (same magnitude response, \|H\|²); the first 1 s after a (re)start is skipped |
| Amplitude calibration | Every valid window of the recording | Ring buffer of the last 600 valid windows (5 min), recomputed at each window with the release's own math (gated p95 + prior for 2.6.0). Status `calibrating` until 60 windows (30 s) |
| High-RMS rule | Nearest clean window on either side within 6 windows | Latest clean window **before** it within 6 windows, otherwise Neutral |

**Measured on the synthetic baseline session:**
- With the same calibration as offline, labels agree on at least 93% of windows (the test threshold).
- With online calibration, labels agree on at least 90% of windows in the second half of the session.

## Engagement (`OnlineEngagement`)

**Same pipeline as offline** (`pipeline_cognition.md`): causal NLMS, 0.5–30 Hz bandpass, Welch band powers, `Engagement = beta / (alpha + theta)`.

**Differences:**
- **Channel.** Only EAR_R is decorrelated. T9 and T10 are ±EAR_R/2, so their band powers are identical, and the result equals offline.
- **Filter.** Run twice forward; the first 2 s after a (re)start are skipped.
- **Epoch criteria.** They are judged on each window's two 1 s halves, instead of fixed 1 s epochs.

**Calibrated output.** `engagement_calibrated` is a 0–1 score relative to the last two minutes (RE-AK online calibration specification, `PercentileMinMaxNormalizer` on `log10(Engagement)`):
- 240-update ring buffer;
- 10th / 90th percentiles mapped to 0 / 1;
- 60-update warm-up;
- values outside [0, 1] folded back with a sigmoid (k = 10);
- output smoothing 0.35.

**Measured on the synthetic baseline:** correlation with offline Engagement is above 0.95, and the median ratio is within 5%.

## Cost

One headset (emotions and engagement together) uses about 1.6% of one core of a desktop CPU in real time (150 s of data in 2.4 s). Most of it is the per-sample NLMS.
