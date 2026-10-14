# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Cognition Processor

Computes cognitive metrics from EEG data using the cleaned temporal pipeline:
optional NLMS adaptive decorrelation, bandpass filtering (0.5–30 Hz), epoch-level
artefact rejection on T9/T10 (three criteria), Welch band-power extraction on T9
and T10, and Engagement computation from bilateral temporal averages.

The current output (``results/Cognition.csv``) emits the Engagement metric only.
Only Engagement is computed (decision D5); the bilateral band averages are archived in
``features/cognition/temporalBandPowers.csv``. There is no 60 Hz notch after the 0.5–30 Hz bandpass
(decision D4).

Feature traceability: per-window EEG band powers, per-epoch artefact metrics, and
an artefact waveform plot are written under the session's ``features/cognition/``
folder.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch

from reak_core.constants import EXG_SAMPLING_RATE
from reak_core.eeg.nlms import CausalNLMSFilter, nlms_nan_safe
from reak_core.io.exg import derive_eeg_channels, load_exg, mark_invalid
from reak_core.sync.resample import resample_to_timebase

logger = logging.getLogger(__name__)

# Gaps in the hardware timestamp stream that are at least this long (seconds)
# are treated as true recording breaks; all windows that straddle a break are
# emitted as NaN rows (matching the convention used by shimmerResampler).
GAP_THRESHOLD_S = 5.0

# NOTE: np.seterr(all='raise') was previously set here. Removed because it
# converts benign floating-point events (e.g. 0/0 in normalisation) into
# exceptions that silently escape the pipeline's broad except-as-warning
# handler. Errors are now reported explicitly at each failure point instead.


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def fill_nans_with_interpolation(series):
    """Fill NaNs in a Series by linear interpolation."""
    return series.interpolate(method="linear")


def bandpass_filter(df, sf, lowcut=0.5, highcut=30, notch_freq=60):
    """
    Apply bandpass and optional notch filtering to EEG data.

    The default upper cutoff is 30 Hz, which covers delta, theta, alpha, and
    beta bands while excluding the gamma / high-beta range that is dominated
    by temporal muscle (EMG) noise at T9/T10.  The 60 Hz notch is applied
    only when ``notch_freq`` falls within the passband (i.e. notch_freq < highcut).

    Args:
        df: DataFrame with one column per EEG channel
        sf: Sampling frequency in Hz

    Returns:
        Filtered DataFrame with the same columns.
    """
    nyq = 0.5 * sf

    if lowcut >= highcut or highcut >= nyq:
        raise ValueError(
            f"[cognitionProcessor] Invalid cutoff frequencies: lowcut={lowcut}, highcut={highcut}, nyq={nyq}"
        )

    if notch_freq >= nyq:
        raise ValueError(f"[cognitionProcessor] Notch frequency {notch_freq} must be below Nyquist {nyq}")

    df = df.copy()

    for col in df.columns:
        series = df[col]

        if series.isna().any() or np.isinf(series.to_numpy()).any():
            logger.warning(f"[cognitionProcessor] Column '{col}' contains NaNs or Infs — interpolating")

            if pd.isna(series.iloc[0]):
                first_valid = series.dropna().iloc[0] if not series.dropna().empty else 0
                series.iloc[0] = first_valid

            if pd.isna(series.iloc[-1]):
                last_valid = series.dropna().iloc[-1] if not series.dropna().empty else 0
                series.iloc[-1] = last_valid

            series = fill_nans_with_interpolation(series)
            df[col] = series

    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")

    if len(df) < (3 * max(len(a), len(b))):
        logger.warning(
            f"[cognitionProcessor] Data too short for filtering ({len(df)} samples). Returning unfiltered data."
        )
        return df

    arr = filtfilt(b, a, df.to_numpy(), axis=0)

    # Only apply the notch when its frequency falls inside the passband.
    # With highcut=30 Hz the 60 Hz notch is beyond the passband and has no effect.
    if notch_freq < highcut:
        b_notch, a_notch = iirnotch(notch_freq / nyq, 30)
        arr = filtfilt(b_notch, a_notch, arr, axis=0)

    return pd.DataFrame(arr, columns=df.columns)


# ---------------------------------------------------------------------------
# Artefact rejection
# ---------------------------------------------------------------------------


def reject_temporal_artefacts(filtered_df, hardware_invalid, sf=EXG_SAMPLING_RATE):
    """
    Reject 1-second non-overlapping epochs on T9 and T10 using three signal-quality
    criteria.  The EMG power-ratio criterion (formerly Criterion 4) has been removed
    because the bandpass upper cutoff is now 30 Hz, which already excludes the
    30–45 Hz muscle noise band from the signal.

    An epoch is flagged if *either* channel fails *any* of the following:

        1. NaN fraction in raw signal (from hardware_invalid) > 20 %
        2. Peak absolute amplitude in filtered signal > 500 µV
        3. Peak-to-peak in filtered signal > 500 µV

    All signal metrics are measured for every epoch regardless of pass/fail so
    that the per-epoch report can be used to calibrate thresholds.

    Args:
        filtered_df: Bandpass-filtered DataFrame; must contain 'T9' and 'T10' columns.
        hardware_invalid: Boolean 1-D array (same length as filtered_df) marking
            samples that were NaN before filtering (hardware disconnects or gaps).
        sf: Sampling frequency in Hz (default 250).

    Returns:
        Tuple (rejected_mask, artefact_stats, epoch_metrics):
            - rejected_mask:  boolean numpy array, True for every sample belonging
              to a rejected 1-second epoch.
            - artefact_stats: single-row summary DataFrame written to
              ``features/cognition/artefactStats.csv``.
            - epoch_metrics:  per-epoch DataFrame written to
              ``features/cognition/epochMetrics.csv`` — one row per epoch with
              measured signal metrics for both channels plus rejection outcome.
    """
    for ch in ("T9", "T10"):
        if ch not in filtered_df.columns:
            logger.warning(
                f"[cognitionProcessor] reject_temporal_artefacts: '{ch}' not found — skipping artefact rejection"
            )
            n = len(filtered_df)
            return np.zeros(n, dtype=bool), None, None

    n_samples = len(filtered_df)
    epoch_len = int(sf)  # 1 s = 250 samples at 250 Hz
    n_epochs = n_samples // epoch_len

    rejected_mask = np.zeros(n_samples, dtype=bool)

    t9 = filtered_df["T9"].to_numpy()
    t10 = filtered_df["T10"].to_numpy()

    # Per-criterion rejection counters (first criterion that triggered rejection)
    criterion_counts = {"nan": 0, "peak_amp": 0, "ptp": 0}

    epoch_rows = []

    for ei in range(n_epochs):
        s, e = ei * epoch_len, (ei + 1) * epoch_len
        epoch_bad = False
        cause = ""
        rejecting_channel = ""

        # Measure all metrics for both channels unconditionally so the per-epoch
        # report captures the full signal picture even for kept epochs.
        nan_frac = float(hardware_invalid[s:e].mean())
        t9_sig = t9[s:e]
        t10_sig = t10[s:e]
        t9_peak = float(np.nanmax(np.abs(t9_sig)))
        t10_peak = float(np.nanmax(np.abs(t10_sig)))
        t9_ptp = float(np.nanmax(t9_sig) - np.nanmin(t9_sig))
        t10_ptp = float(np.nanmax(t10_sig) - np.nanmin(t10_sig))

        # Evaluate criteria in order; first failure sets cause + rejecting_channel.
        if nan_frac > 0.20:
            cause = "nan"
            rejecting_channel = "T9/T10"
            epoch_bad = True
        elif t9_peak > 500.0:
            cause = "peak_amp"
            rejecting_channel = "T9"
            epoch_bad = True
        elif t10_peak > 500.0:
            cause = "peak_amp"
            rejecting_channel = "T10"
            epoch_bad = True
        elif t9_ptp > 500.0:
            cause = "ptp"
            rejecting_channel = "T9"
            epoch_bad = True
        elif t10_ptp > 500.0:
            cause = "ptp"
            rejecting_channel = "T10"
            epoch_bad = True

        if epoch_bad:
            rejected_mask[s:e] = True
            criterion_counts[cause] += 1

        epoch_rows.append(
            {
                "epoch_idx": ei,
                "time_s": round(s / sf, 3),
                "nan_frac": round(nan_frac, 4),
                "T9_peak_amp": round(t9_peak, 3),
                "T10_peak_amp": round(t10_peak, 3),
                "T9_ptp": round(t9_ptp, 3),
                "T10_ptp": round(t10_ptp, 3),
                "rejected": epoch_bad,
                "cause": cause,
                "rejecting_channel": rejecting_channel,
            }
        )

    n_rejected_epochs = int(rejected_mask[: n_epochs * epoch_len].reshape(n_epochs, epoch_len).any(axis=1).sum())
    pct_epochs = round(100.0 * n_rejected_epochs / max(n_epochs, 1), 1)
    n_samples_flagged = int(rejected_mask.sum())
    pct_samples = round(100.0 * n_samples_flagged / max(n_samples, 1), 1)

    logger.info(
        f"[cognitionProcessor] Artefact rejection: {n_rejected_epochs}/{n_epochs} epochs rejected "
        f"({pct_epochs} % of epochs, {pct_samples} % of samples) — "
        f"nan={criterion_counts['nan']}, peak={criterion_counts['peak_amp']}, "
        f"ptp={criterion_counts['ptp']}"
    )

    artefact_stats = pd.DataFrame(
        [
            {
                "n_epochs_total": n_epochs,
                "n_epochs_rejected": n_rejected_epochs,
                "pct_epochs_rejected": pct_epochs,
                "n_rejected_by_nan": criterion_counts["nan"],
                "n_rejected_by_peak_amp": criterion_counts["peak_amp"],
                "n_rejected_by_ptp": criterion_counts["ptp"],
                "n_samples_flagged": n_samples_flagged,
                "pct_samples_flagged": pct_samples,
            }
        ]
    )

    epoch_metrics = pd.DataFrame(epoch_rows)

    return rejected_mask, artefact_stats, epoch_metrics


# ---------------------------------------------------------------------------
# Power band extraction
# ---------------------------------------------------------------------------


def compute_eeg_power_bands(
    df,
    sf=EXG_SAMPLING_RATE,
    window_duration=2,
    bands_definitions=[[0, 4], [4, 8], [8, 13], [13, 22]],
    overlap=0.75,
    timestamps=None,
    hardware_invalid=None,
):
    """
    Compute EEG power in different frequency bands using Welch's method.

    Args:
        df: Preprocessed EEG DataFrame (samples × channels)
        sf: Sampling frequency (default EXG_SAMPLING_RATE)
        window_duration: Duration of analysis window in seconds
        bands_definitions: List of [low, high] frequency ranges for each band
        overlap: Fraction of window overlap (0–1)
        timestamps: Optional 1-D array of per-sample timestamps (seconds).
            When provided, the Timestamp of each window is set to the midpoint
            of the first and last sample's original timestamps rather than being
            derived from the post-preprocessing array index.
        hardware_invalid: Optional boolean 1-D array (same length as df).
            Windows where any sample is marked True are emitted as NaN rows
            instead of running Welch, preserving timeline alignment.

    Returns:
        DataFrame with columns [Timestamp, channel, band, power], or None on failure.
        Windows with lost samples have NaN in the power column.
    """
    logger.info("[cognitionProcessor] Computing EEG power bands")

    channel_names = df.columns.tolist()
    arr = df.to_numpy()
    n_samples = len(df)

    window_samples = int(window_duration * sf)
    step_samples = int(window_samples * (1 - overlap))
    n_windows = (n_samples - window_samples) // step_samples + 1

    if n_windows <= 0:
        logger.error(
            f"[cognitionProcessor] compute_eeg_power_bands: not enough data for a single window "
            f"(samples={n_samples}, window={window_samples}). Recording may be too short."
        )
        return None

    band_names = ["delta", "theta", "alpha", "beta"]
    rows = []

    for win_idx in range(n_windows):
        start_idx = win_idx * step_samples
        end_idx = start_idx + window_samples

        # Representative timestamp for this window: median of the recorded
        # hardware timestamps within the window.  The median is more robust
        # to per-sample jitter than a simple midpoint and correctly tracks
        # any long-term drift between the hardware clock and nominal rate.
        # Fall back to sample-index / sf when no timestamps are available.
        if timestamps is not None:
            ts_window = timestamps[start_idx:end_idx]
            timestamp = float(np.median(ts_window))
        else:
            timestamp = (start_idx + window_samples / 2) / sf

        # Windows where more than 25 % of samples are flagged (hardware-invalid
        # or artefact-rejected) are emitted as NaN rows so the timeline stays
        # intact while bad data is clearly marked.
        if hardware_invalid is not None and hardware_invalid[start_idx:end_idx].mean() > 0.25:
            for ch_name in channel_names:
                for band_name in band_names:
                    rows.append(
                        {
                            "Timestamp": timestamp,
                            "channel": ch_name,
                            "band": band_name,
                            "power": np.nan,
                        }
                    )
            continue

        window_data = arr[start_idx:end_idx, :]

        for ch_idx, ch_name in enumerate(channel_names):
            try:
                freqs, psd = welch(window_data[:, ch_idx], fs=sf, nperseg=min(256, window_samples))
            except Exception as e:
                logger.warning(f"[cognitionProcessor] Welch failed on window {win_idx} channel {ch_name}: {e}")
                continue

            for band_idx, (low, high) in enumerate(bands_definitions):
                freq_mask = (freqs >= low) & (freqs <= high)
                power = np.trapezoid(psd[freq_mask], freqs[freq_mask]) if freq_mask.any() else 0.0

                rows.append(
                    {
                        "Timestamp": timestamp,
                        "channel": ch_name,
                        "band": band_names[band_idx],
                        "power": power,
                    }
                )

    if not rows:
        logger.error(
            "[cognitionProcessor] compute_eeg_power_bands: no rows produced — "
            "Welch failed on every window/channel combination."
        )
        return None

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cognitive index computation
# ---------------------------------------------------------------------------


def compute_cognitive_indexes(df):
    """
    Compute Engagement from bilateral T9/T10 EEG power bands.

    Engagement (canonical metric):
        avg_beta / (avg_alpha + avg_theta)    [Pope et al. 1995]

    The following metrics are currently deferred (exploration phase) and are
    not emitted in the output.  Their formulas are retained here for reference:
        Focus        : avg_beta / avg_alpha
        CognitiveLoad: (avg_theta × avg_beta) / avg_alpha²   [Borghini et al. 2014]
        Frontal      : mean(β/(α+θ) for AF7, β/(α+θ) for AF8)
        Lateralization: log(clip(LeftHemi_eng, 1e-12)) − log(clip(RightHemi_eng, 1e-12))

    Args:
        df: DataFrame with columns [Timestamp, channel, band, power].
            Only T9 and T10 rows are required.

    Returns:
        Tuple (result, temporal_bands):
            result         — DataFrame [Timestamp, Engagement], or None on failure.
            temporal_bands — DataFrame with per-timestamp bilateral temporal band
                             averages and relative powers (for feature storage), or
                             None on failure.
    """
    logger.info("[cognitionProcessor] Computing cognitive indexes")

    required_bands = {"alpha", "beta", "theta", "delta"}
    present_bands = set(df["band"].unique()) if "band" in df.columns else set()
    missing_bands = required_bands - present_bands
    if missing_bands:
        logger.error(
            f"[cognitionProcessor] compute_cognitive_indexes: missing bands {missing_bands}. Present: {present_bands}"
        )
        return None, None

    try:
        df_pivot = df.pivot_table(
            index=["Timestamp", "channel"],
            columns="band",
            values="power",
            aggfunc="mean",
        ).reset_index()
        df_pivot.columns.name = None
    except Exception as e:
        logger.error(f"[cognitionProcessor] pivot_table failed: {e}")
        logger.error(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None, None

    required_channels = {"T9", "T10"}
    present_channels = set(df_pivot["channel"].unique())
    missing_channels = required_channels - present_channels
    if missing_channels:
        logger.error(
            f"[cognitionProcessor] compute_cognitive_indexes: missing channels {missing_channels}. "
            f"Present: {present_channels}"
        )
        return None, None

    try:
        # --- Bilateral temporal averages (T9 + T10) ---
        temporal_df = df_pivot[df_pivot["channel"].isin(["T9", "T10"])]
        temporal_avg = temporal_df.groupby("Timestamp")[["alpha", "beta", "theta", "delta"]].mean()

        avg_alpha = temporal_avg["alpha"]
        avg_beta = temporal_avg["beta"]
        avg_theta = temporal_avg["theta"]
        avg_delta = temporal_avg["delta"]
        all_power = avg_beta + avg_alpha + avg_theta + avg_delta

        engagement = avg_beta / (avg_alpha + avg_theta + 1e-8)

        # --- Assemble results (Engagement only) ---
        ts = temporal_avg.index

        result = pd.DataFrame(
            {
                "Timestamp": ts,
                "Engagement": engagement.values,
            }
        )

        # Temporal band powers saved to features (not in Cognition.csv)
        temporal_bands = pd.DataFrame(
            {
                "Timestamp": ts,
                "avg_beta": avg_beta.values,
                "avg_alpha": avg_alpha.values,
                "avg_theta": avg_theta.values,
                "avg_delta": avg_delta.values,
                "all_power": all_power.values,
                "rel_beta": (avg_beta / (all_power + 1e-12)).values,
                "rel_alpha": (avg_alpha / (all_power + 1e-12)).values,
                "rel_theta": (avg_theta / (all_power + 1e-12)).values,
                "rel_delta": (avg_delta / (all_power + 1e-12)).values,
            }
        )

    except Exception as e:
        logger.error(f"[cognitionProcessor] compute_cognitive_indexes failed: {e}")
        logger.error(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None, None

    return result, temporal_bands


# ---------------------------------------------------------------------------
# Regression-based noise removal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature export helpers
# ---------------------------------------------------------------------------


def save_filtered_eeg_csv(filtered_df, timestamps, out_path):
    """
    Save the bandpass-filtered EEG signal to a CSV file.

    The output has a leading ``Timestamp`` column followed by one column per
    EEG channel, with the same number of rows as the original recording.

    Args:
        filtered_df: Bandpass-filtered EEG DataFrame (samples × channels).
        timestamps: 1-D array of per-sample timestamps in seconds (same length
            as filtered_df).  When None a synthetic 0-based index / sf column
            is written instead.
        out_path: Destination file path.
    """
    try:
        n = len(filtered_df)
        sf = EXG_SAMPLING_RATE

        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=float)
        else:
            ts = np.arange(n, dtype=float) / sf

        out_df = filtered_df.copy()
        out_df.insert(0, "Timestamp", ts)
        out_df.to_csv(out_path, index=False)
    except Exception as exc:
        logger.warning(f"[cognitionProcessor] save_filtered_eeg_csv failed: {exc}")


def save_eeg_artefact_plot(filtered_df, timestamps, rejected_mask, out_path):
    """
    Save a waveform plot of the filtered EEG with artefact regions highlighted.

    One subplot is drawn per EEG channel.  Each subplot shows:
      - The full filtered signal as a black line.
      - Contiguous rejected sample ranges filled in red (semi-transparent).

    A shared legend is placed on the first subplot only.

    Args:
        filtered_df: Bandpass-filtered EEG DataFrame (samples × channels).
        timestamps: 1-D array of per-sample timestamps in seconds.  Falls back
            to a synthetic time axis when None.
        rejected_mask: Boolean 1-D numpy array (same length as filtered_df)
            where True marks samples belonging to a rejected epoch (combined
            hardware invalids + signal-quality artefacts).
        out_path: Destination file path (.png).
    """
    try:
        channel_names = filtered_df.columns.tolist()
        n_channels = len(channel_names)
        n_samples = len(filtered_df)
        sf = EXG_SAMPLING_RATE

        if timestamps is not None:
            t = np.asarray(timestamps, dtype=float)
        else:
            t = np.arange(n_samples, dtype=float) / sf

        mask = np.asarray(rejected_mask, dtype=bool)

        # Identify contiguous rejected runs as (start_time, end_time) spans.
        # Pad the mask with False on both ends so diff detects the first/last edge.
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(int))
        run_starts = np.where(diff == 1)[0]  # indices into original t array
        run_ends = np.where(diff == -1)[0]  # exclusive end indices

        # Convert sample indices to time values (clipped to valid range).
        def _idx_to_t(idx):
            idx = min(idx, n_samples - 1)
            return float(t[idx])

        spans = [(_idx_to_t(s), _idx_to_t(min(e, n_samples - 1))) for s, e in zip(run_starts, run_ends)]

        row_height = 2.0
        fig_height = max(4.0, n_channels * row_height)
        fig, axes = plt.subplots(
            n_channels,
            1,
            figsize=(14, fig_height),
            sharex=True,
        )

        if n_channels == 1:
            axes = [axes]

        for ax_idx, (ax, ch) in enumerate(zip(axes, channel_names)):
            signal = filtered_df[ch].to_numpy(dtype=float)

            # Draw signal in black.
            ax.plot(t, signal, color="black", linewidth=0.6, label="Preserved")

            # Overlay rejected spans in red.
            for i, (t0, t1) in enumerate(spans):
                ax.axvspan(
                    t0,
                    t1,
                    color="red",
                    alpha=0.30,
                    label="Rejected" if (ax_idx == 0 and i == 0) else "_nolegend_",
                )

            ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if ax_idx == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        axes[-1].set_xlabel("Time (s)", fontsize=9)

        pct_rejected = round(100.0 * mask.sum() / max(n_samples, 1), 1)
        fig.suptitle(
            f"Filtered EEG — artefact overlay  ({pct_rejected}% samples rejected)",
            fontsize=11,
            fontweight="bold",
            y=1.01,
        )

        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"[cognitionProcessor] save_eeg_artefact_plot failed: {exc}")


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def computeCognitiveIndexes(recpath, apply_nlms: bool = True):
    """
    Compute cognitive indexes from EEG data using the cleaned temporal pipeline.

    Pipeline steps
    --------------
    1. Load raw EEG; T9/T10 channels derived via midpoint re-reference.
       Optional NLMS adaptive decorrelation applied when apply_nlms=True (default).
    2. Detect hardware timestamp gaps (≥ 5 s).
    3. Bandpass filter (0.5–30 Hz, 4th-order Butterworth zero-phase).
       60 Hz notch is not active because its frequency exceeds the 30 Hz highcut.
    4. Artefact rejection on T9/T10 — 1 s epochs, three signal-quality criteria:
       (1) NaN fraction > 20 %, (2) peak amplitude > 500 µV, (3) ptp > 500 µV.
       The former EMG power-ratio criterion was removed; the 30 Hz bandpass already
       excludes the 30–45 Hz muscle noise band.
       Bad epochs → sample mask; > 25 % of a 2 s analysis window flagged → NaN row.
    5. Welch PSD on T9 and T10 only (2 s windows, 75 % overlap).
       Bands: delta (0–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–22 Hz).
    6. Compute Engagement (bilateral T9+T10 β / (α + θ)) — the sole emitted metric.
    7. Resample to 2 Hz (0.5 s grid).

    Incremental caching
    -------------------
    - ``results/Cognition.csv`` exists → step skipped entirely.
    - ``features/cognition/powerBands.csv`` exists *and* contains T9/T10 channels
      → band powers loaded from cache; steps 1–5 are skipped.
    - Any other case → full recomputation.

    Outputs
    -------
    - ``results/Cognition.csv``                     — primary cognitive metric (Engagement)
    - ``features/cognition/powerBands.csv``         — per-window T9/T10 band powers (long format)
    - ``features/cognition/artefactStats.csv``      — epoch-level rejection summary
    - ``features/cognition/epochMetrics.csv``       — per-epoch signal metrics
    - ``features/cognition/temporalBandPowers.csv`` — bilateral temporal band averages
    - ``features/cognition/eegArtefactPlot.png``    — filtered EEG waveform with rejected regions

    Args:
        recpath: Path to the recording directory.
        apply_nlms: Apply causal NLMS adaptive decorrelation before filtering (default True).
    """
    logger.info("[cognitionProcessor] Computing Cognitive Indexes")

    cognitive_path = os.path.join(recpath, "results", "Cognition.csv")
    features_dir = os.path.join(recpath, "features", "cognition")
    powerbands_path = os.path.join(features_dir, "powerBands.csv")
    artefact_path = os.path.join(features_dir, "artefactStats.csv")
    temporal_bp_path = os.path.join(features_dir, "temporalBandPowers.csv")

    if os.path.isfile(cognitive_path):
        logger.info("[cognitionProcessor] Cognitive indexes already computed, using cached results")
        return

    try:
        powerbands = None

        if os.path.isfile(powerbands_path):
            logger.info(f"[cognitionProcessor] Loading cached power bands from {powerbands_path}")
            powerbands = pd.read_csv(powerbands_path)
            required_cols = {"Timestamp", "channel", "band", "power"}
            if not required_cols.issubset(set(powerbands.columns)):
                logger.error(
                    f"[cognitionProcessor] Cached powerBands.csv missing columns "
                    f"{required_cols - set(powerbands.columns)} — will recompute"
                )
                powerbands = None
            elif not {"T9", "T10"}.issubset(set(powerbands["channel"].unique())):
                logger.info(
                    "[cognitionProcessor] Cached powerBands.csv uses old channel layout "
                    "(missing T9/T10) — recomputing from EEG"
                )
                powerbands = None
            else:
                logger.info("[cognitionProcessor] Cache is valid, skipping EEG reprocessing")

        if powerbands is None:
            logger.info("[cognitionProcessor] Loading EEG data...")
            recording = load_exg(recpath)
            if recording is None or recording.data.shape[1] < 5:
                logger.error("[cognitionProcessor] No usable EXG data")
                return
            # Invalid raw samples (D6, D24) become NaN; the derivation below only uses ch1, ch2 and
            # ch5, so a sample is invalid for cognition only when one of those is (D23).
            raw = mark_invalid(recording.data)
            if apply_nlms:
                logger.info("[cognitionProcessor] NLMS adaptive decorrelation enabled")
                nlms = CausalNLMSFilter(n_channels=raw.shape[1], fs=EXG_SAMPLING_RATE)
                raw = nlms_nan_safe(nlms.push_batch, raw)
            original_timestamps = recording.timestamps
            eegData = derive_eeg_channels(raw)

            sf = EXG_SAMPLING_RATE

            if original_timestamps is None:
                logger.warning(
                    "[cognitionProcessor] Hardware timestamps unavailable — "
                    "falling back to synthetic timestamps (sample index / fs). "
                    "Gap detection is disabled."
                )
                original_timestamps = np.arange(len(eegData), dtype=float) / sf
                gap_sample_mask = np.zeros(len(eegData), dtype=bool)
            else:
                dt = np.diff(original_timestamps)
                gap_indices = np.where(dt >= GAP_THRESHOLD_S)[0]
                gap_sample_mask = np.zeros(len(original_timestamps), dtype=bool)
                if len(gap_indices):
                    logger.warning(
                        f"[cognitionProcessor] {len(gap_indices)} hardware timestamp gap(s) "
                        f">= {GAP_THRESHOLD_S:.0f} s detected — affected windows will be NaN."
                    )
                    gap_sample_mask[gap_indices + 1] = True

            # Capture hardware invalids BEFORE any interpolation so artefact
            # rejection can assess raw NaN fraction per epoch.
            hardware_invalid = eegData.isna().any(axis=1).to_numpy() | gap_sample_mask

            logger.info("[cognitionProcessor] Filtering EEG (bandpass + notch)...")
            try:
                filtered_eeg = bandpass_filter(eegData, sf)
            except BaseException as e:
                logger.error(f"[cognitionProcessor] bandpass_filter failed: {type(e).__name__}: {e}")
                logger.error(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
                return

            logger.info("[cognitionProcessor] Running temporal artefact rejection...")
            artefact_mask, artefact_stats, epoch_metrics = reject_temporal_artefacts(filtered_eeg, hardware_invalid, sf)

            # Combine hardware invalids with artefact-rejected samples.
            # The 25 % window threshold in compute_eeg_power_bands treats both equally.
            combined_invalid = hardware_invalid | artefact_mask

            os.makedirs(features_dir, exist_ok=True)

            filtered_eeg_path = os.path.join(features_dir, "filteredEEG.csv")
            artefact_plot_path = os.path.join(features_dir, "eegArtefactPlot.png")
            epoch_metrics_path = os.path.join(features_dir, "epochMetrics.csv")

            if not os.path.isfile(filtered_eeg_path):
                save_filtered_eeg_csv(filtered_eeg, original_timestamps, filtered_eeg_path)

            if not os.path.isfile(artefact_plot_path):
                save_eeg_artefact_plot(filtered_eeg, original_timestamps, combined_invalid, artefact_plot_path)

            logger.info("[cognitionProcessor] Computing power bands...")
            powerbands = compute_eeg_power_bands(
                filtered_eeg[["T9", "T10"]],
                timestamps=original_timestamps,
                hardware_invalid=combined_invalid,
            )
            if powerbands is None:
                logger.error("[cognitionProcessor] compute_eeg_power_bands returned None — cannot proceed")
                return

            os.makedirs(features_dir, exist_ok=True)
            powerbands.to_csv(powerbands_path, index=False)

            if artefact_stats is not None:
                artefact_stats.to_csv(artefact_path, index=False)

            if epoch_metrics is not None:
                epoch_metrics.to_csv(epoch_metrics_path, index=False)

        logger.info("[cognitionProcessor] Computing cognitive indexes...")
        result, temporal_bands = compute_cognitive_indexes(powerbands)
        if result is None:
            logger.error("[cognitionProcessor] compute_cognitive_indexes returned None — cannot proceed")
            return

        if temporal_bands is not None:
            os.makedirs(features_dir, exist_ok=True)
            temporal_bands.to_csv(temporal_bp_path, index=False)

        result = resample_to_timebase(result, target_interval=0.5)

        os.makedirs(os.path.dirname(cognitive_path), exist_ok=True)
        result.to_csv(cognitive_path, index=False)

        logger.info("[cognitionProcessor] Cognitive index computation completed")

    except Exception as e:
        logger.error(f"[cognitionProcessor] Unhandled error: {e}")
        logger.error(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")


def loadCognitiveIndexes(src):
    """
    Load cognitive index data from CSV file.

    Args:
        src: Path to recording directory

    Returns:
        pandas DataFrame containing cognitive indexes, or None if not found.
    """
    cognitive_path = os.path.join(src, "results", "Cognition.csv")

    if not os.path.isfile(cognitive_path):
        logger.warning(f"[cognitionProcessor] Cognitive file not found: {cognitive_path}")
        return None

    return pd.read_csv(cognitive_path)
