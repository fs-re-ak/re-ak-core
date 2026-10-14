# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""PPG: SQI and MAD artifact rejection, peaks, missed beats, HRV (NucleusKit implementation, D11–D15)."""

from reak_core.ppg.heart import apply_ppg_artifact_rejection, computeHeartDynamics

compute_heart_dynamics = computeHeartDynamics

__all__ = ["apply_ppg_artifact_rejection", "computeHeartDynamics", "compute_heart_dynamics"]
