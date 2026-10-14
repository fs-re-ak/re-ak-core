# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""EDA: artifact rejection, cvxEDA tonic / phasic decomposition, SCR events, arousal."""

from reak_core.eda.arousal import computeArousal, loadArousal

compute_arousal = computeArousal

__all__ = ["computeArousal", "compute_arousal", "loadArousal"]
