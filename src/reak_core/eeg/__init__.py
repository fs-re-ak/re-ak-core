# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""EEG: NLMS decorrelation, bandpass, epoch artefact rejection, band powers, Engagement (D4, D5, D23)."""

from reak_core.eeg.cognition import computeCognitiveIndexes, loadCognitiveIndexes
from reak_core.eeg.nlms import CausalNLMSFilter, nlms_nan_safe

compute_cognition = computeCognitiveIndexes

__all__ = ["CausalNLMSFilter", "computeCognitiveIndexes", "compute_cognition", "loadCognitiveIndexes", "nlms_nan_safe"]
