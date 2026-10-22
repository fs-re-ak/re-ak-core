# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
RE-AK open processing core.

Signal processing and models for Hermès EEG / fEMG and Shimmer EDA / PPG recordings, shared by
the Nucleus-Kit toolkit (local) and the re-akcas online processor (cloud). Nothing in this
package depends on a wrapper or on a proprietary module; extensions plug in through the
``reak_core.steps``, ``reak_core.models`` and ``reak_core.exporters`` entry points.

The package logs through the standard :mod:`logging` module under the ``reak_core`` logger
and never opens a network connection.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

import logging

__version__ = "1.0.1"

logging.getLogger(__name__).addHandler(logging.NullHandler())
