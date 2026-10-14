# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Matplotlib setup for the diagnostic figures (always written, decision Phase 3 question 2).

The core only saves figures to files, so it selects the non-interactive ``Agg`` backend, unless the
host application imported ``matplotlib.pyplot`` first, in which case its choice is kept.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import sys

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt"]
