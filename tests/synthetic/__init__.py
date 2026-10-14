# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Deterministic synthetic sessions for reference recording and non-regression tests."""

from .cases import CASES, Case, build, by_name
from .generator import Session, write_session

__all__ = ["CASES", "Case", "Session", "build", "by_name", "write_session"]
