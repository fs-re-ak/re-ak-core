# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Robust CSV helpers for raw recording files (decision D2: NucleusKit behaviour).

Raw files exist with and without a header row, and an interrupted recording or upload can
leave an incomplete last line. Both are accepted.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import numpy as np


def has_text_header(filepath: str, delimiter: str = ",") -> bool:
    """True when the first non-empty line has at least one field that is not a number."""
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    [float(x) for x in line.split(delimiter)]
                    return False
                except ValueError:
                    return True
    return False


def read_header(filepath: str, delimiter: str = ",") -> list[str] | None:
    """Return the header fields when the file has a text header, else ``None``."""
    if not has_text_header(filepath, delimiter):
        return None
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                return [cell.strip() for cell in line.strip().split(delimiter)]
    return None


def loadtxt_drop_last_if_incomplete(filepath: str, delimiter: str = ",") -> np.ndarray:
    """Load a numeric CSV into an array.

    Tries, in order: the file as is; without its last line (incomplete last line); without its
    first line (text header); without both. Raises ``ValueError`` when none of these parses.
    """
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    for candidate in (lines, lines[:-1], lines[1:], lines[1:-1]):
        try:
            return np.loadtxt(candidate, delimiter=delimiter)
        except ValueError:
            continue
    raise ValueError(f"Failed to load '{filepath}': tried with and without header and last line.")
