# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Session folder: layout, discovery of raw streams, output directories.

A session root holds ``rawData/`` (inputs), ``features/`` (intermediate and diagnostic outputs),
``results/`` (tables on the shared 2 Hz timebase), ``meta/`` and ``experimentConfigs/``.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

from reak_core.io.events import find_events_file
from reak_core.io.exg import find_exg_file
from reak_core.io.positioning import GPS_FILENAMES, UWB_FILENAMES
from reak_core.io.shimmer import find_shimmer_file

logger = logging.getLogger(__name__)

RESERVED_DIRS: frozenset[str] = frozenset({"rawData", "features", "results", "experimentConfigs", "meta"})
OUTPUT_DIRS: tuple[str, ...] = ("features", "results", "meta", "experimentConfigs")
PROCESSING_LOG = os.path.join("results", "processingLog.txt")


@dataclass(frozen=True)
class Session:
    """A session folder on disk. Paths are absolute."""

    root: str

    @classmethod
    def open(cls, root: str, *, fix_layout: bool = True) -> Session:
        """Open a session folder; when ``rawData/`` is missing, optionally move the raw files into it."""
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        if fix_layout:
            ensure_rawdata_layout(root)
        return cls(root)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "rawData")

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def modalities(self) -> dict[str, str | None]:
        """Raw file found for each stream (``None`` when absent)."""
        return {
            "exg": find_exg_file(self.root),
            "shimmer": find_shimmer_file(self.root),
            "events": find_events_file(self.root),
            "uwb": _first_existing(self.raw_dir, UWB_FILENAMES),
            "gps": _first_existing(self.raw_dir, GPS_FILENAMES),
        }

    def prepare(self) -> None:
        """Create the output directories and start a new processing log."""
        prepare_directories(self.root)


def _first_existing(directory: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def ensure_rawdata_layout(session_root: str) -> None:
    """When ``rawData/`` is missing, create it and move every non-reserved top-level entry into it."""
    session_root = os.path.abspath(session_root)
    raw = os.path.join(session_root, "rawData")
    if os.path.exists(raw):
        return
    entries = os.listdir(session_root)
    os.makedirs(raw, exist_ok=True)
    for name in entries:
        if name not in RESERVED_DIRS:
            shutil.move(os.path.join(session_root, name), raw)


def prepare_directories(session_root: str) -> None:
    ensure_rawdata_layout(session_root)
    for name in OUTPUT_DIRS:
        os.makedirs(os.path.join(session_root, name), exist_ok=True)
    started = datetime.now().strftime("%d_%m_%y-%H_%M_%S")
    try:
        with open(os.path.join(session_root, PROCESSING_LOG), "w", encoding="utf-8") as fh:
            fh.write("Processing log for Analysis started at " + started)
    except OSError:
        logger.warning("Cannot write %s, skipping", PROCESSING_LOG)
