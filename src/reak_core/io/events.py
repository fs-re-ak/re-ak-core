# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Raw event files: loading and the processed-event writer (decision D3: Prime behaviour).

Event files are ``timestamp,eventName,data`` rows; Vizia writes JSON payloads unquoted, so a
row can contain more than three comma-separated cells (they are re-joined into the payload).
A header row is optional. Timestamps are re-based on an optional ``TIMESTAMP`` synchronisation
event.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import csv
import logging
import os

logger = logging.getLogger(__name__)

EVENT_FILENAMES: tuple[str, ...] = ("rawEvents.csv", "events_0.csv", "event.csv")
"""Raw event files, in discovery order."""


def find_events_file(session_root: str) -> str | None:
    for name in EVENT_FILENAMES:
        path = os.path.join(session_root, "rawData", name)
        if os.path.exists(path):
            return path
    return None


def _correct_timestamps(rock_ref: float | None, pc_ref: float | None, events: list) -> None:
    for event in events:
        event[0] = float(str(event[0]).strip("'"))
        if rock_ref is not None:
            event[0] += rock_ref - pc_ref
            event[0] -= rock_ref - 1


def read_raw_events(path: str) -> list[list] | None:
    """Parse one raw event file into ``[time, name, payload]`` rows, or ``None`` when absent."""
    if not os.path.isfile(path):
        return None
    events: list[list] = []
    pc_ref = rock_ref = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        no_timestamp_header = True
        for row in csv.reader(fh):
            row = [cell.strip("'").strip() for cell in row]  # Vizia used to insert quotes in event types
            if len(row) > 3:
                row[2] = ",".join(row[2:])
                row = row[:3]
            if no_timestamp_header and len(row) >= 1:
                try:
                    float(row[0])
                except ValueError:
                    continue  # header row
            events.append(row)
            if len(row) >= 3:
                if row[1] == "TIMESTAMP":
                    rock_ref, pc_ref = float(row[2]), float(row[0])
                elif row[2] == "TIMESTAMP":  # badly formatted synchronisation event
                    rock_ref, pc_ref = float(row[1]), float(row[0])
                elif no_timestamp_header:
                    rock_ref = pc_ref = float(row[0])
                no_timestamp_header = False
    _correct_timestamps(rock_ref, pc_ref, events)
    if events and (events[0][1] == "TIMESTAMP" or (len(events[0]) > 2 and events[0][2] == "TIMESTAMP")):
        events.remove(events[0])
    return events


def load_raw_events(session_root: str) -> list[list] | None:
    """Load the session's raw events (first file found in EVENT_FILENAMES), or ``None``."""
    path = find_events_file(session_root)
    if path is None:
        logger.info("No raw events file in %s (tried %s)", os.path.join(session_root, "rawData"), EVENT_FILENAMES)
        return None
    return read_raw_events(path)


def write_events(events: list[list], filename: str) -> None:
    """Write processed events as ``name,PROCESSED,time_ms,,payload`` rows."""
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for time, name, value in (event[:3] for event in events):
            payload = "" if str(value) == "[]" else value.replace('"', "")
            writer.writerow([name, "PROCESSED", str(time * 1000), "", payload])
