# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Events: processed feature events and seeded playback annotations.

- :func:`process_feature_events` splits the raw events into feature events and web events and
  writes them to ``features/events/`` (decision D16), dropping administrative events (D3).
  Raw events are discovered among ``rawEvents.csv``, ``events_0.csv`` and ``event.csv`` (D3).
- :func:`seed_playback_annotations` turns ``rawData/event.csv`` into
  ``features/events/playback_annotations.json`` once, so manual edits are never overwritten.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid

from reak_core.io.events import load_raw_events, write_events

logger = logging.getLogger(__name__)

EVENTS_DIR = os.path.join("features", "events")
ADMIN_EVENTS = frozenset({"RECIPE_CONFIG", "SYSTEM_CONFIG", "CONSENT_INFO", "RECORDING_ONSET"})
"""Pipeline-configuration events: never part of the analysis events nor of the annotations."""
WEB_EVENTS = frozenset({"WEB_SCROLL", "WEB_TAB_UPDATE", "WEB_INITIAL_SCROLL_POSITION"})
BEHAVIOURAL_EVENTS = frozenset({"NO_KEYPRESS", "KEYPRESS", "ANSWER"})

ANNOTATIONS_FILE = "playback_annotations.json"
ANNOTATIONS_SCHEMA_VERSION = 1
_ANNOTATION_SOURCE = "event.csv"
_RECORDING_ONSET = "RECORDING_ONSET"


def process_feature_events(session_root: str) -> tuple[int, int] | None:
    """Write ``processedFeatureEvents.csv`` and ``processedWebFeatureEvents.csv``.

    Returns ``(feature events, web events)``, or ``None`` when the session has no raw events.
    """
    raw = load_raw_events(session_root)
    if raw is None:
        logger.info("No raw events, skipping feature events")
        return None
    feature, web = [], []
    for event in raw:
        kind = event[1] if len(event) > 1 else ""
        if kind in ADMIN_EVENTS:
            continue
        if kind in WEB_EVENTS:
            web.append(event)
            if kind == "WEB_TAB_UPDATE" and len(event) > 2 and "complete" in event[2]:
                feature.append(event)
        elif kind in BEHAVIOURAL_EVENTS:
            continue
        else:
            feature.append(event)
    out_dir = os.path.join(session_root, EVENTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    write_events(list(feature), os.path.join(out_dir, "processedFeatureEvents.csv"))
    write_events(list(web), os.path.join(out_dir, "processedWebFeatureEvents.csv"))
    logger.info("Processed %d feature events and %d web events", len(feature), len(web))
    return len(feature), len(web)


def _read_event_rows(path: str) -> list[tuple[float, str]] | None:
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for raw in csv.reader(fh):
            if len(raw) < 2:
                continue
            try:
                rows.append((float(raw[0].strip()), raw[1].strip()))
            except ValueError:
                continue
    return rows or None


def build_annotations(rows: list[tuple[float, str]]) -> dict | None:
    """Annotation document with one point per non-administrative event, or ``None`` without RECORDING_ONSET."""
    onset = next((ts for ts, name in rows if name == _RECORDING_ONSET), None)
    if onset is None:
        return None
    points = [
        {"id": str(uuid.uuid4()), "t": round(ts - onset, 6), "label": name, "visible": True}
        for ts, name in rows
        if name not in ADMIN_EVENTS
    ]
    return {"version": ANNOTATIONS_SCHEMA_VERSION, "points": points, "zones": []}


def seed_playback_annotations(session_root: str) -> bool:
    """Write ``features/events/playback_annotations.json`` from ``rawData/event.csv`` if it does not exist yet."""
    output = os.path.join(session_root, EVENTS_DIR, ANNOTATIONS_FILE)
    if os.path.isfile(output):
        logger.info("%s already exists, keeping it", ANNOTATIONS_FILE)
        return False
    rows = _read_event_rows(os.path.join(session_root, "rawData", _ANNOTATION_SOURCE))
    if rows is None:
        logger.info("%s not found or empty, no annotations seeded", _ANNOTATION_SOURCE)
        return False
    annotations = build_annotations(rows)
    if annotations is None:
        logger.warning("RECORDING_ONSET not found in %s, no annotations seeded", _ANNOTATION_SOURCE)
        return False
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(annotations, indent=2))
    logger.info("Wrote %d annotation(s)", len(annotations["points"]))
    return True
