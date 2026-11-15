# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Events: processed feature events and seeded playback annotations.

- :func:`process_feature_events` splits the raw events into feature events and web events and
  writes them to ``features/events/`` (decision D16), dropping administrative events (D3).
  Raw events are discovered among ``rawEvents.csv``, ``events_0.csv`` and ``event.csv`` (D3).
- :func:`seed_playback_annotations` turns ``rawData/event.csv`` into
  ``features/events/playback_annotations.json`` once, so manual edits are never overwritten.
- :func:`recording_zones` returns the zones recorded with ``ZONE_START`` / ``ZONE_STOP`` tags in the raw events,
  in the time base of the session results (seconds from the first EXG sample).

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid

from reak_core.io.events import find_events_file, load_raw_events, write_events
from reak_core.io.exg import find_exg_file, load_exg_matrix

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

ZONE_START = "ZONE_START"
ZONE_STOP = "ZONE_STOP"
"""Zone tags (sent by Prime's live zones): ``Values`` = ``<zoneId>|<name>``; a start and a stop with the same id
make one zone."""


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


def _read_event_rows(path: str, lenient: bool = False) -> list[tuple[float, str, str]] | None:
    """``(time, name, value)`` rows of an event file; an unquoted value holding commas is joined back.

    ``lenient`` also accepts the rows as the session API returns them (``t, 'NAME', value``: spaces after the
    commas, names in quotes).
    """
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for raw in csv.reader(fh, skipinitialspace=lenient):
            if len(raw) < 2:
                continue
            name, value = raw[1].strip(), ",".join(raw[2:]).strip()
            if lenient:
                name, value = name.strip("'\""), value.strip("'")
            try:
                rows.append((float(raw[0].strip().strip("'")), name, value))
            except ValueError:
                continue
    return rows or None


def _zone_tag(value: str) -> tuple[str, str]:
    """``(zone id, name)`` of a zone tag value ``<zoneId>|<name>``; a value without ``|`` is both."""
    zone_id, sep, name = value.partition("|")
    zone_id, name = zone_id.strip(), name.strip()
    if not sep:
        name = zone_id
    return zone_id, name or zone_id or "Zone"


def pair_zone_events(
    rows: list[tuple], start: float | None = None, end: float | None = None
) -> tuple[list[dict], set[int]]:
    """Pair ``ZONE_START`` / ``ZONE_STOP`` rows by zone id.

    Returns the zones (``zone_id``, ``label``, ``t0``, ``t1``, in the rows' time) and the indices of the rows they
    used. A start without a stop ends at ``end``, and a stop without a start begins at ``start``; when that bound
    is ``None`` the row is not used, so it stays a point event. The label is the start's name (the stop's when the
    start is missing).
    """
    zones: list[dict] = []
    used: set[int] = set()
    open_zones: dict[str, tuple[float, str, int]] = {}
    for i, row in enumerate(rows):
        name = row[1]
        if name not in (ZONE_START, ZONE_STOP):
            continue
        zone_id, label = _zone_tag(row[2] if len(row) > 2 else "")
        t = float(row[0])
        if name == ZONE_START:
            if zone_id not in open_zones:
                open_zones[zone_id] = (t, label, i)
            continue
        if zone_id in open_zones:
            t0, label, j = open_zones.pop(zone_id)
            used.update((i, j))
        elif start is not None:
            t0 = start
            used.add(i)
        else:
            continue
        if t > t0:
            zones.append({"zone_id": zone_id, "label": label, "t0": t0, "t1": t})
    if end is not None:
        for zone_id, (t0, label, j) in open_zones.items():
            used.add(j)
            if end > t0:
                zones.append({"zone_id": zone_id, "label": label, "t0": t0, "t1": end})
    zones.sort(key=lambda z: (z["t0"], z["t1"]))
    return zones, used


def build_annotations(rows: list[tuple]) -> dict | None:
    """Annotation document from ``(time, name[, value])`` rows, or ``None`` without RECORDING_ONSET.

    Paired ``ZONE_START`` / ``ZONE_STOP`` tags become zones; every other non-administrative event is a point.
    """
    onset = next((row[0] for row in rows if row[1] == _RECORDING_ONSET), None)
    if onset is None:
        return None
    zones, used = pair_zone_events(rows)
    points = [
        {"id": str(uuid.uuid4()), "t": round(row[0] - onset, 6), "label": row[1], "visible": True}
        for i, row in enumerate(rows)
        if row[1] not in ADMIN_EVENTS and i not in used
    ]
    zone_items = [
        {
            "id": str(uuid.uuid4()),
            "t0": round(z["t0"] - onset, 6),
            "t1": round(z["t1"] - onset, 6),
            "label": z["label"],
            "visible": True,
        }
        for z in zones
    ]
    return {"version": ANNOTATIONS_SCHEMA_VERSION, "points": points, "zones": zone_items}


def _epoch_seconds(value: float) -> float | None:
    """Unix time in seconds from an epoch timestamp in s, ms, µs or ns; ``None`` for a relative clock."""
    value = float(value)
    for limit, scale in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3), (1e8, 1.0)):
        if abs(value) > limit:
            return value / scale
    return None


def recording_zones(session_root: str) -> list[dict] | None:
    """Zones recorded with ``ZONE_START`` / ``ZONE_STOP`` tags in the session's raw events.

    The events come from the first raw event file found (``rawEvents.csv`` as downloaded by Prime,
    ``events_0.csv``, ``event.csv`` as written by Vizia).

    Times are seconds from the first EXG sample, the time base of the session results. A zone still open at the
    end of the recording ends with it; a stop whose start came before the recording begins at 0. Zones are
    clipped to the recording.

    Returns ``[{"zone_id", "label", "start_s", "end_s"}]`` sorted by start, ``[]`` when the session has no zone
    tag, or ``None`` when the zones cannot be placed (no EXG file, or clocks that are not Unix time).
    """
    events_path = find_events_file(session_root)
    rows = _read_event_rows(events_path, lenient=True) if events_path else None
    if not rows or not any(row[1] in (ZONE_START, ZONE_STOP) for row in rows):
        return []
    path = find_exg_file(session_root)
    if path is None:
        logger.warning("Zone tags found but no EXG file, zones not placed")
        return None
    try:
        matrix = load_exg_matrix(path)
    except (OSError, ValueError) as exc:
        logger.error("Could not load EXG file %s: %s", path, exc)
        return None
    if matrix.ndim < 2 or matrix.shape[0] == 0:
        logger.warning("Empty EXG file %s, zones not placed", path)
        return None
    origin, last = _epoch_seconds(matrix[0, 0]), _epoch_seconds(matrix[-1, 0])
    events = [(_epoch_seconds(t), name, value) for t, name, value in rows]
    if origin is None or last is None or any(t is None for t, _, _ in events):
        logger.warning("EXG or event times are not Unix time, zones not placed")
        return None
    zones, _ = pair_zone_events(events, start=origin, end=last)
    duration = last - origin
    out = []
    for z in zones:
        start_s = min(max(z["t0"] - origin, 0.0), duration)
        end_s = min(max(z["t1"] - origin, 0.0), duration)
        if end_s > start_s:
            out.append(
                {"zone_id": z["zone_id"], "label": z["label"], "start_s": round(start_s, 3), "end_s": round(end_s, 3)}
            )
    logger.info("Found %d recorded zone(s)", len(out))
    return out


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
