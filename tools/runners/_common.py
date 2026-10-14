# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Helpers shared by the reference runners.

The runners execute the *current* NucleusKit and Prime code on a synthetic session, one
system per process, so that the two code bases never share ``sys.modules``.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback


def setup_io() -> None:
    """Both systems print non-ASCII characters; piped output on Windows defaults to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")


def block_network() -> None:
    """Make every outgoing connection fail: the runners must never reach an API."""

    def _refuse(*args, **kwargs):
        raise RuntimeError("network access is disabled while recording references")

    socket.socket.connect = _refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse  # type: ignore[method-assign]
    socket.create_connection = _refuse  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="checkout of the system to run")
    parser.add_argument("--session", required=True, help="session root (contains rawData/)")
    parser.add_argument("--model", required=True, help="emotion model release key, e.g. classical-emotion-2.6.0")
    return parser.parse_args()


def run_steps(steps: list, session: str, report_path: str, context: dict) -> int:
    """Run ``(name, callable)`` steps in order, record status and duration, never stop on error."""
    report = dict(context, steps=[])
    for name, fn in steps:
        started = time.time()
        entry = {"step": name}
        try:
            fn(session)
            entry["status"] = "ok"
        except BaseException as exc:  # noqa: BLE001 — the systems themselves catch BaseException per step
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
        entry["seconds"] = round(time.time() - started, 2)
        report["steps"].append(entry)
        print(f"[runner] {name}: {entry['status']} ({entry['seconds']} s)", file=sys.stderr)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return 0
