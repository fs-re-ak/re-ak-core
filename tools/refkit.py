# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Reference toolkit: generate synthetic sessions, run a system on them, collect and compare
outputs.

Used by ``record_references.py`` (record), ``report_divergences.py`` (NK versus Prime) and
by the non-regression tests of NucleusKit and Prime (a system versus its own stored
references, through :func:`check_system`).

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES_DIR = os.path.join(CORE_ROOT, "tests", "references")
RUNNERS = {
    "nk": os.path.join(CORE_ROOT, "tools", "runners", "run_nk.py"),
    "prime": os.path.join(CORE_ROOT, "tools", "runners", "run_prime.py"),
}
MODELS = ("classical-emotion-2.6.0", "classical-emotion-2.5.0")
PRIMARY_MODEL = MODELS[0]

sys.path.insert(0, os.path.join(CORE_ROOT, "tests"))
from synthetic import CASES, build, by_name, write_session  # noqa: E402

# Outputs kept as references. Images, logs, large intermediates (filteredEEG.csv,
# ppg_normalized.csv, eda_resampled.csv) and outputs dropped by the Phase 1 decisions
# (temporalBandPowers.csv, D5) are not stored.
FULL_OUTPUTS = (
    "results/Emotions.csv",
    "results/Cognition.csv",
    "results/Arousal.csv",
    "results/HeartDynamics.csv",
    "results/emotion_model.json",
    "features/metainfo.json",
    "meta/metainfo.json",
    "features/cognition/powerBands.csv",
    "features/cognition/artefactStats.csv",
    "features/cognition/epochMetrics.csv",
    "features/eda/SCR_events.csv",
    "features/ppg/ppg_resampled.csv",
    "features/emotions/emotionClassifierInputs.csv",
    "features/emotions/rmsSignals.csv",
    "features/emotions/avgRms.csv",
    "features/emotions/relabeled-samples.csv",
    "features/emotions/model_info.json",
    "features/processedFeatureEvents.csv",
    "features/events/processedFeatureEvents.csv",
    "features/events/playback_annotations.json",
)
EMOTION_OUTPUTS = (
    "results/Emotions.csv",
    "results/emotion_model.json",
    "features/emotions/emotionClassifierInputs.csv",
    "features/emotions/rmsSignals.csv",
    "features/emotions/avgRms.csv",
    "features/emotions/relabeled-samples.csv",
    "features/emotions/model_info.json",
)

# JSON values that change on every run and are replaced by a placeholder when collected.
VOLATILE_JSON_KEYS = {"processed_at", "generated_at", "created_at"}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

# Outputs longer than this are stored as a summary (shape and column statistics) instead of
# the full table. Sessions last 150 s, so only a misread time axis (D1) gets there.
MAX_STORED_ROWS = 50_000

# Tolerances for a system against its own references (same code, same environment).
RTOL = 1e-6
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Generate and run
# ---------------------------------------------------------------------------


def generate(case_name: str, dest: str) -> dict:
    """Write one synthetic case under ``dest`` and return its raw-file hashes."""
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    return write_session(build(by_name(case_name)), dest)


def run_system(system: str, repo: str, session: str, model: str) -> dict:
    """Run one system on one session in a separate process and return its run report."""
    cmd = [sys.executable, RUNNERS[system], "--repo", repo, "--session", session, "--model", model]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        cmd, cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800
    )
    with open(os.path.join(session, "run_output.log"), "w", encoding="utf-8") as fh:
        fh.write(proc.stdout)
        fh.write(proc.stderr)
    report_path = os.path.join(session, "reference_run.json")
    if proc.returncode != 0 or not os.path.isfile(report_path):
        raise RuntimeError(f"{system} runner failed on {session}:\n{proc.stderr[-4000:]}")
    with open(report_path, encoding="utf-8") as fh:
        return json.load(fh)


def run_case(system: str, repo: str, case_name: str, workdir: str) -> dict:
    """Generate one case and run the system with every model. Returns {model: session dir}."""
    raw = os.path.join(workdir, "raw", case_name)
    generate(case_name, raw)
    sessions = {}
    previous = None
    for model in MODELS:
        session = os.path.join(workdir, system, model, case_name)
        if os.path.isdir(session):
            shutil.rmtree(session)
        # The second model starts from the first run, so the cached non-emotion steps are reused
        shutil.copytree(previous or raw, session)
        for leftover in ("reference_run.json", "run_output.log"):
            path = os.path.join(session, leftover)
            if os.path.isfile(path):
                os.remove(path)
        run_system(system, repo, session, model)
        sessions[model] = session
        previous = session
    return sessions


# ---------------------------------------------------------------------------
# Collect and sanitize
# ---------------------------------------------------------------------------


def _path_patterns(paths: list[str]) -> list[re.Pattern]:
    pats = []
    for p in paths:
        p = os.path.abspath(p).rstrip("\\/")
        variants = {p, p.replace("\\", "/"), p.replace("/", "\\"), p.replace("\\", "\\\\")}
        for v in sorted(variants, key=len, reverse=True):
            pats.append(re.compile(re.escape(v), re.IGNORECASE))
    return pats


def sanitize_text(text: str, replacements: dict) -> str:
    """Replace absolute local paths (session, repos, home) so nothing personal is published."""
    for placeholder, path in sorted(replacements.items(), key=lambda kv: len(kv[1]), reverse=True):
        for pat in _path_patterns([path]):
            text = pat.sub(placeholder, text)
    return text


def normalize_json(value):
    """Replace run-dependent JSON values (timestamps, random UUIDs) with stable placeholders."""
    if isinstance(value, dict):
        return {k: ("<volatile>" if k in VOLATILE_JSON_KEYS else normalize_json(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    if isinstance(value, str) and _UUID.match(value):
        return "<uuid>"
    return value


def assert_no_local_paths(text: str, where: str) -> None:
    home = os.path.expanduser("~")
    for needle in (home, home.replace("\\", "/"), "AppData", "/Users/", "\\Users\\"):
        if needle and needle.lower() in text.lower():
            raise ValueError(f"local path left in {where}: {needle!r}")


def collect(session: str, dest: str, outputs: tuple, replacements: dict) -> list[str]:
    """Copy the listed outputs and the run report of a run into ``dest``, sanitized. Returns the stored paths."""
    stored = collect_outputs(session, dest, outputs, replacements)
    with open(os.path.join(session, "reference_run.json"), encoding="utf-8") as fh:
        run = json.load(fh)
    for step in run["steps"]:
        step.pop("traceback", None)  # keep the error line only: tracebacks hold local paths
        if "error" in step:
            step["error"] = sanitize_text(step["error"], replacements)
    text = json.dumps(run, indent=2)
    assert_no_local_paths(text, "reference_run.json")
    with open(os.path.join(dest, "reference_run.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return stored


def collect_outputs(session: str, dest: str, outputs: tuple, replacements: dict) -> list[str]:
    """Copy the listed outputs of a run into ``dest`` (CSV gzipped), sanitized. Returns the stored paths."""
    stored = []
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    for rel in outputs:
        src = os.path.join(session, rel)
        if not os.path.isfile(src):
            continue
        with open(src, encoding="utf-8", errors="replace") as fh:
            text = sanitize_text(fh.read(), replacements)
        assert_no_local_paths(text, rel)
        target = os.path.join(dest, rel.replace("/", "__"))
        if rel.endswith(".csv"):
            frame = pd.read_csv(io.StringIO(text), na_values=["NULL"], keep_default_na=True)
            if len(frame) > MAX_STORED_ROWS:
                # A 150 s session never needs that many rows: the time axis was misread (D1).
                with open(target + ".summary.json", "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(summarize(frame), fh, indent=2)
                    fh.write("\n")
                stored.append(rel)
                continue
            target += ".gz"
            with gzip.GzipFile(target, "wb", mtime=0) as gz:  # mtime=0: byte-identical re-records
                gz.write(text.encode("utf-8"))
        else:
            text = json.dumps(normalize_json(json.loads(text)), indent=2) + "\n"
            with open(target, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        stored.append(rel)
    return stored


def reference_dir(system: str, model: str, case_name: str, root: str = REFERENCES_DIR) -> str:
    return os.path.join(root, system, model, case_name)


# ---------------------------------------------------------------------------
# Load and compare
# ---------------------------------------------------------------------------


def summarize(frame: pd.DataFrame) -> dict:
    """Shape and per-column statistics of an output too large to store (see MAX_STORED_ROWS)."""
    columns = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            columns[col] = {
                "nan": int(np.isnan(values).sum()),
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std()) if finite.size else None,
            }
        else:
            columns[col] = {"distinct": int(series.nunique(dropna=True)), "missing": int(series.isna().sum())}
    return {"summary": True, "rows": len(frame), "columns": columns}


def _compare_summaries(name: str, a: dict, b: dict, rtol: float, atol: float) -> list[Difference]:
    if a["rows"] != b["rows"] or list(a["columns"]) != list(b["columns"]):
        return [Difference(name, "shape", f"summary: {a['rows']} rows against {b['rows']} rows or columns differ")]
    diffs = []
    for col, sa in a["columns"].items():
        sb = b["columns"][col]
        for key, va in sa.items():
            vb = sb.get(key)
            same = va == vb or (
                isinstance(va, float) and isinstance(vb, float) and abs(va - vb) <= atol + rtol * abs(vb)
            )
            if not same:
                diffs.append(Difference(name, "values", f"summary of column {col}: {key} {va} against {vb}"))
    return diffs


def load_output(path: str):
    """Load a stored (``.csv.gz`` / ``.json`` / ``.summary.json``) or live (``.csv`` / ``.json``) output."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return pd.read_csv(io.StringIO(fh.read()), na_values=["NULL"], keep_default_na=True)
    return pd.read_csv(path, na_values=["NULL"], keep_default_na=True, encoding_errors="replace")


@dataclass
class Difference:
    output: str
    kind: str  # missing, schema, shape, values, json
    detail: str
    stats: dict = field(default_factory=dict)


def compare_frames(
    name: str, a: pd.DataFrame, b: pd.DataFrame, rtol: float = RTOL, atol: float = ATOL
) -> list[Difference]:
    diffs = []
    if list(a.columns) != list(b.columns):
        only_a = [c for c in a.columns if c not in b.columns]
        only_b = [c for c in b.columns if c not in a.columns]
        diffs.append(Difference(name, "schema", f"columns differ (only in first: {only_a}, only in second: {only_b})"))
    if len(a) != len(b):
        diffs.append(Difference(name, "shape", f"{len(a)} rows against {len(b)} rows"))
    n = min(len(a), len(b))
    for col in [c for c in a.columns if c in b.columns]:
        x, y = a[col].iloc[:n], b[col].iloc[:n]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            nan_x, nan_y = np.isnan(xv), np.isnan(yv)
            nan_mismatch = int(np.sum(nan_x != nan_y))
            both = ~nan_x & ~nan_y
            delta = np.abs(xv[both] - yv[both])
            bad = int(np.sum(delta > atol + rtol * np.abs(yv[both]))) if both.any() else 0
            if nan_mismatch or bad:
                diffs.append(
                    Difference(
                        name,
                        "values",
                        f"column {col}: {bad} values differ, {nan_mismatch} NaN mismatches",
                        {
                            "column": col,
                            "differing": bad,
                            "nan_mismatch": nan_mismatch,
                            "max_abs": float(delta.max()) if delta.size else 0.0,
                            "rows": n,
                        },
                    )
                )
        else:
            mismatch = int(np.sum(x.astype(str).to_numpy() != y.astype(str).to_numpy()))
            if mismatch:
                diffs.append(
                    Difference(
                        name,
                        "values",
                        f"column {col}: {mismatch} labels differ",
                        {"column": col, "differing": mismatch, "rows": n},
                    )
                )
    return diffs


def compare_dirs(
    first: str, second: str, rtol: float = RTOL, atol: float = ATOL, outputs: tuple | None = None
) -> list[Difference]:
    """Compare two output sets. Each side is a stored reference dir or a live session dir."""

    def index(root: str) -> dict:
        found = {}
        for rel in outputs or FULL_OUTPUTS:
            for candidate in (
                os.path.join(root, rel.replace("/", "__") + ".summary.json"),
                os.path.join(root, rel.replace("/", "__") + ".gz"),
                os.path.join(root, rel.replace("/", "__")),
                os.path.join(root, rel),
            ):
                if os.path.isfile(candidate):
                    found[rel] = candidate
                    break
        return found

    a, b = index(first), index(second)
    diffs = []
    for rel in sorted(set(a) | set(b)):
        if rel not in a or rel not in b:
            diffs.append(Difference(rel, "missing", "only in first" if rel in a else "only in second"))
            continue
        x, y = load_output(a[rel]), load_output(b[rel])
        is_summary = [isinstance(v, dict) and v.get("summary") is True for v in (x, y)]
        if any(is_summary):
            x = x if is_summary[0] else summarize(x) if isinstance(x, pd.DataFrame) else x
            y = y if is_summary[1] else summarize(y) if isinstance(y, pd.DataFrame) else y
            if isinstance(x, dict) and isinstance(y, dict) and x.get("summary") and y.get("summary"):
                diffs.extend(_compare_summaries(rel, x, y, rtol, atol))
            else:
                diffs.append(Difference(rel, "schema", "table on one side, other content on the other"))
        elif isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            diffs.extend(compare_frames(rel, x, y, rtol, atol))
        elif x != y:
            diffs.append(Difference(rel, "json", "JSON content differs"))
    return diffs


# ---------------------------------------------------------------------------
# Non-regression check used by NucleusKit and Prime
# ---------------------------------------------------------------------------


def check_system(system: str, repo: str, workdir: str, cases: list[str] | None = None) -> dict:
    """Run ``system`` from ``repo`` on the cases and compare with its stored references.

    Returns ``{case/model: [Difference, ...]}``; an empty dict means no regression.
    Volatile outputs (sidecar timestamps, file paths) are compared too, after sanitizing,
    so the stored references must have been recorded by :mod:`record_references`.
    """
    failures = {}
    for case_name in cases or [c.name for c in CASES]:
        sessions = run_case(system, repo, case_name, workdir)
        for model, session in sessions.items():
            outputs = FULL_OUTPUTS if model == PRIMARY_MODEL else EMOTION_OUTPUTS
            live = os.path.join(workdir, "collected", system, model, case_name)
            collect(session, live, outputs, {"<session>": session, "<repo>": repo})
            diffs = compare_dirs(reference_dir(system, model, case_name), live, outputs=outputs)
            if diffs:
                failures[f"{case_name}/{model}"] = diffs
    return failures
