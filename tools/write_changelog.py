# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Measure the core against both current systems and write the generated part of CHANGELOG.md.

Runs ``Pipeline.default()`` on every synthetic case (2.6.0, then 2.5.0 on a copy), compares each output
with the NucleusKit and the Prime references, and lists every difference with the decisions that cause it
(``tests/core/expectations.py``). Rewrites the block between the GENERATED markers of CHANGELOG.md.

Usage: python tools/write_changelog.py [--workdir DIR]

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refkit  # noqa: E402

ROOT = refkit.CORE_ROOT
sys.path.insert(0, os.path.join(ROOT, "tests", "core"))
from expectations import ALTERNATES, EXCEPTIONS, EXPECTATIONS  # noqa: E402

import reak_core  # noqa: E402
from reak_core.pipeline import Pipeline  # noqa: E402

CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
BEGIN, END = "<!-- GENERATED:BEGIN -->", "<!-- GENERATED:END -->"
SYSTEMS = {"nk": "NucleusKit", "prime": "Prime"}


def run_core(workdir: str) -> dict:
    collected = {}
    primary, secondary = refkit.MODELS
    for case in refkit.CASES:
        p_root = os.path.join(workdir, primary, case.name)
        refkit.generate(case.name, p_root)
        Pipeline.default(emotion_model=primary).run(p_root)
        s_root = os.path.join(workdir, secondary, case.name)
        shutil.copytree(p_root, s_root)
        Pipeline.default(emotion_model=secondary).run(s_root)
        for model, root in ((primary, p_root), (secondary, s_root)):
            dest = os.path.join(workdir, "collected", model, case.name)
            refkit.collect_outputs(root, dest, tuple(EXPECTATIONS), {"<session>": root, "<source>": p_root})
            collected[(model, case.name)] = dest
        print(f"[changelog] {case.name}", file=sys.stderr)
    return collected


def _json_diff(ref_dir: str, core_dir: str, output: str) -> str:
    name = output.replace("/", "__")
    a = refkit.load_output(os.path.join(ref_dir, name))
    b = refkit.load_output(os.path.join(core_dir, name))
    keys = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k) and k != "basepath"]
    return "; ".join(f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in keys) or "same values (paths only)"


def _describe(diffs) -> str:
    kinds = {d.kind for d in diffs}
    if "missing" in kinds:
        return "; ".join(d.detail.replace("first", "reference").replace("second", "core") for d in diffs)
    parts = []
    for d in diffs:
        if d.kind == "values" and d.stats:
            extra = f" (max abs diff {d.stats['max_abs']:.4g})" if d.stats.get("max_abs") else ""
            parts.append(f"{d.stats['column']}: {d.stats['differing']} of {d.stats['rows']} rows{extra}")
        elif d.kind == "schema" and "only in" in d.detail:
            parts.append(_short_schema(d.detail))
        else:
            parts.append(d.detail)
    return "; ".join(parts[:6]) + (f"; and {len(parts) - 6} more" if len(parts) > 6 else "")


def _short_schema(detail: str) -> str:
    """Count the columns of a schema difference instead of listing them."""
    import ast

    first = detail.split("only in first: ")[1].split(", only in second: ")[0]
    second = detail.split("only in second: ")[1].rstrip(")")
    a, b = ast.literal_eval(first), ast.literal_eval(second)
    shown = lambda cols: ", ".join(cols) if len(cols) <= 3 else f"{len(cols)} columns ({cols[0]}, …)"  # noqa: E731
    return f"columns only in the reference: {shown(a) or 'none'}; only in the core: {shown(b) or 'none'}"


def _null_rows(directory: str) -> int | None:
    """All-NULL rows of Emotions.csv (null windows on the 2 Hz grid); written by both systems and the core."""
    path = os.path.join(directory, "results__Emotions.csv.gz")
    if not os.path.isfile(path):
        return None
    df = refkit.load_output(path)
    return int(df.drop(columns=["Timestamp"]).isna().all(axis=1).sum())


def measure(collected: dict) -> tuple[dict, list[str]]:
    """``{decision: [line, ...]}`` of differences, and the outputs identical to the designated system."""
    by_decision: dict[str, list[str]] = defaultdict(list)
    identical: list[str] = []
    for output, (rule, decisions, models) in EXPECTATIONS.items():
        for model in models:
            for case in refkit.CASES:
                core_dir = collected[(model, case.name)]
                designated = ALTERNATES.get(output, {}).get(case.name, rule)
                for system in ("nk", "prime"):
                    ref = refkit.reference_dir(system, model, case.name)
                    diffs = refkit.compare_dirs(ref, core_dir, outputs=(output,))
                    if not diffs:
                        if system == designated:
                            identical.append(f"{output} / {model[-5:]} / {case.name}")
                        continue
                    if system == designated:
                        causes = EXCEPTIONS.get(output, {}).get(case.name) or decisions or ("unexplained",)
                    elif rule == "new":
                        causes = decisions
                    else:
                        continue  # differs from the non-designated system: expected by the decision itself
                    detail = (
                        _json_diff(ref, core_dir, output)
                        if output.endswith(".json") and all(d.kind == "json" for d in diffs)
                        else _describe(diffs)
                    )
                    line = f"`{output}`, model {model[-5:]}, case `{case.name}`, against {SYSTEMS[system]}: {detail}"
                    for decision in causes:
                        by_decision[decision].append(line)
    return by_decision, identical


def null_window_table(collected: dict) -> list[str]:
    lines = ["| Case | Model | NucleusKit | Prime | Core |", "|---|---|---|---|---|"]
    for case in refkit.CASES:
        for model in refkit.MODELS:
            counts = [
                _null_rows(refkit.reference_dir("nk", model, case.name)),
                _null_rows(refkit.reference_dir("prime", model, case.name)),
                _null_rows(collected[(model, case.name)]),
            ]
            if len({c for c in counts if c is not None}) > 1:
                cells = ["n/a" if c is None else str(c) for c in counts]
                lines.append(f"| {case.name} | {model[-5:]} | {' | '.join(cells)} |")
    return lines


def render(by_decision: dict, identical: list[str], null_lines: list[str]) -> str:
    manifest = json.load(open(os.path.join(refkit.REFERENCES_DIR, "manifest.json"), encoding="utf-8"))
    commits = {
        system: ", ".join(sorted({c["recorded_with"][system][:7] for c in manifest["cases"].values()}))
        for system in SYSTEMS
    }
    out = [
        BEGIN,
        "",
        f"Measured by `tools/write_changelog.py` with reak_core {reak_core.__version__} on the "
        f"{len(refkit.CASES)} synthetic cases of `tests/synthetic` (models 2.6.0 and 2.5.0), against the references "
        f"of NucleusKit `{commits['nk']}` and Prime `{commits['prime']}` (same processing code as NucleusKit 1.1 and "
        "Prime 3.1.0). Do not edit this block by hand.",
        "",
        f"- {len(identical)} output / model / case combinations are identical to the system the decisions designate "
        "(within rtol 1e-6, atol 1e-9).",
        "- Every other difference is listed below under the decision that causes it. Differences with the system a "
        "decision did not retain are not repeated: they follow from the decision itself (see "
        "`tests/references/DIVERGENCES.md`).",
        "",
        "### Null emotion rows (2 Hz grid) where the systems disagree",
        "",
        *null_lines,
        "",
    ]
    for decision in sorted(by_decision, key=lambda d: (len(d), d)):
        out += [f"### {decision}", ""]
        out += [f"- {line}" for line in sorted(set(by_decision[decision]))]
        out.append("")
    out.append(END)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    workdir = args.workdir or tempfile.mkdtemp(prefix="reak-changelog-")
    collected = run_core(workdir)
    by_decision, identical = measure(collected)
    block = render(by_decision, identical, null_window_table(collected))
    with open(CHANGELOG, encoding="utf-8") as fh:
        text = fh.read()
    start, end = text.index(BEGIN), text.index(END) + len(END)
    with open(CHANGELOG, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text[:start] + block + text[end:])
    print(f"[changelog] wrote {CHANGELOG}; unexplained differences: {len(by_decision.get('unexplained', []))}")
    return 1 if by_decision.get("unexplained") else 0


if __name__ == "__main__":
    sys.exit(main())
