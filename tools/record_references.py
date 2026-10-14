# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Record the reference outputs of the current NucleusKit and Prime pipelines.

For every synthetic case and every supported emotion model, runs each system from its
checkout (neighbouring clones by default), stores the selected outputs under
``tests/references/<system>/<model>/<case>/`` and writes ``tests/references/manifest.json``
(commits, environment, generator hashes). Prime's outputs are recorded on synthetic data
only and contain no proprietary code.

Usage:
    python tools/record_references.py [--nk PATH] [--prime PATH] [--cases a,b] [--systems nk,prime]

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refkit  # noqa: E402

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "neurokit2", "cvxopt", "matplotlib")


def git_state(repo: str) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(terse=True), "packages": versions}


def main() -> int:
    parent = os.path.dirname(refkit.CORE_ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nk", default=os.path.join(parent, "NucleusKit-ProcessingPipeline"))
    parser.add_argument("--prime", default=os.path.join(parent, "primeV3"))
    parser.add_argument("--systems", default="nk,prime")
    parser.add_argument("--cases", default=",".join(c.name for c in refkit.CASES))
    parser.add_argument("--workdir", default=None, help="scratch directory (default: a temporary one)")
    args = parser.parse_args()

    repos = {"nk": os.path.abspath(args.nk), "prime": os.path.abspath(args.prime)}
    systems = args.systems.split(",")
    cases = args.cases.split(",")
    workdir = args.workdir or tempfile.mkdtemp(prefix="reak-references-")
    manifest_path = os.path.join(refkit.REFERENCES_DIR, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    for system in systems:
        state = git_state(repos[system])
        if state["dirty"]:
            print(f"WARNING: {system} checkout has uncommitted changes; recorded as dirty", file=sys.stderr)
        manifest.setdefault("systems", {})[system] = state
    manifest["environment"] = environment()
    manifest["models"] = list(refkit.MODELS)
    manifest["recorded_on"] = dt.date.today().isoformat()
    manifest.setdefault("cases", {})

    for case_name in cases:
        case = refkit.by_name(case_name)
        hashes = refkit.generate(case_name, os.path.join(workdir, "raw", case_name))
        manifest["cases"][case_name] = {
            "seed": case.seed,
            "targets": list(case.targets),
            "description": case.description,
            "raw_sha256": hashes,
        }
        for system in systems:
            print(f"[record] {system} / {case_name}", file=sys.stderr)
            sessions = refkit.run_case(system, repos[system], case_name, workdir)
            for model, session in sessions.items():
                outputs = refkit.FULL_OUTPUTS if model == refkit.PRIMARY_MODEL else refkit.EMOTION_OUTPUTS
                refkit.collect(
                    session,
                    refkit.reference_dir(system, model, case_name),
                    outputs,
                    {"<session>": session, "<repo>": repos[system], "<home>": os.path.expanduser("~")},
                )

    os.makedirs(refkit.REFERENCES_DIR, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[record] done; scratch data in {workdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
