# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""
Emotion model registry.

Built-in releases live in ``reak_core/models/releases/<key>/`` (``manifest.json``, ``config.json``,
``pipeline.joblib``, the release's ``exg_emotion`` runtime). Extensions add models through the
``reak_core.models`` entry point group: each entry point is a callable ``factory(key) -> EmotionModel``
registered under the model key.

Aliases: ``standard-latest`` is the default model of the core (decision D9, 2.6.0);
``classical-emotion`` is the historical NucleusKit alias for 2.5.0.

Author(s):
    RE-AK Technologies Inc., Fall 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from reak_core.models.base import EmotionModel, PostprocessSpec, PreprocessSpec

logger = logging.getLogger(__name__)

RELEASES_DIR = Path(__file__).parent / "releases"
DEFAULT_MODEL = "classical-emotion-2.6.0"
ALIASES: dict[str, str] = {
    "standard-latest": DEFAULT_MODEL,
    "classical-emotion": "classical-emotion-2.5.0",
}
ENTRY_POINT_GROUP = "reak_core.models"


class ReleaseIntegrityError(RuntimeError):
    """A release file does not match the checksum recorded in its manifest."""


@dataclass(frozen=True)
class Release:
    """A built-in model release on disk."""

    key: str
    path: Path
    manifest: dict

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.key)

    @property
    def version(self) -> str:
        return str(self.manifest.get("version") or "")

    def config(self) -> dict:
        try:
            return json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def preprocess(self) -> PreprocessSpec:
        """Bandpass and notch declared in ``config.json``; 15–40 Hz without notch when undeclared."""
        config = self.config()
        band = config.get("bandpass_hz")
        if not (isinstance(band, (list, tuple)) and len(band) == 2):
            return PreprocessSpec(used_fallback=True)
        notch_raw = config.get("notch_hz")
        try:
            notch = float(notch_raw) if notch_raw not in (None, "", 0, 0.0) else None
        except (TypeError, ValueError):
            notch = None
        return PreprocessSpec(bandpass_low=float(band[0]), bandpass_high=float(band[1]), notch_hz=notch)

    def postprocess(self) -> PostprocessSpec:
        """Post-classifier defaults declared in the manifest (decision D8)."""
        spec = self.manifest.get("reak_core", {}).get("postprocessing", {})
        high = spec.get("high_rms", {})
        low = spec.get("low_confidence", {})
        return PostprocessSpec(
            one_hot=bool(spec.get("one_hot", False)),
            high_rms_enabled=bool(high.get("enabled", True)),
            high_rms_threshold_uv=float(high.get("threshold_uv", 200.0)),
            high_rms_max_gap_windows=int(high.get("max_gap_windows", 6)),
            low_confidence_enabled=bool(low.get("enabled", False)),
            low_confidence_threshold=float(low.get("threshold", 0.3)),
        )

    def verify(self) -> None:
        """Check every file against the checksums of the manifest; raise ReleaseIntegrityError otherwise."""
        expected = self.manifest.get("reak_core", {}).get("checksums", {})
        if not expected:
            raise ReleaseIntegrityError(f"{self.key}: manifest has no checksums")
        for rel, digest in expected.items():
            path = self.path / rel
            if not path.is_file():
                raise ReleaseIntegrityError(f"{self.key}: missing file {rel}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ReleaseIntegrityError(f"{self.key}: {rel} does not match its checksum")


def builtin_releases() -> dict[str, Release]:
    """``{key: Release}`` for every folder of RELEASES_DIR that has a manifest."""
    found: dict[str, Release] = {}
    if not RELEASES_DIR.is_dir():
        return found
    for child in sorted(RELEASES_DIR.iterdir()):
        manifest_path = child / "manifest.json"
        if child.is_dir() and manifest_path.is_file():
            try:
                found[child.name] = Release(child.name, child, json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring release %s: unreadable manifest (%s)", child.name, exc)
    return found


def _extension_factories() -> dict:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def resolve_key(key: str | None) -> str:
    """Canonical model key for a key or alias (``None`` gives the default model)."""
    key = key or "standard-latest"
    return ALIASES.get(key, key)


def available_models() -> list[str]:
    """Keys accepted by :func:`load_model`, aliases included."""
    return sorted({*ALIASES, *builtin_releases(), *_extension_factories()})


def load_model(key: str | None = None, *, verify: bool = True) -> EmotionModel:
    """Load an emotion model by key or alias."""
    canonical = resolve_key(key)
    releases = builtin_releases()
    if canonical in releases:
        from reak_core.models.classical import ClassicalReleaseModel

        release = releases[canonical]
        if verify:
            release.verify()
        return ClassicalReleaseModel.load(release)
    factories = _extension_factories()
    if canonical in factories:
        return factories[canonical].load()(canonical)
    raise ValueError(f"Unknown emotion model {key!r}. Available: {available_models()}")
