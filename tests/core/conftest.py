# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Shared fixtures for the reak_core tests: synthetic sessions generated once per test run."""

from __future__ import annotations

import os
import shutil
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic import build, by_name, write_session  # noqa: E402


@pytest.fixture(scope="session")
def _generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    cache: dict[str, str] = {}

    def get(case: str) -> str:
        if case not in cache:
            path = str(root / case)
            write_session(build(by_name(case)), path)
            cache[case] = path
        return cache[case]

    return get


@pytest.fixture
def session_copy(_generated, tmp_path):
    """Return a function giving a fresh, writable copy of a generated synthetic case."""

    def copy(case: str) -> str:
        dest = str(tmp_path / case)
        shutil.copytree(_generated(case), dest)
        return dest

    return copy
