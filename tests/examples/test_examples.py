# SPDX-License-Identifier: Apache-2.0
"""The documented examples run as written (each in its own process with its own Ray)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name,needs_db,expect", [
  ("router_stepping.py", False, "stale actions accepted: 0"),
  ("mode_b_evaluation.py", True, "COMPLETED, complete=True, coverage=1.0"),
  ("mode_a_rollout.py", True, "COMPLETED; gaps=0"),
  ("mode_c_sim2sim.py", True, "conservative transfer ratio"),
])
def test_example(name, needs_db, expect, tmp_path):
  if needs_db and not os.environ.get("RSB_TEST_DSN"):
    pytest.skip("RSB_TEST_DSN not set")
  p = subprocess.run([sys.executable, name, "--root", f"file://{tmp_path}/artifacts"] if needs_db else
                     [sys.executable, name], cwd=EXAMPLES, capture_output=True, text=True, timeout=1200)
  assert p.returncode == 0, p.stderr[-3000:]
  assert expect in p.stdout, p.stdout
