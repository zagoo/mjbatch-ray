# SPDX-License-Identifier: Apache-2.0
"""The in-cluster smoke tool is run by operators during releases and rollbacks. A missing
or mistyped run id must stop it with a message naming the problem, not a TypeError from
deep inside a comparison (found by the pyright triage)."""

from __future__ import annotations

import pytest

from raysimbatch.deployment import smoke


@pytest.mark.parametrize("argv,needs", [
  (["--mode", "xresume-finish"], "--run-id"),
  (["--mode", "xresume-compare"], "--run-id"),
  (["--mode", "compare-runs", "--reference-run", "r"], "--run-id"),
  (["--mode", "compare-runs", "--run-id", "r"], "--reference-run"),
])
def test_modes_that_read_stored_runs_require_their_ids(argv, needs, capsys):
  with pytest.raises(SystemExit) as exc:
    smoke.main(argv)  # rejected before any config, Ray or database access
  assert exc.value.code == 2
  assert needs in capsys.readouterr().err


def test_an_unknown_run_is_named():
  class NoRuns:
    def get_run(self, run_id):
      return None

  with pytest.raises(SystemExit, match="run 0123abcd not found"):
    smoke._stored_run(NoRuns(), "0123abcd")
