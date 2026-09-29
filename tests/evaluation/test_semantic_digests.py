# SPDX-License-Identifier: Apache-2.0
"""Cross-run comparison (release rollback, janitor resume) rests on semantic digests:
the same episode computed in another run, attempt or cluster must compare equal, and
any difference in what was computed must not."""

from __future__ import annotations

from raysimbatch.evaluation.schema import IDENTITY_FIELDS, semantic_digests


class FakeDB:
  def __init__(self, rows):
    self.rows = rows

  def episode_rows(self, run_id):
    return self.rows[run_id]


def _row(unit, ordinal, **data):
  base = {"seed": 7, "episode_return": 12.5, "steps": 200, "success": True}
  return {"unit_id": unit, "ordinal": ordinal, "row_data": {**base, **data}}


def test_identity_fields_do_not_affect_the_digest():
  a = [_row("u1", 0, run_id="A", attempt=0, env_id=3, wall_seconds=1.0, hardware_class="x"),
       _row("u1", 1, run_id="A", attempt=0)]
  b = [_row("u1", 0, run_id="B", attempt=2, env_id=9, wall_seconds=7.5, hardware_class="y"),
       _row("u1", 1, run_id="B", attempt=2)]
  d = semantic_digests(FakeDB({"A": a, "B": b}), "A")
  assert d == semantic_digests(FakeDB({"A": a, "B": b}), "B")
  assert set(d) == {("u1", 0), ("u1", 1)}
  assert {"run_id", "attempt", "env_id", "wall_seconds", "hardware_class"} <= IDENTITY_FIELDS


def test_any_computed_difference_changes_the_digest():
  ref = semantic_digests(FakeDB({"A": [_row("u1", 0)]}), "A")
  for change in ({"episode_return": 12.500000001}, {"seed": 8}, {"steps": 199}, {"success": False}):
    assert semantic_digests(FakeDB({"B": [_row("u1", 0, **change)]}), "B") != ref, change


def test_missing_or_extra_episodes_are_visible():
  a = semantic_digests(FakeDB({"A": [_row("u1", 0), _row("u1", 1)]}), "A")
  b = semantic_digests(FakeDB({"B": [_row("u1", 0)]}), "B")
  assert a != b
