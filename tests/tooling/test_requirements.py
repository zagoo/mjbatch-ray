# SPDX-License-Identifier: Apache-2.0
"""The requirement matrix may only cite code, tests and evidence that exist, uses the
four statuses, and never claims VERIFIED without a test or evidence reference."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REQS = yaml.safe_load((ROOT / "docs/qualification/requirements.yaml").read_text())["requirements"]


def test_ids_unique_and_statuses_valid():
  ids = [r["id"] for r in REQS]
  assert len(ids) == len(set(ids))
  assert {r["status"] for r in REQS} <= {"VERIFIED", "IMPLEMENTED", "BLOCKED", "NOT_STARTED"}


def test_every_referenced_path_exists():
  missing = [f"{r['id']}: {p}" for r in REQS for key in ("implementation", "tests", "evidence")
             for p in r[key] if not (ROOT / p).exists()]
  assert not missing, missing


def test_verified_requires_tests_or_evidence_and_blocked_requires_a_procedure():
  for r in REQS:
    if r["status"] == "VERIFIED":
      assert r["tests"] or r["evidence"], r["id"]
    if r["status"] in ("BLOCKED", "IMPLEMENTED"):
      assert r.get("notes"), f"{r['id']}: {r['status']} needs notes explaining what is missing"
