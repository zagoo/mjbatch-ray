# SPDX-License-Identifier: Apache-2.0
"""Render docs/qualification/requirements.yaml as requirements.md.

  python tools/render_requirements.py [--check]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/qualification/requirements.yaml"
OUT = ROOT / "docs/qualification/requirements.md"
STATUSES = ("VERIFIED", "IMPLEMENTED", "BLOCKED", "NOT_STARTED")


def _cell(items) -> str:
  return "<br>".join(f"`{i}`" for i in items) if items else "—"


def render() -> str:
  reqs = yaml.safe_load(SRC.read_text())["requirements"]
  counts = collections.Counter(r["status"] for r in reqs)
  lines = ["# Requirement matrix", "",
           "*Rendered from [requirements.yaml](requirements.yaml) by `tools/render_requirements.py`; "
           "`tests/tooling/test_requirements.py` checks that every referenced path exists.*", "",
           "| Status | Count |", "|---|---:|", *[f"| {s} | {counts.get(s, 0)} |" for s in STATUSES], "",
           "Status meanings: **VERIFIED** implemented and verified here with real components; **IMPLEMENTED** "
           "implemented, verification incomplete (see notes); **BLOCKED** not verifiable in this environment "
           "(procedure in notes); **NOT_STARTED** not implemented.", "",
           "| ID | Design section | Requirement | Implementation | Config | Tests | Evidence | Status | Notes |",
           "|---|---|---|---|---|---|---|---|---|"]
  for r in reqs:
    lines.append("| {id} | {section} | {req} | {impl} | {cfg} | {tests} | {ev} | **{st}** | {notes} |".format(
      id=r["id"], section=r["section"], req=r["requirement"].replace("|", "/"), impl=_cell(r["implementation"]),
      cfg=_cell(r["config"]), tests=_cell(r["tests"]), ev=_cell(r["evidence"]), st=r["status"],
      notes=r.get("notes", "").replace("|", "/")))
  return "\n".join(lines) + "\n"


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", action="store_true")
  args = ap.parse_args(argv)
  text = render()
  if args.check:
    ok = OUT.exists() and OUT.read_text() == text
    print("requirements.md up to date" if ok else "stale: docs/qualification/requirements.md")
    return 0 if ok else 1
  OUT.write_text(text)
  print(f"wrote {OUT.relative_to(ROOT)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
