# SPDX-License-Identifier: Apache-2.0
"""Alert rules (design §11.3) may only reference catalog metrics (as Ray exports them),
Ray native names verified on the pinned Ray version, recording rules defined in the same
file, or standard Kubernetes monitoring series; every alert names a severity and a
runbook section that exists."""

from __future__ import annotations

import re
from pathlib import Path

import ray
import yaml

from raysimbatch.telemetry.metrics import CATALOG

ROOT = Path(__file__).resolve().parents[2]
RULES = yaml.safe_load((ROOT / "deploy/observability/alerts.yaml").read_text())
NATIVE = {ln.strip() for ln in (ROOT / f"deploy/observability/ray-{ray.__version__}-native-metrics.txt").read_text()
          .splitlines() if ln.strip() and not ln.startswith("#")}
PROMQL_WORDS = {"sum", "min", "max", "avg", "rate", "increase", "by", "on", "and", "or", "unless", "without",
                "histogram_quantile", "scalar", "vector", "le", "deriv", "count", "humanizePercentage"}


def _series(expr: str) -> set[str]:
  expr = re.sub(r"\b(by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", "", expr)  # grouping labels
  expr = re.sub(r"\{[^}]*\}", "", expr)  # drop label matchers
  expr = re.sub(r"\[[^\]]*\]", "", expr)  # and range selectors
  return {t for t in re.findall(r"[A-Za-z_:][A-Za-z0-9_:]*", expr) if t not in PROMQL_WORDS}


def test_every_series_is_known():
  records = {r["record"] for g in RULES["groups"] for r in g["rules"] if "record" in r}
  unknown = []
  for g in RULES["groups"]:
    for r in g["rules"]:
      for s in _series(r["expr"]):
        if s.startswith("ray_raysimbatch_"):
          base = re.sub(r"_(bucket|count|sum)$", "", s.removeprefix("ray_raysimbatch_"))
          ok = base in CATALOG and (CATALOG[base] == "histogram") == (base != s.removeprefix("ray_raysimbatch_"))
        elif s.startswith("ray_"):
          ok = s in NATIVE
        elif s.startswith("raysimbatch:"):
          ok = s in records
        else:
          ok = s.startswith(("container_", "kube_"))
        if not ok:
          unknown.append(f"{r.get('alert') or r.get('record')}: {s}")
  assert not unknown, unknown


def test_alerts_have_severity_and_existing_runbook_anchor():
  runbooks = (ROOT / "docs/operations/runbooks.md").read_text()
  anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).strip().replace(" ", "-")
             for h in re.findall(r"^#+ (.+)$", runbooks, re.M)}
  for g in RULES["groups"]:
    for r in g["rules"]:
      if "alert" not in r:
        continue
      assert r["labels"]["severity"] in ("page", "warning"), r["alert"]
      anchor = r["annotations"]["runbook"].split("#", 1)[1]
      assert anchor in anchors, f"{r['alert']}: runbook section #{anchor} missing"


def test_design_page_conditions_are_all_covered():
  pages = {r["alert"] for g in RULES["groups"] for r in g["rules"] if r.get("labels", {}).get("severity") == "page"}
  assert pages == {"RsbObjectStoreOccupancyHigh", "RsbHealthyCapacityLow", "RsbWorkerRestartExhausted",
                   "RsbNativeProgressStalled", "RsbDatabaseCommitsFailing", "RsbDigestConflict",
                   "RsbCoordinatorLeasesLost"}
