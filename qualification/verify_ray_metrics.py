# SPDX-License-Identifier: Apache-2.0
"""Scrape the native metric names a live Ray node exports (design §11.1: Ray native names
are mapped through a version-tested list, never assumed stable across upgrades).

  python qualification/verify_ray_metrics.py --write   # record deploy/observability/ray-<ver>-native-metrics.txt
  python qualification/verify_ray_metrics.py --check   # fail if a name the alert rules use disappeared
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scrape() -> tuple[str, list[str]]:
  import ray

  ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
           _system_config={"metrics_report_interval_ms": 1000})
  try:
    refs = [ray.put(b"x" * (1 << 20)) for _ in range(3)]  # populate object-store series
    time.sleep(12)
    port = ray._private.worker._global_node.metrics_export_port
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    del refs
    names = sorted({ln.split("{")[0].split(" ")[0] for ln in text.splitlines()
                    if ln.startswith("ray_") and "raysimbatch" not in ln})
    return ray.__version__, names
  finally:
    ray.shutdown()


def rule_names() -> set[str]:
  text = (ROOT / "deploy/observability/alerts.yaml").read_text()
  return set(re.findall(r"\b(ray_[a-z_]+?)(?:_bucket|_count|_sum)?\b(?=[{\[ )]|$)", text))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--write", action="store_true")
  ap.add_argument("--check", action="store_true")
  args = ap.parse_args()
  version, names = scrape()
  path = ROOT / f"deploy/observability/ray-{version}-native-metrics.txt"
  if args.write:
    path.write_text(f"# Native metric names exported by Ray {version} (scraped from a live node's metrics endpoint\n"
                    "# by qualification/verify_ray_metrics.py). Alert rules may reference only these or catalog "
                    "metrics.\n" + "\n".join(names) + "\n")
    print(f"wrote {len(names)} names to {path}")
  if args.check:
    used = {n for n in rule_names() if not n.startswith("ray_raysimbatch_")}
    missing = sorted(n for n in used if n not in names and not any(n.startswith(x) for x in names))
    if missing:
      print(f"Ray {version} no longer exports: {missing}", file=sys.stderr)
      return 1
    print(f"Ray {version}: all {len(used)} native names used by alert rules are exported")
  return 0


if __name__ == "__main__":
  sys.exit(main())
