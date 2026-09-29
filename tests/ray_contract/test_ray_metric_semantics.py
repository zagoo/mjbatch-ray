# SPDX-License-Identifier: Apache-2.0
"""The recording rules derive the catalog's object_spilled_bytes_total and
object_restored_bytes_total from Ray's native ray_spill_manager_objects_bytes with
State="Spilled"/"Restored" (deploy/observability/alerts.yaml). That is correct only if
those series are the raylet's cumulative bytes, as observed on Ray 2.58.0: this test
forces spills and restores on a separate small Ray instance and checks it, so a Ray
upgrade that changes the semantics fails here instead of silently skewing dashboards."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.ray, pytest.mark.slow]

PROBE = r"""
import json, re, time, urllib.request
import numpy as np, ray
ray.init(num_cpus=1, object_store_memory=80 * 2**20, include_dashboard=False, log_to_driver=False,
         _system_config={"metrics_report_interval_ms": 500})
port = ray._private.worker._global_node.metrics_export_port
def scrape():
  out = {}
  for line in urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode().splitlines():
    if line.startswith("ray_spill_manager_objects_bytes{"):
      state = re.search(r'State="([^"]*)"', line).group(1)
      out[state] = out.get(state, 0.0) + float(line.rsplit(" ", 1)[1])
  return out
def when(pred):  # the exporter publishes every 0.5 s: wait for the state, then one more export
  for _ in range(60):
    time.sleep(0.5)
    if pred(scrape()):
      break
  time.sleep(1.5)
  return scrape()
refs = [ray.put(np.ones(10 * 2**20 // 8)) for _ in range(20)]  # 200 MiB into an 80 MiB store
after_puts = when(lambda s: s.get("Spilled", 0.0) >= 120 * 2**20)
for r in refs[:10]:
  ray.get(r)
after_gets = when(lambda s: s.get("Restored", 0.0) >= 95 * 2**20)
print("RESULT " + json.dumps({"puts": after_puts, "gets": after_gets}))
ray.shutdown()
"""


def test_spill_manager_bytes_are_cumulative_spilled_and_restored_bytes():
  out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=240)
  line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
  assert line, out.stderr[-2000:]
  r = json.loads(line[len("RESULT "):])
  mib = 2**20
  # 200 MiB put into an 80 MiB store: at least the overflow was spilled, and the counter
  # keeps it after restores (cumulative, not "currently spilled").
  assert r["puts"]["Spilled"] >= 120 * mib, r
  assert r["gets"]["Spilled"] == r["puts"]["Spilled"], r
  # Ten 10 MiB objects read back: each restore adds its bytes.
  assert r["puts"].get("Restored", 0.0) < mib, r
  assert 95 * mib <= r["gets"]["Restored"] <= 110 * mib, r
