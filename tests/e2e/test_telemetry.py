# SPDX-License-Identifier: Apache-2.0
"""Telemetry (design §11.1-11.2) observed through the real exporters of a Mode B run:
catalog metrics scraped from Ray's node metrics endpoint, structured JSON logs in the
actors' log files, and the reproducibility manifest in the artifact store."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
import uuid
from pathlib import Path

import pytest
import ray
from test_mode_b import build  # tests/e2e is on sys.path (rootdir import mode)

from raysimbatch.control.driver import ClusterDriver, RunRequest
from raysimbatch.protocol import Artifact
from raysimbatch.storage.blob import open_blob_store

pytestmark = [pytest.mark.ray, pytest.mark.postgres, pytest.mark.slow]


def _scrape(port: int) -> dict[str, float]:
  """Sum each ray_raysimbatch_* series over Ray's per-process labels (WorkerId etc.)."""
  text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
  out: dict[str, float] = {}
  for line in text.splitlines():
    m = re.match(r"^(ray_raysimbatch_[a-z0-9_]+)(\{[^}]*\})? ([0-9.eE+-]+)$", line)
    if m:
      out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
  return out


def test_mode_b_emits_catalog_metrics_logs_and_manifest(ray_session, tmp_path):
  cfg, _ = build(tmp_path)
  cfg = cfg.model_copy(update={"evaluation": cfg.evaluation.model_copy(update={"seeds": tuple(range(8))}),
                               "models": (cfg.models[0].model_copy(update={"max_control_steps": 2000}),)})
  driver = ClusterDriver(cfg, namespace=f"rsb-tel-{uuid.uuid4().hex[:6]}")
  port = ray._private.worker._global_node.metrics_export_port
  want = {"ray_raysimbatch_control_steps_total", "ray_raysimbatch_physics_steps_total",
          "ray_raysimbatch_episodes_total", "ray_raysimbatch_batch_step_seconds_count",
          "ray_raysimbatch_publish_seconds_count", "ray_raysimbatch_control_interval_seconds_count",
          "ray_raysimbatch_eval_episodes_committed_total", "ray_raysimbatch_storage_commit_seconds_count",
          "ray_raysimbatch_healthy_capacity_ratio",
          # 1.9: liveness and resources (watchdog-polled health), capacity, suite summaries,
          # and the sampled adapter cost.
          "ray_raysimbatch_heartbeat_age_seconds", "ray_raysimbatch_worker_rss_bytes",
          "ray_raysimbatch_cpu_busy_ratio", "ray_raysimbatch_slots_quarantined",
          "ray_raysimbatch_eval_success_ratio", "ray_raysimbatch_eval_return_mean",
          "ray_raysimbatch_observe_seconds_count"}
  # Ray exports every ~10 s and drops a process's series when it exits, so scrape while
  # the run's actors are alive (observed: a run shorter than one interval exports nothing).
  seen: dict[str, float] = {}

  async def go():
    done = None
    rid = await driver.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2}), uuid.uuid4().hex)
    while True:
      for k, v in (await asyncio.to_thread(_scrape, port)).items():
        seen[k] = max(v, seen.get(k, 0.0))
      st = await driver.status(rid)
      if want <= set(seen):
        return rid, await driver.wait(rid, timeout_s=300)
      if st.state in ("COMPLETED", "FAILED", "CANCELED"):
        done = done or asyncio.get_running_loop().time()
        if asyncio.get_running_loop().time() - done > 5.0:  # final gauges export while the controller lingers
          return rid, st
      await asyncio.sleep(1.0)

  rid, st = asyncio.run(go())
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  assert want <= set(seen), sorted(want - set(seen))
  assert seen["ray_raysimbatch_physics_steps_total"] >= 4 * seen["ray_raysimbatch_control_steps_total"] > 0

  # Reproducibility manifest: content-addressed, linked from the results, audited.
  art = Artifact(**st.artifacts["run_manifest"])
  man = json.loads(open_blob_store(cfg.artifact_root).get(art))
  assert man["schema"] == "rsb-run-manifest/1" and man["run_id"] == rid
  assert man["config"]["hash"] == driver.cs.canonical_hash(cfg.model_copy(update={"run_id": rid}))
  assert man["counts"]["units"] == {"SUCCEEDED": st.committed_units}
  assert man["software"]["mjbatch_revision"] and man["software"]["engine_abi"]
  assert man["artifacts"]["aggregate_json"] == st.artifacts["aggregate_json"]["sha256"]
  assert "run_manifest" in [e["event"] for e in driver.store.db.audit_events(rid)]

  # Structured JSON logs in the actors' stderr files (log_to_driver is off in tests).
  logs = Path(ray._private.worker._global_node.get_session_dir_path()) / "logs"
  records = []
  for f in logs.glob("worker-*.err"):
    for line in f.read_text(errors="replace").splitlines():
      if line.startswith("{") and rid in line:
        records.append(json.loads(line))
  states = [r["message"].split()[0] for r in records if r["event"] == "run_state"]
  assert states[0] == "VALIDATING" and states[-1] == "COMPLETED", states
  assert all({"timestamp", "severity", "event", "run_id"} <= set(r) for r in records)
