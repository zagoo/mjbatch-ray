# SPDX-License-Identifier: Apache-2.0
"""Mode B end to end on a real Ray runtime, PostgreSQL and blob storage."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import numpy as np
import pytest

from raysimbatch.config.models import EvalConfig
from raysimbatch.control.driver import ClusterDriver, RunRequest
from raysimbatch.policy.bundle import make_bundle
from raysimbatch.policy.export import cartpole_bundle, linear_policy
from raysimbatch.protocol import Artifact
from raysimbatch.storage.blob import open_blob_store
from raysimbatch.testing import fixtures as F
from raysimbatch.testing import runs as R

pytestmark = [pytest.mark.ray, pytest.mark.postgres, pytest.mark.slow]


def weak_policy():
  exe = linear_policy(np.zeros((1, 4)))  # does nothing: the pole falls
  return make_bundle("zero", 1, "onnx", exe, 4, 1, certified_batch_sizes=(1, 4, 8, 16, 32, 64, 128)), exe


def build(tmp_path, **eval_kw):
  if not os.environ.get("RSB_TEST_DSN"):
    pytest.skip("RSB_TEST_DSN not set")
  root = f"file://{tmp_path}/artifacts"
  fx = F.cartpole(max_control_steps=150)
  g = R.group_from_fixture(root, fx)
  good, gexe = cartpole_bundle()
  bad, bexe = weak_policy()
  pols = [R.policy_spec(root, good, gexe, "lqr"), R.policy_spec(root, bad, bexe, "zero")]
  task = R.task_spec(root, "balance", [g])
  ev = EvalConfig(seeds=(0, 1, 2), episodes_per_seed=3, bootstrap_replicates=2000, wave_actors=2,
                  **eval_kw)
  cfg = R.run_config(root, "env:RSB_TEST_DSN", [g], pols, [task], mode="eval", evaluation=ev,
                     worker={"envs": 8, "slice_size": 8, "slice_min": 8})
  return cfg, (good, bad)


def test_mode_b_end_to_end(ray_session, tmp_path):
  cfg, (good, bad) = build(tmp_path)
  driver = ClusterDriver(cfg, namespace=f"rsb-e2e-{uuid.uuid4().hex[:6]}")

  async def go():
    key = uuid.uuid4().hex
    rid = await driver.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2}), key)
    again = await driver.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2}), key)
    assert again == rid  # idempotent submission
    return rid, await driver.wait(rid, timeout_s=300)

  rid, st = asyncio.run(go())
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  assert st.committed_units == 6 and st.failed_units == 0 and st.missing_units == 0
  arts = st.artifacts
  assert arts["complete"] and arts["coverage"] == 1.0
  store = open_blob_store(cfg.artifact_root)
  man = json.loads(store.get(Artifact(**arts["manifest"])))
  assert man["expected_episodes"] == 18 and man["committed_episodes"] == 18
  agg = json.loads(store.get(Artifact(**arts["aggregate_json"])))
  succ = {a["policy_hash"]: a for a in agg["aggregates"] if a["metric"] == "success"}
  assert succ[good.policy_hash]["mean"] == 1.0
  assert succ[bad.policy_hash]["mean"] == 0.0
  assert succ[good.policy_hash]["valid_seeds"] == 3 and succ[good.policy_hash]["low_seed_count"]
  paired = [p for p in agg["paired"] if p["metric"] == "success"]
  assert paired and abs(abs(paired[0]["mean_difference"]) - 1.0) < 1e-9
  md = store.get(Artifact(**arts["scorecard_markdown"])).decode()
  assert "complete" in md and "Paired policy differences" in md
  # Durable rows are unique per (unit, ordinal) and deterministic by seed.
  rows = driver.store.db.episode_rows(rid)
  assert len({(r["unit_id"], r["ordinal"]) for r in rows}) == 18


def test_finished_run_controller_exits_and_status_stays_durable(ray_session, tmp_path):
  """A detached controller that outlives its run keeps a node busy forever (found in
  kind). It exits after a linger; status and results are then served from the store."""
  import ray

  cfg, _ = build(tmp_path)
  driver = ClusterDriver(cfg, namespace=f"rsb-e2e-{uuid.uuid4().hex[:6]}")
  rid = asyncio.run(driver.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2, "controller_linger_s": 1}),
                                  uuid.uuid4().hex))
  st = asyncio.run(driver.wait(rid, timeout_s=300))
  assert st.state == "COMPLETED" and st.artifacts["complete"], (st.error, st.detail.get("traceback"))
  import time

  deadline = time.monotonic() + 30
  while True:
    try:
      ray.get_actor(f"rsb-run-{rid}", namespace=driver.namespace)
    except ValueError:
      break
    assert time.monotonic() < deadline, "controller of a finished run is still alive"
    time.sleep(0.25)
  after = asyncio.run(driver.status(rid))
  assert after.state == "COMPLETED" and after.artifacts == st.artifacts
