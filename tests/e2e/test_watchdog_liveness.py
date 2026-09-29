# SPDX-License-Identifier: Apache-2.0
"""The per-node watchdog in the run path (design §8.4, AM-16).

Two failures that command deadlines alone handle late or never:

* a worker holding the GIL inside a command: the coordinator's call deadline is 60 s
  here, so recovery before it can only come from the watchdog's heartbeat timeout;
* an idle worker that dies: no command is outstanding, so without the watchdog nothing
  notices until work is sent to it.

Both runs must still produce exactly the episodes of a clean run."""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest
import ray

from raysimbatch.config.models import EvalConfig
from raysimbatch.control.driver import ClusterDriver, RunRequest
from raysimbatch.evaluation.schema import semantic_digests
from raysimbatch.policy.export import cartpole_bundle
from raysimbatch.testing import fixtures as F
from raysimbatch.testing import runs as R

pytestmark = [pytest.mark.ray, pytest.mark.postgres, pytest.mark.slow]

DEADLINE_S = 60.0


def config(tmp_path):
  if not os.environ.get("RSB_TEST_DSN"):
    pytest.skip("RSB_TEST_DSN not set")
  root = f"file://{tmp_path}/artifacts"
  fx = F.cartpole(max_control_steps=30000)  # the LQR policy balances: episodes run to the limit
  g = R.group_from_fixture(root, fx)
  b, exe = cartpole_bundle()
  task = R.task_spec(root, "balance", [g])
  # Four episodes fit one worker's eight slots: the second worker stays idle.
  ev = EvalConfig(seeds=(0, 1), episodes_per_seed=2, bootstrap_replicates=200, wave_actors=2)
  return R.run_config(root, "env:RSB_TEST_DSN", [g], [R.policy_spec(root, b, exe, "lqr")], [task], mode="eval",
                      evaluation=ev, worker={"envs": 8, "slice_size": 8, "slice_min": 8},
                      failure={"heartbeat_s": 0.5, "suspect_s": 2.0, "cancel_grace_s": 1.0,
                               "native_max_deadline_s": DEADLINE_S})


def run(driver, cfg, options, chaos=None):
  async def go():
    rid = await driver.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2, **options}), uuid.uuid4().hex)
    if chaos:
      await chaos(driver, rid)
    return rid, await driver.wait(rid, timeout_s=300)
  return asyncio.run(go())


def lost(driver, rid):
  return [(e["ts"], e["payload"]["worker"], e["payload"]["reason"]) for e in driver.store.db.audit_events(rid)
          if e["event"] == "worker_lost"]


@pytest.fixture(scope="module")
def reference(ray_session, tmp_path_factory):
  cfg = config(tmp_path_factory.mktemp("ref"))
  d = ClusterDriver(cfg, namespace=f"rsb-wdref-{uuid.uuid4().hex[:6]}")
  t0 = time.monotonic()
  rid, st = run(d, cfg, {})
  assert st.state == "COMPLETED", st.error
  assert not lost(d, rid)
  return semantic_digests(d.store.db, rid), time.monotonic() - t0


def test_gil_holding_command_is_recovered_before_its_deadline(reference, tmp_path):
  digests, _ = reference
  cfg = config(tmp_path)
  d = ClusterDriver(cfg, namespace=f"rsb-wdgil-{uuid.uuid4().hex[:6]}")
  t0 = time.monotonic()
  rid, st = run(d, cfg, {"worker_class": "raysimbatch.testing.faults:RemoteGilHoldOnce"})
  elapsed = time.monotonic() - t0
  assert st.state == "COMPLETED", st.error
  assert semantic_digests(d.store.db, rid) == digests  # same episodes, each exactly once
  reasons = [r for _, _, r in lost(d, rid)]
  # Only the worker that ran episodes held the GIL; the watchdog, not the 60 s call
  # deadline, started its recovery.
  assert reasons == ["watchdog HEARTBEAT_TIMEOUT: fenced and killed"], reasons
  assert elapsed < DEADLINE_S, elapsed


@pytest.mark.parametrize("restart_watchdog", [False, True], ids=["watchdog", "restarted-watchdog"])
def test_idle_worker_death_is_noticed_without_a_command(reference, tmp_path, restart_watchdog):
  """With ``restart_watchdog`` the node's watchdog is killed first and comes back empty:
  the coordinator must register its workers again (``UNWATCHED`` or a failed ``inspect``)
  or the idle worker's death goes unnoticed."""
  digests, ref_s = reference
  cfg = config(tmp_path)
  d = ClusterDriver(cfg, namespace=f"rsb-wdidle-{uuid.uuid4().hex[:6]}")
  killed = {}

  async def chaos(driver, rid):
    coord = None
    t0 = time.monotonic()
    while True:  # wait until the episodes are running and one worker holds none of them
      assert time.monotonic() - t0 < 60, "no worker became busy"
      try:
        coord = coord or ray.get_actor(f"rsb-coord-{rid[:8]}-0", namespace=driver.namespace)
        workers = ray.get(coord.status.remote())["workers"]
      except (ValueError, ray.exceptions.RayActorError):
        workers = {}
      if len(workers) == 2 and any(w["active"] for w in workers.values()):
        break
      await asyncio.sleep(0.1)
    idle = [n for n, w in workers.items() if not w["active"]]
    assert len(idle) == 1, workers
    if restart_watchdog:
      node = ray.get_runtime_context().get_node_id()
      ray.kill(ray.get_actor(f"rsb-watchdog-{node[:16]}", namespace=driver.namespace), no_restart=False)
      await asyncio.sleep(3.0)  # restarted with no registrations; its owner must notice
    killed["name"], killed["at"] = idle[0], time.monotonic() - t0
    ray.get(coord.kill_worker.remote(idle[0]))
    t1 = time.monotonic()
    while not lost(driver, rid):
      assert time.monotonic() - t1 < 30, "the idle worker's death went unnoticed"
      await asyncio.sleep(0.2)
    st = await driver.status(rid)
    assert st.state == "RUNNING", f"run ended ({st.state}) before the check; lengthen its episodes (ref {ref_s:.0f} s)"

  rid, st = run(d, cfg, {}, chaos)
  assert st.state == "COMPLETED", st.error
  assert semantic_digests(d.store.db, rid) == digests
  events = lost(d, rid)
  assert len(events) == 1 and events[0][1] == killed["name"], events
  assert events[0][2] in ("watchdog: WORKER_DIED", "watchdog: WORKER_RESTARTED"), events
