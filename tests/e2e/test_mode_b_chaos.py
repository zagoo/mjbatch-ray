# SPDX-License-Identifier: Apache-2.0
"""Mode B under failure: worker, coordinator and controller loss mid-run.

Every committed episode must be identical (semantic digest) to a clean run with the
same configuration, committed exactly once, with full coverage."""

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
from raysimbatch.testing import fixtures as F
from raysimbatch.testing import runs as R

pytestmark = [pytest.mark.ray, pytest.mark.postgres, pytest.mark.slow]


def config(tmp_path, run_seed=11):
  if not os.environ.get("RSB_TEST_DSN"):
    pytest.skip("RSB_TEST_DSN not set")
  root = f"file://{tmp_path}/artifacts"
  fx = F.quadruped(max_control_steps=400)
  g = R.group_from_fixture(root, fx)
  from raysimbatch.policy.export import gaussian_bundle

  b, exe = gaussian_bundle(36, 8, seed=3)
  pol = R.policy_spec(root, b, exe, "mlp")
  task = R.task_spec(root, "walk", [g])
  ev = EvalConfig(seeds=tuple(range(12)), episodes_per_seed=6, bootstrap_replicates=500, wave_actors=2,
                  lease_s=6.0, lease_renew_s=1.0, max_attempts=4)
  return R.run_config(root, "env:RSB_TEST_DSN", [g], [pol], [task], mode="eval", evaluation=ev, run_seed=run_seed,
                      worker={"envs": 8, "slice_size": 8, "slice_min": 8}, failure={"lease_s": 6.0,
                                                                                   "lease_renew_s": 1.0})


def digests(driver, rid):
  return semantic_digests(driver.store.db, rid)


def run(driver, cfg, chaos=None, timeout=600):
  async def go():
    rid = await driver.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2}), uuid.uuid4().hex)
    if chaos:
      await chaos(driver, rid)
    return rid, await driver.wait(rid, timeout_s=timeout)
  return asyncio.run(go())


@pytest.fixture(scope="module")
def reference(ray_session, tmp_path_factory):
  cfg = config(tmp_path_factory.mktemp("ref"))
  d = ClusterDriver(cfg, namespace=f"rsb-ref-{uuid.uuid4().hex[:6]}")
  rid, st = run(d, cfg)
  assert st.state == "COMPLETED", st.error
  return digests(d, rid)


async def wait_committed(driver, rid, n, within_s=120):
  t0 = time.monotonic()
  while driver.store.db.episode_count(rid) < n:
    assert time.monotonic() - t0 < within_s, "run made no progress"
    await asyncio.sleep(0.2)


def coordinators(driver, rid):
  ctl = ray.get_actor(f"rsb-run-{rid}", namespace=driver.namespace)
  return ctl


def test_worker_loss_mid_run(reference, tmp_path):
  cfg = config(tmp_path)
  d = ClusterDriver(cfg, namespace=f"rsb-c1-{uuid.uuid4().hex[:6]}")

  async def chaos(driver, rid):
    await wait_committed(driver, rid, 2)
    coord = ray.get_actor(f"rsb-coord-{rid[:8]}-0", namespace=driver.namespace)
    names = ray.get(coord.worker_names.remote())
    ray.get(coord.kill_worker.remote(names[0]))

  rid, st = run(d, cfg, chaos)
  assert st.state == "COMPLETED", st.error
  assert st.artifacts["complete"], st.artifacts
  got = digests(d, rid)
  assert got == reference  # same episodes, same results, each exactly once
  events = [e["event"] for e in d.store.db.audit_events(rid)]
  assert "worker_lost" in events


def test_coordinator_loss_mid_run(reference, tmp_path):
  cfg = config(tmp_path)
  d = ClusterDriver(cfg, namespace=f"rsb-c2-{uuid.uuid4().hex[:6]}")

  async def chaos(driver, rid):
    await wait_committed(driver, rid, 2)
    coord = ray.get_actor(f"rsb-coord-{rid[:8]}-0", namespace=driver.namespace)
    ray.kill(coord, no_restart=True)

  rid, st = run(d, cfg, chaos)
  assert st.state == "COMPLETED", st.error
  assert st.artifacts["complete"], st.artifacts
  assert digests(d, rid) == reference
  events = [e["event"] for e in d.store.db.audit_events(rid)]
  assert "coordinator_lost" in events and "coordinator_replaced" in events


def test_controller_loss_and_resume(reference, tmp_path):
  cfg = config(tmp_path)
  d = ClusterDriver(cfg, namespace=f"rsb-c3-{uuid.uuid4().hex[:6]}")

  async def go():
    rid = await d.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2}), uuid.uuid4().hex)
    await wait_committed(d, rid, 2)
    ray.kill(ray.get_actor(f"rsb-run-{rid}", namespace=d.namespace), no_restart=True)  # client stays alive
    await asyncio.sleep(7.0)  # the dead controller's run and unit leases expire
    before = d.store.db.episode_count(rid)
    st = await d.resume(rid, cfg)
    st = await d.wait(rid, timeout_s=600)
    return rid, st, before

  rid, st, before = asyncio.run(go())
  assert 2 <= before < 72
  assert st.state == "COMPLETED", st.error
  assert digests(d, rid) == reference
  units = d.store.db.units(rid)
  assert any(u["attempt"] > 1 for u in units)  # interrupted units were re-leased


def test_deterministic_worker_error_terminates_the_run(ray_session, tmp_path):
  """Found while wiring telemetry: an exception inside a worker command killed the
  coordinator, the controller replaced it, and the run stayed RUNNING forever. Now a
  worker error is a worker fault (restart, then quarantine), units retry up to
  max_attempts, coordinator replacement is bounded by failure.max_restarts, and the run
  terminates either way with the root cause recorded: FAILED when the replacement bound
  is hit first, or COMPLETED with complete=false when every unit exhausted first
  (allow_partial, design §10)."""
  cfg = config(tmp_path)
  cfg = cfg.model_copy(update={"failure": cfg.failure.model_copy(update={"circuit_open_s": 1.0, "max_restarts": 1})})
  d = ClusterDriver(cfg, namespace=f"rsb-c4-{uuid.uuid4().hex[:6]}")

  async def go():
    rid = await d.submit(RunRequest("eval", cfg, cfg.tasks, "test",
                                    {"actors": 2, "worker_class": "raysimbatch.testing.faults:RemoteFailingWorker"}),
                         uuid.uuid4().hex)
    return rid, await d.wait(rid, timeout_s=300)

  rid, st = asyncio.run(go())
  assert st.state in ("FAILED", "COMPLETED"), (st.state, st.error)
  if st.state == "FAILED":
    assert "coordinator 0 failed" in st.error and "injected worker fault" in st.error, st.error
  else:
    assert st.artifacts["complete"] is False and st.artifacts["coverage"] == 0.0
    assert set(d.store.db.unit_counts(rid)) == {"EXHAUSTED"}
  audit = d.store.db.audit_events(rid)
  lost = [e["payload"]["reason"] for e in audit if e["event"] == "worker_lost"]
  assert lost and all("injected worker fault" in r for r in lost)
  for e in audit:  # a coordinator that gives up names the root cause, not a lease symptom
    if e["event"] == "coordinator_lost":
      assert "injected worker fault" in e["payload"]["error"], e["payload"]["error"][-400:]


def test_worker_class_override_is_restricted_to_testing(ray_session, tmp_path):
  from raysimbatch.control.coordinator import _worker_class

  with pytest.raises(ValueError, match="limited to raysimbatch.testing"):
    _worker_class("os:system")


def _orphan(d, cfg, cluster: str):
  """Submit, let it commit a little, then kill the controller and let its lease lapse."""
  async def go():
    rid = await d.submit(RunRequest("eval", cfg, cfg.tasks, "test", {"actors": 2, "cluster": cluster}),
                         uuid.uuid4().hex)
    await wait_committed(d, rid, 2)
    ray.kill(ray.get_actor(f"rsb-run-{rid}", namespace=d.namespace), no_restart=True)
    await asyncio.sleep(cfg.failure.lease_s + 1.0)
    return rid, await d.status(rid)
  return asyncio.run(go())


def test_janitor_resumes_an_orphaned_run(reference, tmp_path):
  """Design §8.1 RunController loss: lease expiry, then a new controller reconciles. Found
  in kind: an OOM-killed controller left its run orphaned until someone noticed."""
  from raysimbatch.recovery.watchdog import janitor_pass

  cfg = config(tmp_path)
  d = ClusterDriver(cfg, namespace=f"rsb-c5-{uuid.uuid4().hex[:6]}")
  cluster = f"test-{uuid.uuid4().hex[:8]}"
  rid, st = _orphan(d, cfg, cluster)
  assert "orphaned" in st.detail  # status says so instead of repeating a stale state
  res = asyncio.run(janitor_pass(d.store, d.namespace, cluster=cluster))
  assert res["resumed_runs"] == [rid]
  st = asyncio.run(d.wait(rid, timeout_s=600))
  assert st.state == "COMPLETED" and st.artifacts["complete"], st.error
  assert digests(d, rid) == reference  # same episodes from the original seeds, each once
  assert "janitor_resumed_run" in [e["event"] for e in d.store.db.audit_events(rid)]


def test_janitor_bounds_orphan_resumes(tmp_path):
  from raysimbatch.recovery.watchdog import janitor_pass

  cfg = config(tmp_path)
  cfg = cfg.model_copy(update={"failure": cfg.failure.model_copy(update={"max_restarts": 0})})
  d = ClusterDriver(cfg, namespace=f"rsb-c6-{uuid.uuid4().hex[:6]}")
  cluster = f"test-{uuid.uuid4().hex[:8]}"
  rid, _ = _orphan(d, cfg, cluster)
  res = asyncio.run(janitor_pass(d.store, d.namespace, cluster=cluster))
  assert res["failed_orphans"] == [rid] and res["resumed_runs"] == []
  st = asyncio.run(d.status(rid))
  assert st.state == "FAILED" and "lost 1 times" in st.error
