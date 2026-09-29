# SPDX-License-Identifier: Apache-2.0
"""Liveness (design §8.4, §13.3): a native hang with the GIL released (heartbeat alive,
no progress) and a GIL-holding extension call (heartbeat dead) are both detected
externally, fenced durably and recovered by kill-with-restart."""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import numpy as np
import pytest
import ray

from raysimbatch import protocol as P
from raysimbatch.config.models import FailureConfig, WorkerConfig
from raysimbatch.recovery.watchdog import RemoteWatchdog, RestartManager
from raysimbatch.testing import fixtures as F
from raysimbatch.testing.faults import RemoteHangable
from raysimbatch.worker.actor import worker_options

pytestmark = [pytest.mark.ray, pytest.mark.postgres]


def start(fx, name):
  run_id = uuid.uuid4().hex
  spec, mjb = F.worker_spec(fx, run_id, name, num_slots=16, threads=2)
  opts = worker_options(WorkerConfig.from_obj(spec.worker), FailureConfig(), name=name, namespace="rsb-wd",
                        memory_bytes=256 * 2**20)
  h = RemoteHangable.options(**opts).remote(spec.to_dict(), "test://")
  ray.get(h.initialize.remote(ray.put(mjb), None, {"owner": "t", "epoch": 1, "expires_unix_ns": time.time_ns() + 10**12}))
  s = np.arange(16, dtype=np.int32)
  ray.get(h.reset.remote({"command_id": P.new_id(), "generation": 1, "slots": s, "env_id": s.astype(np.uint64),
                          "episode_id": np.zeros(16, np.uint64)}))
  return h, run_id


@pytest.fixture(scope="module")
def store():
  dsn = os.environ.get("RSB_TEST_DSN")
  if not dsn:
    pytest.skip("RSB_TEST_DSN not set")
  from raysimbatch.storage.blob import FileBlobStore
  from raysimbatch.storage.db import Database
  from raysimbatch.storage.store import DurableStore

  db = Database(dsn)
  db.migrate()
  import tempfile

  return DurableStore(db, FileBlobStore(tempfile.mkdtemp()))


@pytest.mark.parametrize("fault,code", [("hang_native", "NATIVE_PROGRESS_STALLED"), ("hold_gil", "HEARTBEAT_TIMEOUT")])
def test_watchdog_detects_and_recovers(ray_session, store, fault, code):
  name = f"wd-{fault}-{uuid.uuid4().hex[:6]}"
  h, run_id = start(F.quadruped(max_control_steps=10**6), name)
  pid = ray.get(h.health.remote())["pid"]
  cfg = FailureConfig(heartbeat_s=0.5, suspect_s=2.0).model_dump(mode="json")
  wd = RemoteWatchdog.options(namespace="rsb-wd").remote("node", cfg, "rsb-wd")
  ray.get(wd.register.remote(name, 1, 1.0))
  getattr(h, fault).remote()  # never returns on its own
  deadline = time.monotonic() + 60
  faults = ()
  while not faults:
    assert time.monotonic() < deadline, "watchdog did not detect the fault"
    time.sleep(0.5)
    faults = ray.get(wd.inspect.remote())
  assert faults[0]["code"] == code and faults[0]["affected"] == [name] or faults[0]["affected"] == (name,)
  rm = RestartManager(run_id, store, FailureConfig(), namespace="rsb-wd")
  res = asyncio.run(rm.recover(name, faults[0], family="quadruped", slots_in_flight=16))
  assert res.plan == "restart_reinitialize" and res.new_generation >= 1
  events = [e["event"] for e in store.db.audit_events(run_id)]
  assert "worker_fenced" in events
  # The actor comes back as a fresh process: constructor only, no state restored.
  deadline = time.monotonic() + 60
  while True:
    try:
      hh = ray.get(h.health.remote(), timeout=5)
      if hh["pid"] != pid:
        break
    except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
      pass
    assert time.monotonic() < deadline
    time.sleep(0.3)
  assert hh["state"] == P.WorkerState.CREATED
  ray.get(wd.stop.remote())
  ray.kill(h, no_restart=True)


def test_shared_watchdog_serves_each_owner_its_own_faults(ray_session):
  """One watchdog serves every coordinator on its node. ``inspect(names)`` must return and
  clear only those workers' faults, name the generation each fault is about (owners
  ignore faults of a generation they already replaced), report a name it does not watch
  (it lost its registrations) and stop watching a worker that is permanently gone."""
  names = [f"wd-own-{uuid.uuid4().hex[:6]}" for _ in range(2)]
  handles = [start(F.cartpole(max_control_steps=10**6), n)[0] for n in names]
  cfg = FailureConfig(heartbeat_s=0.5, suspect_s=2.0).model_dump(mode="json")
  wd = RemoteWatchdog.options(namespace="rsb-wd").remote("node", cfg, "rsb-wd")
  for n in names:
    ray.get(wd.register.remote(n, 1, 30.0))
  for h in handles:
    ray.kill(h, no_restart=False)  # dies, then comes back as a fresh, uninitialized process
  got: dict[str, list] = {n: [] for n in names}
  deadline = time.monotonic() + 30
  while not all(got.values()):
    assert time.monotonic() < deadline, got
    for n in names:
      got[n] += ray.get(wd.inspect.remote([n]))
    time.sleep(0.3)
  for n, faults in got.items():
    assert all(tuple(f["affected"]) == (n,) for f in faults), faults  # nobody else's
    assert {f["code"] for f in faults} <= {"WORKER_DIED", "WORKER_RESTARTED"}, faults
    assert all(f["detail"]["generation"] == 1 for f in faults), faults
  ghost = ray.get(wd.inspect.remote(["ghost"]))
  assert [(f["code"], tuple(f["affected"])) for f in ghost] == [("UNWATCHED", ("ghost",))]
  ray.kill(handles[0], no_restart=True)
  deadline = time.monotonic() + 30
  while not any(f["code"] == "UNWATCHED" for f in ray.get(wd.inspect.remote([names[0]]))):
    assert time.monotonic() < deadline, "a permanently dead worker is still watched"
    time.sleep(0.3)
  assert not ray.get(wd.inspect.remote([names[1]]))  # the restarted one is still watched, reported once
  ray.get(wd.stop.remote())
  ray.kill(handles[1], no_restart=True)
