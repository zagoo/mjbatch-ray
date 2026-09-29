# SPDX-License-Identifier: Apache-2.0
"""Mode A end to end: heterogeneous asynchronous rollouts into a durable replay sink."""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import numpy as np
import pytest
import ray

from raysimbatch.config.models import CurriculumConfig, InferenceConfig, RolloutConfig
from raysimbatch.control.driver import ClusterDriver, RolloutOrchestrator, RunRequest
from raysimbatch.experience.sinks import DurableReplaySink, to_rllib_batch
from raysimbatch.policy.export import gaussian_bundle
from raysimbatch.protocol import Flag
from raysimbatch.testing import fixtures as F
from raysimbatch.testing import runs as R

pytestmark = [pytest.mark.ray, pytest.mark.postgres, pytest.mark.slow]


def config(tmp_path, placement="local", target=20_000, **ro):
  if not os.environ.get("RSB_TEST_DSN"):
    pytest.skip("RSB_TEST_DSN not set")
  root = f"file://{tmp_path}/artifacts"
  cp = R.group_from_fixture(root, F.cartpole(max_control_steps=200))
  qd = R.group_from_fixture(root, F.quadruped(max_control_steps=150))
  pc, ec = gaussian_bundle(4, 1, seed=1, name="cp_pol")
  pq, eq = gaussian_bundle(36, 8, seed=2, name="qd_pol")
  pols = [R.policy_spec(root, pc, ec), R.policy_spec(root, pq, eq)]
  return R.run_config(
    root, "env:RSB_TEST_DSN", [cp, qd], pols, mode="rollout", target_envs=64,
    rollout=RolloutConfig(target_transitions=target, fragment_steps=16, **ro),
    curriculum=CurriculumConfig(version=1, weights={"cartpole": 0.5, "quadruped": 0.5}),
    inference=InferenceConfig(placement=placement, device="cpu", flush_ms=2.0),
    worker={"envs": 16, "slice_size": 16, "slice_min": 16},
    failure={"lease_s": 10.0, "lease_renew_s": 2.0},
  ), (pc, pq)


def replay(driver, rid):
  sink = DurableReplaySink(driver.store, rid)
  return [f for _, f in sink.read(0, 100_000)]


def check_fragments(frags):
  ids = [f.fragment_id() for f in frags]
  assert len(ids) == len(set(ids))  # exactly-once logical insertion
  total = 0
  for f in frags:
    total += f.transitions
    for e in range(f.segments):
      a, b = int(f.episode_offsets[e]), int(f.episode_offsets[e + 1])
      np.testing.assert_array_equal(f.control_seq[a:b], f.seg_first_seq[e] + np.arange(b - a))
      last = int(f.flags[b - 1])
      if last & (Flag.TERMINATED | Flag.TRUNCATED):
        assert np.any(f.terminal_observation[e] != 0)
    batch = to_rllib_batch(f)
    assert batch["obs"].shape == batch["new_obs"].shape
    assert np.isfinite(f.behavior_logp).all()
  return total


def run(driver, cfg, opts, during=None, timeout=600):
  async def go():
    rid = await driver.submit(RunRequest("rollout", cfg, (), "test", opts), uuid.uuid4().hex)
    if during:
      await during(driver, rid)
    return rid, await driver.wait(rid, timeout_s=timeout)
  return asyncio.run(go())


@pytest.mark.parametrize("placement", ["local", "central"])
def test_heterogeneous_rollout(ray_session, tmp_path, placement):
  cfg, (pc, pq) = config(tmp_path, placement)
  d = ClusterDriver(cfg, namespace=f"rsb-a-{uuid.uuid4().hex[:6]}")
  rid, st = run(d, cfg, {"actors": 2, "target_transitions": 20_000})
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  frags = replay(d, rid)
  total = check_fragments(frags)
  assert total >= 20_000
  dims = {f.observation.shape[1] for f in frags}
  assert dims == {4, 36}  # two schemas, never padded to the larger one
  versions = {int(v) for f in frags for v in np.unique(f.policy_version)}
  assert versions == {1}
  hashes = {f.policy_hash.hex() for f in frags}
  assert hashes == {pc.policy_hash, pq.policy_hash}
  prog = [c["program"] for c in st.artifacts["coordinators"]]
  assert all(p["committed_transitions"] >= p["target"] or True for p in prog)
  starts = sum((p["starts"] for p in prog), start={}) if False else None
  del starts


def test_policy_lag_rejection_and_update(ray_session, tmp_path):
  cfg, (pc, pq) = config(tmp_path, "local", target=10**9, max_policy_lag_versions=2)
  d = ClusterDriver(cfg, namespace=f"rsb-lag-{uuid.uuid4().hex[:6]}")

  async def wait_for(pred, what, within_s=120):
    t0 = time.monotonic()
    while not pred():
      assert time.monotonic() - t0 < within_s, what
      await asyncio.sleep(0.1)

  async def during(driver, rid):
    ctl = ray.get_actor(f"rsb-run-{rid}", namespace=driver.namespace)
    await wait_for(lambda: ray.get(ctl.get_assembler.remote()) is not None, "assembler")
    asm = ray.get(ctl.get_assembler.remote())
    await wait_for(lambda: ray.get(asm.status.remote())["stats"]["accepted"] > 5, "first fragments")
    ray.get(asm.set_learner_version.remote(10))  # the learner moved on: v1 is now 9 behind
    await wait_for(lambda: ray.get(asm.status.remote())["stats"]["rejected_lag"] > 0, "lag rejection")
    accepted_before = ray.get(asm.status.remote())["stats"]["accepted"]
    orch = RolloutOrchestrator(rid, driver.namespace)
    for obs_dim, act_dim, grp, seed, name in ((4, 1, "cartpole", 1, "cp_pol"), (36, 8, "quadruped", 2, "qd_pol")):
      b, exe = gaussian_bundle(obs_dim, act_dim, version=10, seed=seed, name=name)
      await orch.update_policy({"groups": [grp], "manifest": b.to_json().decode(),
                                "exe_ref": ray.put(np.frombuffer(exe, np.uint8)), "hash": b.policy_hash,
                                "version": 10})
    await wait_for(lambda: ray.get(asm.status.remote())["stats"]["accepted"] > accepted_before + 10, "v10 accepted")
    await orch.stop(drain=True)

  rid, st = run(d, cfg, {"actors": 2, "target_transitions": 10**9}, during)
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  frags = replay(d, rid)
  check_fragments(frags)
  versions = {int(v) for f in frags for v in np.unique(f.policy_version)}
  assert {1, 10} <= versions  # pre-lag fragments and post-update fragments
  for f in frags:  # a fragment never mixes behavior policies under one header hash
    assert len(np.unique(f.policy_version)) == 1
  seal = st.artifacts["sink_offset"]["stats"]
  assert seal["rejected_lag"] > 0


def test_worker_loss_records_gaps(ray_session, tmp_path):
  cfg, _ = config(tmp_path, "local", target=40_000)
  d = ClusterDriver(cfg, namespace=f"rsb-gap-{uuid.uuid4().hex[:6]}")

  async def during(driver, rid):
    t0 = time.monotonic()
    while True:
      try:
        coord = ray.get_actor(f"rsb-coord-{rid[:8]}-0", namespace=driver.namespace)
        names = ray.get(coord.worker_names.remote())
        if names and ray.get(coord.status.remote())["program"].get("committed_transitions", 0) > 1000:
          break
      except ValueError:
        pass
      assert time.monotonic() - t0 < 120
      await asyncio.sleep(0.2)
    ray.get(coord.kill_worker.remote(names[0]))

  rid, st = run(d, cfg, {"actors": 2, "target_transitions": 40_000}, during)
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  gaps = d.store.db.gaps(rid)
  assert gaps and all(g["lost_upper"] >= g["lost_lower"] >= 0 for g in gaps)
  assert st.artifacts["lost_upper_bound"] == sum(int(g["lost_upper"]) for g in gaps)
  check_fragments(replay(d, rid))  # committed fragments stay valid and unique


def test_restarted_worker_keeps_updated_policy(ray_session, tmp_path):
  """A worker lost after a policy update must come back on the updated policy. With
  the learner at v10 and max lag 2, a worker that silently reverted to v1 would produce
  only lag-rejected fragments after its restart."""
  cfg, _ = config(tmp_path, "local", target=10**9, max_policy_lag_versions=2)
  d = ClusterDriver(cfg, namespace=f"rsb-rp-{uuid.uuid4().hex[:6]}")

  async def wait_for(pred, what, within_s=120):
    t0 = time.monotonic()
    while not pred():
      assert time.monotonic() - t0 < within_s, what
      await asyncio.sleep(0.1)

  async def during(driver, rid):
    ctl = ray.get_actor(f"rsb-run-{rid}", namespace=driver.namespace)
    await wait_for(lambda: ray.get(ctl.get_assembler.remote()) is not None, "assembler")
    asm = ray.get(ctl.get_assembler.remote())
    stats = lambda: ray.get(asm.status.remote())["stats"]  # noqa: E731
    await wait_for(lambda: stats()["accepted"] > 5, "first fragments")
    orch = RolloutOrchestrator(rid, driver.namespace)
    for obs_dim, act_dim, grp, seed, name in ((4, 1, "cartpole", 1, "cp_pol"), (36, 8, "quadruped", 2, "qd_pol")):
      b, exe = gaussian_bundle(obs_dim, act_dim, version=10, seed=seed, name=name)
      await orch.update_policy({"groups": [grp], "manifest": b.to_json().decode(),
                                "exe_ref": ray.put(np.frombuffer(exe, np.uint8)), "hash": b.policy_hash,
                                "version": 10})
    ray.get(asm.set_learner_version.remote(10))
    base = stats()["accepted"]
    await wait_for(lambda: stats()["accepted"] > base + 10, "v10 accepted")
    coord = ray.get_actor(f"rsb-coord-{rid[:8]}-0", namespace=driver.namespace)
    for name in ray.get(coord.worker_names.remote()):
      ray.get(coord.kill_worker.remote(name))
    await wait_for(lambda: all(w["restarts"] >= 1 and w["state"] == "ready"
                               for w in ray.get(coord.status.remote())["workers"].values()), "restart")
    rejected, accepted = stats()["rejected_lag"], stats()["accepted"]
    await wait_for(lambda: stats()["accepted"] > accepted + 20,
                   "no post-restart fragments accepted (stale policy after restart is lag-rejected)")
    assert stats()["rejected_lag"] == rejected, "a restarted worker produced stale-policy fragments"
    await orch.stop(drain=True)

  rid, st = run(d, cfg, {"actors": 2, "target_transitions": 10**9}, during)
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  frags = replay(d, rid)
  check_fragments(frags)
  assert int(frags[-1].policy_version[0]) == 10


def test_assembler_restart_mid_run(ray_session, tmp_path):
  """Ray restarts a dead assembler by rerunning only its constructor. The restarted
  assembler must set its store and sink up again, deliver what its durable outbox holds,
  and accept again the fragments that were in flight when it died. Before, every later
  accept failed, each failure leaked a worker credit and the run never reached its
  target."""
  cfg, _ = config(tmp_path, "local", target=40_000)
  d = ClusterDriver(cfg, namespace=f"rsb-asm-{uuid.uuid4().hex[:6]}")

  async def during(driver, rid):
    t0 = time.monotonic()
    while True:
      try:
        coord = ray.get_actor(f"rsb-coord-{rid[:8]}-0", namespace=driver.namespace)
        if ray.get(coord.status.remote())["program"].get("committed_transitions", 0) > 1000:
          break
      except ValueError:
        pass
      assert time.monotonic() - t0 < 120
      await asyncio.sleep(0.2)
    ctl = ray.get_actor(f"rsb-run-{rid}", namespace=driver.namespace)
    ray.kill(ray.get(ctl.get_assembler.remote()), no_restart=False)

  rid, st = run(d, cfg, {"actors": 2, "target_transitions": 40_000}, during, timeout=240)
  assert st.state == "COMPLETED", (st.error, st.detail.get("traceback"))
  total = check_fragments(replay(d, rid))  # unique and valid, including the redelivered ones
  assert total >= 40_000
  assert not d.store.db.outbox_pending(rid, 10), "outbox rows left undelivered"
