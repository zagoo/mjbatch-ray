# SPDX-License-Identifier: Apache-2.0
"""HeteroBatchRouter on real actors: asynchronous ready subsets, barrier semantics,
fencing, and exact migration."""

from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
import pytest

from raysimbatch import protocol as P
from raysimbatch.protocol import Flag
from raysimbatch.routing.fleet import FleetMember, start_fleet, stop_fleet
from raysimbatch.routing.router import DeficitRoundRobin, HeteroBatchRouter, MigrationState
from raysimbatch.testing import fixtures as F

pytestmark = pytest.mark.ray

HEAVY = F.asset("quadruped.xml").replace(
  b"</worldbody>",
  b"".join(
    b'<body pos="%f %f %f"><freejoint/><geom type="box" size=".05 .05 .05" mass=".1"/></body>'
    % (0.3 + 0.11 * (i % 6), 0.3 + 0.11 * ((i // 6) % 6), 0.1 + 0.11 * (i // 36))
    for i in range(108)
  ) + b"</worldbody>",
)


def member(fx, n, slots=8, **kw):
  run_id = uuid.uuid4().hex
  spec, mjb = F.worker_spec(fx, run_id, fx.name, num_slots=slots, threads=1, **kw)
  return FleetMember(spec, mjb, n)


def actions_for(router: HeteroBatchRouter, frame: P.HeteroFrame, rng=None, scale=0.0):
  groups = []
  for g in frame.groups:
    live = (g.flags & Flag.VALID) != 0
    live &= (g.flags & (Flag.TERMINATED | Flag.TRUNCATED | Flag.ABORTED)) == 0
    if not live.any():
      continue
    name = router.routes[int(g.env_id[live][0])].actor_name
    a = router.actors[name].action_dim
    act = np.zeros((int(live.sum()), a), np.float32) if rng is None else (
      rng.uniform(-scale, scale, (int(live.sum()), a)).astype(np.float32))
    groups.append(P.ActionGroup(
      group_hash=g.group_hash, schema_hash=g.schema_hash, policy_hash=b"\0" * 32,
      env_id=g.env_id[live].copy(), episode_id=g.episode_id[live].copy(),
      expected_control_seq=g.control_seq[live].copy(), slot_generation=g.slot_generation[live].copy(),
      slot=g.slot[live].copy(), action=act, behavior_logp=np.zeros(int(live.sum()), np.float32),
      policy_version=np.zeros(int(live.sum()), np.uint64)))
  return P.ActionFrame(router.run_bytes, P.new_id(), frame.routing_epoch, time.time_ns(), P.new_id(), tuple(groups))


def test_heterogeneous_async_send_recv(ray_session):
  run_id = uuid.uuid4().hex
  cp, qd = member(F.cartpole(max_control_steps=10**6), 1), member(F.quadruped(max_control_steps=10**6), 1)
  router, _ = start_fleet(run_id, [cp, qd])
  router.run_id, router.run_bytes = run_id, bytes.fromhex(run_id)

  async def go():
    envs = sorted(router.routes)
    frame = await router.reset(envs, [0] * len(envs))
    assert {g.observation.shape[1] for g in frame.groups} == {4, 36}
    last_seq = {}
    sub = await router.send(actions_for(router, frame))
    assert sub.rejected.size == 0 and sub.accepted.size == len(envs)
    steps = 0
    while steps < 200:
      f, _ = await router.recv(min_ready=1, timeout_s=1.0)
      for g in f.groups:
        for e, s in zip(g.env_id.tolist(), g.control_seq.tolist(), strict=True):
          assert s == last_seq.get(e, 0) + 1  # per-env sequences arrive in order
          last_seq[e] = s
        steps += 1
      await router.send(actions_for(router, f))
    # Stale: replaying the first frame's actions (seq 0) is rejected before mutation.
    stale = await router.send(actions_for(router, frame))
    assert stale.accepted.size == 0
    assert set(stale.codes.tolist()) <= {P.FaultCode.SEQUENCE_MISMATCH, P.FaultCode.DUPLICATE_ROW}
    return last_seq

  seqs = asyncio.run(go())
  assert len(seqs) == 16
  stop_fleet(router)


def _rate(router, envs, seconds):
  async def go():
    f = await router.reset(envs, [0] * len(envs))
    await router.send(actions_for(router, f))
    t0, n = time.monotonic(), 0
    while time.monotonic() - t0 < seconds:
      fr, _ = await router.recv(min_ready=1, timeout_s=0.5)
      for g in fr.groups:
        if router.routes[int(g.env_id[0])].actor_name.startswith("cartpole"):
          n += g.rows
      await router.send(actions_for(router, fr))
    # drain
    while any(a.inflight is not None for a in router.actors.values()):
      await router.recv(min_ready=10**9, timeout_s=0.2)
    return n / (time.monotonic() - t0)
  return asyncio.run(go())


def test_slow_actor_does_not_gate_the_logical_batch(ray_session):
  """§13.2: a much slower environment group must not slow unrelated actors."""
  fast = member(F.cartpole(max_control_steps=10**9), 2, slots=16)
  q = F.quadruped()
  heavy_fx = F.Fixture("heavy", HEAVY, q.group.model_copy(update={"iterations": 100, "initial_keyframe": -1}),
                       q.contract, {**q.task_doc, "spec": {**q.task_doc["spec"], "min_height": -1e9,
                                                           "min_up_z": -2.0}},
                       q.reset_distribution, (), -1)
  slow = member(heavy_fx, 1, slots=16)
  run_id = uuid.uuid4().hex
  r1, _ = start_fleet(run_id, [fast])
  base = _rate(r1, sorted(r1.routes), 4.0)
  stop_fleet(r1)
  r2, _ = start_fleet(run_id, [fast, slow])
  mixed = _rate(r2, sorted(r2.routes), 4.0)
  stop_fleet(r2)
  assert mixed >= 0.95 * base, (base, mixed)


def test_barrier_step_reports_every_env_and_never_autoresets(ray_session):
  run_id = uuid.uuid4().hex
  router, _ = start_fleet(run_id, [member(F.cartpole(max_control_steps=3), 1, slots=4)])

  async def go():
    envs = sorted(router.routes)
    f = await router.reset(envs, [0] * 4)
    for _ in range(3):
      res = await router.barrier_step(actions_for(router, f), router.epoch, deadline_s=10)
      assert res.complete and set(res.status) == set(envs)
      f = res.frame
    flags = np.concatenate([g.flags for g in f.groups])
    assert (flags & Flag.TRUNCATED).all()  # time limit reached inside the barrier
    res = await router.barrier_step(actions_for(router, f), router.epoch, deadline_s=5)
    # No hidden autoreset: truncated envs have no valid action rows to step.
    assert not res.complete or all(v != "ok" for v in res.status.values()) or len(res.status) == 0

  asyncio.run(go())
  stop_fleet(router)


def test_exact_migration_between_actors(ray_session):
  fx = F.quadruped(history=2, max_control_steps=10**9)
  run_id = uuid.uuid4().hex
  a = member(fx, 2, slots=4)
  router, _ = start_fleet(run_id, [a])
  ref, _ = start_fleet(run_id, [member(fx, 1, slots=4)])
  rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)

  async def drive(r, steps, rng, env):
    f = await r.reset([env], [0])
    out = []
    for _ in range(steps):
      acts = actions_for(r, f, rng, 1.0)
      await r.send(acts)
      f, _ = await r.recv(min_ready=1, timeout_s=10)
      out.append(np.concatenate([g.observation for g in f.groups]))
    return f, out

  async def go():
    f, first = await drive(router, 10, rng1, 0)
    names = sorted(router.actors)
    src = router.routes[0].actor_name
    dst = [n for n in names if n != src][0]
    router.retire([router.slot_env[(dst, 3)]])  # free the destination slot
    plan = await router.migrate(0, dst, 3)
    assert plan.states[-2:] == [MigrationState.COMMITTED, MigrationState.RETIRED]
    assert router.routes[0].actor_name == dst and router.routes[0].slot == 3
    stale = await router.send(actions_for(router, f, np.random.default_rng(9), 1.0))
    assert stale.accepted.size == 0 and (stale.codes == P.FaultCode.STALE_ROUTING_EPOCH).all()
    np.testing.assert_array_equal(plan.frame.groups[0].observation, first[-1])  # same current observation
    f = plan.frame
    later = []
    for _ in range(10):
      await router.send(actions_for(router, f, rng1, 1.0))
      f, _ = await router.recv(min_ready=1, timeout_s=10)
      later.append(np.concatenate([g.observation for g in f.groups]))
    _, whole = await drive(ref, 20, rng2, 0)
    for x, y in zip(first + later, whole, strict=True):
      np.testing.assert_array_equal(x, y)  # bit-exact continuation after migration

  asyncio.run(go())
  stop_fleet(router)
  stop_fleet(ref)


def test_deficit_round_robin_and_rebalance_plan():
  drr = DeficitRoundRobin({"a": 0.75, "b": 0.25})
  for _ in range(400):
    g = drr.next_group(["a", "b"])
    drr.record(g, starts=1, transitions=10 if g == "a" else 1)  # "a" is much faster
  sh = drr.shares()
  assert abs(sh["starts"]["a"] - 0.75) < 0.01  # weights refer to episode starts
  assert sh["transitions"]["a"] > 0.9  # transition share reported separately
  r = HeteroBatchRouter("00" * 16, {}, {})
  plan = r.rebalance_plan({"a": 0.5, "b": 0.5}, {f"w{i}": "a" for i in range(4)}, version=2)
  assert len(plan.moves) == 2 and all(m["to_group"] == "b" for m in plan.moves)
  with pytest.raises(ValueError):
    r.rebalance_plan({"a": 1.0}, {}, 3, boundary="now")


def test_fleet_routes_follow_each_members_env_count(ray_session):
  """Found by static analysis: start_fleet sized every actor's routes from the *last*
  member's worker.envs, so a smaller member got routes to slots it does not have."""
  import collections
  import dataclasses

  from raysimbatch.worker.actor import WorkerSpec

  run_id = uuid.uuid4().hex
  small, large = member(F.cartpole(max_control_steps=100), 1, slots=4), member(F.quadruped(max_control_steps=100), 1)
  members = [dataclasses.replace(m, spec=WorkerSpec.from_dict({**m.spec.to_dict(), "num_slots": None}))
             for m in (small, large)]  # sized by each member's worker.envs
  router, _ = start_fleet(run_id, members)
  try:
    per_actor = collections.Counter(r.actor_name for r in router.routes.values())
    assert sorted(per_actor.values()) == [4, 8]
  finally:
    stop_fleet(router)
