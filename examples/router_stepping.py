# SPDX-License-Identifier: Apache-2.0
"""Explicit stepping over a heterogeneous logical batch with HeteroBatchRouter.

  python examples/router_stepping.py

A caller that owns its own control loop (for example an RL library's environment
runner) resets environments, sends actions and receives frames asynchronously: each
group advances as soon as its actions arrive, and per-environment control sequences
guard against stale or duplicate actions. No database is needed.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
from _common import start_ray

from raysimbatch import protocol as P
from raysimbatch.routing.fleet import FleetMember, start_fleet, stop_fleet
from raysimbatch.testing import fixtures as F


def actions(router, frame: P.HeteroFrame, rng) -> P.ActionFrame:
  """One action row per live environment of each group (random, for illustration)."""
  groups = []
  for g in frame.groups:
    live = ((g.flags & P.Flag.VALID) != 0) & ((g.flags & (P.Flag.TERMINATED | P.Flag.TRUNCATED | P.Flag.ABORTED)) == 0)
    if not live.any():
      continue
    width = router.actors[router.routes[int(g.env_id[live][0])].actor_name].action_dim
    n = int(live.sum())
    groups.append(P.ActionGroup(
      group_hash=g.group_hash, schema_hash=g.schema_hash, policy_hash=b"\0" * 32,
      env_id=g.env_id[live].copy(), episode_id=g.episode_id[live].copy(),
      expected_control_seq=g.control_seq[live].copy(), slot_generation=g.slot_generation[live].copy(),
      slot=g.slot[live].copy(), action=rng.uniform(-0.3, 0.3, (n, width)).astype(np.float32),
      behavior_logp=np.zeros(n, np.float32), policy_version=np.zeros(n, np.uint64)))
  return P.ActionFrame(router.run_bytes, P.new_id(), frame.routing_epoch, time.time_ns(), P.new_id(), tuple(groups))


def main() -> None:
  start_ray(None)
  run_id = uuid.uuid4().hex
  members = []
  for fx in (F.cartpole(max_control_steps=10**6), F.quadruped(max_control_steps=10**6)):
    spec, mjb = F.worker_spec(fx, run_id, fx.name, num_slots=8, threads=1)
    members.append(FleetMember(spec, mjb, 1))
  router, _ = start_fleet(run_id, members)
  router.run_id, router.run_bytes = run_id, bytes.fromhex(run_id)
  rng = np.random.default_rng(0)

  async def go():
    envs = sorted(router.routes)
    frame = await router.reset(envs, [0] * len(envs))
    print("groups:", [(g.rows, g.observation.shape[1]) for g in frame.groups])  # (envs, observation width)
    await router.send(actions(router, frame, rng))
    for _ in range(100):
      f, _ = await router.recv(min_ready=1, timeout_s=1.0)  # whichever groups are ready
      await router.send(actions(router, f, rng))
    stale = await router.send(actions(router, frame, rng))  # replaying the first frame's actions
    print(f"stale actions accepted: {stale.accepted.size}, rejected: {stale.rejected.size}")

  asyncio.run(go())
  stop_fleet(router)


if __name__ == "__main__":
  main()
