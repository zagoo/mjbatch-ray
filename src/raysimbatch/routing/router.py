# SPDX-License-Identifier: Apache-2.0
"""HeteroBatchRouter: explicit asynchronous stepping over a heterogeneous logical batch
(design §6.3, §12.1).

* ``send`` routes an ActionFrame's rows by env id to their actors, rejecting stale
  epoch/generation, mismatched episode/sequence, duplicate rows, wrong schema and
  nonfinite actions before any mutation. At most one action is outstanding per env and
  at most one mutating call per actor; further rows queue.
* ``recv`` returns whatever rows are ready (arrival order, per-env sequences ordered),
  possibly fewer than ``min_ready`` with ``timed_out=True``: slow actors never gate the
  logical batch.
* ``barrier_step`` is separately named: it steps every requested env, pins the policy
  and inputs, returns a row or an explicit timeout/fault for *every* env, and performs
  no hidden autoreset. Different timesteps still do not represent equal simulated time.
* ``migrate`` moves a quiescent env to another actor's slot through
  PLANNED -> DRAINING -> SNAPSHOTTED -> INSTALLED -> COMMITTED -> RETIRED; any failure
  before COMMITTED is ABORTED and the old route stays authoritative.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray

from raysimbatch import protocol as P
from raysimbatch.protocol import FaultCode, Route


class MigrationState(enum.Enum):
  PLANNED = "PLANNED"
  DRAINING = "DRAINING"
  SNAPSHOTTED = "SNAPSHOTTED"
  INSTALLED = "INSTALLED"
  COMMITTED = "COMMITTED"
  RETIRED = "RETIRED"
  ABORTED = "ABORTED"


@dataclass
class Submission:
  accepted: np.ndarray  # env ids
  rejected: np.ndarray  # env ids
  codes: np.ndarray  # uint16 per rejected row


@dataclass
class BarrierResult:
  frame: P.HeteroFrame
  status: dict[int, str]  # env id -> ok | timeout | <fault name>
  complete: bool


@dataclass
class MigrationPlan:
  version: int
  moves: list[dict[str, Any]]
  states: list[MigrationState] = field(default_factory=list)
  frame: P.HeteroFrame | None = None  # the migrated env's current observation


@dataclass
class ActorState:
  name: str
  handle: Any
  generation: int
  group_hash: bytes
  schema_hash: bytes
  obs_dim: int
  action_dim: int
  inflight: Any = None
  inflight_envs: np.ndarray | None = None
  queue: deque = field(default_factory=deque)
  pending_ack: list[int] = field(default_factory=list)


class HeteroBatchRouter:
  def __init__(self, run_id: str, actors: Mapping[str, dict[str, Any]], routes: Mapping[int, Route],
               routing_epoch: int = 0, store=None):
    self.run_id = run_id
    self.run_bytes = bytes.fromhex(run_id)
    self.epoch = routing_epoch
    self.store = store
    self.actors = {
      n: ActorState(n, a["handle"], int(a["generation"]), bytes.fromhex(a["group_hash"]),
                    bytes.fromhex(a["schema_hash"]), int(a["obs_dim"]), int(a["action_dim"]))
      for n, a in actors.items()
    }
    self.routes: dict[int, Route] = dict(routes)
    self.actor_epoch: dict[str, int] = {n: routing_epoch for n in self.actors}
    self.slot_env: dict[tuple[str, int], int] = {(r.actor_name, r.slot): e for e, r in self.routes.items()}
    self.episode: dict[int, int] = {}
    self.seq: dict[int, int] = {}
    self.outstanding: set[int] = set()
    self.ready: deque[tuple[float, P.GroupFrame]] = deque()
    self.fenced: set[str] = set()
    self.stats = defaultdict(int)

  # ------------------------------------------------------------------ routing

  def resolve(self, env_ids: np.ndarray, epoch: int) -> tuple[Route, ...]:
    if epoch != self.epoch:
      raise ValueError(f"stale routing epoch {epoch} (current {self.epoch})")
    return tuple(self.routes[int(e)] for e in env_ids)

  def observe(self, frame: P.HeteroFrame) -> None:
    """Record identities from a published frame (reset/step outputs)."""
    for g in frame.groups:
      for e, ep, sq, _fl in zip(g.env_id.tolist(), g.episode_id.tolist(), g.control_seq.tolist(), g.flags.tolist(),
                               strict=True):
        self.episode[e] = ep
        self.seq[e] = sq

  async def reset(self, env_ids: Sequence[int], episode_ids: Sequence[int]) -> P.HeteroFrame:
    by: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for e, ep in zip(env_ids, episode_ids, strict=True):
      r = self.routes[int(e)]
      by[r.actor_name].append((r.slot, int(e), int(ep)))
    groups = []
    for name, rows in by.items():
      a = self.actors[name]
      rows.sort()
      req = {"command_id": P.new_id(), "generation": a.generation, "slots": np.asarray([x[0] for x in rows], np.int32),
             "env_id": np.asarray([x[1] for x in rows], np.uint64),
             "episode_id": np.asarray([x[2] for x in rows], np.uint64)}
      env = await _await(a.handle.reset.remote(req))
      if env["rejected"]:
        raise RuntimeError(f"reset rejected by {name}: {FaultCode(env['rejected']).name}")
      f = P.HeteroFrame.from_wire(await _await(env["frame_ref"]))
      await _await(a.handle.ack.remote(env["credit_ids"]))
      self.observe(f)
      groups.extend(f.groups)
    return P.HeteroFrame(self.run_bytes, P.new_id(), 0, self.epoch, time.time_ns(), P.new_id(), tuple(groups))

  # --------------------------------------------------------------- send/recv

  async def send(self, frame: P.ActionFrame) -> Submission:
    acc, rej, codes = [], [], []
    per_actor: dict[str, list[tuple[P.ActionGroup, int]]] = defaultdict(list)
    seen: set[int] = set()
    for g in frame.groups:
      for i, e in enumerate(g.env_id.tolist()):
        code = FaultCode.NONE
        r = self.routes.get(e)
        if r is None:
          code = FaultCode.UNKNOWN_ENV
        elif frame.routing_epoch < r.routing_epoch or frame.routing_epoch > self.epoch:
          code = FaultCode.STALE_ROUTING_EPOCH  # this env's route changed after the frame
        elif e in seen:
          code = FaultCode.DUPLICATE_ROW
        elif e in self.outstanding:
          code = FaultCode.DUPLICATE_ROW  # at most one outstanding action per env
        elif r.actor_name in self.fenced or int(g.slot_generation[i]) != self.actors[r.actor_name].generation:
          code = FaultCode.STALE_GENERATION
        elif int(g.episode_id[i]) != self.episode.get(e, -1):
          code = FaultCode.EPISODE_MISMATCH
        elif int(g.expected_control_seq[i]) != self.seq.get(e, -1):
          code = FaultCode.SEQUENCE_MISMATCH
        elif g.schema_hash != self.actors[r.actor_name].schema_hash or g.action.shape[1] != self.actors[
          r.actor_name].action_dim:
          code = FaultCode.SCHEMA_MISMATCH
        elif not np.isfinite(g.action[i]).all():
          code = FaultCode.NONFINITE_ACTION
        seen.add(e)
        if code:
          rej.append(e)
          codes.append(int(code))
        else:
          assert r is not None  # an accepted row has a route (UNKNOWN_ENV otherwise)
          acc.append(e)
          per_actor[r.actor_name].append((g, i))
    for name, rows in per_actor.items():
      self.actors[name].queue.append((frame, rows))
      for g, i in rows:
        self.outstanding.add(int(g.env_id[i]))
    self._pump()
    return Submission(np.asarray(acc, np.uint64), np.asarray(rej, np.uint64), np.asarray(codes, np.uint16))

  def _pump(self) -> None:
    for a in self.actors.values():
      if a.inflight is not None or not a.queue:
        continue
      frame, rows = a.queue.popleft()
      rows.sort(key=lambda gi: self.routes[int(gi[0].env_id[gi[1]])].slot)
      slots = np.asarray([self.routes[int(g.env_id[i])].slot for g, i in rows], np.int32)
      envs = np.asarray([int(g.env_id[i]) for g, i in rows], np.uint64)
      g0 = rows[0][0]
      ag = P.ActionGroup(
        group_hash=a.group_hash, schema_hash=a.schema_hash, policy_hash=g0.policy_hash,
        env_id=envs, episode_id=np.asarray([int(g.episode_id[i]) for g, i in rows], np.uint64),
        expected_control_seq=np.asarray([int(g.expected_control_seq[i]) for g, i in rows], np.uint64),
        slot_generation=np.full(len(rows), a.generation, np.uint64), slot=slots,
        action=np.ascontiguousarray(np.stack([g.action[i] for g, i in rows]), np.float32),
        behavior_logp=np.asarray([float(g.behavior_logp[i]) for g, i in rows], np.float32),
        policy_version=np.asarray([int(g.policy_version[i]) for g, i in rows], np.uint64),
      )
      ep = self.actor_epoch[a.name]
      af = P.ActionFrame(self.run_bytes, P.new_id(), ep, time.time_ns(), frame.trace_id, (ag,))
      cmd = {"command_id": P.new_id(), "expected_generation": a.generation, "routing_epoch": ep,
             "slots": slots, "episode_id": ag.episode_id, "control_seq": ag.expected_control_seq,
             "policy_hash": g0.policy_hash, "deadline_unix_ns": time.time_ns() + int(30e9)}
      a.inflight = a.handle.advance.remote(cmd, af.to_wire())
      a.inflight_envs = envs

  async def _collect(self, timeout_s: float) -> None:
    refs = {a.inflight: a for a in self.actors.values() if a.inflight is not None}
    if not refs:
      return
    ready, _ = ray.wait(list(refs), num_returns=1, timeout=timeout_s, fetch_local=False)
    if ready:
      more, _ = ray.wait([r for r in refs if r not in ready], num_returns=len(refs) - len(ready), timeout=0)
      ready += more
    for ref in ready:
      a = refs[ref]
      a.inflight = None
      envs = a.inflight_envs
      assert envs is not None  # set together with the in-flight ref
      for e in envs.tolist():
        self.outstanding.discard(e)
      env = await _await(ref)
      if env["rejected"]:
        self.stats[f"rejected_{FaultCode(env['rejected']).name}"] += 1
        continue
      f = P.HeteroFrame.from_wire(await _await(env["frame_ref"]))
      await _await(a.handle.ack.remote(env["credit_ids"]))
      self.observe(f)
      now = time.monotonic()
      for g in f.groups:
        self.ready.append((now, g))
    self._pump()

  async def recv(self, min_ready: int = 1, max_ready: int = 1024, timeout_s: float = 0.01) -> tuple[P.HeteroFrame, bool]:
    deadline = time.monotonic() + timeout_s
    while sum(g.rows for _, g in self.ready) < min_ready:
      left = deadline - time.monotonic()
      if left <= 0 or all(a.inflight is None for a in self.actors.values()):
        break
      await self._collect(min(left, 0.01))
    out, rows = [], 0
    while self.ready and rows + self.ready[0][1].rows <= max_ready:
      _, g = self.ready.popleft()
      out.append(g)
      rows += g.rows
    timed_out = rows < min_ready
    return P.HeteroFrame(self.run_bytes, P.new_id(), 0, self.epoch, time.time_ns(), P.new_id(), tuple(out)), timed_out

  async def barrier_step(self, actions: P.ActionFrame, epoch: int, deadline_s: float = 30.0) -> BarrierResult:
    if epoch != self.epoch:
      raise ValueError("stale routing epoch")
    sub = await self.send(actions)
    want = set(sub.accepted.tolist())
    status: dict[int, str] = {int(e): FaultCode(int(c)).name for e, c in zip(sub.rejected.tolist(), sub.codes.tolist(),
                                                                                strict=True)}
    got: list[P.GroupFrame] = []
    deadline = time.monotonic() + deadline_s
    have: set[int] = set()
    while want - have and time.monotonic() < deadline:
      frame, _ = await self.recv(min_ready=len(want - have), timeout_s=min(0.1, deadline - time.monotonic()))
      for g in frame.groups:
        got.append(g)
        for e, fc, _fl in zip(g.env_id.tolist(), g.fault_code.tolist(), g.flags.tolist(), strict=True):
          have.add(e)
          status[e] = "ok" if fc == 0 else FaultCode(fc).name
    for e in want - have:
      status[e] = "timeout"
    return BarrierResult(P.HeteroFrame(self.run_bytes, P.new_id(), 0, self.epoch, time.time_ns(), P.new_id(),
                                       tuple(got)), status, not (want - have) and not sub.rejected.size)

  # --------------------------------------------------------------- migration

  async def migrate(self, env_id: int, dest_actor: str, dest_slot: int, version: int = 0) -> MigrationPlan:
    """Exact same-group migration of a quiescent env (full integration/task/policy state)."""
    from raysimbatch.telemetry import tracing

    with tracing.span("migration", env_id=env_id, to=dest_actor):
      return await self._migrate(env_id, dest_actor, dest_slot, version)

  async def _migrate(self, env_id: int, dest_actor: str, dest_slot: int, version: int) -> MigrationPlan:
    plan = MigrationPlan(version, [{"env_id": env_id, "to": dest_actor, "slot": dest_slot}], [MigrationState.PLANNED])
    src = self.routes[env_id]
    a, b = self.actors[src.actor_name], self.actors[dest_actor]
    try:
      if a.group_hash != b.group_hash:
        raise ValueError("snapshot migration requires the same model group; start a new episode instead")
      if (dest_actor, dest_slot) in self.slot_env:
        raise ValueError("destination slot is occupied")
      plan.states.append(MigrationState.DRAINING)
      # Quiesce both actors: nothing in flight or queued, so the epoch change below
      # cannot strand an issued command.
      while a.inflight is not None or b.inflight is not None or a.queue or b.queue:
        await self._collect(0.01)
      plan.states.append(MigrationState.SNAPSHOTTED)
      snap = await _await(a.handle.snapshot.remote(np.asarray([src.slot], np.int32), a.generation))
      res = await _await(b.handle.restore.remote(snap, np.asarray([dest_slot], np.int32), b.generation))
      if not res["ok"]:
        raise RuntimeError(f"restore failed: {res}")
      plan.states.append(MigrationState.INSTALLED)
      # Fence the source slot before the route commits: the old route can never revive.
      new_epoch = self.epoch + 1
      if self.store is not None:
        self.store.db.upsert_routes(self.run_id, [(env_id, new_epoch, {"actor": dest_actor, "slot": dest_slot,
                                                                        "generation": b.generation})])
      await _await(a.handle.set_routing_epoch.remote(new_epoch))  # fences the source route
      await _await(b.handle.set_routing_epoch.remote(new_epoch))
      self.actor_epoch[a.name] = self.actor_epoch[b.name] = new_epoch
      del self.slot_env[(src.actor_name, src.slot)]
      self.routes[env_id] = Route(dest_actor, 0, dest_slot, b.generation, new_epoch)
      self.slot_env[(dest_actor, dest_slot)] = env_id
      self.epoch = new_epoch
      plan.states.append(MigrationState.COMMITTED)
      env = await _await(b.handle.current.remote(P.new_id(), np.asarray([dest_slot], np.int32), b.generation))
      plan.frame = P.HeteroFrame.from_wire(await _await(env["frame_ref"]))
      plan.frame = P.HeteroFrame(plan.frame.run_id, plan.frame.request_id, plan.frame.producer_generation,
                                 new_epoch, plan.frame.created_unix_ns, plan.frame.trace_id, plan.frame.groups)
      await _await(b.handle.ack.remote(env["credit_ids"]))
      plan.states.append(MigrationState.RETIRED)
    except Exception:
      plan.states.append(MigrationState.ABORTED)  # the old route stays authoritative
      raise
    return plan

  def retire(self, env_ids: Sequence[int]) -> None:
    """Remove quiescent envs from the logical batch, freeing their slots."""
    for e in env_ids:
      if int(e) in self.outstanding:
        raise ValueError(f"env {e} has an action in flight")
      r = self.routes.pop(int(e))
      self.slot_env.pop((r.actor_name, r.slot), None)
      self.episode.pop(int(e), None)
      self.seq.pop(int(e), None)

  def rebalance_plan(self, weights: Mapping[str, float], group_of_actor: Mapping[str, str], version: int,
                     boundary: str = "episode") -> MigrationPlan:
    """Episode-boundary actor reassignment toward target group shares (no dynamics
    state moves: reassigned actors start new episodes of their new group)."""
    if boundary != "episode":
      raise ValueError("curriculum changes apply at episode boundaries")
    groups = sorted(weights)
    total = sum(weights.values())
    n = len(group_of_actor)
    target = {g: round(n * weights[g] / total) for g in groups}
    have = defaultdict(list)
    for a, g in sorted(group_of_actor.items()):
      have[g].append(a)
    moves = []
    surplus = [a for g in groups for a in have[g][target.get(g, 0):]]
    for g in groups:
      need = target[g] - len(have[g])
      while need > 0 and surplus:
        moves.append({"actor": surplus.pop(0), "to_group": g})
        need -= 1
    return MigrationPlan(version, moves, [MigrationState.PLANNED])


class DeficitRoundRobin:
  """Episode-start fairness across groups (design §6.3): tracks target minus actual
  starts; fast tasks cannot dominate a curriculum."""

  def __init__(self, weights: Mapping[str, float], quantum: int = 1):
    self.set_weights(weights)
    self.quantum = quantum
    self.starts: dict[str, int] = defaultdict(int)
    self.transitions: dict[str, int] = defaultdict(int)
    self.sim_seconds: dict[str, float] = defaultdict(float)

  def set_weights(self, weights: Mapping[str, float]) -> None:
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
      raise ValueError("weights must be nonnegative with a positive sum")
    total = sum(weights.values())
    self.weights = {k: v / total for k, v in weights.items()}

  def deficit(self, group: str) -> float:
    total = sum(self.starts.values())
    return total * self.weights.get(group, 0.0) - self.starts[group]

  def next_group(self, candidates: Sequence[str]) -> str | None:
    live = [g for g in candidates if self.weights.get(g, 0) > 0]
    if not live:
      return None
    return max(live, key=lambda g: (self.deficit(g) + self.quantum * self.weights[g], -live.index(g)))

  def record(self, group: str, starts: int = 0, transitions: int = 0, sim_seconds: float = 0.0) -> None:
    self.starts[group] += starts
    self.transitions[group] += transitions
    self.sim_seconds[group] += sim_seconds

  def shares(self) -> dict[str, dict[str, float]]:
    def norm(d):
      t = sum(d.values()) or 1
      return {k: v / t for k, v in d.items()}

    return {"starts": norm(self.starts), "transitions": norm(self.transitions), "sim_seconds": norm(self.sim_seconds)}


async def _await(ref):
  return await asyncio.wrap_future(ref.future())
