# SPDX-License-Identifier: Apache-2.0
"""ClusterDriver and the RaySimBatch facade (design §6.1, §12).

The driver performs no per-step work: it validates, records the run durably
(idempotent on the caller's key), and starts one named detached RunController per run.
Clients may disconnect; ``status``/``wait``/``results`` reconnect by run id.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import ray

from raysimbatch.config.models import RunConfig, TaskSpec
from raysimbatch.config.system import ConfigurationSystem
from raysimbatch.storage.store import DurableStore

TERMINAL = ("COMPLETED", "FAILED", "CANCELED")


@dataclass(frozen=True)
class RunRequest:
  mode: str
  config: RunConfig
  tasks: tuple[TaskSpec, ...] = ()
  caller: str = "unknown"
  options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunStatus:
  run_id: str
  state: str
  epoch: int
  planned_slots: int
  healthy_slots: int
  quarantined_slots: int
  committed_units: int
  missing_units: int
  failed_units: int
  artifacts: dict[str, Any]
  error: str | None = None
  detail: dict[str, Any] = field(default_factory=dict)


class ClusterDriver:
  def __init__(self, config: RunConfig, store: DurableStore | None = None, namespace: str = "raysimbatch"):
    self.config = config
    self.store = store or DurableStore.from_config(config)
    self.namespace = namespace
    self.cs = ConfigurationSystem()

  def _controller(self, run_id: str):
    return ray.get_actor(f"rsb-run-{run_id}", namespace=self.namespace)

  async def submit(self, request: RunRequest, idempotency_key: str) -> str:
    cfg = request.config
    if cfg.mode != request.mode:
      cfg = cfg.model_copy(update={"mode": request.mode})
    rep = self.cs.validate(cfg)
    rep.raise_for_errors()
    run_id = cfg.run_id or uuid.uuid4().hex
    cfg = cfg.model_copy(update={"run_id": run_id})
    h = self.cs.canonical_hash(cfg)
    run_id, created = await asyncio.to_thread(
      self.store.db.create_run, run_id, h, cfg.mode,
      {"resolved": self.cs.resolved(cfg), "tasks": [t.model_dump(mode="json") for t in request.tasks]},
      idempotency_key, request.caller)
    if not created:
      return run_id  # idempotent resubmission
    await self._start(run_id, cfg, request.tasks, {**request.options, "config_hash": h})
    return run_id

  async def _start(self, run_id: str, cfg: RunConfig, tasks, options: dict[str, Any]) -> None:
    from raysimbatch.control.controller import RemoteRunController

    ctl = RemoteRunController.options(name=f"rsb-run-{run_id}", namespace=self.namespace, lifetime="detached",
                                      get_if_exists=False, **_off_head()).remote(
      run_id, cfg.model_dump_json(), json.dumps([t.model_dump(mode="json") for t in tasks]),
      {**options, "namespace": self.namespace})
    await asyncio.wrap_future(ctl.start.remote().future())

  async def status(self, run_id: str) -> RunStatus:
    row = await asyncio.to_thread(self.store.db.get_run, run_id)
    if row is None:
      raise KeyError(run_id)
    counts = await asyncio.to_thread(self.store.db.unit_counts, run_id)
    live: dict[str, Any] = {}
    try:
      live = await asyncio.wait_for(asyncio.wrap_future(self._controller(run_id).status.remote().future()), 10)
    except Exception:  # noqa: BLE001 - controller gone: durable state is authoritative
      live = {}
    if live.get("state") == "REFUSED":  # a start refused by another owner's lease: report the run itself
      live = {"error": live.get("error"), "detail": {"refused_controller": True}}
    orphaned = False
    if not live and row["state"] not in TERMINAL:
      lease = await asyncio.to_thread(self.store.db.lease_state, f"run/{run_id}")
      orphaned = lease is None or bool(lease["expired"])
    healthy = planned = 0
    for c in live.get("coordinators", []):
      for w in (c.get("workers") or {}).values():
        planned += 1
        healthy += w.get("state") == "ready"
    return RunStatus(
      run_id=run_id, state=live.get("state", row["state"]), epoch=int(row["epoch"]),
      planned_slots=planned, healthy_slots=healthy, quarantined_slots=planned - healthy,
      committed_units=counts.get("SUCCEEDED", 0),
      missing_units=sum(v for k, v in counts.items() if k not in ("SUCCEEDED", "EXHAUSTED", "CANCELED")),
      # A finished run's controller exits after a short linger; its results are durable.
      failed_units=counts.get("EXHAUSTED", 0), artifacts=live.get("results") or row["payload"].get("results") or {},
      error=live.get("error") or row["payload"].get("error"),
      detail={**(live.get("detail") or row["payload"].get("detail", {})),
              **({"orphaned": "no controller holds the run lease; the janitor or `resume` restarts it"}
                 if orphaned else {})},
    )

  async def wait(self, run_id: str, timeout_s: float = 60.0, poll_s: float = 1.0) -> RunStatus:
    deadline = time.monotonic() + timeout_s
    while True:
      st = await self.status(run_id)
      if st.state in TERMINAL or time.monotonic() >= deadline or st.detail.get("refused_controller"):
        return st
      await asyncio.sleep(poll_s)

  async def cancel(self, run_id: str, reason: str, drain: bool = True) -> None:
    await asyncio.wrap_future(self._controller(run_id).cancel.remote(reason, drain).future())

  async def resume(self, run_id: str, config: RunConfig, options: dict[str, Any] | None = None) -> RunStatus:
    """Resume under a new attempt epoch; any semantic config difference is refused here
    (it must be a linked new run, never an overwrite under the old hash)."""
    row = await asyncio.to_thread(self.store.db.get_run, run_id)
    if row is None:
      raise KeyError(run_id)
    prev = RunConfig.model_validate_json(json.dumps(row["payload"]["resolved"]))
    diff = self.cs.diff(prev, config.model_copy(update={"run_id": run_id}))
    if diff.classification in ("new_run", "forbidden"):
      raise ValueError(f"resume refused: {diff.classification} changes {diff.paths(diff.classification)}")
    try:
      old = self._controller(run_id)
      ray.kill(old, no_restart=True)
    except ValueError:
      pass
    # The run's state and epoch change only once the new controller holds the run lease
    # (a driver is not an owner: found in kind, a refused takeover mislabeled a live run).
    tasks = tuple(TaskSpec.from_obj(t) for t in row["payload"].get("tasks", []))
    await self._start(run_id, config.model_copy(update={"run_id": run_id}), tasks,
                      {**(options or {}), "config_hash": row["config_hash"], "resumed": True})
    return await self.status(run_id)

  async def scale(self, run_id: str, target_envs: int):
    """Capacity changes are boundary changes: recorded now, applied at the next wave
    (eval) or episode boundary (rollout)."""
    await asyncio.to_thread(self.store.db.audit, run_id, "scale_request", {"target_envs": target_envs})
    from raysimbatch.control.placement import plan_capacity

    cfg = self.config.model_copy(update={"target_envs": target_envs})
    return plan_capacity(cfg, run_id, {m.name: 1.0 for m in cfg.models})


class RaySimBatch:
  """Public facade (design §12); dispatches to the driver, adds no second control plane."""

  def __init__(self, config: RunConfig, *, namespace: str = "raysimbatch", options: dict[str, Any] | None = None):
    self.config = config
    self.driver = ClusterDriver(config, namespace=namespace)
    self.options = dict(options or {})

  async def evaluate(self, tasks: Sequence[TaskSpec] = (), seeds: Sequence[int] | None = None,
                     idempotency_key: str | None = None) -> str:
    cfg = self.config
    if seeds is not None:
      cfg = cfg.model_copy(update={"evaluation": cfg.evaluation.model_copy(update={"seeds": tuple(seeds)})})
    req = RunRequest("eval", cfg, tuple(tasks) or cfg.tasks, "facade", self.options)
    return await self.driver.submit(req, idempotency_key or uuid.uuid4().hex)

  async def rollout(self, sink: dict[str, Any] | None = None, idempotency_key: str | None = None) -> str:
    req = RunRequest("rollout", self.config, (), "facade", {**self.options, "sink": sink or {"kind": "durable_replay"}})
    return await self.driver.submit(req, idempotency_key or uuid.uuid4().hex)

  async def transfer(self, tasks: Sequence[TaskSpec], seeds: Sequence[int], axes: Sequence[str] = ("",),
                     idempotency_key: str | None = None) -> str:
    cfg = self.config.model_copy(update={"evaluation": self.config.evaluation.model_copy(
      update={"seeds": tuple(seeds)})})
    req = RunRequest("sim2sim", cfg, tuple(tasks), "facade", {**self.options, "axes": tuple(axes)})
    return await self.driver.submit(req, idempotency_key or uuid.uuid4().hex)

  async def results(self, run_id: str) -> dict[str, Any]:
    return (await self.driver.status(run_id)).artifacts

  async def wait(self, run_id: str, timeout_s: float = 60.0) -> RunStatus:
    return await self.driver.wait(run_id, timeout_s)

  async def cancel(self, run_id: str, drain: bool = True) -> None:
    await self.driver.cancel(run_id, "client request", drain)


class RolloutOrchestrator:
  """Mode A controls routed to the run's controller (design §6.1)."""

  def __init__(self, run_id: str, namespace: str = "raysimbatch"):
    self.run_id, self.namespace = run_id, namespace

  def _ctl(self):
    return ray.get_actor(f"rsb-run-{self.run_id}", namespace=self.namespace)

  async def update_curriculum(self, version: int, weights: dict[str, float]) -> None:
    await asyncio.wrap_future(self._ctl().update_curriculum.remote(version, weights).future())

  async def update_policy(self, policy: dict[str, Any]) -> None:
    await asyncio.wrap_future(self._ctl().update_policy.remote(policy).future())

  async def stop(self, drain: bool = True) -> dict[str, Any]:
    """Drained stop -> COMPLETED; ``drain=False`` cancels (CANCELED)."""
    ctl = self._ctl()
    if drain:
      await asyncio.wrap_future(ctl.stop.remote().future())
    else:
      await asyncio.wrap_future(ctl.cancel.remote("stop", False).future())
    while True:
      st = await asyncio.wrap_future(ctl.status.remote().future())
      if st["state"] in TERMINAL:
        return st
      await asyncio.sleep(0.5)


def _off_head() -> dict[str, Any]:
  """Place a run controller on the alive non-head node with the most memory; default
  placement only when there is no such node.

  A 0-CPU actor is otherwise placed next to the submitting driver, usually on the head;
  found in kind, controllers doing admission there were killed by Ray's memory monitor
  alongside GCS and the dashboard. The affinity is hard: a worker raylet seconds after
  registering answered a *soft* lease with spillback, and the GCS put the controller on
  the head (kind, twice). Hard affinity queues on that node instead; if the node dies
  first, creation fails loudly and the run is resumed like any orphan."""
  from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

  nodes = _non_head_nodes()
  if not nodes:
    return {}
  best = max(nodes, key=lambda n: n["Resources"].get("memory", 0.0))
  return {"scheduling_strategy": NodeAffinitySchedulingStrategy(best["NodeID"], soft=False)}


def _non_head_nodes() -> list[dict[str, Any]]:
  return [n for n in ray.nodes() if n.get("Alive") and "node:__internal_head__" not in n.get("Resources", {})]


def wait_for_non_head_node(timeout_s: float, poll_s: float = 1.0) -> bool:
  """Wait until a non-head node is alive, so `_off_head` has somewhere to place a
  controller; False after `timeout_s` (a head-only or scaled-to-zero cluster then keeps
  the soft fallback). Found in kind: the janitor sidecar starts with the head, and an
  orphan resumed before any worker node registered put its controller on the head, where
  the memory monitor killed it."""
  deadline = time.monotonic() + timeout_s
  while not _non_head_nodes():
    if time.monotonic() >= deadline:
      return False
    time.sleep(poll_s)
  return True
