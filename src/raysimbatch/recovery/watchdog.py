# SPDX-License-Identifier: Apache-2.0
"""Watchdog, RestartManager and the cluster janitor (design §6.8, §8.2-8.4).

The Watchdog polls each registered worker's control-group ``health`` every
``heartbeat_s``. Heartbeat alone does not prove progress: a worker is SUSPECT when an
active command is older than its deadline while the native-completion counter has not
advanced for ``suspect_s``. A worker whose control group does not answer at all (e.g.
an extension holding the GIL) is also SUSPECT, as is one that died or came back as a
fresh, uninitialized process. Suspicion becomes a FaultRecord.

In a run there is one Watchdog per node (``watchdog_for_node``). Each coordinator
registers its workers after every (re)initialization and collects their faults; it owns
the in-flight commands, so it runs the containment (cancel, fence, kill with restart).
The RestartManager is the same fence-and-kill path for callers without a coordinator.

The janitor reconciles detached controllers and placement groups against database
leases: owners whose leases expired are fenced and their Ray resources removed; no
live lease is ever destroyed.
"""

from __future__ import annotations

import asyncio
import collections
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import ray
import ray.exceptions  # explicit: pyright does not see the submodule through `import ray`

from raysimbatch.compat.ray_typing import actor_class, get_one
from raysimbatch.config.models import FailureConfig
from raysimbatch.protocol import WorkerState
from raysimbatch.recovery.circuit import Circuit


@dataclass(frozen=True)
class FaultRecord:
  correlation_id: str
  component: str
  code: str
  time_unix_ns: int
  retryable: bool
  affected: tuple[str, ...]
  last_committed_seq: int | None = None
  diagnostic_uri: str | None = None
  detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Entry:
  name: str
  handle: Any
  generation: int
  deadline_s: float
  last_progress: int = -1
  last_progress_at: float = field(default_factory=time.monotonic)
  last_heartbeat_at: float = field(default_factory=time.monotonic)
  unresponsive_since: float | None = None
  suspect: bool = False


class Watchdog:
  def __init__(self, node_id: str, config: dict[str, Any], namespace: str | None = None,
               telemetry: dict[str, Any] | None = None, idle_exit_s: float | None = None):
    self.node_id = node_id
    self.cfg = FailureConfig.from_obj(config)
    self.namespace = namespace
    self.telemetry = telemetry
    self.idle_exit_s = idle_exit_s  # per-node deployment: exit when nothing is registered
    self.entries: dict[str, _Entry] = {}
    self.faults: collections.deque[FaultRecord] = collections.deque(maxlen=4096)
    self.metrics = None
    self._task: asyncio.Task | None = None
    self._stopped = False
    self._idle_since = time.monotonic()

  async def register(self, actor_name: str, generation: int, deadline_s: float) -> None:
    """Watch ``actor_name`` at ``generation``; registering again replaces the entry."""
    handle = ray.get_actor(actor_name, namespace=self.namespace)
    self.entries[actor_name] = _Entry(actor_name, handle, generation, deadline_s)
    if self._task is None:
      from raysimbatch import telemetry

      telemetry.setup("raysimbatch-watchdog", self.telemetry)
      self.metrics = telemetry.exporter(actor_role="watchdog", node=self.node_id[:16])
      self._task = asyncio.get_running_loop().create_task(self._loop())

  async def unregister(self, actor_name: str) -> None:
    self.entries.pop(actor_name, None)

  async def _loop(self) -> None:
    while not self._stopped:
      await self._poll_once()
      if self.entries:
        self._idle_since = time.monotonic()
      elif self.idle_exit_s is not None and time.monotonic() - self._idle_since >= self.idle_exit_s:
        from raysimbatch.control.assets import _exit_self

        _exit_self()  # the next registration re-creates it; lets an idle node scale down
      await asyncio.sleep(self.cfg.heartbeat_s)

  async def _poll_once(self) -> None:
    # Concurrently: a node's polls must not queue behind unresponsive workers' timeouts.
    await asyncio.gather(*(self._poll(e) for e in list(self.entries.values())))
    if self.metrics is not None:
      now = time.monotonic()
      self.metrics.gauge("heartbeat_age_seconds",
                         max((now - e.last_heartbeat_at for e in self.entries.values()), default=0.0))

  async def _poll(self, e: _Entry) -> None:
    try:
      h = await asyncio.wait_for(asyncio.wrap_future(e.handle.health.remote().future()), self.cfg.heartbeat_s)
    except TimeoutError:
      now = time.monotonic()
      e.unresponsive_since = e.unresponsive_since or now
      if now - e.unresponsive_since >= self.cfg.suspect_s and not e.suspect:
        self._fault(e, "HEARTBEAT_TIMEOUT", {"unresponsive_s": now - e.unresponsive_since})
      return
    except Exception as exc:  # noqa: BLE001 - dead actor
      if not e.suspect:
        self._fault(e, "WORKER_DIED", {"error": str(exc)[:300]})
      if not await asyncio.to_thread(self._named_actor_exists, e.name):
        self.entries.pop(e.name, None)  # permanently gone (its owner ended): stop watching
      return
    now = time.monotonic()
    e.unresponsive_since, e.last_heartbeat_at = None, now
    if h["generation"] < e.generation or h["state"] == WorkerState.CREATED:
      # Restarted as a fresh process (constructor only): idle, it would fail only when
      # its next command arrives.
      if not e.suspect:
        self._fault(e, "WORKER_RESTARTED", {"pid": h["pid"], "reported_generation": h["generation"]})
      return
    if h["generation"] > e.generation:  # the owner re-initialized it
      e.generation = h["generation"]
      e.suspect = False
    if h["progress"] != e.last_progress:
      e.last_progress, e.last_progress_at = h["progress"], now
      e.suspect = False
      return
    active = h["active_command"] is not None
    if active and h["active_age_s"] > e.deadline_s and now - e.last_progress_at >= self.cfg.suspect_s and not e.suspect:
      self._fault(e, "NATIVE_PROGRESS_STALLED", {"active_age_s": h["active_age_s"],
                                                 "progress_age_s": now - e.last_progress_at})

  def _named_actor_exists(self, name: str) -> bool:
    try:
      ray.get_actor(name, namespace=self.namespace)
      return True
    except ValueError:
      return False

  def _fault(self, e: _Entry, code: str, detail: dict[str, Any]) -> None:
    e.suspect = True
    self.faults.append(FaultRecord(f"{e.name}:{e.generation}:{time.time_ns()}", "SimulationWorker", code,
                                   time.time_ns(), True, (e.name,), detail={**detail, "generation": e.generation}))

  async def inspect(self, actor_names: list[str] | None = None) -> tuple[dict[str, Any], ...]:
    """Take the pending faults (only those of ``actor_names`` when given: the watchdog is
    shared by every coordinator on the node). A given name that is not registered, e.g.
    after this actor restarted and lost its entries, is reported as ``UNWATCHED`` so its
    owner registers it again."""
    if actor_names is None:
      out = tuple(asdict(f) for f in self.faults)
      self.faults.clear()
      return out
    names = set(actor_names)
    mine = [f for f in self.faults if f.affected[0] in names]
    self.faults = collections.deque((f for f in self.faults if f.affected[0] not in names), maxlen=self.faults.maxlen)
    now = time.time_ns()
    mine += [FaultRecord(f"{n}:unwatched:{now}", "SimulationWorker", "UNWATCHED", now, True, (n,))
             for n in sorted(names - self.entries.keys())]
    return tuple(asdict(f) for f in mine)

  async def poll_now(self) -> tuple[dict[str, Any], ...]:
    await self._poll_once()
    return await self.inspect()

  async def stop(self) -> None:
    self._stopped = True
    if self._task is not None:
      self._task.cancel()


RemoteWatchdog = actor_class(Watchdog, num_cpus=0, max_restarts=-1, max_task_retries=0)


def watchdog_for_node(node_id: str, failure: dict[str, Any], namespace: str | None = None,
                      telemetry: dict[str, Any] | None = None, idle_exit_s: float = 300.0):
  """Get-or-create the named watchdog pinned to ``node_id`` (design §5.1: one per node).
  The first creator's failure settings apply to everyone on the node."""
  from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

  name = f"rsb-watchdog-{node_id[:16]}"
  try:
    return ray.get_actor(name, namespace=namespace)
  except ValueError:
    opts: dict[str, Any] = {"name": name, "get_if_exists": True, "lifetime": "detached",
                            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id, soft=False)}
    if namespace:
      opts["namespace"] = namespace
    return RemoteWatchdog.options(**opts).remote(node_id, failure, namespace, telemetry, idle_exit_s=idle_exit_s)


@dataclass(frozen=True)
class RecoveryResult:
  new_generation: int
  plan: str
  lost_tail_upper_bound: int
  circuit_state: int


class RestartManager:
  """Fence then kill with restart allowed; bounded by a per-family circuit breaker."""

  def __init__(self, run_id: str, store, config: FailureConfig | None = None, namespace: str | None = None):
    self.run_id, self.store = run_id, store
    self.cfg = config or FailureConfig()
    self.namespace = namespace
    self.circuits: dict[str, Circuit] = {}

  async def fence(self, actor_name: str, generation: int) -> int:
    new = await asyncio.to_thread(self.store.db.bump_generation, f"workergen/{self.run_id}/{actor_name}", "restart")
    await asyncio.to_thread(self.store.db.audit, self.run_id, "worker_fenced",
                            {"worker": actor_name, "generation": generation, "new_generation": new})
    try:
      h = ray.get_actor(actor_name, namespace=self.namespace)
      await asyncio.wait_for(asyncio.wrap_future(h.fence.remote(generation).future()), 1.0)
    except Exception:  # noqa: BLE001 - a hung actor cannot answer; the durable fence stands
      pass
    return new

  async def recover(self, actor_name: str, fault: dict[str, Any], family: str = "default",
                    slots_in_flight: int = 0, local_steps: int = 8) -> RecoveryResult:
    from raysimbatch.telemetry import tracing

    with tracing.span("recovery", force=True, actor=actor_name, code=str(fault.get("code"))):
      return await self._recover(actor_name, fault, family, slots_in_flight, local_steps)

  async def _recover(self, actor_name: str, fault: dict[str, Any], family: str, slots_in_flight: int,
                     local_steps: int) -> RecoveryResult:
    c = self.circuits.setdefault(family, Circuit(self.cfg))
    state = c.record_restart()
    gen = await self.fence(actor_name, int(fault.get("detail", {}).get("generation", 0)))
    plan = "none"
    if c.admits_restart():
      try:
        h = ray.get_actor(actor_name, namespace=self.namespace)
        # ray.cancel(force=True) is rejected for actor tasks; killing with restart
        # allowed is the external path for blocked native work.
        ray.kill(h, no_restart=False)
        plan = "restart_reinitialize"
      except ValueError:
        plan = "recreate"
    else:
      plan = "circuit_open"
    return RecoveryResult(gen, plan, slots_in_flight * local_steps, state)

  async def reconcile(self) -> dict[str, Any]:
    return await janitor_pass(self.store, self.namespace)


async def janitor_pass(store, namespace: str | None = None, cluster: str | None = None) -> dict[str, Any]:
  """Reconcile detached run controllers and placement groups with durable leases."""
  from ray.util import placement_group_table
  from ray.util.placement_group import PlacementGroup, remove_placement_group

  expired = await asyncio.to_thread(store.db.expired_leases, "run/")
  fenced, removed_pgs = [], []
  live_runs = {r["run_id"] for r in await asyncio.to_thread(store.db.list_runs,
                                                            ("VALIDATING", "RESERVING", "RUNNING", "DRAINING",
                                                             "DEGRADED", "CANCELING", "PAUSED"))}
  for row in expired:
    run_id = row["key"].split("/", 1)[1]
    try:
      ctl = ray.get_actor(f"rsb-run-{run_id}", namespace=namespace)
    except ValueError:
      ctl = None
    if ctl is not None:
      ray.kill(ctl, no_restart=True)  # its lease expired: a zombie owner
      fenced.append(run_id)
      await asyncio.to_thread(store.db.audit, run_id, "janitor_fenced_controller", {"owner": row["owner"]})
  for pg_id, info in placement_group_table().items():
    name = info.get("name") or ""
    if not name.startswith("rsb-") or info.get("state") == "REMOVED":
      continue
    prefix = name.split("-")[1] if "-" in name else ""
    if not any(r.startswith(prefix) for r in live_runs):
      remove_placement_group(PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id))))
      removed_pgs.append(name)
  # Asset-cache holds of runs that are no longer live (a controller that died without
  # cleanup) would pin their bytes, and the node, forever.
  # Caches have deterministic per-node names; no State API (dashboard) dependency.
  released = 0
  for node in ray.nodes():
    if not node.get("Alive"):
      continue
    try:
      cache = ray.get_actor(f"rsb-asset-cache-{node['NodeID'][:16]}", namespace=namespace)
      released += get_one(cache.retain_only.remote(sorted(live_runs)), timeout=10)
    except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
      pass  # no cache on that node, or it is exiting (it then holds nothing)
  resumed, failed = await _resume_orphans(store, namespace, cluster)
  return {"fenced_controllers": fenced, "removed_placement_groups": removed_pgs, "released_asset_holds": released,
          "resumed_runs": resumed, "failed_orphans": failed}


LIVE_RUN_STATES = ("VALIDATING", "RESERVING", "RUNNING", "DRAINING", "DEGRADED", "CANCELING", "PAUSED")


async def _resume_orphans(store, namespace: str | None, cluster: str | None = None) -> tuple[list[str], list[str]]:
  """Design §8.1 "RunController loss": a live run of *this* cluster whose run lease has
  expired and whose controller is gone gets a new controller (resume from the durable
  store and original seeds). Bounded by the run's failure.max_restarts, then FAILED with
  the cause (found in kind: an OOM-killed controller left its run orphaned)."""
  import json
  import os

  from raysimbatch.config.models import RunConfig
  from raysimbatch.control.driver import ClusterDriver

  cluster = cluster or os.environ.get("RSB_CLUSTER_NAME")
  resumed, failed = [], []
  for row in await asyncio.to_thread(store.db.list_runs, LIVE_RUN_STATES):
    rid = row["run_id"]
    if row["payload"].get("cluster") != cluster:
      continue  # another cluster's run: its own janitor (or an operator's resume) owns it
    lease = await asyncio.to_thread(store.db.lease_state, f"run/{rid}")
    if lease is not None and not lease["expired"]:
      continue
    try:
      ray.get_actor(f"rsb-run-{rid}", namespace=namespace)
      continue  # a controller exists (starting, or a zombie the fence pass above handles)
    except ValueError:
      pass
    cfg = RunConfig.model_validate_json(json.dumps(row["payload"]["resolved"]))
    prior = sum(1 for e in await asyncio.to_thread(store.db.audit_events, rid) if e["event"] == "janitor_resumed_run")
    if prior >= cfg.failure.max_restarts:
      await asyncio.to_thread(store.db.set_run_state, rid, "FAILED", None,
                              {"error": f"run controller lost {prior + 1} times; not resumed again"})
      await asyncio.to_thread(store.db.audit, rid, "janitor_failed_orphan", {"resumes": prior})
      failed.append(rid)
      continue
    await ClusterDriver(cfg, namespace=namespace or "raysimbatch", store=store).resume(rid, cfg, {"cluster": cluster})
    await asyncio.to_thread(store.db.audit, rid, "janitor_resumed_run", {"attempt": prior + 1})
    resumed.append(rid)
  return resumed, failed
