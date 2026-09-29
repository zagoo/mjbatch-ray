# SPDX-License-Identifier: Apache-2.0
"""ShardCoordinator: owns up to ``coordinator_fanout`` workers and runs the B.1 loop.

* At most one mutating command in flight per actor; ready envelopes are collected with
  ``ray.wait(pending, num_returns=1, timeout=0.01, fetch_local=False)`` and a zero-time
  drain; frame/fragment refs are forwarded without materializing their arrays.
* Every command carries a deadline max(native_min, multiplier x p99) <= native_max.
  An overdue command is cancelled cooperatively; after ``cancel_grace_s`` the worker's
  generation is fenced durably and the actor is killed with restart allowed. Restarts
  create a new generation and re-initialize from the node asset cache; a circuit
  breaker bounds reconstruction.
* Every worker is registered with its node's Watchdog (design §8.4). A watchdog fault
  starts that sequence at once for a busy worker, and recovers an idle worker that
  died, restarted or stopped answering, which no command deadline would notice.
* Programs: ``EvalProgram`` (Mode B/C units) and ``RolloutProgram`` (Mode A), each with
  local (worker-side) or central (PolicyInferenceService) inference.
"""

from __future__ import annotations

import collections
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import ray
import ray.exceptions  # explicit: pyright does not see the submodule through `import ray`

from raysimbatch import protocol as P
from raysimbatch import telemetry
from raysimbatch.compat.ray_typing import actor_class, get_one
from raysimbatch.config.models import RunConfig
from raysimbatch.evaluation.schema import EPISODE_SCHEMA_VERSION, EpisodeResult
from raysimbatch.protocol import FaultCode, SlotStatus
from raysimbatch.recovery.circuit import CLOSED, UNAVAILABLE, Circuit
from raysimbatch.recovery.watchdog import watchdog_for_node
from raysimbatch.worker.actor import RemoteSimulationWorker, worker_options

if TYPE_CHECKING:
  from raysimbatch.storage.store import DurableStore

FREE, RESETTING, ACTIVE, WAIT_INFER, READY_STEP, ENDED, DEAD = range(7)

# Coordinator events that are also catalog counters (design §11.1).
_EVENT_COUNTERS = {"worker_restarted": "worker_restarts_total", "worker_quarantined": "worker_restart_exhausted_total",
                   "coordinator_lease_lost": "leases_expired_total"}
_EVENT_LEVELS = {"worker_lost": 30, "worker_suspect": 30, "worker_quarantined": 40, "coordinator_failed": 40,
                 "coordinator_lease_lost": 40, "capacity_degraded": 40, "assembler_error": 40, "assembler_retry": 30,
                 "unit_commit_rejected": 30, "command_rejected": 30, "watchdog_fault": 30,
                 "watchdog_unavailable": 40}


_ACTOR_GONE = (ray.exceptions.RayActorError, ray.exceptions.ActorDiedError, ray.exceptions.ActorUnavailableError)


def _worker_class(path: str | None):
  """The simulation actor class. Overrides exist only for fault-injection qualification
  and are restricted to ``raysimbatch.testing`` (never arbitrary code from run options)."""
  if not path:
    return RemoteSimulationWorker
  if not path.startswith("raysimbatch.testing."):
    raise ValueError(f"worker_class {path!r}: overrides are limited to raysimbatch.testing fault injection")
  import importlib

  module, _, name = path.partition(":")
  return getattr(importlib.import_module(module), name)


def _ack(rec, credits: list[int]) -> None:
  """Release frame/fragment credits on a worker. A worker that died or restarted
  meanwhile took its credits with it; its loss is handled when its next command fails,
  never here in the middle of processing an envelope (which would crash the coordinator)."""
  try:
    ray.get(rec.actor.ack.remote(credits))
  except _ACTOR_GONE:
    pass


@dataclass
class Slice:
  slots: np.ndarray
  state: int = FREE
  frame_ref: Any = None
  status: dict | None = None
  infer_ref: Any = None
  infer_started: float = 0.0
  action_ref: Any = None


@dataclass
class WorkerRec:
  name: str
  group: str
  n: int
  actor: Any
  generation: int = 0
  state: str = "new"
  restarts: int = 0
  inflight: Any = None
  inflight_kind: str = ""
  inflight_cmd: dict | None = None
  inflight_deadline: float = 0.0
  inflight_started: float = 0.0
  inflight_slice: int = -1
  cancel_at: float = 0.0
  suspect_reason: str = ""  # set when a watchdog fault, not the deadline, made the command suspect
  node_id: str = ""
  durations: collections.deque = field(default_factory=lambda: collections.deque(maxlen=256))
  slot_state: np.ndarray = None  # type: ignore[assignment]
  env_id: np.ndarray = None  # type: ignore[assignment]
  episode_id: np.ndarray = None  # type: ignore[assignment]
  seq: np.ndarray = None  # type: ignore[assignment]
  assign: dict[int, Any] = field(default_factory=dict)  # slot -> program-specific episode assignment
  slices: list[Slice] = field(default_factory=list)
  ready: dict | None = None
  ready_slots: int = 0
  infer: dict = field(default_factory=dict)  # inference ref -> request metadata
  ready_steps: collections.deque = field(default_factory=collections.deque)  # (command, action_ref)


class ShardCoordinator:
  def __init__(self, run_id: str, coordinator_id: int, config_json: str, groups: dict[str, Any],
               plan: dict[str, Any], program: dict[str, Any], namespace: str | None = None):
    self.run_id = run_id
    self.cid = coordinator_id
    self.cfg = RunConfig.model_validate_json(config_json)
    self.groups = groups
    self.plan = plan
    self.program_spec = program
    self.namespace = namespace
    self.name = f"coord-{run_id[:8]}-{coordinator_id}"
    self.workers: dict[str, WorkerRec] = {}
    self.circuits: dict[str, Circuit] = collections.defaultdict(lambda: Circuit(self.cfg.failure))
    self.stop_requested = False
    self.stop_drain = True
    self.events: collections.deque = collections.deque(maxlen=2000)
    self.counters: collections.Counter = collections.Counter()
    self.lock = threading.Lock()
    # Late initialization: install() sets these before the loop or any command uses them
    # (control methods that may run earlier check for None).
    self.store: DurableStore = None  # type: ignore[assignment]
    self.lease: P.Lease = None  # type: ignore[assignment]
    self.program: EvalProgram | RolloutProgram = None  # type: ignore[assignment]
    self.inference = program.get("inference_actor")
    self.error: str | None = None
    self.finished = False
    self.metrics = None
    self.last_worker_fault = ""
    self.watchdogs: dict[str, Any] = {}  # node id -> that node's Watchdog
    self._wd_pending: dict[str, tuple[Any, float]] = {}  # node id -> (inspect ref, sent at)
    self._wd_at = 0.0

  # ------------------------------------------------------------------ install

  @ray.method(concurrency_group="loop")
  def install(self) -> dict[str, Any]:
    from raysimbatch.storage.store import DurableStore
    telemetry.setup("raysimbatch-coordinator", self.cfg.telemetry)
    self.metrics = telemetry.exporter(mode=self.cfg.mode, actor_role="coordinator")
    self.store = DurableStore.from_config(self.cfg, migrate=False)
    self.lease = self.store.db.acquire_lease(f"coord/{self.run_id}/{self.cid}", self.name, self.cfg.failure.lease_s)
    self.lease_renewed = time.monotonic()
    pg = self.plan.get("placement_group")
    for i, w in enumerate(self.plan["workers"]):
      g = self.groups[w["group"]]
      spec = dict(g["spec"])
      spec.update(actor_name=w["actor_name"], num_slots=int(w["slots"]), cpu_set=tuple(w.get("cpu_set", ())),
                  coordinator_id=self.cid, shard_id=i)
      from raysimbatch.config.models import FailureConfig, WorkerConfig

      wc = WorkerConfig.from_obj(spec["worker"])
      strat = None
      if pg is not None:
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        strat = PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=int(w["bundle_index"]),
                                                 placement_group_capture_child_tasks=False)
      opts = worker_options(wc, FailureConfig.from_obj(spec["failure"]), name=w["actor_name"],
                            namespace=self.namespace, scheduling_strategy=strat,
                            memory_bytes=int(w.get("memory_bytes", wc.memory_gib * 2**30)),
                            resources=w.get("resources"))
      actor = _worker_class(self.program_spec.get("worker_class")).options(**opts).remote(
        spec, f"{self.cfg.artifact_root}/runs/{self.run_id}")
      n = int(w["slots"])
      rec = WorkerRec(w["actor_name"], w["group"], n, actor)
      rec.slot_state = np.full(n, FREE, np.int8)
      rec.env_id = np.zeros(n, np.uint64)
      rec.episode_id = np.zeros(n, np.uint64)
      rec.seq = np.zeros(n, np.uint64)
      ss = self.cfg.worker.slice_size
      rec.slices = [Slice(np.arange(a, min(n, a + ss), dtype=np.int32)) for a in range(0, n, ss)]
      self.workers[rec.name] = rec
    readies = {}
    for rec in self.workers.values():
      readies[rec.name] = self._initialize(rec)
    kind = self.program_spec["kind"]
    self.program = (EvalProgram if kind == "eval" else RolloutProgram)(self, self.program_spec)
    self._event("installed", workers=len(self.workers))
    self._check_capacity()
    return {"coordinator": self.name, "lease_epoch": self.lease.epoch, "workers": readies}

  def _initialize(self, rec: WorkerRec) -> dict[str, Any]:
    g = self.groups[rec.group]
    # Every (re)initialization is a new generation allocated durably.
    gen = self.store.db.bump_generation(f"workergen/{self.run_id}/{rec.name}", self.name)
    lease = {"owner": self.name, "epoch": gen, "expires_unix_ns": self.lease.expires_unix_ns, "routing_epoch": 0}
    pol = g.get("policy") if self.program_spec.get("placement", "local") == "local" else None
    args = [g["mjb_ref"], pol["exe_ref"] if pol else None, lease, pol["manifest"] if pol else None]
    ready = get_one(rec.actor.initialize.remote(*args), timeout=max(120.0, self.cfg.failure.native_max_deadline_s))
    rec.generation = gen
    rec.state = "ready"
    rec.ready = ready
    rec.slot_state[:] = FREE
    rec.assign.clear()
    for s in rec.slices:
      s.state, s.frame_ref, s.infer_ref, s.action_ref = FREE, None, None, None
    rec.infer.clear()
    rec.ready_steps.clear()
    rec.inflight = None
    self._watch(rec)
    return ready

  # --------------------------------------------------------------------- loop

  @ray.method(concurrency_group="loop")
  def run(self, stop_epoch: int = 1 << 62) -> dict[str, Any]:
    try:
      while not self.program.done() and not (self.stop_requested and self._quiescent()):
        self._renew_lease()
        if not self.stop_requested:
          self.program.dispatch()
        self._collect()
        self._check_deadlines()
        self._check_watchdogs()
      self.program.close()
      for rec in self.workers.values():
        self._unwatch(rec)
      self.finished = True
      return self._status()
    except Exception as exc:  # noqa: BLE001 - reported through status and the controller
      self.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
      self._event("coordinator_failed", error=str(exc))
      raise

  def _quiescent(self) -> bool:
    return not any(r.inflight is not None or r.infer or r.ready_steps for r in self.workers.values())

  def _renew_lease(self) -> None:
    if time.monotonic() - self.lease_renewed < self.cfg.failure.lease_renew_s:
      return
    lease = self.store.db.renew_lease(f"coord/{self.run_id}/{self.cid}", self.name, self.lease.epoch,
                                      self.cfg.failure.lease_s)
    if lease is None:
      self._event("coordinator_lease_lost")
      raise RuntimeError("coordinator lease lost: fenced by a newer owner")
    self.lease = lease
    self.lease_renewed = time.monotonic()

  def deadline_for(self, rec: WorkerRec, steps: int = 1) -> float:
    f = self.cfg.failure
    if len(rec.durations) >= 16:
      p99 = float(np.quantile(np.asarray(rec.durations), 0.99))
      d = max(f.native_min_deadline_s, f.native_p99_multiplier * p99)
    else:
      d = f.native_max_deadline_s  # calibration phase
    return time.monotonic() + min(f.native_max_deadline_s, d)

  def submit(self, rec: WorkerRec, kind: str, ref: Any, cmd: dict, slice_index: int, steps: int = 1) -> None:
    rec.inflight, rec.inflight_kind, rec.inflight_cmd = ref, kind, cmd
    rec.inflight_started = time.monotonic()
    rec.inflight_deadline = self.deadline_for(rec, steps)
    rec.inflight_slice = slice_index
    rec.cancel_at = 0.0
    rec.suspect_reason = ""
    self.counters[f"submitted_{kind}"] += 1

  def _collect(self) -> None:
    pending = []
    index = {}
    for rec in self.workers.values():
      if rec.inflight is not None:
        pending.append(rec.inflight)
        index[rec.inflight] = ("mut", rec, -1)
      for ref in rec.infer:
        pending.append(ref)
        index[ref] = ("inf", rec, -1)
    if not pending:
      time.sleep(0.001)
      return
    ready, _ = ray.wait(pending, num_returns=1, timeout=0.01, fetch_local=False)
    if ready:
      more, _ = ray.wait([p for p in pending if p not in ready], num_returns=len(pending) - len(ready), timeout=0,
                         fetch_local=False)
      ready += more
    for ref in ready:
      kind, rec, i = index[ref]
      if kind == "mut":
        self._on_mutation(rec, ref)
      else:
        self.program.on_inference(rec, ref)

  def _on_mutation(self, rec: WorkerRec, ref: Any) -> None:
    kind, cmd, sl = rec.inflight_kind, rec.inflight_cmd, rec.inflight_slice
    assert cmd is not None  # submitted together with the in-flight ref
    rec.inflight = None
    if self.metrics is not None:
      self.metrics.observe("control_interval_seconds", time.monotonic() - rec.inflight_started, stage=kind)
    try:
      env = get_one(ref)
    except ray.exceptions.TaskCancelledError:
      # The method may have completed: recover its cached output by command id.
      self.counters["cancelled_commands"] += 1
      self._event("command_cancelled", worker=rec.name, kind=kind)
      try:
        method = getattr(rec.actor, kind)
        env = get_one(method.remote(*_resend_args(kind, cmd)), timeout=self.cfg.failure.native_max_deadline_s)
      except Exception as exc:  # noqa: BLE001
        self._worker_lost(rec, f"cancel recovery failed: {exc}")
        return
    except (ray.exceptions.RayActorError, ray.exceptions.ActorDiedError, ray.exceptions.ActorUnavailableError) as exc:
      self._worker_lost(rec, f"{type(exc).__name__}: {exc}")
      return
    except ray.exceptions.RayTaskError as exc:
      # An application error inside a worker command is a worker fault (restart, then
      # quarantine), never a coordinator failure (found: it crash-looped coordinators).
      self._worker_lost(rec, f"worker command {kind} failed: {str(exc)[-500:]}")
      return
    rec.durations.append(time.monotonic() - rec.inflight_started)
    if env.get("rejected"):
      self.counters[f"rejected_{FaultCode(env['rejected']).name}"] += 1
      self._event("command_rejected", worker=rec.name, code=FaultCode(env["rejected"]).name, message=env.get("message"))
      if env["rejected"] in (FaultCode.STALE_GENERATION, FaultCode.REPLAY_REQUIRED):
        self._worker_lost(rec, "stale generation / replay required: reconstructing")
        return
      self.program.on_rejected(rec, kind, cmd, sl, env)
      return
    st = env["status"]
    rej = st["status"] == SlotStatus.REJECTED
    if rej.any():
      self.counters["rejected_rows"] += int(rej.sum())
    self.program.on_envelope(rec, kind, cmd, sl, env)

  def _check_deadlines(self) -> None:
    now = time.monotonic()
    if self.metrics is not None and now - getattr(self, "_age_at", 0.0) >= 1.0:
      self._age_at = now  # active native-call age; alerts compare it with the call deadline
      ages = [now - r.inflight_started for r in self.workers.values() if r.inflight is not None]
      self.metrics.gauge("worker_progress_age_seconds", max(ages, default=0.0))
    for rec in list(self.workers.values()):
      if rec.inflight is None or now <= rec.inflight_deadline:
        continue
      if rec.cancel_at == 0.0:
        rec.cancel_at = now
        self.counters["suspect_commands"] += 1
        self._event("worker_suspect", worker=rec.name, kind=rec.inflight_kind, age=now - rec.inflight_started,
                    reason=rec.suspect_reason or "deadline")
        try:
          ray.cancel(rec.inflight, force=False, recursive=False)
        except Exception:  # noqa: BLE001
          pass
      elif now - rec.cancel_at > self.cfg.failure.cancel_grace_s:
        # Fence the generation durably, then kill with restart allowed.
        self._fence(rec)
        ray.kill(rec.actor, no_restart=False)
        rec.inflight = None
        self._worker_lost(rec, f"{rec.suspect_reason or 'native deadline exceeded'}: fenced and killed")

  def _fence(self, rec: WorkerRec) -> None:
    self.store.db.bump_generation(f"workergen/{self.run_id}/{rec.name}", self.name)
    self.store.db.audit(self.run_id, "worker_fenced", {"worker": rec.name, "generation": rec.generation})
    try:
      ray.get(rec.actor.fence.remote(rec.generation), timeout=1.0)
    except Exception:  # noqa: BLE001 - a hung or dead actor cannot answer; the durable fence stands
      pass

  def _worker_lost(self, rec: WorkerRec, reason: str) -> None:
    self.counters["worker_failures"] += 1
    self.last_worker_fault = f"{rec.name}: {reason}"
    self._event("worker_lost", worker=rec.name, reason=reason, generation=rec.generation)
    self.store.db.audit(self.run_id, "worker_lost", {"worker": rec.name, "reason": reason, "generation": rec.generation})
    lost_slots = np.flatnonzero(rec.slot_state != FREE)
    self.program.on_worker_lost(rec, lost_slots, reason)
    circuit = self.circuits[rec.group]
    state = circuit.record_restart()
    if self.metrics is not None:
      self.metrics.gauge("circuit_state", float(state), model_family=rec.group)
    rec.restarts += 1
    for s in rec.slices:
      s.state, s.frame_ref, s.infer_ref, s.action_ref = FREE, None, None, None
    rec.infer.clear()
    rec.ready_steps.clear()
    rec.slot_state[:] = FREE
    rec.assign.clear()
    rec.inflight = None
    if rec.restarts > self.cfg.failure.max_restarts or circuit.state == UNAVAILABLE:
      rec.state = "quarantined"
      self._unwatch(rec)
      self.counters["worker_restart_exhausted"] += 1
      self._event("worker_quarantined", worker=rec.name)
      self._check_capacity()
      return
    deadline = time.monotonic() + 60.0
    while circuit.state not in (CLOSED,) and not circuit.admits_restart():
      if time.monotonic() > deadline:
        break
      self._renew_lease()  # waiting for the circuit must not cost this coordinator its lease
      time.sleep(0.5)
    for attempt in range(60):
      self._renew_lease()
      try:
        self._initialize(rec)
        self.counters["worker_restarts"] += 1
        self._event("worker_restarted", worker=rec.name, generation=rec.generation)
        self._check_capacity()
        return
      except (ray.exceptions.RayActorError, ray.exceptions.ActorDiedError, ray.exceptions.GetTimeoutError,
              ray.exceptions.ActorUnavailableError) as exc:
        if "restart" in str(exc).lower() and "exhaust" in str(exc).lower():
          break
        time.sleep(min(2.0, 0.2 * (attempt + 1)))
    rec.state = "quarantined"
    self._unwatch(rec)
    self.counters["worker_restart_exhausted"] += 1
    self._event("worker_quarantined", worker=rec.name)
    self._check_capacity()

  # ----------------------------------------------------------------- watchdog

  def _watch(self, rec: WorkerRec) -> None:
    """Register a (re)initialized worker at its generation with its node's watchdog."""
    node = rec.ready.get("node_id") if rec.ready else None
    if not node:
      return
    rec.node_id = node
    error: Exception | None = None
    for _ in range(3):  # a per-node watchdog exits when idle; the next lookup re-creates it
      self._renew_lease()  # registration must not cost this coordinator its lease
      try:
        wd = self.watchdogs.get(node) or watchdog_for_node(
          node, self.cfg.failure.model_dump(mode="json"), self.namespace, self.cfg.telemetry.model_dump(mode="json"))
        ray.get(wd.register.remote(rec.name, rec.generation, self.cfg.failure.native_max_deadline_s), timeout=10.0)
        self.watchdogs[node] = wd
        return
      except Exception as exc:  # noqa: BLE001
        self.watchdogs.pop(node, None)
        error = exc
        time.sleep(0.2)
    # Command deadlines still apply; only the watchdog's detection is missing.
    self._event("watchdog_unavailable", worker=rec.name, node=node[:16], error=str(error)[:300])

  def _unwatch(self, rec: WorkerRec) -> None:
    wd = self.watchdogs.get(rec.node_id)
    if wd is not None:
      try:
        wd.unregister.remote(rec.name)
      except Exception:  # noqa: BLE001 - a watchdog that is gone watches nothing
        pass

  def _check_watchdogs(self) -> None:
    """Collect the watchdogs' faults for this coordinator's workers without blocking the
    loop: at most one ``inspect`` per node in flight, given up after ``suspect_s``."""
    f = self.cfg.failure
    now = time.monotonic()
    for node, (ref, sent) in list(self._wd_pending.items()):
      done, _ = ray.wait([ref], timeout=0)
      if done:
        del self._wd_pending[node]
        try:
          faults = ray.get(ref)
        except Exception as exc:  # noqa: BLE001 - the watchdog died or exited
          self._watchdog_lost(node, exc)
          continue
        for fault in faults:
          self._on_watchdog_fault(fault)
      elif now - sent > f.suspect_s:
        del self._wd_pending[node]
        self._watchdog_lost(node, "inspect unanswered")
    if now - self._wd_at < f.heartbeat_s:
      return
    self._wd_at = now
    names: dict[str, list[str]] = collections.defaultdict(list)
    for rec in self.workers.values():
      if rec.state == "ready" and rec.node_id in self.watchdogs:
        names[rec.node_id].append(rec.name)
    for node, ns in names.items():
      if node not in self._wd_pending:
        self._wd_pending[node] = (self.watchdogs[node].inspect.remote(ns), now)

  def _watchdog_lost(self, node: str, reason: Any) -> None:
    self.watchdogs.pop(node, None)
    self._event("watchdog_unavailable", node=node[:16], error=str(reason)[:300])
    alive = any(n["NodeID"] == node and n["Alive"] for n in ray.nodes())
    for rec in list(self.workers.values()):
      if rec.state != "ready" or rec.node_id != node:
        continue
      if alive:
        self._watch(rec)  # the watchdog restarted or exited: a new one watches the node
      elif rec.inflight is None:
        # The node is gone with its watchdog: nothing else would report its idle workers
        # (busy ones fail through their commands). Never re-register on a dead node.
        self._worker_lost(rec, "node lost (its watchdog is gone)")

  def _on_watchdog_fault(self, fault: dict[str, Any]) -> None:
    rec = self.workers.get(fault["affected"][0])
    if rec is None or rec.state != "ready":
      return
    code = fault["code"]
    if code == "UNWATCHED":  # the watchdog restarted and lost its registrations
      self._watch(rec)
      return
    if fault["detail"].get("generation") != rec.generation:
      return  # about a generation this coordinator has already replaced
    self.counters[f"watchdog_{code.lower()}"] += 1
    self._event("watchdog_fault", worker=rec.name, code=code, generation=rec.generation)
    if rec.inflight is not None:
      # Busy: start the cancel -> fence -> kill sequence now, not at the command deadline.
      rec.inflight_deadline = min(rec.inflight_deadline, time.monotonic())
      rec.suspect_reason = f"watchdog {code}"
      return
    # Idle: nothing would fail before its next command.
    if code in ("HEARTBEAT_TIMEOUT", "NATIVE_PROGRESS_STALLED"):
      self._fence(rec)
      ray.kill(rec.actor, no_restart=False)
    self._worker_lost(rec, f"watchdog: {code}")

  def _check_capacity(self) -> None:
    total = sum(r.n for r in self.workers.values())
    healthy = sum(r.n for r in self.workers.values() if r.state == "ready")
    frac = healthy / total if total else 0.0
    if self.metrics is not None:
      self.metrics.gauge("healthy_capacity_ratio", frac)
      self.metrics.gauge("slots_quarantined", float(sum(r.n for r in self.workers.values() if r.state == "quarantined")))
    if frac < self.cfg.failure.minimum_healthy_fraction:
      self._event("capacity_degraded", healthy_fraction=frac)
      if healthy == 0:
        raise RuntimeError(f"no healthy simulation capacity left in this coordinator; last worker fault: "
                           f"{self.last_worker_fault}")

  def _event(self, event: str, **kw: Any) -> None:
    rec = {"t": time.time(), "event": event, **{k: (v if isinstance(v, int | float | str | None) else str(v))
                                                for k, v in kw.items()}}
    with self.lock:
      self.events.append(rec)
    from raysimbatch.telemetry import logs

    fields = {k: v for k, v in rec.items() if k not in ("t", "event", "worker", "generation")}
    logs.log(event, _EVENT_LEVELS.get(event, 20), run_id=self.run_id, coordinator_id=self.cid,
             actor_name=kw.get("worker"), generation=kw.get("generation"),
             job_epoch=self.lease.epoch if self.lease else None,
             message=f"{event} " + " ".join(f"{k}={v}" for k, v in fields.items()))
    if self.metrics is not None:
      if event in _EVENT_COUNTERS:
        self.metrics.count(_EVENT_COUNTERS[event])
      elif event == "command_rejected" and kw.get("code") in ("STALE_GENERATION", "REPLAY_REQUIRED"):
        self.metrics.count("stale_commands_total")

  # ------------------------------------------------------------------ control

  @ray.method(concurrency_group="control")
  def status(self) -> dict[str, Any]:
    return self._status()

  def _status(self) -> dict[str, Any]:  # also called locally, from run()
    with self.lock:
      ev = list(self.events)[-50:]
    return {
      "coordinator": self.name,
      "finished": self.finished,
      "error": self.error,
      "counters": dict(self.counters),
      "workers": {n: {"state": r.state, "generation": r.generation, "restarts": r.restarts,
                      "active": int((r.slot_state != FREE).sum()) if r.slot_state is not None else 0}
                  for n, r in self.workers.items()},
      "program": self.program.status() if self.program else {},
      "events": ev,
    }

  @ray.method(concurrency_group="control")
  def request_stop(self, drain: bool = True) -> None:
    self.stop_requested, self.stop_drain = True, drain

  @ray.method(concurrency_group="control")
  def update_policy(self, policy: dict[str, Any]) -> None:
    if self.program is not None:
      with self.lock:  # taken by the loop thread in RolloutProgram._apply_policy
        self.program.pending_policy = policy

  @ray.method(concurrency_group="control")
  def update_curriculum(self, version: int, weights: dict[str, float]) -> None:
    if isinstance(self.program, RolloutProgram):  # evaluations have no curriculum
      self.program.set_curriculum(version, weights)

  @ray.method(concurrency_group="control")
  def kill_worker(self, name: str) -> int:
    """Chaos hook used by qualification: kill one worker process (restart allowed)."""
    rec = self.workers[name]
    ray.kill(rec.actor, no_restart=False)
    return rec.generation

  @ray.method(concurrency_group="control")
  def worker_names(self) -> list[str]:
    return list(self.workers)

  @ray.method(concurrency_group="control")
  def shutdown_workers(self) -> None:
    for r in self.workers.values():
      self._unwatch(r)
      try:
        ray.kill(r.actor, no_restart=True)
      except Exception:  # noqa: BLE001
        pass
    if self.store is not None and self.lease is not None:
      self.store.db.release_lease(f"coord/{self.run_id}/{self.cid}", self.name, self.lease.epoch)


def _resend_args(kind: str, cmd: dict) -> tuple:
  if kind == "advance":
    return (cmd["command"], cmd["actions"])
  return (cmd,)


RemoteShardCoordinator = actor_class(ShardCoordinator, concurrency_groups={"loop": 1, "control": 2}, max_restarts=0,
                                     max_task_retries=0)


# ============================================================ Mode B / C programs


@dataclass
class UnitRun:
  unit: Any
  pending: collections.deque
  running: set
  results: list
  infra_retries: collections.Counter
  failed: bool = False


class EvalProgram:
  """Evaluation units -> episodes on free slots; durable per-unit commits."""

  def __init__(self, co: ShardCoordinator, spec: dict[str, Any]):
    from raysimbatch.evaluation.aggregator import ResultAggregator
    from raysimbatch.evaluation.scheduler import EvalScheduler

    self.co = co
    self.spec = spec
    self.cfg = co.cfg
    self.placement = spec.get("placement", "local")
    self.aggregator = ResultAggregator(co.run_id, co.cid, co.store, replicates=self.cfg.evaluation.bootstrap_replicates)
    self.scheduler = EvalScheduler(co.run_id, self.cfg.evaluation, co.store, self.aggregator)
    self.units: dict[str, UnitRun] = {}
    self.exhausted_scheduler = False
    self.last_renew = time.monotonic()
    self.committed = 0
    self.health_failures = 0
    self.pending_policy: dict[str, Any] | None = None
    self.policy_hash = {g: (bytes.fromhex(v["policy"]["hash"]) if v.get("policy") else b"\0" * 32)
                        for g, v in co.groups.items()}
    self.hardware = spec.get("hardware_class", "")
    self.max_steps = self.cfg.worker.local_control_steps
    self._next_env = 0
    self.next_poll = 0.0

  # -- scheduling
  def _refill(self) -> None:
    free = sum(int((r.slot_state == FREE).sum()) for r in self.co.workers.values() if r.state == "ready")
    queued = sum(len(u.pending) for u in self.units.values())
    if self.exhausted_scheduler or queued >= free:
      return
    if time.monotonic() < self.next_poll:
      return
    units = _run_async(self.scheduler.lease(self.co.name, self.cfg.evaluation.unit_block))
    if not units:
      # Nothing leasable *now*; units leased by a lost peer become leasable when their
      # leases expire. Only when every unit is durably terminal is the matrix done.
      if not self.units and _run_async(self.scheduler.done()):
        self.exhausted_scheduler = True
      else:
        self.next_poll = time.monotonic() + min(1.0, self.cfg.evaluation.lease_renew_s)
      return
    for u in units:
      done = _run_async(self.scheduler.committed_ordinals(u.unit_id))
      todo = collections.deque(k for k in range(u.episode_count) if k not in done)
      self.units[u.unit_id] = UnitRun(u, todo, set(), [], collections.Counter())
      _run_async(_to_thread(self.co.store.db.mark_running, self.co.run_id, u.unit_id, u.attempt, u.lease_epoch))
      if not todo:
        self._maybe_complete(self.units[u.unit_id])

  def dispatch(self) -> None:
    if time.monotonic() - self.last_renew > self.cfg.evaluation.lease_renew_s:
      self._renew()
    self._refill()
    for rec in self.co.workers.values():
      if rec.state != "ready" or rec.inflight is not None:
        continue
      free = np.flatnonzero(rec.slot_state == FREE)
      if free.size:
        take = self._assign(rec, free)
        if take.size:
          self._reset(rec, take)
          continue
      if self.placement == "local":
        active = np.flatnonzero(rec.slot_state == ACTIVE).astype(np.int32)
        if active.size:
          cmd = {"command_id": P.new_id(), "expected_generation": rec.generation, "slots": active,
                 "policy_hash": self.policy_hash[rec.group], "deadline_unix_ns": time.time_ns() + int(30e9),
                 "max_control_steps": self.max_steps, "autoreset": False}
          self.co.submit(rec, "run_local", rec.actor.run_local.remote(cmd), cmd, -1, self.max_steps)
      else:
        self._central_dispatch(rec)

  def _assign(self, rec: WorkerRec, free: np.ndarray) -> np.ndarray:
    take = []
    base = rec.group.split("@")[0]
    ph = self.co.groups[rec.group]["policy"]["hash"]
    for uid, ur in self.units.items():
      if ur.failed or ur.unit.payload["group"] != base or ur.unit.policy_hash != ph:
        continue
      while ur.pending and len(take) < free.size:
        k = ur.pending.popleft()
        slot = int(free[len(take)])
        take.append(slot)
        rec.assign[slot] = (uid, k)
        ur.running.add(k)
      if len(take) == free.size:
        break
    return np.asarray(sorted(take), np.int32)

  def _reset(self, rec: WorkerRec, slots: np.ndarray) -> None:
    seeds_ = np.asarray([self.units[rec.assign[int(s)][0]].unit.seed for s in slots], np.uint64)
    ords = np.asarray([rec.assign[int(s)][1] for s in slots], np.uint64)
    start = self.co.store.db.allocate_episode_ids(self.co.run_id, slots.size)
    eps = np.arange(start, start + slots.size, dtype=np.uint64)
    env = np.asarray([self._env_for(rec, int(s)) for s in slots], np.uint64)
    req = {"command_id": P.new_id(), "generation": rec.generation, "slots": slots, "env_id": env,
           "episode_id": eps, "key_a": seeds_, "key_b": ords}
    rec.slot_state[slots] = RESETTING
    self.co.submit(rec, "reset", rec.actor.reset.remote(req), req, -1)

  def _env_for(self, rec: WorkerRec, slot: int) -> int:
    base = sorted(self.co.workers).index(rec.name) * 1_000_000 + self.co.cid * 1_000_000_000
    return base + slot

  def _central_dispatch(self, rec: WorkerRec) -> None:
    _central_step(self, rec)

  def on_inference(self, rec: WorkerRec, ref: Any) -> None:
    _central_on_inference(self, rec, ref)

  # -- results
  def on_envelope(self, rec: WorkerRec, kind: str, cmd: dict, sl: int, env: dict) -> None:
    st = env["status"]
    if self.placement == "local":
      _ack(rec, env["credit_ids"])  # the coordinator is the frame's only consumer
    if kind == "run_local" and "slot_state" in env:
      stuck = st["slot"][(env["slot_state"] != P.SlotState.READY) & (env["slot_state"] != P.SlotState.TERMINAL)]
      for s in stuck.tolist():
        if s in rec.assign and rec.slot_state[s] == ACTIVE and not any(e["slot"] == s for e in env["episodes"]):
          self._requeue(rec, s)
          rec.slot_state[s] = FREE
    if kind == "reset":
      ok = st["status"] == SlotStatus.OK
      for j, s in enumerate(st["slot"].tolist()):
        if ok[j]:
          rec.slot_state[s] = ACTIVE
          rec.seq[s] = 0
          rec.episode_id[s] = st["episode_id"][j]
          rec.env_id[s] = st["env_id"][j]
        else:
          self._episode_failed_to_start(rec, s, int(st["fault_code"][j]))
      if self.placement == "central":
        _central_after_frame(self, rec, {k: v[ok] for k, v in st.items()}, env["frame_ref"],
                             frame_credit=env["frame_credit"])
      return
    for e in env["episodes"]:
      self._episode_done(rec, e)
    if self.placement == "central":
      if kind == "advance":
        _central_row_rejections(self, rec, cmd, st)
      live = st["status"] == SlotStatus.OK
      rec.seq[st["slot"]] = st["control_seq"]
      if live.any():
        _central_after_frame(self, rec, {k: v[live] for k, v in st.items()}, env["frame_ref"],
                             frame_credit=env["frame_credit"])
      else:
        _ack(rec, [env["frame_credit"]])

  def on_rejected(self, rec, kind, cmd, sl, env) -> None:
    _on_rejected(self, rec, kind, cmd, env)

  def _episode_failed_to_start(self, rec: WorkerRec, slot: int, fault: int) -> None:
    uid, k = rec.assign.pop(slot)
    ur = self.units[uid]
    ur.running.discard(k)
    ur.results.append(self._result(ur, k, rec, {
      "outcome": "health_failure", "success": None, "fault_code": fault, "return_sum": None, "control_steps": 0,
      "physics_steps": 0, "simulated_seconds": 0.0, "started_ns": time.time_ns(), "finished_ns": time.time_ns(),
      "env_id": int(rec.env_id[slot]), "episode_id": int(rec.episode_id[slot]), "generation": rec.generation,
      "randomization_hash": "", "tracking_rmse": None, "energy_joules": None, "saturation_fraction": None,
      "reward_components": {}}))
    rec.slot_state[slot] = FREE
    self._maybe_complete(ur)

  def _episode_done(self, rec: WorkerRec, e: dict) -> None:
    slot = int(e["slot"])
    if slot not in rec.assign:
      return
    uid, k = rec.assign.pop(slot)
    rec.slot_state[slot] = FREE
    ur = self.units.get(uid)
    if ur is None or ur.failed:
      return
    ur.running.discard(k)
    ur.results.append(self._result(ur, k, rec, e))
    if e["outcome"] == "health_failure":
      self.health_failures += 1
      q = self.co.store.db.quarantine_event(
        self.co.run_id, f"{ur.unit.model_hash[:16]}:{ur.unit.task_hash[:16]}:{e['randomization_hash'][:8]}:{e['fault_code']}",
        int(e["fault_code"]), native=int(e["fault_code"]) in (10, 11, 12, 13), health=True,
        native_limit=self.cfg.failure.native_quarantine_count, health_limit=self.cfg.failure.health_quarantine_count,
        window=self.cfg.failure.health_quarantine_window)
      if q:
        self.co._event("quarantine_bin", unit=uid, fault=e["fault_code"])
    self._maybe_complete(ur)

  def _result(self, ur: UnitRun, k: int, rec: WorkerRec, e: dict) -> EpisodeResult:
    u = ur.unit
    rc = {str(a): float(b) for a, b in (e.get("reward_components") or {}).items()}
    fc = int(e.get("fault_code", 0))
    return EpisodeResult(
      schema_version=EPISODE_SCHEMA_VERSION, episode_ordinal=k, attempt=u.attempt, run_id=self.co.run_id,
      unit_id=u.unit_id, policy_hash=u.policy_hash, model_hash=u.model_hash, task_hash=u.task_hash,
      adapter_hash=u.adapter_hash, config_hash=self.spec.get("config_hash", ""), seed=u.seed,
      env_id=int(e["env_id"]), episode_id=int(e["episode_id"]), generation=int(e["generation"]),
      started_at=int(e["started_ns"]), finished_at=int(e["finished_ns"]), physics_steps=int(e["physics_steps"]),
      control_steps=int(e["control_steps"]), simulated_seconds=float(e["simulated_seconds"]),
      wall_seconds=max(0.0, (int(e["finished_ns"]) - int(e["started_ns"])) * 1e-9),
      return_sum=e.get("return_sum"), success=e.get("success"), outcome=e["outcome"],
      fault_code=FaultCode(fc).name if fc else "", experiment_axis=u.experiment_axis,
      randomization_hash=e.get("randomization_hash", ""), hardware_class=self.hardware,
      tracking_rmse=e.get("tracking_rmse"), energy_joules=e.get("energy_joules"),
      saturation_fraction=e.get("saturation_fraction"), reward_components=rc,
      task_metrics={"episode_success": float(bool(e.get("success")))} if e.get("success") is not None else {},
      source_episode_key=f"{u.seed}:{k}",
    )

  def _maybe_complete(self, ur: UnitRun) -> None:
    if ur.pending or ur.running or ur.failed:
      return
    ack = _run_async(self.scheduler.complete(ur.unit, ur.results))
    if ack.rejected:
      self.co._event("unit_commit_rejected", unit=ur.unit.unit_id, reason=ack.rejected)
    else:
      self.committed += len(ack.committed)
      if self.co.metrics is not None:
        self.co.metrics.count("eval_episodes_committed_total", len(ack.committed))
        if ack.duplicates:
          self.co.metrics.count("duplicate_commits_total", len(ack.duplicates))
    del self.units[ur.unit.unit_id]

  def _requeue(self, rec: WorkerRec, slot: int) -> None:
    uid, k = rec.assign.pop(slot, (None, None))
    if uid is None or uid not in self.units:
      return
    ur = self.units[uid]
    ur.running.discard(k)
    ur.infra_retries[k] += 1
    if ur.infra_retries[k] > self.cfg.evaluation.max_attempts:
      ur.failed = True
      _run_async(self.scheduler.fail(ur.unit, {"code": "INFRASTRUCTURE", "ordinal": k,
                                               "retries": ur.infra_retries[k]}))
      self.units.pop(uid, None)
      return
    ur.pending.appendleft(k)  # restart from the original seed: same keys, same episode

  def on_worker_lost(self, rec: WorkerRec, slots: np.ndarray, reason: str) -> None:
    for s in slots.tolist():
      self._requeue(rec, s)

  def _renew(self) -> None:
    self.last_renew = time.monotonic()
    live = [u.unit for u in self.units.values()]
    if not live:
      return
    ok = set(_run_async(self.scheduler.renew(self.co.name, live)))
    for u in live:
      if u.unit_id not in ok:  # lease lost: nothing of this attempt may commit
        self.co._event("unit_lease_lost", unit=u.unit_id)
        ur = self.units.pop(u.unit_id)
        ur.failed = True

  def done(self) -> bool:
    return self.exhausted_scheduler and not self.units and all(
      r.inflight is None for r in self.co.workers.values())

  def close(self) -> None:
    pass

  def status(self) -> dict[str, Any]:
    return {"kind": "eval", "units_active": len(self.units), "committed_episodes": self.committed,
            "health_failures": self.health_failures, "scheduler_exhausted": self.exhausted_scheduler}


# ================================================================ Mode A program


class RolloutProgram:
  """Asynchronous rollout generation: fragments -> assembler -> durable sink."""

  def __init__(self, co: ShardCoordinator, spec: dict[str, Any]):
    self.co = co
    self.spec = spec
    self.cfg = co.cfg
    self.placement = spec.get("placement", "local")
    self.assembler = spec["assembler"]
    self.target = int(spec.get("target_transitions", self.cfg.rollout.target_transitions))
    self.committed = 0
    self.rejected_fragments = 0
    self.pending_acks: dict[Any, tuple[WorkerRec, int, dict, Any, int]] = {}  # (rec, credit, meta, ref, attempt)
    self.policy_hash = {g: bytes.fromhex(v["policy"]["hash"]) for g, v in co.groups.items() if v.get("policy")}
    self.policy_version = {g: int(v["policy"]["version"]) for g, v in co.groups.items() if v.get("policy")}
    self.pending_policy: dict | None = None
    self.id_block: collections.deque = collections.deque()
    self.started = False
    self.curriculum_version = self.cfg.curriculum.version
    self.weights = dict(self.cfg.curriculum.weights) or {g: 1.0 for g in co.groups}
    self.starts: collections.Counter = collections.Counter()
    self.transitions_by_group: collections.Counter = collections.Counter()
    self.episodes_done = 0
    self.lost_upper = 0

  def _episode_ids(self, n: int) -> np.ndarray:
    while len(self.id_block) < n:
      start = self.co.store.db.allocate_episode_ids(self.co.run_id, 1024)
      self.id_block.extend(range(start, start + 1024))
    return np.asarray([self.id_block.popleft() for _ in range(n)], np.uint64)

  def set_curriculum(self, version: int, weights: dict[str, float]) -> None:
    total = sum(weights.values())
    self.weights = {k: v / total for k, v in weights.items()}
    self.curriculum_version = version
    self.co.store.db.audit(self.co.run_id, "curriculum_update", {"version": version, "weights": self.weights})

  def _allow_start(self, group: str) -> bool:
    """Deficit round robin over episode starts: a group more than one quantum ahead of
    its weighted share waits while others catch up (no fictitious padding)."""
    total = sum(self.starts.values())
    w = self.weights.get(group, 0.0)
    if w <= 0:
      return False
    live = [g for g in self.weights if self.weights[g] > 0 and any(
      r.group == g and r.state == "ready" for r in self.co.workers.values())]
    if len(live) <= 1:
      return True
    wsum = sum(self.weights[g] for g in live)
    return self.starts[group] - total * w / wsum <= 64

  def dispatch(self) -> None:
    self._drain_acks(block=False)  # release fragment credits as soon as the sink ACKs
    if self.pending_policy is not None:
      self._apply_policy()
    for rec in self.co.workers.values():
      if rec.state != "ready" or rec.inflight is not None:
        continue
      free = np.flatnonzero(rec.slot_state == FREE).astype(np.int32)
      if free.size and self._allow_start(rec.group) and self.committed < self.target:
        env = self._env_ids(rec, free)
        eps = self._episode_ids(free.size)
        req = {"command_id": P.new_id(), "generation": rec.generation, "slots": free, "env_id": env,
               "episode_id": eps}
        rec.slot_state[free] = RESETTING
        self.starts[rec.group] += free.size
        self.co.submit(rec, "reset", rec.actor.reset.remote(req), req, -1)
        continue
      if self.committed >= self.target:
        continue
      if self.placement == "local":
        # Rotate slices so one slice's fragment wait does not stall the others.
        for i, s in enumerate(rec.slices):
          act = s.slots[rec.slot_state[s.slots] == ACTIVE]
          if act.size:
            eps = self._episode_ids(act.size)
            cmd = {"command_id": P.new_id(), "expected_generation": rec.generation, "slots": act,
                   "policy_hash": self.policy_hash[rec.group], "deadline_unix_ns": time.time_ns() + int(30e9),
                   "max_control_steps": self.cfg.worker.local_control_steps, "autoreset": True,
                   "episode_ids": eps}
            self.starts[rec.group] += 0  # autoreset starts are counted from envelopes
            self.co.submit(rec, "run_local", rec.actor.run_local.remote(cmd), cmd, i,
                           self.cfg.worker.local_control_steps)
            rec.slices.append(rec.slices.pop(i))
            break
      else:
        _central_step(self, rec)

  def _env_ids(self, rec: WorkerRec, slots: np.ndarray) -> np.ndarray:
    have = rec.env_id[slots]
    missing = have == 0
    if missing.any():
      start = self.co.store.db.allocate_env_ids(self.co.run_id, int(missing.sum())) + 1
      rec.env_id[slots[missing]] = np.arange(start, start + int(missing.sum()), dtype=np.uint64)
    return rec.env_id[slots].copy()

  def on_envelope(self, rec: WorkerRec, kind: str, cmd: dict, sl: int, env: dict) -> None:
    st = env["status"]
    frame_credit = env["frame_credit"]
    if kind == "reset":
      ok = st["status"] == SlotStatus.OK
      rec.slot_state[st["slot"][ok]] = ACTIVE
      rec.slot_state[st["slot"][~ok]] = FREE
      rec.episode_id[st["slot"]] = st["episode_id"]
      rec.seq[st["slot"]] = 0
    else:
      ended = st["slot"][np.isin(st["status"], (SlotStatus.TERMINATED, SlotStatus.TRUNCATED,
                                                SlotStatus.NATIVE_FAILURE, SlotStatus.HEALTH_FAILURE))]
      self.episodes_done += len(env["episodes"])
      if kind == "run_local":
        # autoreset consumed ids; slots that could not restart are re-reset by the loop
        rec.episode_id[st["slot"]] = st["episode_id"]
        if "slot_state" in env:
          bad = st["slot"][(env["slot_state"] != P.SlotState.READY)]
          rec.slot_state[bad] = FREE
        restarted = len(cmd.get("episode_ids", ())) - len(env.get("unused_episode_ids", ()))
        self.starts[rec.group] += max(0, restarted)
        for eid in env.get("unused_episode_ids", ()):
          self.id_block.appendleft(int(eid))
      else:
        rec.slot_state[ended] = FREE
        _central_row_rejections(self, rec, cmd, st)
      rec.seq[st["slot"]] = st["control_seq"]
    # Forward fragments to the assembler; ACK their credits after durable acceptance.
    for ref, meta in zip(env["fragment_refs"], env["fragments"], strict=True):
      m = {**meta, "actor_name": rec.name, "generation": rec.generation, "group": rec.group}
      self.pending_acks[self.assembler.accept.remote(ref, m)] = (rec, int(meta["credit_id"]), m, ref, 1)
    if self.placement == "central" and kind != "run_local":
      live = st["status"] == SlotStatus.OK
      if kind == "reset":
        live = st["status"] == SlotStatus.OK
      if live.any():
        _central_after_frame(self, rec, {k: v[live] for k, v in st.items()}, env["frame_ref"],
                             frame_credit=frame_credit)
        return
    _ack(rec, [frame_credit])
    self._drain_acks(block=False)

  def _drain_acks(self, block: bool) -> None:
    if not self.pending_acks:
      return
    refs = list(self.pending_acks)
    ready, _ = ray.wait(refs, num_returns=len(refs) if block else 1, timeout=None if block else 0, fetch_local=True)
    for r in ready:
      rec, credit, meta, ref, attempt = self.pending_acks.pop(r)
      try:
        ack = ray.get(r)
      except _ACTOR_GONE as exc:
        if attempt < 3:
          # The assembler died with this fragment in flight. A restarted assembler
          # recovers from its durable outbox and accept is idempotent: send it again
          # (the worker holds the fragment until its credit is acknowledged).
          self.co._event("assembler_retry", error=str(exc)[:300], attempt=attempt)
          self.pending_acks[self.assembler.accept.remote(ref, meta)] = (rec, credit, meta, ref, attempt + 1)
          continue
        self.co._event("assembler_error", error=str(exc))
        self.rejected_fragments += 1
        continue
      except Exception as exc:  # noqa: BLE001 - assembler failure: the fragment is not acknowledged
        self.co._event("assembler_error", error=str(exc))
        self.rejected_fragments += 1
        continue
      if ack.get("rejected_reason"):
        self.rejected_fragments += 1
      else:
        self.committed += int(ack["accepted_transitions"])
        self.transitions_by_group[rec.group] += int(ack["accepted_transitions"])
      _ack(rec, [credit])

  def on_inference(self, rec: WorkerRec, ref: Any) -> None:
    _central_on_inference(self, rec, ref)

  def on_rejected(self, rec, kind, cmd, sl, env) -> None:
    _on_rejected(self, rec, kind, cmd, env)

  def on_worker_lost(self, rec: WorkerRec, slots: np.ndarray, reason: str) -> None:
    """Uncommitted tails of the lost generation become explicit gap records with
    conservative bounds; committed fragments stay valid (design §8.3)."""
    for s in slots.tolist():
      seq = int(rec.seq[s])
      bound = seq + self.cfg.worker.local_control_steps
      self.lost_upper += bound
      if self.co.metrics is not None:
        self.co.metrics.count("transitions_lost_upper_bound_total", bound)
      self.co.store.db.record_gap(self.co.run_id, int(rec.env_id[s]), int(rec.episode_id[s]), committed_prefix=0,
                                  lost_lower=0, lost_upper=bound, reason=reason[:200], generation=rec.generation)
    # Pending assembler acks for this worker stay: the assembler still owns those
    # fragments; only their worker credits are gone (_ack tolerates the dead worker).

  def _take_policy(self) -> dict | None:
    with self.co.lock:  # set by the control thread: read and clear in one step
      pol, self.pending_policy = self.pending_policy, None
    return pol

  def _apply_policy(self) -> None:
    pol = self._take_policy()
    if pol is None:
      return
    if self.placement == "local":
      # A worker re-initialized after a loss must load the *current* policy, not the
      # run's initial one (found by review of the restart path). The coordinator owns a
      # copy of the executable so it outlives the process that published the update.
      pol = {**pol, "exe_ref": ray.put(ray.get(pol["exe_ref"]))}
      for g in pol["groups"]:
        if g in self.co.groups:
          self.co.groups[g]["policy"] = {**(self.co.groups[g].get("policy") or {}), "manifest": pol["manifest"],
                                         "exe_ref": pol["exe_ref"]}
    for rec in self.co.workers.values():
      if rec.group not in pol["groups"]:
        continue
      if self.placement == "local":
        while rec.inflight is not None:  # command boundary
          self.co._collect()
        ray.get(rec.actor.load_policy.remote(pol["manifest"], pol["exe_ref"]))
      self.policy_hash[rec.group] = bytes.fromhex(pol["hash"])
      self.policy_version[rec.group] = int(pol["version"])
    self.co._event("policy_updated", version=pol["version"], hash=pol["hash"][:12])

  def done(self) -> bool:
    if self.committed >= self.target:
      self._drain_acks(block=True)
      return all(r.inflight is None for r in self.co.workers.values()) and not self.pending_acks
    return False

  def close(self) -> None:
    self._drain_acks(block=True)

  def status(self) -> dict[str, Any]:
    return {"kind": "rollout", "committed_transitions": self.committed, "target": self.target,
            "rejected_fragments": self.rejected_fragments, "episodes": self.episodes_done,
            "starts": dict(self.starts), "transitions_by_group": dict(self.transitions_by_group),
            "curriculum_version": self.curriculum_version, "lost_transitions_upper_bound": self.lost_upper}


# ======================================================== central inference glue


def _central_after_frame(prog, rec: WorkerRec, st: dict, frame_ref: Any, frame_credit: int | None = None) -> None:
  """Send a published frame to the inference service by reference (never fetched here).
  Each slot is in at most one of: awaiting inference, ready to step, stepping."""
  slots = np.asarray(st["slot"], np.int32)
  if slots.size == 0:
    if frame_credit is not None:
      _ack(rec, [frame_credit])
    return
  req = {"request_id": P.new_id(), "policy_hash": prog.policy_hash[rec.group],
         "deadline_unix_ns": time.time_ns() + int(max(prog.cfg.inference.deadline_s * 20, 1.0) * 1e9)}
  ref = prog.co.inference.infer.remote(req, frame_ref)
  rec.infer[ref] = {"slots": slots, "episode_id": np.asarray(st["episode_id"], np.uint64),
                    "control_seq": np.asarray(st["control_seq"], np.uint64), "frame_credit": frame_credit,
                    "started": time.monotonic()}
  rec.slot_state[slots] = WAIT_INFER


def _central_on_inference(prog, rec: WorkerRec, ref: Any) -> None:
  meta = rec.infer.pop(ref, None)
  if meta is None:
    return
  if meta.get("frame_credit") is not None:
    _ack(rec, [meta["frame_credit"]])  # the inference actor consumed the frame
  prog.co.metrics and prog.co.metrics.observe("inference_seconds", time.monotonic() - meta["started"])
  try:
    ray.get(ref)  # small ActionFrame wire; the step passes the ref itself to the worker
  except Exception as exc:  # noqa: BLE001 - no substituted actions: the slots' episodes are aborted
    prog.co.counters["inference_errors"] += 1
    prog.co._event("inference_error", worker=rec.name, error=str(exc)[:200])
    rec.slot_state[meta["slots"]] = FREE
    if isinstance(prog, EvalProgram):
      for sl in meta["slots"].tolist():
        prog._requeue(rec, sl)
    return
  cmd = {"command_id": P.new_id(), "expected_generation": rec.generation, "routing_epoch": 0,
         "slots": meta["slots"], "episode_id": meta["episode_id"], "control_seq": meta["control_seq"],
         "policy_hash": prog.policy_hash[rec.group], "deadline_unix_ns": time.time_ns() + int(30e9)}
  rec.slot_state[meta["slots"]] = READY_STEP
  rec.ready_steps.append({"command": cmd, "actions": ref})


def _central_step(prog, rec: WorkerRec) -> None:
  """One mutating advance per idle actor, oldest ready slice first."""
  if rec.inflight is not None or not rec.ready_steps:
    return
  cmd = rec.ready_steps.popleft()
  rec.slot_state[cmd["command"]["slots"]] = ACTIVE
  prog.co.submit(rec, "advance", rec.actor.advance.remote(cmd["command"], cmd["actions"]), cmd, -1)


def _on_rejected(prog, rec: WorkerRec, kind: str, cmd: dict, env: dict) -> None:
  """Whole-command rejection. CREDIT_EXHAUSTED is backpressure: nothing mutated, so the
  same step (same sequence, same action frame) is re-queued under a new command id.
  Anything else aborts the affected episodes explicitly."""
  code = env.get("rejected")
  if kind == "advance":
    slots = np.asarray(cmd["command"]["slots"])
    if code == FaultCode.CREDIT_EXHAUSTED:
      c = dict(cmd["command"], command_id=P.new_id())
      rec.slot_state[slots] = READY_STEP
      rec.ready_steps.append({"command": c, "actions": cmd["actions"]})
      return
  else:
    slots = np.asarray(cmd["slots"])
  _abort_slots(prog, rec, slots, f"rejected {FaultCode(code).name if code else 'rows'}")


def _abort_slots(prog, rec: WorkerRec, slots: np.ndarray, reason: str) -> None:
  if isinstance(prog, EvalProgram):
    for s in slots.tolist():
      prog._requeue(rec, s)
  else:
    for s in slots.tolist():
      if rec.slot_state[s] not in (FREE, RESETTING):
        prog.co.store.db.record_gap(prog.co.run_id, int(rec.env_id[s]), int(rec.episode_id[s]), 0, 0,
                                    int(rec.seq[s]), reason[:200], rec.generation)
  rec.slot_state[slots] = FREE


def _central_row_rejections(prog, rec: WorkerRec, cmd: dict, st: dict) -> None:
  """Row-level rejections in a central advance: transient credit exhaustion re-queues
  the unchanged rows; stale/invalid rows abort their episodes."""
  rej = st["status"] == SlotStatus.REJECTED
  if not rej.any():
    return
  transient = rej & (st["fault_code"] == FaultCode.CREDIT_EXHAUSTED)
  if transient.any():
    c = cmd["command"]
    keep = np.isin(c["slots"], st["slot"][transient])
    rc: dict[str, Any] = dict(c, command_id=P.new_id(), slots=c["slots"][keep], episode_id=c["episode_id"][keep],
              control_seq=c["control_seq"][keep])
    rec.slot_state[rc["slots"]] = READY_STEP
    rec.ready_steps.append({"command": rc, "actions": cmd["actions"]})
  hard = rej & ~transient
  if hard.any():
    _abort_slots(prog, rec, st["slot"][hard], "rejected rows")


# ------------------------------------------------------------------- helpers


def _run_async(coro):
  import asyncio

  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None
  if loop is None:
    return asyncio.run(coro)
  raise RuntimeError("coordinator programs run on the loop thread, not in an event loop")


async def _to_thread(fn, *args):
  return fn(*args)
