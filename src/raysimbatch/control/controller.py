# SPDX-License-Identifier: Apache-2.0
"""RunController: the detached per-run owner (design §6.1).

State machine: NEW -> VALIDATING -> RESERVING -> RUNNING -> DRAINING -> COMPLETED;
validation/reservation errors -> FAILED; RUNNING may enter DEGRADED, PAUSED or
CANCELING; CANCELING ends at CANCELED after fencing and cleanup. COMPLETED, FAILED and
CANCELED are terminal; resuming creates a new attempt epoch under the same run id.

The controller holds a PostgreSQL lease (renewed every ``lease_renew_s``, expiring after
``lease_s``). A replacement controller fences the old one by taking that lease. The
submitting client may disconnect at any time.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

import numpy as np
import ray

from raysimbatch.compat.ray_typing import actor_class
from raysimbatch.config.models import RunConfig, TaskSpec
from raysimbatch.config.system import ClusterInventory, ConfigurationSystem

if TYPE_CHECKING:
  from raysimbatch.protocol import Lease
  from raysimbatch.storage.store import DurableStore

TERMINAL = ("COMPLETED", "FAILED", "CANCELED")


class RunController:
  def __init__(self, run_id: str, config_json: str, tasks_json: str = "[]", options: dict[str, Any] | None = None):
    self.run_id = run_id
    self.cfg = RunConfig.model_validate_json(config_json)
    self.tasks = tuple(TaskSpec.from_obj(t) for t in json.loads(tasks_json))
    self.opts = dict(options or {})
    self.namespace = self.opts.get("namespace")
    self.state = "NEW"
    self.error: str | None = None
    self.detail: dict[str, Any] = {}
    self.coordinators: list[Any] = []
    self.coord_refs: dict[Any, int] = {}
    self.inference: list[Any] = []
    self.assembler = None
    self.pg = None
    self.cancel_requested: tuple[str, bool] | None = None
    self.stop_requested = False  # graceful drained stop (COMPLETED), unlike cancel
    self.lock = threading.Lock()
    self.results: dict[str, Any] | None = None
    self.thread: threading.Thread | None = None
    # Late initialization: _main sets these before anything uses them.
    self.store: DurableStore = None  # type: ignore[assignment]
    self.lease: Lease = None  # type: ignore[assignment]
    import uuid

    # A per-incarnation owner identity: a replacement controller fences this one by
    # taking the run lease after it expires, never by sharing its identity.
    self.name = f"rsb-run-{run_id}:{uuid.uuid4().hex[:8]}"
    self.events: list[dict] = []
    self.curriculum: tuple[int, dict] | None = None
    self.policy_update: dict | None = None
    self.metrics = None
    self._gauges_at = 0.0
    self.coord_failures: dict[int, int] = {}
    self._lease_lost: str | None = None

  # ----------------------------------------------------------------- lifecycle

  def start(self) -> str:
    if self.thread is None:
      self.thread = threading.Thread(target=self._main, name="rsb-run", daemon=True)
      self.thread.start()
    return self.state

  def _set(self, state: str, **detail: Any) -> None:
    with self.lock:
      self.state = state
      self.detail.update(detail)
      self.events.append({"t": time.time(), "state": state, **{k: str(v) for k, v in detail.items()}})
    from raysimbatch.telemetry import logs

    logs.log("run_state", 40 if state == "FAILED" else 20, run_id=self.run_id,
             job_epoch=self.lease.epoch if self.lease else None, error_code=detail.get("error") and "RUN_FAILED",
             message=f"{state} " + " ".join(f"{k}={v}" for k, v in detail.items() if k != "traceback"))
    if self.store is not None:
      patch: dict[str, Any] = {"detail": {k: str(v) for k, v in detail.items()}}
      if state in TERMINAL:  # served by status() after this controller exits
        patch.update(results=json.loads(json.dumps(self.results, default=str)) if self.results else None,
                     error=self.error)
      self.store.db.set_run_state(self.run_id, state, payload_patch=patch)
      self.store.db.audit(self.run_id, "run_state", {"state": state, **{k: str(v) for k, v in detail.items()}})

  def _main(self) -> None:
    from raysimbatch import telemetry
    from raysimbatch.storage.db import LeaseHeld
    from raysimbatch.storage.store import DurableStore
    from raysimbatch.telemetry import tracing

    try:
      telemetry.setup("raysimbatch-controller", self.cfg.telemetry)
      self.metrics = telemetry.exporter(mode=self.cfg.mode, actor_role="controller")
      self.store = DurableStore.from_config(self.cfg)
      try:
        with tracing.span("lease", key="run"):
          self.lease = self._acquire_run_lease()
      except LeaseHeld as exc:
        # Not the owner: never write the run's state (found in kind: a refused resume
        # overwrote a live run as FAILED). The attempt is recorded append-only.
        self.error = f"{type(exc).__name__}: {exc}"
        self.store.db.audit(self.run_id, "controller_start_refused", {"controller": self.name, "error": self.error})
        with self.lock:
          self.state = "REFUSED"
        return
      # The owning cluster: its janitor resumes this run if the controller is lost.
      cluster = self.opts.get("cluster") or os.environ.get("RSB_CLUSTER_NAME")
      resumed = bool(self.opts.get("resumed"))
      self.store.db.set_run_state(self.run_id, "VALIDATING", payload_patch={"cluster": cluster, "resumed": resumed},
                                  bump_attempt=resumed)  # a new owner epoch, taken under the lease
      self._renewed = time.monotonic()
      # Renew on a keeper thread: gang reservation (up to gang_timeout_s) and coordinator
      # installation block the main thread far longer than the lease (found in kind:
      # the controller fenced itself while an autoscaled node started).
      threading.Thread(target=self._lease_keeper, daemon=True, name="rsb-run-lease").start()
      self._set("VALIDATING")
      cs = ConfigurationSystem()
      rep = cs.validate(self.cfg, ClusterInventory(production=bool(self.opts.get("production"))))
      if not rep.ok:
        raise ValueError("; ".join(f"{i.path}:{i.code}" for i in rep.errors()))
      from raysimbatch.control.admission import admit

      with tracing.span("admission", mode=self.cfg.mode):
        groups, task_groups = admit(self.cfg, self.tasks)
      self._set("RESERVING")
      if self.cfg.mode in ("eval", "sim2sim"):
        self._run_eval(groups, task_groups)
      else:
        self._run_rollout(groups)
    except Exception as exc:  # noqa: BLE001
      self.error = f"{type(exc).__name__}: {exc}"
      self.detail["traceback"] = traceback.format_exc()
      try:
        self._cleanup()
      finally:
        if self.state not in TERMINAL:
          self._set("FAILED", error=self.error)
    finally:
      if self.state in TERMINAL or self.state == "REFUSED":
        # A detached controller of a finished run would keep its node busy forever
        # (found in kind: the autoscaler never scaled down). Results are durable.
        threading.Timer(float(self.opts.get("controller_linger_s", 30.0)), _exit_self).start()

  def _progress_gauges(self, force: bool = False) -> None:
    """Mode B backlog and suite gauges from the durable tables, at most every 5 s (and once
    more when the run's work is done: the controller lingers, so its final values export)."""
    if self.metrics is None or self.cfg.mode not in ("eval", "sim2sim"):
      return
    if not force and time.monotonic() - self._gauges_at < 5.0:
      return
    self._gauges_at = time.monotonic()
    c = self.store.db.unit_counts(self.run_id)
    total = sum(c.values())
    m = self.metrics
    m.gauge("eval_units_pending", float(c.get("PENDING", 0) + c.get("RETRYABLE", 0)))
    m.gauge("eval_units_running", float(sum(c.get(k, 0) for k in ("LEASED", "RUNNING", "COMMITTING"))))
    m.gauge("eval_units_exhausted", float(c.get("EXHAUSTED", 0)))
    m.gauge("eval_coverage_ratio", c.get("SUCCEEDED", 0) / total if total else 0.0)
    from raysimbatch.evaluation.stats import VALID_OUTCOMES

    e = self.store.db.episode_summary(self.run_id, VALID_OUTCOMES)
    if e["valid"]:  # suite-level: success among valid episodes (the conditional rate), mean return
      m.gauge("eval_success_ratio", int(e["successes"]) / int(e["valid"]))
    if e["return_mean"] is not None:
      m.gauge("eval_return_mean", float(e["return_mean"]))

  def _acquire_run_lease(self):
    """Take the run lease, waiting out a previous owner that may still be dying.

    Found in kind: deleting a release's pods (even with --force --grace-period=0) only
    removes the API objects; the kubelet still stops the containers gracefully, and the
    Ray head keeps running and renewing until SIGKILL at terminationGracePeriodSeconds.
    The resume therefore waits up to that grace plus two lease lifetimes for the lease
    to lapse (fencing stays intact); a holder still renewing after that is alive, and
    the resume is refused explicitly."""
    from raysimbatch.storage.db import LeaseHeld

    f = self.cfg.failure
    wait_s = self.cfg.cluster.termination_grace_s + 2 * f.lease_s + f.lease_renew_s
    deadline = time.monotonic() + wait_s
    while True:
      try:
        return self.store.db.acquire_lease(f"run/{self.run_id}", self.name, f.lease_s)
      except LeaseHeld as exc:
        if time.monotonic() >= deadline:
          raise LeaseHeld(f"{exc}; it kept renewing for {wait_s:.0f}s, so its owner is alive: "
                          "cancel the run there or wait for that controller to stop") from exc
        time.sleep(min(1.0, f.lease_renew_s))

  def _lease_keeper(self) -> None:
    while self.state not in TERMINAL and self._lease_lost is None:
      time.sleep(min(1.0, self.cfg.failure.lease_renew_s / 2))
      if time.monotonic() - self._renewed < self.cfg.failure.lease_renew_s:
        continue
      try:
        lease = self.store.db.renew_lease(f"run/{self.run_id}", self.name, self.lease.epoch,
                                          self.cfg.failure.lease_s)
      except Exception as exc:  # noqa: BLE001 - storage outage: retried until the lease itself expires
        if time.monotonic() - self._renewed > self.cfg.failure.lease_s:
          self._lease_lost = f"run lease expired during a storage outage: {exc}"
        continue
      if lease is None:
        self._lease_lost = "run lease lost: another controller owns this run"
      else:
        self.lease, self._renewed = lease, time.monotonic()

  def _renew(self) -> None:
    """Called from the main loops: surfaces a lease loss detected by the keeper."""
    if self._lease_lost is not None:
      raise RuntimeError(self._lease_lost)

  # ------------------------------------------------------------- shared setup

  def _group_runtimes(self, groups, placement: str) -> dict[str, dict[str, Any]]:
    """Worker spec templates plus node-cache ObjectRefs for model/policy bytes."""
    import hashlib

    out = {}
    for name, g in groups.items():
      mjb_ref = self._cache_put(g.mjb_sha, g.mjb)
      pname, (bundle, exe) = next(iter(g.policies.items()))
      exe_sha = hashlib.sha256(exe).hexdigest()
      exe_ref = self._cache_put(exe_sha, exe)
      spec = {
        "run_id": self.run_id, "actor_name": "", "shard_id": 0, "coordinator_id": 0, "mode": self.cfg.mode,
        "group": g.group.model_dump(mode="json"), "worker": self.cfg.worker.model_dump(mode="json"),
        "failure": self.cfg.failure.model_dump(mode="json"), "flow": self.cfg.flow.model_dump(mode="json"),
        "rollout": self.cfg.rollout.model_dump(mode="json"), "contract": g.contract.model_dump(mode="json"),
        "task_doc": {"name": g.binding.name, "version": g.binding.version, "spec": dict(g.binding.spec)},
        "group_hash": g.group_hash, "config_hash": self.opts.get("config_hash", ""), "run_seed": self.cfg.run_seed,
        "reset_distribution": g.reset_distribution, "envelope": g.envelope, "normalizer": g.normalizer,
        "cpu_set": (), "artifact_root": self.cfg.artifact_root,
        "collect_fragments": self.cfg.mode == "rollout", "curriculum_version": self.cfg.curriculum.version,
        "telemetry": self.cfg.telemetry.model_dump(mode="json"),
      }
      out[name] = {
        "spec": spec, "mjb_ref": mjb_ref, "mjb_sha": g.mjb_sha, "model_hash": g.model_hash,
        "task_hash": g.binding.task_hash, "adapter_hash": g.compiled.adapter_hash, "obs_dim": g.compiled.obs_dim,
        "action_dim": g.compiled.num_actions,
        "policy": {"name": pname, "hash": bundle.policy_hash, "version": bundle.version,
                   "manifest": bundle.to_json().decode(), "exe_ref": exe_ref, "exe_sha": exe_sha},
        "all_policies": {n: {"hash": b.policy_hash, "version": b.version, "manifest": b.to_json().decode(),
                             "exe_sha": hashlib.sha256(e).hexdigest()} for n, (b, e) in g.policies.items()},
        "_bundles": g.policies,
      }
    return out

  def _cache(self):
    from raysimbatch.control.assets import asset_cache_for_node

    return asset_cache_for_node(ray.get_runtime_context().get_node_id(), self.cfg.artifact_root, self.namespace)

  def _cache_put(self, sha256: str, data: bytes):
    """The node cache owns the bytes; this run holds them until cleanup."""
    from raysimbatch.control.assets import call_node_helper

    return call_node_helper(self._cache, lambda c: c.put_bytes.remote(sha256, data, self.run_id))

  def _start_inference(self, runtimes) -> Any:
    from raysimbatch.policy.service import RemotePolicyInferenceService, inference_options

    inf = self.cfg.inference
    svc = RemotePolicyInferenceService.options(
      **inference_options(inf, f"rsb-inf-{self.run_id[:8]}-0", namespace=self.namespace)
    ).remote(inf.model_dump(mode="json"))
    loaded = set()
    for rt in runtimes.values():
      for _, (bundle, exe) in rt["_bundles"].items():
        if bundle.policy_hash in loaded:
          continue
        ray.get(svc.load.remote(bundle.to_json().decode(), np.frombuffer(exe, np.uint8)))
        loaded.add(bundle.policy_hash)
    self.inference.append(svc)
    return svc

  def _reserve(self, plan):
    from raysimbatch.control.placement import GangTimeout, reserve

    deadline = time.monotonic() + self.cfg.cluster.gang_timeout_s
    started = time.monotonic()

    def check() -> None:  # called while waiting: keep the lease and report the backlog
      self._renew()
      if self.metrics is not None:
        self.metrics.gauge("scheduler_backlog_seconds", time.monotonic() - started)

    while True:
      try:
        pg, wait_s = reserve(plan, max(1.0, deadline - time.monotonic()), f"rsb-{self.run_id[:8]}-wave",
                             check=check)
        self.detail["placement_ready_seconds"] = wait_s
        if self.metrics is not None:
          self.metrics.observe("placement_ready_seconds", wait_s)
          self.metrics.gauge("scheduler_backlog_seconds", 0.0)
        return pg
      except GangTimeout:
        if time.monotonic() >= deadline:
          self._set("RESERVING", deferred_wave=True)
          raise

  def _start_coordinators(self, runtimes, plan, program: dict[str, Any]) -> None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from raysimbatch.control.coordinator import RemoteShardCoordinator

    nworkers = len(plan.assignments)
    for cid, assigns in sorted(plan.by_coordinator().items()):
      opts: dict[str, Any] = {"name": f"rsb-coord-{self.run_id[:8]}-{cid}", "num_cpus": 1}
      if self.namespace:
        opts["namespace"] = self.namespace
      if self.pg is not None:
        opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
          self.pg, placement_group_bundle_index=nworkers + cid, placement_group_capture_child_tasks=False)
      groups_arg = {k: {kk: vv for kk, vv in v.items() if kk != "_bundles"} for k, v in runtimes.items()}
      c = RemoteShardCoordinator.options(**opts).remote(
        self.run_id, cid, self.cfg.model_dump_json(), groups_arg,
        {"workers": [a.__dict__ for a in assigns], "placement_group": self.pg, "routing_epoch": plan.routing_epoch},
        program, self.namespace)
      self.coordinators.append(c)
    ray.get([c.install.remote() for c in self.coordinators])

  def _supervise(self) -> None:
    refs = {c.run.remote(): i for i, c in enumerate(self.coordinators)}
    self.coord_refs = refs
    pending = list(refs)
    while pending:
      self._renew()
      self._progress_gauges()
      if self.cancel_requested is not None or self.stop_requested:
        drain = True if self.cancel_requested is None else self.cancel_requested[1]
        for c in self.coordinators:
          c.request_stop.remote(drain)
      cur, pu = self._take_updates()
      if cur is not None:
        for c in self.coordinators:
          c.update_curriculum.remote(*cur)
      if pu is not None:
        for c in self.coordinators:
          c.update_policy.remote(pu)
      ready, pending = ray.wait(pending, num_returns=1, timeout=1.0)
      for r in ready:
        try:
          ray.get(r)
        except Exception as exc:  # noqa: BLE001 - coordinator loss: fence and continue
          self.store.db.audit(self.run_id, "coordinator_lost", {"index": refs[r], "error": _tail(str(exc))})
          n = self.coord_failures[refs[r]] = self.coord_failures.get(refs[r], 0) + 1
          if n > self.cfg.failure.max_restarts:
            # Bounded: a deterministic failure must end the run, not crash-loop it.
            raise RuntimeError(f"coordinator {refs[r]} failed {n} times; last error: {_tail(str(exc), 600)}") from exc
          self._set(self.state, degraded=f"coordinator {refs[r]} lost")
          if self.cfg.mode in ("eval", "sim2sim") and self.cancel_requested is None:
            pending.append(self._replace_coordinator(refs[r]))
          else:
            self.detail.setdefault("lost_coordinators", []).append(refs[r])

  def _replace_coordinator(self, index: int):
    """New coordinator epoch: its workers are recreated (the old ones died with their
    owner); expired unit leases are re-leased from the original seeds."""
    old = self.coordinators[index]
    try:
      ray.kill(old, no_restart=True)
    except Exception:  # noqa: BLE001
      pass
    from raysimbatch.control.coordinator import RemoteShardCoordinator

    spec = self._coord_specs[index]
    spec["plan"]["workers"] = [{**w, "actor_name": w["actor_name"] + "r"} for w in spec["plan"]["workers"]]
    c = RemoteShardCoordinator.options(name=f"rsb-coord-{self.run_id[:8]}-{index}r{int(time.time())}",
                                       num_cpus=1, **({"namespace": self.namespace} if self.namespace else {})).remote(
      self.run_id, index, self.cfg.model_dump_json(), spec["groups"], spec["plan"], spec["program"], self.namespace)
    ray.get(c.install.remote())
    self.coordinators[index] = c
    ref = c.run.remote()
    self.coord_refs[ref] = index
    self.store.db.audit(self.run_id, "coordinator_replaced", {"index": index})
    return ref

  # ------------------------------------------------------------------- Mode B

  def _run_eval(self, groups, task_groups) -> None:
    import asyncio

    from raysimbatch.control.placement import plan_capacity
    from raysimbatch.evaluation.aggregator import ResultAggregator
    from raysimbatch.evaluation.scheduler import EvalPolicy, EvalScheduler, EvalTask

    ev = self.cfg.evaluation
    agg = ResultAggregator(self.run_id, 0, self.store, replicates=ev.bootstrap_replicates, confidence=ev.confidence,
                           task_weights=ev.task_weights, exports=ev.exports)
    sched = EvalScheduler(self.run_id, ev, self.store, agg)
    tasks = []
    weights_by_group: dict[str, float] = {}
    axis_of_group = dict(self.opts.get("axis_of_group", {}))
    for t in self.tasks or self.cfg.tasks:
      for gname in task_groups[t.name]:
        g = groups[gname]
        tasks.append(EvalTask(t.name, g.binding.task_hash, gname, g.group_hash, g.compiled.adapter_hash,
                              ev.task_weights.get(t.name, 1.0), axis_of_group.get(gname)))
    policies = []
    for p in self.cfg.policies:
      for g in groups.values():
        if p.name in g.policies:
          b = g.policies[p.name][0]
          policies.append(EvalPolicy(p.name, b.policy_hash, b.version))
          break
    axes = tuple(self.opts.get("axes", ("",)))
    n = asyncio.run(sched.enumerate(policies, tasks, ev.seeds, axes))  # persisted before execution
    for t in tasks:
      weights_by_group[t.group] = weights_by_group.get(t.group, 0) + 1.0
    self.detail["units"] = n
    placement = self.cfg.inference.placement if self.cfg.inference.placement != "auto" else self.opts.get(
      "resolved_placement", "local")
    runtimes = self._group_runtimes(groups, placement)
    # Multiple policies evaluated on one group: each policy is its own worker group.
    runtimes = self._policy_runtimes(runtimes, groups)
    weights = {k: 1.0 for k in runtimes}
    plan = plan_capacity(self.cfg, self.run_id, weights, actors=self.opts.get("actors"),
                         slots_per_actor=self.opts.get("slots_per_actor"),
                         memory_bytes=self.opts.get("worker_memory_bytes"),
                         cluster_resources=ray.cluster_resources())
    if ev.gang:
      self.pg = self._reserve(plan)
    svc = self._start_inference(runtimes) if placement == "central" else None
    program = {"kind": "eval", "placement": placement, "inference_actor": svc,
               "config_hash": self.opts.get("config_hash", ""), "hardware_class": self.opts.get("hardware_class", ""),
               "worker_class": self.opts.get("worker_class")}
    self._coord_specs = {cid: {"groups": {k: {kk: vv for kk, vv in v.items() if kk != "_bundles"}
                                          for k, v in runtimes.items()},
                               "plan": {"workers": [a.__dict__ for a in assigns], "placement_group": None,
                                        "routing_epoch": 0},
                               "program": program}
                         for cid, assigns in plan.by_coordinator().items()}
    self._start_coordinators(runtimes, plan, program)
    self._set("RUNNING", workers=len(plan.assignments), coordinators=len(self.coordinators))
    self._supervise()
    self._progress_gauges(force=True)
    if self.cancel_requested is not None:
      self.store.db.cancel_units(self.run_id)
      self._set("CANCELING")
    counts = self.store.db.unit_counts(self.run_id)
    open_units = {k: v for k, v in counts.items() if k not in ("SUCCEEDED", "EXHAUSTED", "CANCELED")}
    if open_units:
      raise RuntimeError(f"coordinators stopped with non-terminal units {open_units}; resume the run")
    self._set("DRAINING")
    arts = agg.finalize_sync()
    self.results = {k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in arts.__dict__.items()}
    self.results["run_manifest"] = self._write_manifest({"units": dict(counts)}, results=self.results)
    self._cleanup()
    if self.cancel_requested:
      final = "CANCELED"
    elif arts.complete or ev.allow_partial:
      final = "COMPLETED"  # a partial result is exported but labeled complete=false
    else:
      final = "FAILED"
    self._set(final, complete=arts.complete, coverage=arts.coverage, unit_states=counts)

  def _policy_runtimes(self, runtimes, groups) -> dict[str, dict[str, Any]]:
    """Units bind (policy, group); worker groups are keyed by group so a unit's policy
    must equal its group's worker policy. With several policies per group, clone the
    group runtime per policy (local inference loads one policy per worker)."""
    import hashlib

    out = {}
    for name, rt in runtimes.items():
      pols = rt["_bundles"]
      if len(pols) == 1:
        out[name] = rt
        continue
      for pname, (bundle, exe) in pols.items():
        r = dict(rt)
        sha = hashlib.sha256(exe).hexdigest()
        r["policy"] = {"name": pname, "hash": bundle.policy_hash, "version": bundle.version,
                       "manifest": bundle.to_json().decode(), "exe_ref": self._cache_put(sha, exe),
                       "exe_sha": sha}
        r["_bundles"] = {pname: (bundle, exe)}
        out[f"{name}@{pname}"] = r
    self._group_alias = {k: k.split("@")[0] for k in out}
    return out

  # ------------------------------------------------------------------- Mode A

  def _run_rollout(self, groups) -> None:
    from raysimbatch.control.placement import plan_capacity
    from raysimbatch.experience.assembler import RemoteExperienceAssembler

    placement = self.cfg.inference.placement if self.cfg.inference.placement != "auto" else self.opts.get(
      "resolved_placement", "local")
    runtimes = self._group_runtimes(groups, placement)
    weights = dict(self.cfg.curriculum.weights) or {g: 1.0 for g in runtimes}
    plan = plan_capacity(self.cfg, self.run_id, weights, actors=self.opts.get("actors"),
                         slots_per_actor=self.opts.get("slots_per_actor"),
                         memory_bytes=self.opts.get("worker_memory_bytes"),
                         cluster_resources=ray.cluster_resources())
    self.pg = self._reserve(plan) if self.opts.get("gang", False) else None
    sink = self.opts.get("sink") or {"kind": "durable_replay"}
    aopts: dict[str, Any] = {"name": f"rsb-asm-{self.run_id[:8]}", "num_cpus": 1}
    if self.namespace:
      aopts["namespace"] = self.namespace
    self.assembler = RemoteExperienceAssembler.options(**aopts).remote(
      self.run_id, self.cfg.model_dump_json(), sink)
    ray.get(self.assembler.ready.remote())
    svc = self._start_inference(runtimes) if placement == "central" else None
    program = {"kind": "rollout", "placement": placement, "inference_actor": svc, "assembler": self.assembler,
               "target_transitions": self.opts.get("target_transitions", self.cfg.rollout.target_transitions)}
    self._coord_specs = {}
    self._start_coordinators(runtimes, plan, program)
    self._set("RUNNING", workers=len(plan.assignments), coordinators=len(self.coordinators))
    self._supervise()
    self._set("DRAINING")
    offset = ray.get(self.assembler.seal.remote(time.time_ns() + int(60e9)))
    statuses = []
    for c in self.coordinators:
      try:
        statuses.append(ray.get(c.status.remote(), timeout=10))
      except Exception:  # noqa: BLE001
        statuses.append({"error": "coordinator unavailable"})
    gaps = self.store.db.gaps(self.run_id)
    self.results = {"sink_offset": offset, "coordinators": statuses, "gaps": len(gaps),
                    "lost_upper_bound": sum(int(g["lost_upper"]) for g in gaps)}
    committed = sum(int((s.get("program") or {}).get("committed_transitions", 0)) for s in statuses)
    self.results["run_manifest"] = self._write_manifest(
      {"committed_transitions": committed, "sink": offset}, results={"gaps": len(gaps)},
      failures={"gaps": [dict(g) for g in gaps], "lost_upper_bound": self.results["lost_upper_bound"]})
    self._cleanup()
    self._set("CANCELED" if self.cancel_requested else "COMPLETED")

  def _write_manifest(self, counts: dict[str, Any], *, results: dict[str, Any],
                      failures: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reproducibility manifest (design §11.2), content-addressed and audited."""
    from raysimbatch.telemetry import manifest

    cs = ConfigurationSystem()
    artifacts = {k: v["sha256"] for k, v in results.items() if isinstance(v, dict) and "sha256" in v}
    audit = [e["event"] for e in self.store.db.audit_events(self.run_id)]
    fails = {"worker_lost": audit.count("worker_lost"), "worker_fenced": audit.count("worker_fenced"),
             "coordinator_lost": audit.count("coordinator_lost"), **(failures or {})}
    doc = manifest.build(
      requested=None, resolved=cs.resolved(self.cfg), config_hash=cs.canonical_hash(self.cfg), run_id=self.run_id,
      artifacts=artifacts, counts=counts,
      seeds={"run_seed": self.cfg.run_seed, "eval_seeds": list(self.cfg.evaluation.seeds),
             "episodes_per_seed": self.cfg.evaluation.episodes_per_seed},
      failures=fails,
      # Replay determinism is certified for the build by qualification (bitwise parity,
      # chaos digests), not re-certified inside each run.
      deterministic_replay_certified=False,
      extra={"job_epoch": self.lease.epoch if self.lease else None, "mode": self.cfg.mode,
             "controller": self.name},
    )
    art = self.store.blobs.put(manifest.dumps(doc), "application/json")
    self.store.db.audit(self.run_id, "run_manifest", {"sha256": art.sha256, "uri": art.uri})
    return art.__dict__

  # ------------------------------------------------------------------ control

  def status(self) -> dict[str, Any]:
    counts = {}
    if self.store is not None:
      try:
        counts = self.store.db.unit_counts(self.run_id)
      except Exception:  # noqa: BLE001
        counts = {}
    coords = []
    for c in self.coordinators:
      try:
        coords.append(ray.get(c.status.remote(), timeout=2))
      except Exception:  # noqa: BLE001
        coords.append({"error": "unavailable"})
    with self.lock:
      return {"run_id": self.run_id, "state": self.state, "error": self.error, "detail": dict(self.detail),
              "units": counts, "coordinators": coords, "results": self.results,
              "lease_epoch": None if self.lease is None else self.lease.epoch}

  def cancel(self, reason: str, drain: bool = True) -> str:
    self.cancel_requested = (reason, drain)
    if self.state in ("RUNNING", "DEGRADED", "PAUSED"):
      self._set("CANCELING", reason=reason)
    return self.state

  def stop(self) -> str:
    """Drain and finish normally (open-ended rollouts); results stay COMPLETED."""
    self.stop_requested = True
    return self.state

  def update_curriculum(self, version: int, weights: dict[str, float]) -> None:
    with self.lock:
      self.curriculum = (version, weights)

  def update_policy(self, policy: dict[str, Any]) -> None:
    with self.lock:
      self.policy_update = policy

  def _take_updates(self) -> tuple[tuple[int, dict[str, float]] | None, dict[str, Any] | None]:
    """Updates arrive on other actor threads (max_concurrency): read and clear each slot
    in one step under the lock, so an update landing in between is never erased."""
    with self.lock:
      cur, self.curriculum = self.curriculum, None
      pu, self.policy_update = self.policy_update, None
    return cur, pu

  def get_assembler(self):
    return self.assembler

  def _cleanup(self) -> None:
    from raysimbatch.control.placement import release

    for c in self.coordinators:
      try:
        ray.get(c.shutdown_workers.remote(), timeout=30)
      except Exception:  # noqa: BLE001
        pass
      try:
        ray.kill(c, no_restart=True)
      except Exception:  # noqa: BLE001
        pass
    for s in self.inference:
      try:
        ray.kill(s, no_restart=True)
      except Exception:  # noqa: BLE001
        pass
    if self.assembler is not None:
      try:
        ray.kill(self.assembler, no_restart=True)
      except Exception:  # noqa: BLE001
        pass
    release(self.pg)
    self.pg = None
    try:
      ray.get(self._cache().release_holder.remote(self.run_id), timeout=30)
    except Exception:  # noqa: BLE001 - the janitor releases holds of terminal runs
      pass
    if self.store is not None and self.lease is not None:
      self.store.db.release_lease(f"run/{self.run_id}", self.name, self.lease.epoch)


RemoteRunController = actor_class(RunController, num_cpus=0, max_restarts=0, max_task_retries=0, max_concurrency=8)


def _tail(text: str, n: int = 2000) -> str:
  """Keep the end of a remote traceback: the exception type and message are last."""
  return text if len(text) <= n else "..." + text[-n:]


def _exit_self() -> None:
  ray.kill(ray.get_runtime_context().current_actor, no_restart=True)
