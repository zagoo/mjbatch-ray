# SPDX-License-Identifier: Apache-2.0
"""SimulationWorker: the Ray actor that owns one MJBatch shard (design §6.2).

Concurrency: ``mutate`` (one thread) runs every physics/reset/snapshot call; ``control``
(one thread) serves health, ACKs, fencing and drain intent. The control thread reads
only plain counters and never touches MJBatch views; the credit table and command
cache it shares with the mutation thread are guarded by a lock that is never held
across a native call or any RPC.

Publication (design §7.2): outputs are assembled into freshly allocated arrays, then
``ray.put`` snapshots them into immutable Ray-owned storage before anything can mutate
them again; the returned envelope carries only small metadata and ObjectRefs. The
worker is the owner of those objects: they are transient, and durable data is
committed downstream (assembler/aggregator).
"""

from __future__ import annotations

import io
import json
import os
import resource
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import ray

from raysimbatch import protocol as P
from raysimbatch import telemetry
from raysimbatch.adapter import Normalizer, compile_adapter
from raysimbatch.compat.ray_typing import actor_class
from raysimbatch.config.models import (
  AdapterContract,
  FailureConfig,
  FlowConfig,
  ModelGroup,
  RolloutConfig,
  WorkerConfig,
)
from raysimbatch.policy.bundle import PolicyBundle
from raysimbatch.policy.runtime import LocalPolicyRuntime
from raysimbatch.protocol import FaultCode, Flag, SlotState, SlotStatus, WorkerState
from raysimbatch.seeds import derive_keys
from raysimbatch.tasks import bind_task
from raysimbatch.telemetry import procfs, tracing
from raysimbatch.worker.affinity import AffinityPlan
from raysimbatch.worker.core import AdvanceResult, CoreSpec, MJBatchCore
from raysimbatch.worker.fragments import FragmentBuilder
from raysimbatch.worker.model import ModelMetadata, from_mjb


@dataclass(frozen=True)
class WorkerSpec:
  run_id: str
  actor_name: str
  shard_id: int
  coordinator_id: int
  mode: str
  group: dict[str, Any]
  worker: dict[str, Any]
  failure: dict[str, Any]
  flow: dict[str, Any]
  rollout: dict[str, Any]
  contract: dict[str, Any]
  task_doc: dict[str, Any]
  group_hash: str
  config_hash: str
  run_seed: int = 0
  reset_distribution: dict[str, Any] | None = None
  envelope: dict[str, Any] | None = None
  normalizer: dict[str, list[float]] | None = None
  cpu_set: tuple[int, ...] = ()
  artifact_root: str | None = None
  collect_fragments: bool = False
  curriculum_version: int = 0
  num_slots: int | None = None
  telemetry: dict[str, Any] = field(default_factory=dict)  # TelemetryConfig (logs/traces/metrics prefix)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> WorkerSpec:
    d = dict(d)
    d["cpu_set"] = tuple(d.get("cpu_set", ()))
    return cls(**d)


class CreditTable:
  """Output credits: FREE -> RESERVED -> WRITING -> PUBLISHED -> ACKED -> FREE (B.1)."""

  FREE, RESERVED, WRITING, PUBLISHED, ACKED, ABORTED, LOST = range(7)

  def __init__(self, frames: int, fragments: int):
    self._next = 0
    self.state: dict[int, int] = {}
    self.kind: dict[int, str] = {}
    self.need: dict[int, int] = {}
    self.acked: dict[int, int] = {}
    self.capacity = {"frame": frames, "fragment": fragments}

  def in_use(self, kind: str) -> int:
    return sum(1 for c, k in self.kind.items() if k == kind and self.state[c] != self.FREE)

  def reserve(self, kind: str) -> int | None:
    if self.in_use(kind) >= self.capacity[kind]:
      return None
    self._next += 1
    cid = self._next
    self.state[cid], self.kind[cid], self.acked[cid] = self.RESERVED, kind, 0
    return cid

  def writing(self, cid: int) -> None:
    self.state[cid] = self.WRITING

  def publish(self, cid: int, consumers: int) -> None:
    self.state[cid], self.need[cid] = self.PUBLISHED, consumers

  def release(self, cid: int) -> None:  # RESERVED -> FREE before mutation, or ABORTED
    self._free(cid)

  def ack(self, cid: int, consumer_bit: int) -> bool:
    if self.state.get(cid) != self.PUBLISHED:
      return False  # duplicate or unknown ACKs never increase credits
    self.acked[cid] |= consumer_bit
    if self.acked[cid] & self.need[cid] == self.need[cid]:
      self._free(cid)
      return True
    return False

  def _free(self, cid: int) -> None:
    for d in (self.state, self.kind, self.need, self.acked):
      d.pop(cid, None)


class WorkerFenced(RuntimeError):
  pass


def _rss_bytes() -> int:
  if sys.platform.startswith("linux"):
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) * 1024
  r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return int(r if sys.platform == "darwin" else r * 1024)


class SimulationWorker:
  """Create with ``make_worker_class(...)`` / ``SimulationWorker.options``; see
  raysimbatch.control.placement for the resource options derived from config."""

  def __init__(self, spec: dict[str, Any], manifest_uri: str):
    self.spec = WorkerSpec.from_dict(spec)
    self.manifest_uri = manifest_uri
    self.cfg = WorkerConfig.from_obj(self.spec.worker)
    self.failure = FailureConfig.from_obj(self.spec.failure)
    self.flow = FlowConfig.from_obj(self.spec.flow)
    self.rollout = RolloutConfig.from_obj(self.spec.rollout)
    self.group = ModelGroup.from_obj(self.spec.group)
    self.N = self.spec.num_slots or self.cfg.envs
    self.state = WorkerState.CREATED
    self.generation = 0
    self.routing_epoch = 0
    self.lease_expires_ns = 0
    self.core: MJBatchCore | None = None
    self.policy: LocalPolicyRuntime | None = None
    self.policy_version = 0
    self.behavior_hash = b"\0" * 32  # policy that produced the most recent actions
    self.curriculum_version = self.spec.curriculum_version
    self._lock = threading.Lock()  # credits/cache/intent; never held across native calls
    self._fenced_generation = -1
    self._drain_deadline_ns: int | None = None
    self._active_command: str | None = None
    self._active_since_ns = 0
    self._cache: OrderedDict[bytes, dict[str, Any]] = OrderedDict()
    self._evicted: OrderedDict[bytes, None] = OrderedDict()
    self._credits = CreditTable(self.flow.inflight_frames_per_worker, self.flow.fragment_slabs_per_worker)
    self._credit_refs: dict[int, list[ray.ObjectRef]] = {}
    self.lost_rows = 0
    self._native_calls = 0
    self.aborted_episodes: list[tuple[int, int]] = []
    self.affinity = AffinityPlan(self.cfg.affinity, self.spec.cpu_set, self.cfg.threads)
    self.affinity_report: dict[str, Any] = {}
    telemetry.setup("raysimbatch-simulation", self.spec.telemetry)
    self.metrics = telemetry.exporter(mode=self.spec.mode, model_family=self.group.name, actor_role="simulation")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    self.run_id_bytes = bytes.fromhex(self.spec.run_id)

  # --------------------------------------------------------------- lifecycle

  @ray.method(concurrency_group="mutate")
  def initialize(
    self,
    model_blob: np.ndarray,
    policy_blob: np.ndarray | None,
    lease: dict[str, Any],
    policy_manifest: str | None = None,
  ) -> dict[str, Any]:
    """Build the shard from immutable asset bytes (auto-dereferenced ObjectRefs)."""
    self.state = WorkerState.LOADING
    self.generation = int(lease["epoch"])
    self.lease_expires_ns = int(lease["expires_unix_ns"])
    self.routing_epoch = int(lease.get("routing_epoch", 0))
    model = from_mjb(model_blob)
    meta = ModelMetadata.from_model(model)
    contract = AdapterContract.from_obj(self.spec.contract)
    binding = bind_task(self.spec.task_doc, meta)
    norm = None
    if self.spec.normalizer:
      norm = Normalizer(np.asarray(self.spec.normalizer["mean"]), np.asarray(self.spec.normalizer["var"]))
    compiled = compile_adapter(contract, meta, num_commands=binding.num_commands, normalizer=norm,
                               envelope=self.spec.envelope)
    self.state = WorkerState.WARMING
    if self.cfg.affinity == "numa_exclusive" and not self.spec.cpu_set:
      # Lease physical cores from this node's agent (never from Ray logical IDs).
      from raysimbatch.control.assets import call_node_helper
      from raysimbatch.worker.affinity_agent import agent_for_node

      ctx = ray.get_runtime_context()
      cpus = call_node_helper(lambda: agent_for_node(ctx.get_node_id(), namespace=ctx.namespace),
                              lambda a: a.lease.remote(self.spec.actor_name, self.cfg.threads, os.getpid()))
      self.affinity = AffinityPlan(self.cfg.affinity, tuple(cpus), self.cfg.threads)
    self.affinity.before_construction()
    factory = None
    if self.cfg.native_backend == "scalar":
      from raysimbatch.worker.scalar import ScalarHandle

      factory = ScalarHandle
    self.core = MJBatchCore(
      model, meta, compiled, binding,
      CoreSpec(
        num_slots=self.N,
        threads=self.cfg.threads,
        decimation=self.group.decimation,
        timestep_ns=self.group.timestep_ns,
        keyframe=self.group.initial_keyframe,
        max_control_steps=self.group.max_control_steps,
        forward_at_control_boundary=self.cfg.forward_at_control_boundary,
        fuse_substeps=self.cfg.fuse_substeps,
        health=self.group.health,
        run_seed=self.spec.run_seed,
        randomization=self.group.randomization,
        reset_distribution=self.spec.reset_distribution,
      ),
      handle_factory=factory,
    )
    self.affinity_report = self.affinity.after_construction().as_dict()
    self.metrics.count("affinity_violations_total", len(self.affinity_report.get("violations", ())))
    if policy_blob is not None:
      if policy_manifest is None:
        raise ValueError("a policy executable needs its bundle manifest")
      self._load_policy(policy_manifest, policy_blob)
    self.component_names = list(binding.component_names)
    self.schema_hash = bytes.fromhex(compiled.adapter_hash)
    self.group_hash = bytes.fromhex(self.spec.group_hash)
    self.task_hash = binding.task_hash
    self.last_obs = np.zeros((self.N, compiled.obs_dim), np.float32)
    self.rec_shapes = {s.name: s.shape for s in self.policy.bundle.recurrent} if self.policy else {}
    self.recurrent = {k: np.zeros((self.N, *shape), np.float32) for k, shape in self.rec_shapes.items()}
    self.action_keys = np.zeros((self.N, 2), np.uint64)
    self.fragments = None
    if self.spec.collect_fragments:
      self.fragments = FragmentBuilder(
        self.run_id_bytes, self.schema_hash, self.N, compiled.obs_dim, compiled.num_actions,
        binding.num_components, self.rollout.fragment_steps, self.rec_shapes,
      )
    self.eval_ids = np.zeros((self.N, 2), np.uint64)  # (unit seed, ordinal) when evaluating
    self._smoke_test()
    self.state = WorkerState.READY
    return {
      "actor_name": self.spec.actor_name,
      "generation": self.generation,
      "native_abi": self.core.compat.abi_fingerprint,
      "mjbatch": self.core.compat.mjbatch_version,
      "mujoco": self.core.compat.mujoco_version,
      "obs_dim": compiled.obs_dim,
      "action_dim": compiled.num_actions,
      "schema_hash": compiled.adapter_hash,
      "task_hash": binding.task_hash,
      "num_components": binding.num_components,
      "component_names": list(binding.component_names),
      "affinity": self.affinity_report,
      "memory": {**self.core.memory_census(), "rss_bytes": _rss_bytes()},
      "threads": self.core.h.num_threads,
      "pid": os.getpid(),
      "node_id": ray.get_runtime_context().get_node_id(),
      "policy_hash": self.policy.policy_hash if self.policy else None,
    }

  def _smoke_test(self) -> None:
    """Known-state check: the MJBatch probes already ran in check_environment();
    here the shard's own model must step slot 0 finitely from its initial state."""
    core = self.core
    assert core is not None
    s = np.array([0], np.int32)
    snap = core.save_state(s)
    r = core.reset_exact(s, np.zeros(1, np.uint64), np.zeros(1, np.uint64),
                         (np.zeros(1, np.uint64), np.zeros(1, np.uint64)))
    if not r.ok.all():
      raise RuntimeError(f"smoke test reset failed: {r.message}")
    out = core.advance_control(s, np.zeros((1, core.A), np.float32))
    if out.status[0] not in (SlotStatus.OK, SlotStatus.TERMINATED, SlotStatus.TRUNCATED):
      raise RuntimeError(f"smoke test step failed: status={out.status[0]} fault={out.fault[0]}")
    core.load_state(s, snap)

  def _load_policy(self, manifest: str, blob: np.ndarray | bytes) -> None:
    bundle = PolicyBundle.from_json(manifest)
    rt = LocalPolicyRuntime(bundle, blob, device="cpu", threads=self.cfg.torch_threads)
    old, self.policy = self.policy, rt
    self.policy_version = bundle.version
    if getattr(self, "fragments", None) is None:
      self.behavior_hash = bytes.fromhex(bundle.policy_hash)
    # Otherwise the next command cuts buffered fragments and publishes them (never here:
    # there is no envelope to carry them).
    if old is not None:
      old.close()

  @ray.method(concurrency_group="mutate")
  def load_policy(self, manifest: str, executable: np.ndarray) -> dict[str, Any]:
    """CPU-local policy update at a command boundary (fragments record the version)."""
    self._check_mutation_allowed(self.generation)
    if self.fragments is not None and self.policy is not None and self.policy.bundle.recurrent:
      raise ValueError("recurrent policies update only at episode boundaries (use reset)")
    self._load_policy(manifest, executable)
    assert self.policy is not None
    return {"policy_hash": self.policy.policy_hash, "version": self.policy_version}

  @ray.method(concurrency_group="mutate")
  def close(self) -> None:
    self.state = WorkerState.STOPPED
    if self.core is not None:
      self.core.close()
    self.core = None

  # ----------------------------------------------------------------- control

  @ray.method(concurrency_group="control")
  def health(self) -> dict[str, Any]:
    core = self.core
    rss = _rss_bytes()
    self.metrics.gauge("worker_rss_bytes", float(rss))
    dropped, _ = telemetry.log_drops()  # telemetry degradation, reported where it is polled
    self.metrics.count("log_dropped_total", dropped - getattr(self, "_log_dropped", 0))
    self._log_dropped = dropped
    spans = tracing.drops()
    self.metrics.count("traces_dropped_total", spans - getattr(self, "_traces_dropped", 0))
    self._traces_dropped = spans
    self._resource_gauges()
    return {
      "actor_name": self.spec.actor_name,
      "state": int(self.state),
      "generation": self.generation,
      "fenced_generation": self._fenced_generation,
      "lease_expires_unix_ns": self.lease_expires_ns,
      "progress": core.progress if core is not None else 0,
      "last_native_monotonic_ns": core.last_native_ns if core is not None else 0,
      "active_command": self._active_command,
      "active_age_s": (time.monotonic_ns() - self._active_since_ns) * 1e-9 if self._active_command else 0.0,
      "cache": len(self._cache),
      "credits_frame": self._credits.in_use("frame"),
      "credits_fragment": self._credits.in_use("fragment"),
      "rss_bytes": rss,
      "draining": self._drain_deadline_ns is not None,
      "lost_rows": self.lost_rows,
      "now_monotonic_ns": time.monotonic_ns(),
      "pid": os.getpid(),
    }

  def _resource_gauges(self) -> None:
    """CPU placement quality and private memory (design §11.1), sampled when the node
    watchdog polls health. cpu_busy_ratio: this process's CPU time (MJBatch pool and all
    other threads) per wall second, per leased thread; it can exceed 1 slightly, from
    Ray's own threads. The /proc readings are Linux only and rate-limited."""
    now, cpu = time.monotonic(), time.process_time()
    last = getattr(self, "_cpu_at", None)
    if last is not None and now > last[0]:
      self.metrics.gauge("cpu_busy_ratio", (cpu - last[1]) / ((now - last[0]) * max(1, self.cfg.threads)))
    self._cpu_at = (now, cpu)
    if now - getattr(self, "_mem_at", -1e9) >= 10.0:
      self._mem_at = now
      private = procfs.read_private_bytes()
      if private is not None:
        self.metrics.gauge("worker_private_bytes", float(private))
    if now - getattr(self, "_numa_at", -1e9) >= 30.0:  # numa_maps walks every mapping
      self._numa_at = now
      ratio = procfs.read_numa_remote_ratio(self.affinity.cpus)
      if ratio is not None:
        self.metrics.gauge("numa_remote_ratio", ratio)

  @ray.method(concurrency_group="control")
  def ack(self, credit_ids: list[int], consumer_bit: int = 1) -> int:
    """Consumers acknowledge ownership/commit; freed credits drop cached refs."""
    freed = 0
    with self._lock:
      for cid in credit_ids:
        if self._credits.ack(int(cid), consumer_bit):
          freed += 1
          self._credit_refs.pop(int(cid), None)
      done = [k for k, env in self._cache.items() if all(c not in self._credits.state for c in env["credit_ids"])]
      for k in done:
        self._cache.pop(k)
        self._remember_evicted(k)
    return freed

  @ray.method(concurrency_group="control")
  def fence(self, generation: int) -> None:
    """A newer owner exists: reject every mutation at or below ``generation``."""
    with self._lock:
      self._fenced_generation = max(self._fenced_generation, int(generation))
      if self.generation <= self._fenced_generation:
        self.state = WorkerState.FENCED

  @ray.method(concurrency_group="control")
  def set_routing_epoch(self, epoch: int) -> int:
    """Route changes (migration) bump the actor's routing epoch; commands issued under
    an older table are rejected before mutation. Monotonic."""
    with self._lock:
      self.routing_epoch = max(self.routing_epoch, int(epoch))
      return self.routing_epoch

  @ray.method(concurrency_group="control")
  def drain(self, deadline_unix_ns: int) -> dict[str, Any]:
    with self._lock:
      if self._drain_deadline_ns is None:
        self._drain_deadline_ns = int(deadline_unix_ns)
        if self.state in (WorkerState.READY, WorkerState.BUSY):
          self.state = WorkerState.DRAINING
      pending = self._active_command is not None or bool(self._credits.state)
      if not pending and self.state == WorkerState.DRAINING:
        self.state = WorkerState.STOPPED
      return {
        "pending": pending,
        "committed_output_ids": [],
        "aborted_episode_ids": [list(x) for x in self.aborted_episodes],
        "final_snapshot_uri": None,
        "outstanding_credits": len(self._credits.state),
      }

  # --------------------------------------------------------- mutation guards

  def _check_mutation_allowed(self, expected_generation: int) -> None:
    with self._lock:
      if self.core is None or self.state in (WorkerState.CREATED, WorkerState.LOADING):
        raise WorkerFenced("worker is not initialized (restarted actors rerun only the constructor)")
      if self.generation <= self._fenced_generation or self.state == WorkerState.FENCED:
        raise WorkerFenced(f"generation {self.generation} is fenced")
      if expected_generation != self.generation:
        raise WorkerFenced(f"stale generation {expected_generation} != {self.generation}")
      if self._drain_deadline_ns is not None:
        raise WorkerFenced("draining: no new mutations")

  def _begin(self, command_id: bytes) -> None:
    self._active_command = command_id.hex()
    self._active_since_ns = time.monotonic_ns()
    self.state = WorkerState.BUSY

  def _end(self) -> None:
    self._active_command = None
    if self.state == WorkerState.BUSY:
      self.state = WorkerState.READY

  def _remember_evicted(self, cid: bytes) -> None:
    self._evicted[cid] = None
    while len(self._evicted) > 4096:
      self._evicted.popitem(last=False)

  def _dedup(self, command_id: bytes) -> dict[str, Any] | None:
    with self._lock:
      if command_id in self._cache:
        return self._cache[command_id]
      if command_id in self._evicted:
        return self._rejection(command_id, None, FaultCode.REPLAY_REQUIRED)
    return None

  def _rejection(self, command_id: bytes, slots: np.ndarray | None, code: FaultCode, message: str = "") -> dict:
    n = 0 if slots is None else slots.size
    st = P.status_table(n)
    if n:
      st["slot"][:] = slots
      st["status"][:] = SlotStatus.REJECTED
      st["fault_code"][:] = code
    return {
      "command_id": command_id,
      "actor_name": self.spec.actor_name,
      "generation": self.generation,
      "frame_ref": None,
      "fragment_refs": (),
      "status": st,
      "episodes": [],
      "credit_ids": [],
      "rejected": int(code),
      "message": message,
      "progress": self.core.progress if self.core else 0,
    }

  # ------------------------------------------------------------------- reset

  @ray.method(concurrency_group="mutate", enable_task_events=False)
  def reset(self, request: dict[str, Any]) -> dict[str, Any]:
    """Start new episodes. request: command_id, generation, slots, env_id, episode_id,
    key_a, key_b (seed identity), optional parameters (ParameterFrame wire), optional
    eval ids."""
    cid = request["command_id"]
    cached = self._dedup(cid)
    if cached is not None:
      return cached
    try:
      self._check_mutation_allowed(int(request["generation"]))
    except WorkerFenced as exc:
      return self._rejection(cid, np.asarray(request["slots"], np.int32), FaultCode.STALE_GENERATION, str(exc))
    core = self.core
    assert core is not None
    slots = np.ascontiguousarray(request["slots"], np.int32)
    credit = self._reserve("frame")
    if credit is None:
      return self._rejection(cid, slots, FaultCode.CREDIT_EXHAUSTED)
    self._begin(cid)
    try:
      if self.fragments is not None:
        # An interrupted episode's open segment ends as an infrastructure truncation:
        # the new episode's observation must never become its bootstrap.
        for s in slots.tolist():
          self.fragments.close_open(s)
      params = None
      if request.get("parameters") is not None:
        params = P.ParameterFrame.from_wire(request["parameters"]).values
      env_id = np.asarray(request["env_id"], np.uint64)
      episode_id = np.asarray(request["episode_id"], np.uint64)
      key_a = np.asarray(request.get("key_a", env_id), np.uint64)
      key_b = np.asarray(request.get("key_b", episode_id), np.uint64)
      self._credits.writing(credit)
      t0 = time.perf_counter()
      r = core.reset_exact(slots, env_id, episode_id, (key_a, key_b), parameters=params)
      self.metrics.observe("reset_seconds", time.perf_counter() - t0)
      if core.set_const_s is not None:
        self.metrics.observe("set_const_seconds", core.set_const_s)
      self.last_obs[slots] = r.observation
      for k in self.recurrent:
        self.recurrent[k][slots] = 0.0
      self.action_keys[slots] = derive_keys(self.spec.run_seed, self.task_hash, key_a, key_b, "action")
      self.eval_ids[slots, 0], self.eval_ids[slots, 1] = key_a, key_b
      if self.fragments is not None:
        for j, s in enumerate(slots.tolist()):
          if r.ok[j]:
            self.fragments.begin(s, int(env_id[j]), int(episode_id[j]), 0,
                                 {k: v[s] for k, v in self.recurrent.items()})
      st = P.status_table(slots.size)
      st["slot"][:] = slots
      st["env_id"][:] = env_id
      st["episode_id"][:] = episode_id
      st["status"][:] = np.where(r.ok, SlotStatus.OK, SlotStatus.NATIVE_FAILURE)
      st["control_seq"][:] = 0
      st["fault_code"][:] = r.fault
      flags = np.where(r.ok, Flag.VALID | Flag.RESET, Flag.ABORTED).astype(np.uint8)
      frame = self._frame(slots, r.observation, np.zeros(slots.size, np.float32), flags, r.fault,
                          np.zeros(slots.size, np.uint64), np.zeros(slots.size, np.int64))
      return self._publish(cid, credit, frame, [], st, [], {"randomization_hash": r.randomization_hash})
    except BaseException:
      self._abort_credit(credit)
      raise
    finally:
      self._end()

  # ----------------------------------------------------------------- advance

  def _validate(
    self, cmd: P.AdvanceCommand, group: P.ActionGroup
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-level validation before any mutation. Returns (accepted mask, fault codes,
    actions aligned with cmd.slots, action-frame row index per command slot)."""
    core = self.core
    assert core is not None
    n = cmd.slots.size
    fault = np.zeros(n, np.uint16)
    b = core.book
    if cmd.routing_epoch != self.routing_epoch:
      fault[:] = FaultCode.STALE_ROUTING_EPOCH
    if time.time_ns() > cmd.deadline_unix_ns:
      fault[fault == 0] = FaultCode.DEADLINE_EXCEEDED
    if group.schema_hash != self.schema_hash or group.group_hash != self.group_hash:
      fault[fault == 0] = FaultCode.SCHEMA_MISMATCH
    if group.policy_hash != cmd.policy_hash:
      fault[fault == 0] = FaultCode.POLICY_MISMATCH
    if group.action.shape[1] != core.A:
      fault[fault == 0] = FaultCode.SCHEMA_MISMATCH
      return fault == 0, fault, np.zeros((n, core.A), np.float32), np.zeros(n, np.int64)
    # Map action rows to command slots.
    rows = np.full(n, -1, np.int64)
    order = np.argsort(group.slot, kind="stable")
    ss = group.slot[order]
    if np.any(np.diff(ss) == 0):
      dup = np.isin(cmd.slots, ss[:-1][np.diff(ss) == 0])
      fault[(fault == 0) & dup] = FaultCode.DUPLICATE_ROW
    pos = np.searchsorted(ss, cmd.slots)
    hit = (pos < ss.size) & (ss[np.minimum(pos, ss.size - 1)] == cmd.slots) if ss.size else np.zeros(n, bool)
    rows[hit] = order[pos[hit]]
    fault[(fault == 0) & ~hit] = FaultCode.UNKNOWN_ENV
    r = np.maximum(rows, 0)
    s = cmd.slots
    ok = fault == 0
    chk = [
      (group.env_id[r] != b["env_id"][s], FaultCode.UNKNOWN_ENV),
      (group.slot_generation[r] != self.generation, FaultCode.STALE_GENERATION),
      (group.episode_id[r] != b["episode_id"][s], FaultCode.EPISODE_MISMATCH),
      (cmd.episode_id != b["episode_id"][s], FaultCode.EPISODE_MISMATCH),
      (group.expected_control_seq[r] != b["control_seq"][s, 0], FaultCode.SEQUENCE_MISMATCH),
      (cmd.control_seq != b["control_seq"][s, 0], FaultCode.SEQUENCE_MISMATCH),
      ((b["slot_state"][s] != SlotState.READY) & (b["slot_state"][s] != SlotState.WAITING_ACTION),
       FaultCode.SLOT_NOT_READY),
      (~np.isfinite(group.action[r]).all(axis=1), FaultCode.NONFINITE_ACTION),
    ]
    for bad, code in chk:
      fault[ok & bad] = code
      ok = fault == 0
    actions = np.ascontiguousarray(group.action[r], np.float32) if group.rows else np.zeros((n, core.A), np.float32)
    return ok, fault, actions, r

  @ray.method(concurrency_group="mutate", enable_task_events=False)
  def advance(self, command: dict[str, Any], actions: dict[str, Any]) -> dict[str, Any]:
    cmd = _command(command)
    cached = self._dedup(cmd.command_id)
    if cached is not None:
      return cached
    try:
      self._check_mutation_allowed(cmd.expected_generation)
    except WorkerFenced as exc:
      return self._rejection(cmd.command_id, cmd.slots, FaultCode.STALE_GENERATION, str(exc))
    frame = P.ActionFrame.from_wire(actions)
    if len(frame.groups) != 1 or frame.run_id != self.run_id_bytes:
      return self._rejection(cmd.command_id, cmd.slots, FaultCode.SCHEMA_MISMATCH, "one group of this run")
    group = frame.groups[0]
    ok, fault, act, rows = self._validate(cmd, group)
    credit = self._reserve("frame")
    if credit is None:
      return self._rejection(cmd.command_id, cmd.slots, FaultCode.CREDIT_EXHAUSTED)
    self._begin(cmd.command_id)
    try:
      pre = self._flush_ready(cmd.slots[ok])
      if self.fragments is not None:
        full = self.fragments.full(cmd.slots)
        fault[ok & full] = FaultCode.CREDIT_EXHAUSTED
        ok &= ~full
      pos = np.flatnonzero(ok)
      slots = cmd.slots[ok]
      self._credits.writing(credit)
      cut = pre + (self._set_behavior(group.policy_hash, slots) if slots.size else [])
      logp = group.behavior_logp[rows] if group.rows else np.zeros(cmd.slots.size, np.float32)
      pver = group.policy_version[rows] if group.rows else np.zeros(cmd.slots.size, np.uint64)
      for name, arr in group.recurrent.items():
        if name in self.recurrent and slots.size:
          self.recurrent[name][slots] = arr[rows[ok]]
      out = self._native(slots, act[ok]) if slots.size else None
      return self._finish_step(cmd.command_id, credit, cmd.slots, ok, fault, pos, out, act, logp, pver,
                               extra_fragments=cut)
    except BaseException:
      self._abort_credit(credit)
      raise
    finally:
      self._end()

  def _finish_step(self, command_id, credit, all_slots, ok, fault, pos, out: AdvanceResult | None, act, logp,
                   pver, extra_fragments=None, episodes=None):
    core = self.core
    assert core is not None
    n = all_slots.size
    st = P.status_table(n)
    st["slot"][:] = all_slots
    st["env_id"][:] = core.book["env_id"][all_slots]
    st["episode_id"][:] = core.book["episode_id"][all_slots]
    st["status"][~ok] = SlotStatus.REJECTED
    st["fault_code"][~ok] = fault[~ok]
    obs = self.last_obs[all_slots].copy()
    reward = np.zeros(n, np.float32)
    flags = np.zeros(n, np.uint8)
    seq = core.book["control_seq"][all_slots, 0].copy()
    tns = core.book["sim_time_ns"][all_slots, 0].copy()
    episodes = list(episodes or [])
    fragments = list(extra_fragments or [])
    if out is not None:
      st["status"][pos] = out.status
      st["fault_code"][pos] = out.fault
      st["completed_substeps"][pos] = out.completed_substeps
      st["failed_substep"][pos] = out.failed_substep
      st["warning_delta"][pos] = out.warning_delta
      st["control_steps"][pos] = 1
      obs[pos], reward[pos], flags[pos] = out.observation, out.reward, out.flags
      seq[pos], tns[pos] = out.control_seq, out.sim_time_ns
      if self.fragments is not None:
        self.fragments.append(
          out.slots, self.last_obs[out.slots], out.action, out.applied_action, out.reward, out.components,
          logp[pos], out.flags, out.control_seq - 1, pver[pos],
          np.full(out.slots.size, self.curriculum_version, np.uint64),
          self.group.timestep_ns * self.group.decimation, out.observation,
        )
      self.last_obs[out.slots] = out.observation
      episodes += self._episode_records(out)
      if self.fragments is not None:
        ready = self.fragments.ready(out.slots)
        if ready.size:
          fragments += self._flush_fragments(ready)
    st["control_seq"][:] = seq
    frame = self._frame(all_slots, obs, reward, flags, st["fault_code"], seq, tns, logp, pver)
    return self._publish(command_id, credit, frame, fragments, st, episodes, {})

  def _set_behavior(self, policy_hash: bytes, slots: np.ndarray | None) -> list[tuple[P.Fragment, int]]:
    """A behavior-policy change cuts buffered fragments first: a fragment's header hash
    must describe every row in it."""
    out: list[tuple[P.Fragment, int]] = []
    frags = getattr(self, "fragments", None)
    if policy_hash != self.behavior_hash and frags is not None:
      pending = np.flatnonzero(frags.count > 0).astype(np.int32)
      if pending.size:
        out = self._flush_fragments(pending)
        if not out:
          raise WorkerFenced("policy change needs a free fragment slab: retry after ACKs")
    self.behavior_hash = policy_hash
    if frags is not None:
      frags.policy_hash = policy_hash
    return out

  def _flush_ready(self, slots: np.ndarray) -> list[tuple[P.Fragment, int]]:
    """Flush buffers that reached ``fragment_steps`` (found: flushing only after a step
    livelocked slots whose buffers filled while every fragment credit was in flight)."""
    if self.fragments is None or slots.size == 0:
      return []
    ready = self.fragments.ready(slots)
    return self._flush_fragments(ready) if ready.size else []

  def _flush_fragments(self, slots: np.ndarray) -> list[tuple[P.Fragment, int]]:
    """Cut a fragment only when a completed-slab credit is free; otherwise keep
    buffering (the bounded builder then rejects further steps for full slots)."""
    assert self.fragments is not None  # callers flush only when fragments are built
    fc = self._reserve("fragment")
    if fc is None:
      return []
    frag = self.fragments.flush(slots, {int(s): self.last_obs[s] for s in slots},
                                {int(s): {k: v[s] for k, v in self.recurrent.items()} for s in slots})
    if frag is None:
      self._abort_credit(fc)
      return []
    return [(frag, fc)]

  def _episode_records(self, out: AdvanceResult) -> list[dict[str, Any]]:
    core = self.core
    assert core is not None
    ended = np.flatnonzero(out.status != SlotStatus.OK)
    recs = []
    now = time.time_ns()
    for j in ended.tolist():
      s = int(out.slots[j])
      stt = int(out.status[j])
      aborted = stt in (SlotStatus.NATIVE_FAILURE, SlotStatus.HEALTH_FAILURE)
      m = out.metrics[j]
      failed_task = bool(m[0] == 1.0) if np.isfinite(m[0]) else True
      success = None if aborted else (not failed_task)
      if aborted:
        outcome = "health_failure"
      elif success:
        outcome = "success"
      elif stt == SlotStatus.TERMINATED:
        outcome = "task_failure"
      else:
        outcome = "time_limit"
      evals = int(core.adapter.state["effort_evals"][s, 0])
      recs.append({
        "slot": s,
        "env_id": int(core.book["env_id"][s]),
        "episode_id": int(core.book["episode_id"][s]),
        "key_a": int(self.eval_ids[s, 0]),
        "key_b": int(self.eval_ids[s, 1]),
        "status": stt,
        "outcome": outcome,
        "success": success,
        "fault_code": int(out.fault[j]),
        "return_sum": float(core.book["return_sum"][s]) if not aborted or out.control_seq[j] > 1 else None,
        "control_steps": int(out.control_seq[j]),
        "physics_steps": int(core.book["physics_steps"][s]),
        "simulated_seconds": float(core.book["sim_time_ns"][s, 0]) * 1e-9,
        "started_ns": int(core.book["episode_started_ns"][s]),
        "finished_ns": now,
        "tracking_rmse": float(m[1]) if np.isfinite(m[1]) else None,
        "energy_joules": float(core.adapter.state["energy"][s, 0]),
        "saturation_fraction": float(core.adapter.state["saturated"][s, 0]) / evals if evals else None,
        "randomization_hash": core.book["randomization_hash"][s].tobytes().hex(),
        "reward_components": {
          name: float(core.book["component_sums"][s, i]) for i, name in enumerate(self.component_names)
        },
        "generation": self.generation,
        "policy_hash": self.policy.policy_hash if self.policy else None,
      })
      if aborted:
        self.aborted_episodes.append((int(core.book["env_id"][s]), int(core.book["episode_id"][s])))
    return recs

  def _frame(self, slots, obs, reward, flags, fault, seq, tns, logp=None, pver=None) -> P.HeteroFrame:
    core = self.core
    assert core is not None
    n = slots.size
    g = P.GroupFrame(
      group_hash=self.group_hash,
      schema_hash=self.schema_hash,
      policy_hash=self.behavior_hash,
      env_id=np.ascontiguousarray(core.book["env_id"][slots]),
      episode_id=np.ascontiguousarray(core.book["episode_id"][slots]),
      control_seq=np.ascontiguousarray(seq, np.uint64),
      sim_time_ns=np.ascontiguousarray(tns, np.int64),
      slot_generation=np.full(n, self.generation, np.uint64),
      slot=np.ascontiguousarray(slots, np.int32),
      observation=np.ascontiguousarray(obs, np.float32),
      reward=np.ascontiguousarray(reward, np.float32),
      discount=np.where(flags & Flag.TERMINATED, 0.0, 1.0).astype(np.float32),
      behavior_logp=np.zeros(n, np.float32) if logp is None else np.ascontiguousarray(logp, np.float32),
      flags=np.ascontiguousarray(flags, np.uint8),
      policy_version=np.full(n, self.policy_version, np.uint64) if pver is None else np.ascontiguousarray(pver, np.uint64),
      curriculum_version=np.full(n, self.curriculum_version, np.uint64),
      fault_code=np.ascontiguousarray(fault, np.uint16),
      recurrent={k: np.ascontiguousarray(v[slots]) for k, v in self.recurrent.items()},
      sample_keys=np.ascontiguousarray(self.action_keys[slots]),
    )
    return P.HeteroFrame(self.run_id_bytes, P.new_id(), self.generation, self.routing_epoch, time.time_ns(),
                         P.new_id(), (g,))

  def _reserve(self, kind: str) -> int | None:
    with self._lock:
      return self._credits.reserve(kind)

  def _abort_credit(self, credit: int) -> None:
    with self._lock:
      self._credits.release(credit)

  def _publish(self, command_id, credit, frame: P.HeteroFrame, fragments: list[tuple[P.Fragment, int]], status, episodes,
               extra, physics_steps: int | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    frame_ref = ray.put(frame.to_wire())  # the immutable snapshot; buffers are free after this
    frag_refs, frag_meta, frag_credits = [], [], []
    for frag, fc in fragments:
      frag_refs.append(ray.put(frag.to_wire()))
      frag_meta.append({"fragment_id": frag.fragment_id(), "digest": frag.digest(), "rows": frag.transitions,
                        "segments": frag.segments, "credit_id": fc})
      frag_credits.append(fc)
    publish_s = time.perf_counter() - t0
    frame_bytes = int(sum(v.nbytes for g in frame.groups for v in g.to_wire().values() if isinstance(v, np.ndarray)))
    self._record(status, episodes, publish_s, frame_bytes, physics_steps)
    with self._lock:
      self._credits.publish(credit, 1)
      self._credit_refs[credit] = [frame_ref]
      for fc, ref in zip(frag_credits, frag_refs, strict=True):
        self._credits.publish(fc, 1)
        self._credit_refs[fc] = [ref]
    env = {
      "command_id": command_id,
      "actor_name": self.spec.actor_name,
      "generation": self.generation,
      "frame_ref": frame_ref,
      "fragment_refs": tuple(frag_refs),
      "fragments": frag_meta,
      "status": status,
      "episodes": episodes,
      "credit_ids": [credit, *frag_credits],
      "frame_credit": credit,
      "fragment_credits": frag_credits,
      "rejected": 0,
      "message": "",
      "progress": self.core.progress if self.core is not None else 0,
      "publish_seconds": publish_s,
      "frame_bytes": frame_bytes,
      **extra,
    }
    with self._lock:
      self._cache[command_id] = env
    return env

  def _record(self, status, episodes, publish_s: float, frame_bytes: int, physics_steps: int | None) -> None:
    """Per-command metrics (never per substep): counters for completed work, publish
    cost, capacity and credit gauges (design §11.1)."""
    m = self.metrics
    m.count("control_steps_total", int(status["control_steps"].sum()))
    m.count("physics_steps_total", int(status["completed_substeps"].sum()) if physics_steps is None else physics_steps)
    warnings = int(status["warning_delta"].sum())
    if warnings:
      m.count("warning_total", warnings)
    health = int((status["status"] == SlotStatus.HEALTH_FAILURE).sum())
    if health:
      m.count("slot_health_failures_total", health)
    for e in episodes:
      m.count("episodes_total", outcome=e["outcome"])
      if e["outcome"] == "health_failure":
        m.count("aborted_episodes_total")
    m.observe("publish_seconds", publish_s)
    m.count("publish_bytes_total", frame_bytes)
    book = self.core.book["slot_state"] if self.core is not None else np.zeros(0)
    ready, waiting = int((book == SlotState.READY).sum()), int((book == SlotState.WAITING_ACTION).sum())
    m.gauge("slots_active", float(ready + waiting))
    m.gauge("slots_ready", float(ready))
    m.gauge("slots_waiting", float(waiting))
    with self._lock:  # ack() mutates the credit table from the control thread
      used = {kind: self._credits.in_use(kind) for kind in ("frame", "fragment")}
    for kind, n in used.items():
      m.gauge("credits_in_use", float(n), stage=kind)
      m.gauge("credits_available", float(self._credits.capacity[kind] - n), stage=kind)

  @ray.method(concurrency_group="mutate", enable_task_events=False)
  def current(self, command_id: bytes, slots: np.ndarray, generation: int) -> dict[str, Any]:
    """Publish the current observation of ``slots`` without mutating anything (the
    'request current observation' path after stale rejections or migration)."""
    cached = self._dedup(command_id)
    if cached is not None:
      return cached
    slots = np.ascontiguousarray(slots, np.int32)
    try:
      self._check_mutation_allowed(generation)
    except WorkerFenced as exc:
      return self._rejection(command_id, slots, FaultCode.STALE_GENERATION, str(exc))
    credit = self._reserve("frame")
    if credit is None:
      return self._rejection(command_id, slots, FaultCode.CREDIT_EXHAUSTED)
    core = self.core
    assert core is not None  # _check_mutation_allowed passed
    st = P.status_table(slots.size)
    st["slot"][:] = slots
    st["env_id"][:] = core.book["env_id"][slots]
    st["episode_id"][:] = core.book["episode_id"][slots]
    st["control_seq"][:] = core.book["control_seq"][slots, 0]
    ready = core.book["slot_state"][slots] == SlotState.READY
    st["status"][:] = np.where(ready, SlotStatus.OK, SlotStatus.REJECTED)
    flags = np.where(ready, Flag.VALID, 0).astype(np.uint8)
    frame = self._frame(slots, self.last_obs[slots], np.zeros(slots.size, np.float32), flags, st["fault_code"],
                        st["control_seq"], core.book["sim_time_ns"][slots, 0].copy())
    self._credits.writing(credit)
    return self._publish(command_id, credit, frame, [], st, [], {})

  # ------------------------------------------------------------------- local

  @ray.method(concurrency_group="mutate", enable_task_events=False)
  def run_local(self, command: dict[str, Any]) -> dict[str, Any]:
    """Up to ``max_control_steps`` local-policy control steps for ``slots``.

    Physics and inference alternate on this thread (the MJBatch pool is parked during
    inference). Yields early on deadline, cancellation, fragment-credit exhaustion or
    when every slot has ended (unless autoreset with pre-allocated episode ids)."""
    cid = command["command_id"]
    cached = self._dedup(cid)
    if cached is not None:
      return cached
    slots = np.ascontiguousarray(command["slots"], np.int32)
    try:
      self._check_mutation_allowed(int(command["expected_generation"]))
    except WorkerFenced as exc:
      return self._rejection(cid, slots, FaultCode.STALE_GENERATION, str(exc))
    if self.policy is None:
      return self._rejection(cid, slots, FaultCode.POLICY_MISMATCH, "no local policy loaded")
    if bytes(command["policy_hash"]) != bytes.fromhex(self.policy.policy_hash):
      return self._rejection(cid, slots, FaultCode.POLICY_MISMATCH)
    credit = self._reserve("frame")
    if credit is None:
      return self._rejection(cid, slots, FaultCode.CREDIT_EXHAUSTED)
    core = self.core
    assert core is not None
    self._begin(cid)
    try:
      self._credits.writing(credit)
      deadline = int(command["deadline_unix_ns"])
      max_steps = int(command.get("max_control_steps", self.cfg.local_control_steps))
      new_eps = list(np.asarray(command.get("episode_ids", ()), np.uint64).tolist())
      autoreset = bool(command.get("autoreset", False))
      episodes: list[dict[str, Any]] = []
      fragments = list(self._set_behavior(bytes.fromhex(self.policy.policy_hash), slots))
      steps = np.zeros(slots.size, np.uint32)
      last_status = P.status_table(slots.size)
      last_status["slot"][:] = slots
      ctx = ray.get_runtime_context()
      canceled = False
      physics = 0  # the status table keeps only each slot's last step; count every substep
      for _ in range(max_steps):
        st = core.book["slot_state"][slots]
        live = slots[(st == SlotState.READY) | (st == SlotState.WAITING_ACTION)]
        if self.fragments is not None and live.size:
          fragments += self._flush_ready(live)  # a full buffer must flush before it can step again
          live = live[~self.fragments.full(live)]
        if live.size == 0 or time.time_ns() > deadline:
          break
        try:
          canceled = ctx.is_canceled()
        except RuntimeError:
          canceled = False
        if canceled:
          break
        out, logp, pver = self._local_step(live)
        physics += int(out.completed_substeps.sum())
        pos = np.searchsorted(slots, live)
        steps[pos] += 1
        last_status["status"][pos] = out.status
        last_status["fault_code"][pos] = out.fault
        last_status["completed_substeps"][pos] = out.completed_substeps
        last_status["failed_substep"][pos] = out.failed_substep
        last_status["warning_delta"][pos] += out.warning_delta
        if self.fragments is not None:
          self.fragments.append(
            out.slots, self.last_obs[out.slots], out.action, out.applied_action, out.reward, out.components,
            logp, out.flags, out.control_seq - 1, pver, np.full(out.slots.size, self.curriculum_version, np.uint64),
            self.group.timestep_ns * self.group.decimation, out.observation,
          )
        self.last_obs[out.slots] = out.observation
        episodes += self._episode_records(out)
        ended = out.slots[out.status != SlotStatus.OK]
        if self.fragments is not None:
          ready = self.fragments.ready(live)
          if ready.size:
            fragments += self._flush_fragments(ready)
        if autoreset and ended.size and new_eps:
          take = ended[: len(new_eps)]
          ids = np.asarray([new_eps.pop(0) for _ in range(take.size)], np.uint64)
          env = core.book["env_id"][take]
          if self.fragments is not None:
            fragments += self._flush_fragments(take)
          r = core.reset_exact(take, env, ids, (env, ids))
          if core.set_const_s is not None:
            self.metrics.observe("set_const_seconds", core.set_const_s)
          self.last_obs[take] = r.observation
          for k in self.recurrent:
            self.recurrent[k][take] = 0.0
          self.action_keys[take] = derive_keys(self.spec.run_seed, self.task_hash, env, ids, "action")
          if self.fragments is not None:
            for j, s in enumerate(take.tolist()):
              if r.ok[j]:
                self.fragments.begin(s, int(env[j]), int(ids[j]), 0, {k: v[s] for k, v in self.recurrent.items()})
      if command.get("flush") and self.fragments is not None:
        ready = self.fragments.ready(slots, force=True)
        if ready.size:
          fragments += self._flush_fragments(ready)
      last_status["env_id"][:] = core.book["env_id"][slots]
      last_status["episode_id"][:] = core.book["episode_id"][slots]
      last_status["control_steps"][:] = steps
      seq = core.book["control_seq"][slots, 0].copy()
      last_status["control_seq"][:] = seq
      tns = core.book["sim_time_ns"][slots, 0].copy()
      flags = np.where(np.isin(core.book["slot_state"][slots], (SlotState.READY,)), Flag.VALID, 0).astype(np.uint8)
      frame = self._frame(slots, self.last_obs[slots], np.zeros(slots.size, np.float32), flags,
                          last_status["fault_code"], seq, tns)
      env = self._publish(cid, credit, frame, fragments, last_status, episodes,
                          {"canceled": canceled, "unused_episode_ids": new_eps,
                           "slot_state": core.book["slot_state"][slots].copy()}, physics_steps=physics)
      return env
    except BaseException:
      self._abort_credit(credit)
      raise
    finally:
      self._end()

  def _local_step(self, live: np.ndarray) -> tuple[AdvanceResult, np.ndarray, np.ndarray]:
    core = self.core
    pol = self.policy
    assert core is not None and pol is not None
    rec = {k: np.ascontiguousarray(v[live]) for k, v in self.recurrent.items()}
    res = pol.infer(self.last_obs[live], rec, sample_keys=self.action_keys[live],
                    sample_seq=core.book["control_seq"][live, 0])
    for k, v in res.recurrent.items():
      self.recurrent[k][live] = v
    out = self._native(live, res.action)
    pver = np.full(live.size, self.policy_version, np.uint64)
    return out, res.logp, pver

  def _native(self, slots: np.ndarray, actions: np.ndarray) -> AdvanceResult:
    """One control step of ``slots`` (all substeps + kernels): the native slice."""
    t0 = time.perf_counter()
    core = self.core
    assert core is not None
    with tracing.span("native_slice", rows=int(slots.size), cost_class=self.group.resource_pool):
      out = core.advance_control(slots, actions)
    self.metrics.observe("batch_step_seconds", time.perf_counter() - t0)
    self._native_calls += 1
    if self._native_calls % 16 == 0:
      # Sampled 1 in 16: one metric call costs ~1.8 us, 5-8% of a light model's serial
      # work per control step (observe_seconds counts are therefore 1/16 of the steps).
      self.metrics.observe("observe_seconds", core.observe_s)
    return out

  # --------------------------------------------------------------- snapshots

  @ray.method(concurrency_group="mutate")
  def snapshot(self, slots: np.ndarray, generation: int) -> dict[str, Any]:
    """Quiescent snapshot of whole slot state; stored durably when an artifact root is
    configured, else returned inline (tests)."""
    self._check_mutation_allowed(generation)
    core = self.core
    assert core is not None  # _check_mutation_allowed passed
    s = np.ascontiguousarray(slots, np.int32)
    snap: dict[str, Any] = core.save_state(s)  # also passed to np.savez as keywords
    snap["worker/last_obs"] = self.last_obs[s].copy()
    for k, v in self.recurrent.items():
      snap[f"worker/recurrent/{k}"] = v[s].copy()
    snap["worker/action_keys"] = self.action_keys[s].copy()
    snap["worker/eval_ids"] = self.eval_ids[s].copy()
    meta = {
      "format": "rsb-snapshot/1",
      "native_abi": core.compat.abi_fingerprint,
      "schema_hash": self.schema_hash.hex(),
      "group_hash": self.group_hash.hex(),
      "policy_hash": self.policy.policy_hash if self.policy else None,
      "slots": s.tolist(),
      "cpu_class": _cpu_class(),
    }
    buf = io.BytesIO()
    np.savez(buf, __meta__=np.frombuffer(json.dumps(meta).encode(), np.uint8), **snap)
    blob = buf.getvalue()
    if self.spec.artifact_root:
      from raysimbatch.storage.blob import open_blob_store

      art = open_blob_store(self.spec.artifact_root).put(blob, "application/x-rsb-snapshot")
      return {"artifact": asdict(art), "meta": meta}
    return {"inline": blob, "meta": meta}

  @ray.method(concurrency_group="mutate")
  def restore(self, snapshot: dict[str, Any], slots: np.ndarray, generation: int) -> dict[str, Any]:
    self._check_mutation_allowed(generation)
    core = self.core
    assert core is not None  # _check_mutation_allowed passed
    blob = snapshot.get("inline")
    if blob is None:
      from raysimbatch.protocol import Artifact
      from raysimbatch.storage.blob import open_blob_store

      if not self.spec.artifact_root:
        raise ValueError("snapshot is an artifact but this worker has no artifact_root to read it from")
      blob = open_blob_store(self.spec.artifact_root).get(Artifact(**snapshot["artifact"]))
    with np.load(io.BytesIO(blob)) as z:
      meta = json.loads(z["__meta__"].tobytes())
      if meta["native_abi"] != core.compat.abi_fingerprint or meta["schema_hash"] != self.schema_hash.hex():
        return {"ok": False, "reason": "incompatible snapshot (engine/extension/schema hash); replay from seed"}
      snap = {k: z[k] for k in z.files if k != "__meta__"}
    s = np.ascontiguousarray(slots, np.int32)
    ok = core.load_state(s, {k: v for k, v in snap.items() if not k.startswith("worker/")})
    self.last_obs[s] = snap["worker/last_obs"]
    for k in self.recurrent:
      self.recurrent[k][s] = snap[f"worker/recurrent/{k}"]
    self.action_keys[s] = snap["worker/action_keys"]
    self.eval_ids[s] = snap["worker/eval_ids"]
    return {"ok": bool(ok.all()), "per_slot": ok.tolist()}


def _command(d: dict[str, Any]) -> P.AdvanceCommand:
  return P.AdvanceCommand(
    command_id=bytes(d["command_id"]),
    expected_generation=int(d["expected_generation"]),
    routing_epoch=int(d["routing_epoch"]),
    slots=np.ascontiguousarray(d["slots"], np.int32),
    episode_id=np.ascontiguousarray(d["episode_id"], np.uint64),
    control_seq=np.ascontiguousarray(d["control_seq"], np.uint64),
    policy_hash=bytes(d["policy_hash"]),
    deadline_unix_ns=int(d["deadline_unix_ns"]),
    credit_token=int(d.get("credit_token", 0)),
  )


def _cpu_class() -> str:
  import platform

  return f"{platform.machine()}:{platform.processor() or platform.system()}"


def worker_options(worker: WorkerConfig, failure: FailureConfig, *, name: str, namespace: str | None = None,
                   memory_bytes: int | None = None, scheduling_strategy: Any = None,
                   resources: dict[str, float] | None = None) -> dict[str, Any]:
  """Ray actor options derived from configuration (design §6.2)."""
  opts: dict[str, Any] = {
    "num_cpus": worker.num_cpus,
    "num_gpus": 0,
    "memory": memory_bytes if memory_bytes is not None else worker.memory_gib * 2**30,
    "max_restarts": failure.max_restarts,
    "max_task_retries": failure.max_task_retries,
    "max_pending_calls": worker.max_pending_calls,
    "name": name,
    "runtime_env": {
      "env_vars": {
        "OMP_NUM_THREADS": str(worker.omp_threads),
        "OPENBLAS_NUM_THREADS": str(worker.blas_threads),
        "MKL_NUM_THREADS": str(worker.blas_threads),
        "NUMEXPR_NUM_THREADS": "1",
      }
    },
  }
  if namespace:
    opts["namespace"] = namespace
  if scheduling_strategy is not None:
    opts["scheduling_strategy"] = scheduling_strategy
  if resources:
    opts["resources"] = resources
  return opts


# Ray 2.58 accepts concurrency_groups only on the decorator, not in .options().
RemoteSimulationWorker = actor_class(SimulationWorker, concurrency_groups={"mutate": 1, "control": 1})
