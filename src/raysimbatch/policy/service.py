# SPDX-License-Identifier: Apache-2.0
"""PolicyInferenceService: central (GPU) inference actor (design §6.4).

* Async ``infer`` calls enqueue and await per-request futures; one dedicated execution
  thread launches one batch at a time and never blocks the event loop.
* Queue caps: ``queue_requests`` and ``queue_rows`` are both enforced; a request that
  would exceed either is rejected immediately (backpressure, never silent drops).
* Batches only combine requests with identical (policy hash, observation schema,
  dtype, deterministic mode, recurrent shapes); at most ``max_batch_rows`` rows; flush
  when full or when the oldest request has waited ``flush_ms``.
* Expired requests return INFERENCE_DEADLINE. Observations are never interpreted with
  a policy other than the requested hash; there is no fallback action.
* Recurrent state travels with requests and responses (the service is stateless
  per environment); stochastic sampling uses the rows' counter keys so a retried
  request yields the same actions.
* Two resident versions per policy family; loading a third waits for retirement.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from raysimbatch import protocol as P
from raysimbatch.config.models import InferenceConfig
from raysimbatch.policy.bundle import PolicyBundle
from raysimbatch.policy.runtime import LocalPolicyRuntime
from raysimbatch.telemetry import tracing


class InferenceRejected(RuntimeError):
  def __init__(self, code: P.FaultCode, message: str):
    super().__init__(message)
    self.code = code


@dataclass
class _Req:
  request_id: bytes
  key: tuple
  frame: P.HeteroFrame
  rows: int
  deadline_ns: int
  enqueued_ns: int
  deterministic: bool | None
  future: asyncio.Future = field(repr=False, default=None)  # type: ignore[assignment]


class PolicyInferenceService:
  def __init__(self, config: dict[str, Any], device: str | None = None, family_limit: int | None = None):
    self.cfg = InferenceConfig.from_obj(config)
    self.device = device or self.cfg.device
    self.family_limit = family_limit or self.cfg.max_resident_versions
    self.policies: dict[str, LocalPolicyRuntime] = {}
    self.families: dict[str, list[str]] = collections.defaultdict(list)
    self.retire_after: dict[str, int] = {}
    self.queues: dict[tuple, collections.deque[_Req]] = collections.OrderedDict()
    self.queued_requests = 0
    self.queued_rows = 0
    self.queued_bytes = 0  # observation bytes of the queued requests
    self.executing: collections.Counter[str] = collections.Counter()  # policy hash -> batches running
    self.completed = 0
    self.deadlines = 0
    self.errors = 0
    self.batches = 0
    self.batch_rows_total = 0
    from raysimbatch import telemetry

    self.metrics = telemetry.exporter(actor_role="inference")
    self._cv = threading.Condition()
    self._stop = False
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread = threading.Thread(target=self._run, name="rsb-inference-exec", daemon=True)
    self._thread.start()
    self.service_samples: list[tuple[int, float]] = []

  # ------------------------------------------------------------------ policies

  async def load(self, bundle_json: str, weights_blob: np.ndarray) -> dict[str, Any]:
    bundle = PolicyBundle.from_json(bundle_json)
    h = bundle.policy_hash
    if h in self.policies:
      return self._ready(bundle)
    fam = self.families[bundle.name]
    if len(fam) >= self.family_limit:
      raise InferenceRejected(P.FaultCode.INFERENCE_ERROR,
                              f"family {bundle.name!r} already has {len(fam)} resident versions; retire one first")
    rt = await asyncio.to_thread(LocalPolicyRuntime, bundle, weights_blob, self.device, 1)
    # Warm every certified size up to the batch limit before routing requests.
    sizes = [s for s in bundle.certified_batch_sizes if s <= self.cfg.max_batch_rows]
    curve = []
    for s in sizes:
      obs = np.zeros((s, bundle.obs_dim), np.float32)
      rec = {st.name: np.zeros((s, *st.shape), np.float32) for st in bundle.recurrent}
      keys = np.zeros((s, 2), np.uint64)
      t0 = time.perf_counter()
      await asyncio.to_thread(rt.infer, obs, rec, keys, np.zeros(s, np.uint64))
      curve.append((s, time.perf_counter() - t0))
    self.policies[h] = rt
    fam.append(h)
    self.service_samples += curve
    return self._ready(bundle, curve)

  def _ready(self, bundle: PolicyBundle, curve=()) -> dict[str, Any]:
    return {"policy_hash": bundle.policy_hash, "supported_batch_sizes": list(bundle.certified_batch_sizes),
            "device": self.device, "service_curve": [list(x) for x in curve]}

  async def retire(self, policy_hash: str, after_unix_ns: int = 0) -> None:
    self.retire_after[policy_hash] = after_unix_ns
    await self._maybe_retire()

  async def _maybe_retire(self) -> None:
    now = time.time_ns()
    for h, after in list(self.retire_after.items()):
      with self._cv:  # a batch taken off a queue is still busy until it has executed
        busy = self.executing[h] > 0 or any(r.key[0] == h for q in self.queues.values() for r in q)
        rt = self.policies.pop(h) if now >= after and not busy and h in self.policies else None
      if rt is not None:
        for fam in self.families.values():
          if h in fam:
            fam.remove(h)
        rt.close()
        del self.retire_after[h]

  # ------------------------------------------------------------------- infer

  async def infer(self, request: dict[str, Any], observations: dict[str, Any]) -> dict[str, Any]:
    """request: request_id, policy_hash (bytes), deadline_unix_ns, optional
    deterministic. ``observations`` is the (auto-dereferenced) HeteroFrame wire."""
    if self._loop is None:
      self._loop = asyncio.get_running_loop()
    frame = P.HeteroFrame.from_wire(observations)
    ph = bytes(request["policy_hash"]).hex()
    rt = self.policies.get(ph)
    if rt is None:
      raise InferenceRejected(P.FaultCode.POLICY_MISMATCH, f"policy {ph[:12]} is not resident")
    if len(frame.groups) != 1:
      raise InferenceRejected(P.FaultCode.SCHEMA_MISMATCH, "one schema group per request")
    g = frame.groups[0]
    if g.observation.shape[1] != rt.bundle.obs_dim:
      raise InferenceRejected(P.FaultCode.SCHEMA_MISMATCH, "observation width does not match the policy")
    det = request.get("deterministic")
    key = (ph, g.schema_hash, "float32", bool(rt.bundle.deterministic_inference if det is None else det),
           tuple((s.name, s.shape) for s in rt.bundle.recurrent))
    req = _Req(bytes(request["request_id"]), key, frame, g.rows, int(request["deadline_unix_ns"]), time.monotonic_ns(),
               det, self._loop.create_future())
    with self._cv:
      if self.queued_requests + 1 > self.cfg.queue_requests or self.queued_rows + g.rows > self.cfg.queue_rows:
        raise InferenceRejected(P.FaultCode.CREDIT_EXHAUSTED, "inference queue full (requests or rows)")
      self.queues.setdefault(key, collections.deque()).append(req)
      self.queued_requests += 1
      self.queued_rows += g.rows
      self.queued_bytes += g.observation.nbytes
      self._cv.notify()
    return await req.future

  def _run(self) -> None:
    flush_ns = int(self.cfg.flush_ms * 1e6)
    while True:
      with self._cv:
        while not self._stop and not any(self.queues.values()):
          self._cv.wait()
        if self._stop:
          return
        # Oldest head across compatible queues.
        key = min((k for k, q in self.queues.items() if q), key=lambda k: self.queues[k][0].enqueued_ns)
        q = self.queues[key]
        rows = sum(r.rows for r in q)
        age = time.monotonic_ns() - q[0].enqueued_ns
        if rows < self.cfg.max_batch_rows and age < flush_ns:
          self._cv.wait(timeout=(flush_ns - age) * 1e-9)
          continue
        batch: list[_Req] = []
        n = 0
        while q and n + q[0].rows <= self.cfg.max_batch_rows:
          r = q.popleft()
          batch.append(r)
          n += r.rows
        if not batch:  # a single request larger than the batch limit runs alone
          batch.append(q.popleft())
          n = batch[0].rows
        self.queued_requests -= len(batch)
        self.queued_rows -= n
        self.queued_bytes -= sum(r.frame.groups[0].observation.nbytes for r in batch)
        oldest = min((q[0].enqueued_ns for q in self.queues.values() if q), default=None)
        backlog = (self.queued_rows, self.queued_bytes,
                   0.0 if oldest is None else (time.monotonic_ns() - oldest) * 1e-9)
        self.executing[key[0]] += 1
      # Once per batch (not per request): the backlog left behind for the next ones.
      self.metrics.gauge("queue_rows", float(backlog[0]), stage="inference")
      self.metrics.gauge("queue_bytes", float(backlog[1]), stage="inference")
      self.metrics.gauge("oldest_request_seconds", backlog[2], stage="inference")
      try:
        self._execute(key, batch)
      finally:
        with self._cv:
          self.executing[key[0]] -= 1

  def _execute(self, key: tuple, batch: list[_Req]) -> None:
    now = time.time_ns()
    live = []
    for r in batch:
      if now > r.deadline_ns:
        self.deadlines += 1
        self.metrics.count("inference_deadlines_total")
        self._resolve(r, InferenceRejected(P.FaultCode.INFERENCE_DEADLINE, "request expired before execution"))
      else:
        live.append(r)
    if not live:
      return
    rt = self.policies.get(key[0])
    try:
      if rt is None:
        raise InferenceRejected(P.FaultCode.POLICY_MISMATCH, "policy retired")
      gs = [r.frame.groups[0] for r in live]
      obs = np.concatenate([g.observation for g in gs])
      rec = {s.name: np.concatenate([g.recurrent[s.name] for g in gs]) for s in rt.bundle.recurrent}
      keys = np.concatenate([g.sample_keys if g.sample_keys is not None else np.zeros((g.rows, 2), np.uint64)
                             for g in gs])
      seq = np.concatenate([g.control_seq for g in gs])
      t0 = time.perf_counter()
      with tracing.span("inference_batch", rows=int(obs.shape[0]), requests=len(live)):
        out = rt.infer(obs, rec, keys, seq, deterministic=key[3])
      self.service_samples.append((obs.shape[0], time.perf_counter() - t0))
      self.metrics.observe("inference_seconds", self.service_samples[-1][1])
      self.metrics.observe("inference_batch_rows", float(obs.shape[0]))
      self.service_samples = self.service_samples[-4096:]
      self.batches += 1
      self.batch_rows_total += obs.shape[0]
      off = 0
      for r, g in zip(live, gs, strict=True):
        sl = slice(off, off + g.rows)
        off += g.rows
        ag = P.ActionGroup(
          group_hash=g.group_hash, schema_hash=g.schema_hash, policy_hash=bytes.fromhex(key[0]),
          env_id=g.env_id, episode_id=g.episode_id, expected_control_seq=g.control_seq,
          slot_generation=g.slot_generation, slot=g.slot, action=np.ascontiguousarray(out.action[sl]),
          behavior_logp=np.ascontiguousarray(out.logp[sl]),
          policy_version=np.full(g.rows, rt.bundle.version, np.uint64),
          value=None if out.value is None else np.ascontiguousarray(out.value[sl]),
          recurrent={k: np.ascontiguousarray(v[sl]) for k, v in out.recurrent.items()},
        )
        af = P.ActionFrame(r.frame.run_id, r.request_id, r.frame.routing_epoch, time.time_ns(), r.frame.trace_id, (ag,))
        self.completed += 1
        self._resolve(r, af.to_wire())
    except Exception as exc:  # noqa: BLE001 - surfaced per request, never substituted
      self.errors += 1
      self.metrics.count("inference_errors_total")
      for r in live:
        self._resolve(r, exc if isinstance(exc, InferenceRejected)
                      else InferenceRejected(P.FaultCode.INFERENCE_ERROR, f"policy execution failed: {exc}"))

  def _resolve(self, r: _Req, value: Any) -> None:
    def done():
      if r.future.done():
        return
      if isinstance(value, BaseException):
        r.future.set_exception(value)
      else:
        r.future.set_result(value)

    assert self._loop is not None
    self._loop.call_soon_threadsafe(done)

  # ------------------------------------------------------------------ health

  async def health(self) -> dict[str, Any]:
    await self._maybe_retire()
    oldest = min((q[0].enqueued_ns for q in self.queues.values() if q), default=None)
    gpu = {}
    if self.device == "cuda":
      try:
        import torch

        gpu = {"memory_allocated": int(torch.cuda.memory_allocated()), "device": torch.cuda.get_device_name()}
      except Exception:  # noqa: BLE001
        gpu = {}
    return {
      "queue_requests": self.queued_requests,
      "queue_rows": self.queued_rows,
      "oldest_request_s": 0.0 if oldest is None else (time.monotonic_ns() - oldest) * 1e-9,
      "resident_policies": sorted(self.policies),
      "completed": self.completed,
      "deadlines": self.deadlines,
      "errors": self.errors,
      "batches": self.batches,
      "mean_batch_rows": self.batch_rows_total / self.batches if self.batches else 0.0,
      "device": self.device,
      "gpu": gpu,
    }

  async def service_curve(self) -> list[tuple[int, float]]:
    return list(self.service_samples)

  def __del__(self) -> None:
    self._stop = True


def inference_options(cfg: InferenceConfig, name: str, *, namespace: str | None = None) -> dict[str, Any]:
  """Whole-GPU replicas; fractional GPU sharing is disabled (design §6.4)."""
  opts: dict[str, Any] = {
    "num_cpus": cfg.cpus_per_replica,
    "num_gpus": cfg.gpus_per_replica if cfg.device == "cuda" else 0,
    "max_restarts": 1,
    "max_task_retries": 0,
    "max_concurrency": cfg.queue_requests + 8,
    "name": name,
  }
  if cfg.device == "cuda":
    opts["resources"] = {"policy_gpu": 1}
  if namespace:
    opts["namespace"] = namespace
  return opts



from raysimbatch.compat.ray_typing import actor_class  # noqa: E402

RemotePolicyInferenceService = actor_class(PolicyInferenceService)
