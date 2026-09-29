# SPDX-License-Identifier: Apache-2.0
"""ExperienceAssembler (design §6.5, §7.4).

* Deduplicates by (fragment_id, digest); a repeated id with a different digest is
  corruption and stops this assembler shard.
* Policy-lag contract: fragments whose behavior policy is more than
  ``max_policy_lag_versions`` learner updates (or ``max_policy_lag_s`` seconds) behind
  are rejected and counted, never silently trained on.
* Durable outbox: the fragment is written content-addressed and its outbox row
  committed *before* delivery; delivery retries until the sink ACKs, and only then is
  the outbox row marked delivered and the worker credit released. With an idempotent
  sink this is exactly-once logical insertion; otherwise at-least-once, which a run
  with ``require_idempotent_sink`` refuses.
* Byte credits: at most ``assembler_bytes`` unacknowledged bytes are held.
* Restart: Ray reruns only the constructor, so the first call after a restart sets the
  store and sink up again and delivers the outbox rows that never got a sink ACK before
  serving anything. Statistics restart from zero; the run's transition count comes from
  the coordinators.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from raysimbatch.compat.ray_typing import actor_class
from raysimbatch.config.models import RunConfig
from raysimbatch.experience.sinks import (
  DurableReplaySink,
  ReplaySink,
  SinkAck,
  SinkCapabilities,
  fragment_bytes,
)
from raysimbatch.protocol import Flag, Fragment
from raysimbatch.storage.db import DigestConflict

if TYPE_CHECKING:
  from raysimbatch.storage.store import DurableStore


class ExperienceAssembler:
  def __init__(self, run_id: str, config_json: str, sink: dict[str, Any]):
    self.run_id = run_id
    self.cfg = RunConfig.model_validate_json(config_json)
    self.sink_spec = sink
    self.seen: dict[str, str] = {}
    self.acks: dict[str, dict[str, Any]] = {}
    self.inflight_bytes = 0
    self.learner_version = 0
    self.learner_version_time = time.time()
    self.version_times: dict[int, float] = {0: time.time()}
    self.stats = {"accepted": 0, "duplicates": 0, "rejected_lag": 0, "rejected_schema": 0, "transitions": 0,
                  "aborted_episodes": 0, "bytes": 0}
    self.lag_versions: list[int] = []
    self.stopped: str | None = None
    # Late initialization: _setup() sets these; every method using them first awaits
    # _ensure_ready() (also after a restart, which reruns only this constructor).
    self.store: DurableStore = None  # type: ignore[assignment]
    self.sink: ReplaySink = None  # type: ignore[assignment]
    self.metrics = None
    self._ready = False
    self._init_lock = asyncio.Lock()
    self._outbox: dict[str, int] = {}  # fragment id -> blob bytes written to the outbox, not yet delivered

  async def ready(self) -> dict[str, Any]:
    out = await self._setup()
    self._ready = True
    return out

  async def _ensure_ready(self) -> None:
    if self._ready:
      return
    async with self._init_lock:  # concurrent calls after a restart wait for one setup
      if not self._ready:
        await self._setup()
        if self.cfg.rollout.durable_outbox:
          await self._redeliver()
        self._ready = True

  async def _setup(self) -> dict[str, Any]:
    from raysimbatch import telemetry
    from raysimbatch.storage.store import DurableStore

    telemetry.setup("raysimbatch-assembler", self.cfg.telemetry)
    self.metrics = telemetry.exporter(mode=self.cfg.mode, actor_role="assembler")

    self.store = await asyncio.to_thread(DurableStore.from_config, self.cfg, migrate=False)
    kind = self.sink_spec.get("kind", "durable_replay")
    if kind == "durable_replay":
      self.sink = await asyncio.to_thread(DurableReplaySink, self.store, self.run_id)
    elif kind == "learner_queue":
      from raysimbatch.experience.sinks import LearnerQueueSink

      self.sink = LearnerQueueSink(self.sink_spec["learner"])
    else:
      raise ValueError(f"unknown sink kind {kind!r}")
    caps: SinkCapabilities = await self.sink.capabilities()
    if self.cfg.rollout.require_idempotent_sink and not caps.idempotent_append:
      raise ValueError("require_idempotent_sink=true but the sink is not idempotent (at-least-once only)")
    self.caps = caps
    return {"capabilities": caps.__dict__}

  async def set_learner_version(self, version: int) -> None:
    """Learners report their update index; lag is measured against it."""
    self.learner_version = int(version)
    self.version_times[int(version)] = time.time()

  async def accept(self, fragment: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    await self._ensure_ready()
    if self.stopped:
      raise RuntimeError(f"assembler stopped: {self.stopped}")
    frag = Fragment.from_wire(fragment)
    fid, digest = frag.fragment_id(), frag.digest()
    if fid != meta["fragment_id"] or digest != meta["digest"]:
      self.stopped = "fragment metadata does not match its content"
      raise DigestConflict(self.stopped)
    if fid in self.seen:
      if self.seen[fid] != digest:
        self.stopped = f"fragment {fid}: repeated id with a different digest"
        raise DigestConflict(self.stopped)
      self.stats["duplicates"] += 1
      if self.metrics is not None:
        self.metrics.count("duplicate_commits_total")
      return self.acks[fid]
    nbytes = sum(v.nbytes for v in fragment.values() if isinstance(v, np.ndarray))
    while self.inflight_bytes + nbytes > self.cfg.flow.assembler_bytes:  # noqa: ASYNC110 - bounded poll
      await asyncio.sleep(0.005)  # byte credits: callers wait; nothing is dropped
    self.inflight_bytes += nbytes
    self._gauge("queue_bytes", self.inflight_bytes, stage="assembler")
    try:
      ack = await self._accept(frag, fid, digest, nbytes)
    finally:
      self.inflight_bytes -= nbytes
      self._gauge("queue_bytes", self.inflight_bytes, stage="assembler")
    self.seen[fid] = digest
    self.acks[fid] = ack
    return ack

  async def _accept(self, frag: Fragment, fid: str, digest: str, nbytes: int) -> dict[str, Any]:
    ro = self.cfg.rollout
    if frag.transitions:
      behavior = int(frag.policy_version.min())
      lag = max(0, self.learner_version - behavior)
      age = time.time() - self.version_times.get(behavior + 1, time.time()) if lag else 0.0
      self.lag_versions.append(lag)
      if self.metrics is not None:
        self.metrics.observe("policy_lag_versions", float(lag))
        self.metrics.observe("policy_lag_seconds", float(age))
      if lag > ro.max_policy_lag_versions or (lag and age > ro.max_policy_lag_s):
        self.stats["rejected_lag"] += 1
        if self.metrics is not None:
          self.metrics.count("fragments_rejected_total", outcome="policy_lag")
        return SinkAck(fid, digest, None, 0, f"policy lag {lag} versions / {age:.2f}s exceeds bound").__dict__
    aborted = int(((frag.flags & Flag.ABORTED) != 0).sum())
    self.stats["aborted_episodes"] += aborted
    if ro.durable_outbox:
      blob = fragment_bytes(frag)
      art = await asyncio.to_thread(self.store.blobs.put, blob, "application/x-rsb-fragment")
      await asyncio.to_thread(self.store.db.outbox_insert, fid, self.run_id, digest, art, frag.transitions)
      self._outbox_add(fid, len(blob))
    t0 = time.perf_counter()
    last: Exception | None = None
    for attempt in range(1 + self.cfg.storage.retry_attempts):
      try:
        ack: SinkAck = await self.sink.append(frag, fid)
        break
      except DigestConflict:
        self.stopped = f"sink digest conflict for {fid}"
        if self.metrics is not None:
          self.metrics.count("storage_errors_total", outcome="digest_conflict")
        raise
      except Exception as exc:  # noqa: BLE001 - retried with backoff; the outbox keeps the data
        last = exc
        await asyncio.sleep(self.cfg.storage.retry_backoff_s[min(attempt, len(self.cfg.storage.retry_backoff_s) - 1)])
    else:
      raise RuntimeError(f"sink delivery failed after retries (outbox retains {fid}): {last}")
    if ro.durable_outbox:
      await asyncio.to_thread(self.store.db.outbox_mark, fid, "DELIVERED", ack.durable_offset)
      self._outbox_done(fid)
    self.stats["accepted"] += 1
    self.stats["transitions"] += ack.accepted_transitions
    self.stats["bytes"] += nbytes
    d = ack.__dict__
    d["sink_seconds"] = time.perf_counter() - t0
    if self.metrics is not None:
      self.metrics.count("fragments_committed_total")
      self.metrics.observe("sink_ack_seconds", d["sink_seconds"])
    return d

  async def abort_episode(self, env_id: int, episode_id: int, reason: str, last_committed_seq: int) -> None:
    await self._ensure_ready()
    await asyncio.to_thread(self.store.db.record_gap, self.run_id, env_id, episode_id, last_committed_seq, 0, 0,
                            reason, 0)

  async def redeliver_pending(self) -> int:
    """Deliver outbox rows that never got a sink ACK (a restart does this by itself)."""
    await self._ensure_ready()
    return await self._redeliver()

  async def _redeliver(self) -> int:
    from raysimbatch.experience.sinks import fragment_from_bytes
    from raysimbatch.protocol import Artifact

    rows = await asyncio.to_thread(self.store.db.outbox_pending, self.run_id, 10_000)
    for r in rows:
      self._outbox_add(r["fragment_id"], int(r["byte_length"]))
    n = 0
    for r in rows:
      blob = await asyncio.to_thread(self.store.blobs.get, Artifact(r["object_uri"], r["object_uri"].rsplit("/", 1)[-1],
                                                                    int(r["byte_length"]), "application/x-rsb-fragment"))
      frag = fragment_from_bytes(blob)
      ack = await self.sink.append(frag, r["fragment_id"])
      await asyncio.to_thread(self.store.db.outbox_mark, r["fragment_id"], "DELIVERED", ack.durable_offset)
      self._outbox_done(r["fragment_id"])
      n += 1
    return n

  def _outbox_add(self, fid: str, nbytes: int) -> None:
    if fid not in self._outbox:
      self._outbox[fid] = nbytes
      self._gauge("outbox_bytes", sum(self._outbox.values()))

  def _outbox_done(self, fid: str) -> None:
    if self._outbox.pop(fid, None) is not None:
      self._gauge("outbox_bytes", sum(self._outbox.values()))

  def _gauge(self, metric: str, value: float, **labels: str) -> None:
    if self.metrics is not None:
      self.metrics.gauge(metric, float(value), **labels)

  async def seal(self, deadline_unix_ns: int) -> dict[str, Any]:
    await self._ensure_ready()
    while self.inflight_bytes and time.time_ns() < deadline_unix_ns:  # noqa: ASYNC110 - deadline-bounded
      await asyncio.sleep(0.01)
    off = await self.sink.flush(self.run_id)
    return {"offset": off.offset, "manifest_uri": off.manifest_uri, "stats": dict(self.stats),
            "lag_histogram": np.bincount(np.asarray(self.lag_versions or [0])).tolist()}

  async def status(self) -> dict[str, Any]:
    return {"stats": dict(self.stats), "inflight_bytes": self.inflight_bytes, "learner_version": self.learner_version,
            "stopped": self.stopped}


RemoteExperienceAssembler = actor_class(ExperienceAssembler, num_cpus=1, max_restarts=1, max_task_retries=0,
                                        max_concurrency=64)
