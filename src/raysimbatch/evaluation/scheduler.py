# SPDX-License-Identifier: Apache-2.0
"""EvalScheduler (design §5.3, §6.6): the checkpoint x task x seed matrix as durable units.

The complete expected unit set is persisted before any unit executes. Units are
leased in blocks with ``FOR UPDATE SKIP LOCKED``; completion is fenced by attempt and
lease epoch; failures retry from the original seeds up to ``max_attempts`` and then
become explicit EXHAUSTED results while unrelated units continue.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raysimbatch.config.models import EvalConfig
from raysimbatch.evaluation.aggregator import ResultAggregator
from raysimbatch.evaluation.schema import EpisodeResult
from raysimbatch.storage.db import CommitAck
from raysimbatch.storage.store import DurableStore


@dataclass(frozen=True)
class EvalTask:
  """A task bound to a model group: what a unit needs to run its episodes."""

  name: str
  task_hash: str
  group: str
  model_hash: str
  adapter_hash: str
  weight: float = 1.0
  axis: str | None = None  # Mode C: this task/group pair is one diagnostic axis


@dataclass(frozen=True)
class EvalPolicy:
  name: str
  policy_hash: str
  version: int = 0


@dataclass(frozen=True)
class EvalUnit:
  unit_id: str
  policy_hash: str
  task_hash: str
  model_hash: str
  adapter_hash: str
  seed: int
  episode_count: int
  experiment_axis: str
  attempt: int
  lease_epoch: int
  lease_deadline: int
  payload: dict[str, Any]


@dataclass(frozen=True)
class ResumeReport:
  committed: int
  retryable: int
  expired: int
  exhausted: int
  pending: int
  running: int
  canceled: int
  succeeded_units: int


def unit_id(policy_hash: str, task_hash: str, model_hash: str, adapter_hash: str, seed: int, episode_count: int,
            axis: str) -> str:
  """Hash of every semantic input, excluding attempt and placement."""
  doc = [policy_hash, task_hash, model_hash, adapter_hash, int(seed), int(episode_count), axis]
  return hashlib.sha256(json.dumps(doc).encode()).hexdigest()


class EvalScheduler:
  def __init__(self, run_id: str, config: EvalConfig, store: DurableStore, aggregator: ResultAggregator | None = None):
    self.run_id, self.config, self.store = run_id, config, store
    self.aggregator = aggregator or ResultAggregator(run_id, 0, store)

  async def enumerate(self, policies: Sequence[EvalPolicy], tasks: Sequence[EvalTask], seeds: Sequence[int],
                      axes: Sequence[str] = ("",)) -> int:
    units = []
    for wave, p in enumerate(policies):
      for t in tasks:
        for axis in ((t.axis,) if t.axis is not None else axes):
          for s in seeds:
            uid = unit_id(p.policy_hash, t.task_hash, t.model_hash, t.adapter_hash, s, self.config.episodes_per_seed,
                          axis)
            payload = {
              "policy": p.name, "policy_hash": p.policy_hash, "policy_version": p.version,
              "task": t.name, "task_hash": t.task_hash, "group": t.group, "model_hash": t.model_hash,
              "adapter_hash": t.adapter_hash, "seed": int(s), "episode_count": self.config.episodes_per_seed,
              "experiment_axis": axis,
            }
            units.append({"unit_id": uid, "payload": payload,
                          "payload_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                          "wave": wave})
    await asyncio.to_thread(self.store.db.insert_units, self.run_id, units)
    await asyncio.to_thread(self.store.db.audit, self.run_id, "units_enumerated", {"count": len(units)})
    return len(units)

  async def lease(self, owner: str, limit: int = 128) -> tuple[EvalUnit, ...]:
    rows = await asyncio.to_thread(self.store.db.lease_units, self.run_id, owner, limit, self.config.lease_s,
                                   self.config.max_attempts)
    out = []
    for r in rows:
      p = r["payload"]
      out.append(EvalUnit(r["unit_id"], p["policy_hash"], p["task_hash"], p["model_hash"], p["adapter_hash"],
                          int(p["seed"]), int(p["episode_count"]), p["experiment_axis"], int(r["attempt"]),
                          int(r["lease_epoch"]), int(r["lease_until_ns"]), p))
    return tuple(out)

  async def renew(self, owner: str, units: Sequence[EvalUnit]) -> list[str]:
    return await asyncio.to_thread(self.store.db.renew_units, self.run_id, owner,
                                   [(u.unit_id, u.attempt, u.lease_epoch) for u in units], self.config.lease_s)

  async def committed_ordinals(self, unit_id_: str) -> set[int]:
    return await asyncio.to_thread(self.store.db.committed_ordinals, self.run_id, unit_id_)

  async def complete(self, unit: EvalUnit, results: Sequence[EpisodeResult]) -> CommitAck:
    return await self.aggregator.append_unit(unit, results)

  async def fail(self, unit: EvalUnit, fault: dict[str, Any]) -> str | None:
    state = await asyncio.to_thread(self.store.db.fail_unit, self.run_id, unit.unit_id, unit.attempt,
                                    unit.lease_epoch, fault, self.config.max_attempts)
    await asyncio.to_thread(self.store.db.audit, self.run_id, "unit_failed",
                            {"unit_id": unit.unit_id, "attempt": unit.attempt, "state": state, "fault": fault})
    return state

  async def resume(self) -> ResumeReport:
    counts = await asyncio.to_thread(self.store.db.unit_counts, self.run_id)
    episodes = await asyncio.to_thread(self.store.db.episode_count, self.run_id)
    return ResumeReport(
      committed=episodes,
      retryable=counts.get("RETRYABLE", 0),
      expired=0,
      exhausted=counts.get("EXHAUSTED", 0),
      pending=counts.get("PENDING", 0),
      running=counts.get("LEASED", 0) + counts.get("RUNNING", 0) + counts.get("COMMITTING", 0),
      canceled=counts.get("CANCELED", 0),
      succeeded_units=counts.get("SUCCEEDED", 0),
    )

  async def done(self) -> bool:
    counts = await asyncio.to_thread(self.store.db.unit_counts, self.run_id)
    return all(k in ("SUCCEEDED", "EXHAUSTED", "CANCELED") for k in counts)
