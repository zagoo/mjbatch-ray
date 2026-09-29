# SPDX-License-Identifier: Apache-2.0
"""Closed result schemas (design §7.5): EpisodeResult v1 and AggregateResult v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pyarrow as pa

EPISODE_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1
Outcome = Literal["success", "task_failure", "time_limit", "health_failure", "infrastructure_failure", "canceled"]
OUTCOMES = ("success", "task_failure", "time_limit", "health_failure", "infrastructure_failure", "canceled")


@dataclass(frozen=True)
class EpisodeResult:
  schema_version: int
  episode_ordinal: int
  attempt: int
  run_id: str
  unit_id: str
  policy_hash: str
  model_hash: str
  task_hash: str
  adapter_hash: str
  config_hash: str
  seed: int
  env_id: int
  episode_id: int
  generation: int
  started_at: int  # UTC ns
  finished_at: int
  physics_steps: int
  control_steps: int
  simulated_seconds: float
  wall_seconds: float
  return_sum: float | None
  success: bool | None
  outcome: str
  fault_code: str = ""
  experiment_axis: str = ""
  randomization_hash: str = ""
  hardware_class: str = ""
  tracking_rmse: float | None = None
  energy_joules: float | None = None
  max_contact_force: float | None = None
  saturation_fraction: float | None = None
  reward_components: Mapping[str, float] = field(default_factory=dict)
  task_metrics: Mapping[str, float] = field(default_factory=dict)
  terminal_state_uri: str | None = None
  diagnostic_uri: str | None = None
  source_episode_key: str | None = None

  def __post_init__(self) -> None:
    if self.outcome not in OUTCOMES:
      raise ValueError(f"unknown outcome {self.outcome!r}")
    for k in ("simulated_seconds", "wall_seconds", "return_sum", "tracking_rmse", "energy_joules",
              "max_contact_force", "saturation_fraction"):
      v = getattr(self, k)
      if v is not None and not math.isfinite(v):
        raise ValueError(f"{k}: NaN/Inf are forbidden; use null")
    if self.outcome == "infrastructure_failure" and self.success is not None:
      raise ValueError("success is null for infrastructure aborts")

  def to_row(self) -> dict[str, Any]:
    d = asdict(self)
    d["reward_components"] = dict(self.reward_components)
    d["task_metrics"] = dict(self.task_metrics)
    return d

  def digest(self) -> str:
    """Identity digest over deterministic semantic fields only: a replay of the same
    episode from its original seed must produce the same digest (wall time, placement
    and attempt are excluded)."""
    sem = {k: v for k, v in self.to_row().items() if k not in _NONSEMANTIC}
    return hashlib.sha256(json.dumps(sem, sort_keys=True, default=str).encode()).hexdigest()


_NONSEMANTIC = frozenset({
  "attempt", "env_id", "episode_id", "generation", "started_at", "finished_at", "wall_seconds", "hardware_class",
  "terminal_state_uri", "diagnostic_uri",
})

_MAP = pa.map_(pa.string(), pa.float64())
EPISODE_ARROW = pa.schema([
  pa.field("schema_version", pa.int32(), False),
  pa.field("episode_ordinal", pa.int32(), False),
  pa.field("attempt", pa.int32(), False),
  pa.field("run_id", pa.string(), False),
  pa.field("unit_id", pa.string(), False),
  pa.field("policy_hash", pa.string(), False),
  pa.field("model_hash", pa.string(), False),
  pa.field("task_hash", pa.string(), False),
  pa.field("adapter_hash", pa.string(), False),
  pa.field("config_hash", pa.string(), False),
  pa.field("seed", pa.uint64(), False),
  pa.field("env_id", pa.uint64(), False),
  pa.field("episode_id", pa.uint64(), False),
  pa.field("generation", pa.uint64(), False),
  pa.field("started_at", pa.timestamp("ns", tz="UTC"), False),
  pa.field("finished_at", pa.timestamp("ns", tz="UTC"), False),
  pa.field("physics_steps", pa.uint64(), False),
  pa.field("control_steps", pa.uint64(), False),
  pa.field("simulated_seconds", pa.float64(), False),
  pa.field("wall_seconds", pa.float64(), False),
  pa.field("return_sum", pa.float64(), True),
  pa.field("success", pa.bool_(), True),
  pa.field("outcome", pa.string(), False),
  pa.field("fault_code", pa.string(), False),
  pa.field("experiment_axis", pa.string(), False),
  pa.field("randomization_hash", pa.string(), False),
  pa.field("hardware_class", pa.string(), False),
  pa.field("tracking_rmse", pa.float64(), True),
  pa.field("energy_joules", pa.float64(), True),
  pa.field("max_contact_force", pa.float64(), True),
  pa.field("saturation_fraction", pa.float64(), True),
  pa.field("reward_components", _MAP, False),
  pa.field("task_metrics", _MAP, False),
  pa.field("terminal_state_uri", pa.string(), True),
  pa.field("diagnostic_uri", pa.string(), True),
  pa.field("source_episode_key", pa.string(), True),
], metadata={"rsb_schema": "EpisodeResult", "version": str(EPISODE_SCHEMA_VERSION)})


def episodes_table(rows: Sequence[EpisodeResult | Mapping[str, Any]]) -> pa.Table:
  cols: dict[str, list[Any]] = {f.name: [] for f in EPISODE_ARROW}
  for r in rows:
    d = r.to_row() if isinstance(r, EpisodeResult) else dict(r)
    for name in cols:
      v = d.get(name)
      if name in ("reward_components", "task_metrics"):
        v = list((v or {}).items())
      cols[name].append(v)
  return pa.Table.from_pydict(cols, schema=EPISODE_ARROW)


def from_row(d: Mapping[str, Any]) -> EpisodeResult:
  kw: dict[str, Any] = {f.name: d.get(f.name) for f in EPISODE_ARROW}  # absent (newer, nullable) fields: None
  for k in ("reward_components", "task_metrics"):
    v = kw[k] or {}
    kw[k] = dict(v) if not isinstance(v, list) else dict(v)
  for k in ("started_at", "finished_at"):
    v = kw[k]
    if hasattr(v, "value"):
      kw[k] = int(v.value)
  return EpisodeResult(**kw)


@dataclass(frozen=True)
class Interval:
  low: float
  high: float
  method: str
  replicates: int
  confidence: float


@dataclass(frozen=True)
class AggregateResult:
  policy_hash: str
  task_hash: str
  task_name: str
  experiment_axis: str
  metric: str
  expected_episodes: int
  completed_episodes: int
  valid_episodes: int
  failed_episodes: int
  canceled_episodes: int
  expected_seeds: int
  valid_seeds: int
  coverage: float
  mean: float | None
  std: float | None
  median: float | None
  p5: float | None
  p95: float | None
  ci: Interval | None
  low_seed_count: bool
  success_rate_conditional: float | None
  success_rate_conservative: float | None
  infrastructure_failure_fraction: float
  fault_counts: Mapping[str, int]
  task_weight: float
  schema_version: int = AGGREGATE_SCHEMA_VERSION
  source_manifest_digest: str = ""

  def to_row(self) -> dict[str, Any]:
    d = asdict(self)
    ci = d.pop("ci")
    d["ci_low"] = None if ci is None else ci["low"]
    d["ci_high"] = None if ci is None else ci["high"]
    d["ci_method"] = None if ci is None else ci["method"]
    d["ci_replicates"] = None if ci is None else ci["replicates"]
    d["ci_confidence"] = None if ci is None else ci["confidence"]
    d["fault_counts"] = dict(self.fault_counts)
    return d


@dataclass(frozen=True)
class PairedResult:
  policy_a: str
  policy_b: str
  task_hash: str
  metric: str
  common_seeds: int
  union_seeds: int
  paired_coverage: float
  mean_difference: float | None
  ci: Interval | None
  p_value: float | None
  holm_adjusted_p: float | None = None
  axis: str | None = None  # Mode C: axis effect versus the aligned baseline


# Fields that identify *where/when* an episode ran rather than *what* it computed.
IDENTITY_FIELDS = frozenset({"run_id", "config_hash", "attempt", "env_id", "episode_id", "generation", "started_at",
                             "finished_at", "wall_seconds", "hardware_class", "terminal_state_uri", "diagnostic_uri"})


def semantic_digests(db, run_id: str) -> dict[tuple[str, int], str]:
  """Semantic content per (unit, ordinal), comparable across runs, attempts and clusters.
  The committed digest also covers run_id/config_hash, which legitimately differ."""
  import hashlib
  import json

  out = {}
  for r in db.episode_rows(run_id):
    sem = {k: v for k, v in r["row_data"].items() if k not in IDENTITY_FIELDS}
    out[(r["unit_id"], r["ordinal"])] = hashlib.sha256(json.dumps(sem, sort_keys=True).encode()).hexdigest()
  return out
