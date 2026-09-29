# SPDX-License-Identifier: Apache-2.0
"""ResultAggregator (design §6.6, §7.5): durable episode commits and final exports.

Commit protocol: the batch's Parquet part is written content-addressed *first*; the
database transaction then commits one row per (unit, ordinal) with the row's semantic
digest and the part URI, fenced by the unit's attempt and lease epoch. A crash before
the commit leaves an orphan part for garbage collection; a crash after it is
deduplicated on retry. The final manifest is written last, after the database's
expected-count check, and lists every part hash.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from raysimbatch.evaluation import stats
from raysimbatch.evaluation.schema import (
  EPISODE_ARROW,
  EPISODE_SCHEMA_VERSION,
  AggregateResult,
  EpisodeResult,
  episodes_table,
  from_row,
)
from raysimbatch.protocol import Artifact
from raysimbatch.storage.db import CommitAck
from raysimbatch.storage.store import DurableStore

METRICS = ("success", "return", "tracking_rmse")


@dataclass(frozen=True)
class ResultSelection:
  policy_hashes: tuple[str, ...] = ()
  task_hashes: tuple[str, ...] = ()
  axes: tuple[str, ...] = ()
  seeds: tuple[int, ...] = ()

  def match(self, r: EpisodeResult) -> bool:
    return ((not self.policy_hashes or r.policy_hash in self.policy_hashes)
            and (not self.task_hashes or r.task_hash in self.task_hashes)
            and (not self.axes or r.experiment_axis in self.axes)
            and (not self.seeds or r.seed in self.seeds))


@dataclass(frozen=True)
class ResultArtifacts:
  manifest: Artifact
  episodes_parquet: Artifact
  aggregate_parquet: Artifact | None
  aggregate_csv: Artifact | None
  aggregate_json: Artifact | None
  scorecard_markdown: Artifact | None
  episodes_csv: Artifact | None
  maps_csv: Artifact | None
  complete: bool
  coverage: float
  diagnostic_prefix: str = ""


def parquet_bytes(table: pa.Table) -> bytes:
  buf = io.BytesIO()
  pq.write_table(table, buf, compression="zstd", write_statistics=True)
  return buf.getvalue()


class ResultAggregator:
  def __init__(self, run_id: str, partition: int, store: DurableStore, *, replicates: int = 10_000,
               confidence: float = 0.95, task_weights: Mapping[str, float] | None = None,
               exports: Sequence[str] = ("parquet", "csv", "json", "markdown")):
    self.run_id, self.partition, self.store = run_id, partition, store
    self.replicates, self.confidence = replicates, confidence
    self.task_weights = dict(task_weights or {})
    self.exports = tuple(exports)

  async def append_unit(self, unit, results: Sequence[EpisodeResult]) -> CommitAck:
    """Commit a unit's episodes (any subset of its ordinals) durably."""
    if not results:
      return CommitAck(())
    table = episodes_table(results)
    part = await asyncio.to_thread(self.store.put_blob_sync, parquet_bytes(table),
                                   "application/vnd.apache.parquet")
    rows = [{"ordinal": r.episode_ordinal, "digest": r.digest(), "object_uri": part.uri, "outcome": r.outcome,
             "row": r.to_row()} for r in results]
    t0 = time.perf_counter()
    ack = await asyncio.to_thread(self.store.db.commit_episodes, self.run_id, unit.unit_id, unit.attempt,
                                  unit.lease_epoch, rows, unit.episode_count)
    self.last_commit_seconds = time.perf_counter() - t0
    return ack

  async def append(self, results: Sequence[EpisodeResult], units: Mapping[str, Any]) -> CommitAck:
    """Design signature: group rows by unit (``units`` maps unit_id -> leased unit)."""
    by: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
      by[r.unit_id].append(r)
    committed, dup, rejected = [], [], None
    for uid, rs in by.items():
      ack = await self.append_unit(units[uid], rs)
      committed += ack.committed
      dup += ack.duplicates
      rejected = rejected or ack.rejected
    return CommitAck(tuple(committed), tuple(dup), rejected)

  # ----------------------------------------------------------------- reading

  def committed_results(self) -> list[EpisodeResult]:
    return [from_row(r["row_data"]) for r in self.store.db.episode_rows(self.run_id)]

  def synthesized_failures(self) -> list[EpisodeResult]:
    """Explicit rows for ordinals that exhausted or were canceled (never silently absent)."""
    out = []
    committed = defaultdict(set)
    for r in self.store.db.episode_rows(self.run_id):
      committed[r["unit_id"]].add(int(r["ordinal"]))
    for u in self.store.db.units(self.run_id):
      if u["state"] not in ("EXHAUSTED", "CANCELED"):
        continue
      p = u["payload"]
      outcome = "canceled" if u["state"] == "CANCELED" else "infrastructure_failure"
      fault = (u["fault"] or {}).get("code", u["state"])
      for k in range(int(p["episode_count"])):
        if k in committed[u["unit_id"]]:
          continue
        out.append(EpisodeResult(
          schema_version=EPISODE_SCHEMA_VERSION, episode_ordinal=k, attempt=int(u["attempt"]), run_id=self.run_id,
          unit_id=u["unit_id"], policy_hash=p["policy_hash"], model_hash=p["model_hash"], task_hash=p["task_hash"],
          adapter_hash=p["adapter_hash"], config_hash="", seed=int(p["seed"]), env_id=0, episode_id=0, generation=0,
          started_at=0, finished_at=0, physics_steps=0, control_steps=0, simulated_seconds=0.0, wall_seconds=0.0,
          return_sum=None, success=None, outcome=outcome, fault_code=str(fault),
          experiment_axis=p["experiment_axis"],
        ))
    return out

  async def summarize(self, selection: ResultSelection, metric: str = "success") -> list[AggregateResult]:
    results = [r for r in await asyncio.to_thread(self.committed_results) if selection.match(r)]
    units = await asyncio.to_thread(self.store.db.units, self.run_id)
    return self._aggregates(results, units, metrics=(metric,))

  def _aggregates(self, results, units, metrics=METRICS, digest: str = "") -> list[AggregateResult]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for u in units:
      p = u["payload"]
      key = (p["policy_hash"], p["task_hash"], p["experiment_axis"])
      g = groups.setdefault(key, {"task": p["task"], "expected": 0, "seeds": set()})
      g["expected"] += int(p["episode_count"])
      g["seeds"].add(int(p["seed"]))
    out = []
    for (ph, th, axis), g in sorted(groups.items()):
      for m in metrics:
        out.append(stats.aggregate(
          results, run_id=self.run_id, policy_hash=ph, task_hash=th, task_name=g["task"], experiment_axis=axis,
          metric=m, expected_episodes=g["expected"], expected_seeds=len(g["seeds"]), replicates=self.replicates,
          confidence=self.confidence, task_weight=self.task_weights.get(g["task"], 1.0),
          source_manifest_digest=digest,
        ))
    return out

  # ---------------------------------------------------------------- finalize

  async def finalize(self) -> ResultArtifacts:
    return await asyncio.to_thread(self.finalize_sync)

  def finalize_sync(self) -> ResultArtifacts:
    db = self.store.db
    units = db.units(self.run_id)
    counts = db.unit_counts(self.run_id)
    expected = sum(int(u["payload"]["episode_count"]) for u in units)
    rows = db.episode_rows(self.run_id)
    committed = [from_row(r["row_data"]) for r in rows]
    # Final expected-count check against the database before anything is exported.
    if db.episode_count(self.run_id) != len(committed):
      raise RuntimeError("episode count changed during finalization")
    explicit = self.synthesized_failures()
    all_rows = committed + explicit
    parts = sorted({r["object_uri"] for r in rows})
    ep_table = episodes_table(all_rows)
    put = self.store.put_blob_sync
    episodes_art = put(parquet_bytes(ep_table), "application/vnd.apache.parquet")
    digest = hashlib.sha256("".join(parts).encode() + episodes_art.sha256.encode()).hexdigest()
    aggs = self._aggregates(all_rows, units, digest=digest)
    pairs = []
    by_task = defaultdict(set)
    for u in units:
      by_task[u["payload"]["task_hash"]].add(u["payload"]["policy_hash"])
    for th, pols in sorted(by_task.items()):
      pols = sorted(pols)
      for i in range(len(pols)):
        for j in range(i + 1, len(pols)):
          for m in ("success", "return"):
            pairs.append(stats.paired(all_rows, run_id=self.run_id, policy_a=pols[i], policy_b=pols[j],
                                      task_hash=th, metric=m, replicates=self.replicates,
                                      confidence=self.confidence))
    # Mode C: paired one-axis effects (axis vs aligned baseline, common seeds) per task.
    axes_by = defaultdict(set)
    for u in units:
      pl = u["payload"]
      axes_by[(pl["policy_hash"], pl["task"])].add(pl["experiment_axis"])
    task_hash_of = {}
    for u in units:
      pl = u["payload"]
      task_hash_of[(pl["policy_hash"], pl["task"], pl["experiment_axis"])] = pl["task_hash"]
    for (ph, tname), axes in sorted(axes_by.items()):
      if "" not in axes or len(axes) < 2:
        continue
      for axis in sorted(axes - {""}):
        for m in ("success", "return"):
          pr = stats.paired_axis(all_rows, run_id=self.run_id, policy_hash=ph, task_name=tname, axis=axis, metric=m,
                                 replicates=self.replicates, confidence=self.confidence)
          pairs.append(pr)
    adj = stats.holm([p.p_value for p in pairs])
    pairs = [replace(p, holm_adjusted_p=a) for p, a in zip(pairs, adj, strict=True)]
    agg_rows = [a.to_row() for a in aggs]
    arts: dict[str, Artifact | None] = {"aggregate_parquet": None, "aggregate_csv": None, "aggregate_json": None,
                                        "scorecard_markdown": None, "episodes_csv": None, "maps_csv": None}
    if "parquet" in self.exports:
      t = pa.Table.from_pylist([{**r, "fault_counts": json.dumps(r["fault_counts"])} for r in agg_rows])
      arts["aggregate_parquet"] = put(parquet_bytes(t), "application/vnd.apache.parquet")
    if "csv" in self.exports:
      arts["aggregate_csv"] = put(_csv(agg_rows, maps=("fault_counts",)), "text/csv")
      scalar = [{k: v for k, v in r.to_row().items() if k not in ("reward_components", "task_metrics")}
                for r in all_rows]
      arts["episodes_csv"] = put(_csv(scalar), "text/csv")
      long = [{"unit_id": r.unit_id, "episode_ordinal": r.episode_ordinal, "map": m, "key": k, "value": v}
              for r in all_rows for m in ("reward_components", "task_metrics")
              for k, v in sorted(getattr(r, m).items())]
      arts["maps_csv"] = put(_csv(long, columns=("unit_id", "episode_ordinal", "map", "key", "value")), "text/csv")
    if "json" in self.exports:
      doc = {"schema": "AggregateResult", "version": 1, "aggregates": agg_rows,
             "paired": [_paired_row(p) for p in pairs], "suite": self._suite(aggs)}
      arts["aggregate_json"] = put(json.dumps(doc, indent=2, sort_keys=True, default=_nan_null).encode(),
                                   "application/json")
    complete = counts.get("SUCCEEDED", 0) == len(units) and len(committed) == expected
    coverage = sum(1 for r in committed if r.outcome in stats.VALID_OUTCOMES) / expected if expected else 0.0
    if "markdown" in self.exports:
      arts["scorecard_markdown"] = put(self._markdown(aggs, pairs, counts, expected, len(committed), complete,
                                                      coverage).encode(), "text/markdown")
    manifest = {
      "schema": "rsb-eval-manifest/1",
      "run_id": self.run_id,
      "expected_episodes": expected,
      "committed_episodes": len(committed),
      "explicit_failure_rows": len(explicit),
      "unit_states": counts,
      "complete": complete,
      "coverage": coverage,
      "infrastructure_failures": sum(1 for r in all_rows if r.outcome == "infrastructure_failure"),
      "model_failures": sum(1 for r in all_rows if r.outcome == "health_failure"),
      "canceled": sum(1 for r in all_rows if r.outcome == "canceled"),
      "parts": parts,
      "episodes_parquet": episodes_art.sha256,
      "exports": {k: (v.sha256 if v else None) for k, v in arts.items()},
      "ci_protocol": {"method": "percentile_bootstrap_over_seeds", "replicates": self.replicates,
                      "confidence": self.confidence, "bootstrap_seed": "sha256(rsb-bootstrap-v1|run|labels)"},
      "episode_schema": EPISODE_ARROW.metadata[b"version"].decode(),
    }
    man = put(json.dumps(manifest, indent=2, sort_keys=True).encode(), "application/json")  # written last
    db.audit(self.run_id, "results_finalized", {"manifest": man.sha256, "complete": complete, "coverage": coverage})
    return ResultArtifacts(man, episodes_art, arts["aggregate_parquet"], arts["aggregate_csv"], arts["aggregate_json"],
                           arts["scorecard_markdown"], arts["episodes_csv"], arts["maps_csv"], complete, coverage,
                           diagnostic_prefix=f"{self.store.blobs.root}/diagnostics/{self.run_id}/")

  def _suite(self, aggs: Sequence[AggregateResult]) -> dict[str, Any]:
    out = {}
    by_policy = defaultdict(list)
    for a in aggs:
      if a.metric == "success" and a.experiment_axis == "":
        by_policy[a.policy_hash].append(a)
    weights = self.task_weights or {a.task_name: 1.0 for a in aggs}
    for p, lst in by_policy.items():
      out[p] = stats.suite_mean(lst, weights)
    return out

  def _markdown(self, aggs, pairs, counts, expected, committed, complete, coverage) -> str:
    lines = [
      f"# Evaluation scorecard — run `{self.run_id}`", "",
      f"- Status: **{'complete' if complete else 'PARTIAL (not complete)'}**",
      f"- Expected episodes: {expected}; committed: {committed}; valid coverage: {coverage:.1%}",
      f"- Unit states: {json.dumps(counts, sort_keys=True)}",
      f"- Confidence intervals: 95% percentile bootstrap over seed clusters, {self.replicates} replicates; "
      "median/p5/p95 are descriptive episode quantiles, not CIs.", "",
      "| policy | task | axis | metric | mean | 95% CI | seeds | coverage | cond. success | conservative success | infra fail |",
      "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
      ci = "n/a (<2 seeds)" if a.ci is None else f"[{a.ci.low:.4g}, {a.ci.high:.4g}]"
      flag = " (low seed count)" if a.low_seed_count else ""
      lines.append(
        f"| `{a.policy_hash[:12]}` | {a.task_name} | {a.experiment_axis or '-'} | {a.metric} | "
        f"{_fmt(a.mean)} | {ci}{flag} | {a.valid_seeds}/{a.expected_seeds} | {a.coverage:.1%} | "
        f"{_fmt(a.success_rate_conditional)} | {_fmt(a.success_rate_conservative)} | "
        f"{a.infrastructure_failure_fraction:.1%} |")
    if pairs:
      lines += ["", "## Paired policy differences (common seeds; Holm-adjusted across the listed comparisons)", "",
                "| A | B | metric | mean diff | 95% CI | paired coverage | p | Holm p |", "|---|---|---|---:|---|---:|---:|---:|"]
      for p in pairs:
        ci = "n/a" if p.ci is None else f"[{p.ci.low:.4g}, {p.ci.high:.4g}]"
        lines.append(f"| `{p.policy_a[:12]}` | `{p.policy_b[:12]}` | {p.metric} | {_fmt(p.mean_difference)} | {ci} | "
                     f"{p.paired_coverage:.1%} | {_fmt(p.p_value)} | {_fmt(p.holm_adjusted_p)} |")
    lines += ["", "Exports: episodes Parquet (authoritative), aggregate Parquet/CSV/JSON; maps are in the long-form CSV."]
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
  return "null" if v is None else f"{v:.4g}"


def _nan_null(o):
  return None


def _paired_row(p) -> dict[str, Any]:
  d = asdict(p)
  ci = d.pop("ci")
  d["ci_low"] = None if ci is None else ci["low"]
  d["ci_high"] = None if ci is None else ci["high"]
  return d


def _csv(rows: Sequence[Mapping[str, Any]], maps: Sequence[str] = (), columns: Sequence[str] | None = None) -> bytes:
  buf = io.StringIO()
  cols = list(columns or (rows[0].keys() if rows else []))
  cols = [c for c in cols if c not in maps]
  w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
  w.writeheader()
  for r in rows:
    w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
  return buf.getvalue().encode()
