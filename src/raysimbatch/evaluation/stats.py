# SPDX-License-Identifier: Apache-2.0
"""Seed-cluster statistics (design §6.6).

Aggregation is defined on seed clusters: episode values are averaged per seed, and
the mean is the equal-weight mean of seed means. The 95% interval is a percentile
bootstrap over seed IDs with replacement (all episodes of a sampled seed travel
together), 10,000 replicates, with a deterministic bootstrap seed derived from the run
ID. Fewer than two valid seeds give ``ci=None``; fewer than ten set
``low_seed_count``. Median/p5/p95 are descriptive episode quantiles (linear
interpolation), not confidence intervals.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from raysimbatch.evaluation.schema import AggregateResult, EpisodeResult, Interval, PairedResult


def bootstrap_seed(run_id: str, *labels: str) -> int:
  h = hashlib.sha256(("rsb-bootstrap-v1|" + run_id + "|" + "|".join(labels)).encode()).digest()
  return int.from_bytes(h[:8], "little")


def seed_means(values: Sequence[tuple[int, float]]) -> dict[int, float]:
  acc: dict[int, list[float]] = defaultdict(list)
  for seed, v in values:
    acc[seed].append(v)
  return {s: float(np.mean(v)) for s, v in sorted(acc.items())}


def bootstrap_ci(xs: np.ndarray, replicates: int, confidence: float, seed: int) -> Interval | None:
  if xs.size < 2:
    return None
  rng = np.random.Generator(np.random.Philox(seed))
  idx = rng.integers(0, xs.size, size=(replicates, xs.size))
  means = xs[idx].mean(axis=1)
  a = (1.0 - confidence) / 2.0
  lo, hi = np.quantile(means, [a, 1.0 - a], method="linear")
  return Interval(float(lo), float(hi), "percentile_bootstrap_over_seeds", replicates, confidence)


def metric_value(r: EpisodeResult, metric: str) -> float | None:
  if metric == "return":
    return r.return_sum
  if metric == "success":
    return None if r.success is None else float(r.success)
  if metric == "tracking_rmse":
    return r.tracking_rmse
  if metric in r.task_metrics:
    return r.task_metrics[metric]
  return None


VALID_OUTCOMES = frozenset({"success", "task_failure", "time_limit"})


def aggregate(
  results: Sequence[EpisodeResult],
  *,
  run_id: str,
  policy_hash: str,
  task_hash: str,
  task_name: str,
  experiment_axis: str,
  metric: str,
  expected_episodes: int,
  expected_seeds: int,
  replicates: int = 10_000,
  confidence: float = 0.95,
  task_weight: float = 1.0,
  source_manifest_digest: str = "",
) -> AggregateResult:
  rs = [r for r in results if r.policy_hash == policy_hash and r.task_hash == task_hash
        and r.experiment_axis == experiment_axis]
  completed = [r for r in rs if r.outcome != "canceled"]
  valid = [r for r in rs if r.outcome in VALID_OUTCOMES]
  model_failed = [r for r in rs if r.outcome == "health_failure"]
  infra = [r for r in rs if r.outcome == "infrastructure_failure"]
  canceled = [r for r in rs if r.outcome == "canceled"]
  vals = [(r.seed, v) for r in valid if (v := metric_value(r, metric)) is not None]
  sm = seed_means(vals)
  xs = np.asarray(list(sm.values()), np.float64)
  ev = np.asarray([v for _, v in vals], np.float64)
  successes = sum(1 for r in valid if r.success)
  faults: dict[str, int] = defaultdict(int)
  for r in rs:
    if r.fault_code:
      faults[r.fault_code] += 1
  ci = bootstrap_ci(xs, replicates, confidence, bootstrap_seed(run_id, policy_hash, task_hash, experiment_axis, metric))
  return AggregateResult(
    policy_hash=policy_hash,
    task_hash=task_hash,
    task_name=task_name,
    experiment_axis=experiment_axis,
    metric=metric,
    expected_episodes=expected_episodes,
    completed_episodes=len(completed),
    valid_episodes=len(valid),
    failed_episodes=len(model_failed) + len(infra),
    canceled_episodes=len(canceled),
    expected_seeds=expected_seeds,
    valid_seeds=int(xs.size),
    coverage=len(valid) / expected_episodes if expected_episodes else 0.0,
    mean=float(xs.mean()) if xs.size else None,
    std=float(xs.std(ddof=1)) if xs.size > 1 else None,
    median=float(np.quantile(ev, 0.5, method="linear")) if ev.size else None,
    p5=float(np.quantile(ev, 0.05, method="linear")) if ev.size else None,
    p95=float(np.quantile(ev, 0.95, method="linear")) if ev.size else None,
    ci=ci,
    low_seed_count=xs.size < 10,
    success_rate_conditional=successes / len(valid) if valid else None,
    # Conservative: unresolved and model-failed scheduled episodes count as failures.
    success_rate_conservative=successes / expected_episodes if expected_episodes else None,
    infrastructure_failure_fraction=len(infra) / expected_episodes if expected_episodes else 0.0,
    fault_counts=dict(faults),
    task_weight=task_weight,
    source_manifest_digest=source_manifest_digest,
  )


def suite_mean(aggs: Sequence[AggregateResult], weights: Mapping[str, float]) -> float | None:
  """Per-task suite average by declared task weights (never by completion speed)."""
  num = den = 0.0
  for a in aggs:
    if a.mean is None:
      return None  # a missing task makes the suite average undefined, not biased
    w = weights.get(a.task_name, 0.0)
    num += w * a.mean
    den += w
  return num / den if den > 0 else None


def paired(
  results: Sequence[EpisodeResult], *, run_id: str, policy_a: str, policy_b: str, task_hash: str, metric: str,
  replicates: int = 10_000, confidence: float = 0.95,
) -> PairedResult:
  def means(p):
    vals = [(r.seed, v) for r in results if r.policy_hash == p and r.task_hash == task_hash
            and r.outcome in VALID_OUTCOMES and (v := metric_value(r, metric)) is not None]
    return seed_means(vals)

  ma, mb = means(policy_a), means(policy_b)
  common = sorted(set(ma) & set(mb))
  union = set(ma) | set(mb)
  d = np.asarray([ma[s] - mb[s] for s in common], np.float64)
  ci = bootstrap_ci(d, replicates, confidence, bootstrap_seed(run_id, "paired", policy_a, policy_b, task_hash, metric))
  p = None
  if d.size >= 2:
    rng = np.random.Generator(np.random.Philox(bootstrap_seed(run_id, "pvalue", policy_a, policy_b, task_hash)))
    boot = d[rng.integers(0, d.size, size=(replicates, d.size))].mean(axis=1)
    p = _bootstrap_p(boot)
  return PairedResult(
    policy_a, policy_b, task_hash, metric, len(common), len(union),
    len(common) / len(union) if union else 0.0, float(d.mean()) if d.size else None, ci, p,
  )


def _bootstrap_p(boot: np.ndarray) -> float:
  """Two-sided bootstrap p-value for "mean difference = 0". Each tail uses the Monte
  Carlo estimate (k + 1) / (B + 1), so it is never reported as exactly zero from
  finitely many resamples (found in an example run: p = 0.0 for identical differences)."""
  b = boot.size
  lo, hi = (int((boot <= 0).sum()) + 1) / (b + 1), (int((boot >= 0).sum()) + 1) / (b + 1)
  return float(min(1.0, 2.0 * min(lo, hi)))


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
  """Holm step-down adjustment over the predeclared comparisons."""
  idx = [i for i, p in enumerate(pvalues) if p is not None]
  order = sorted(idx, key=lambda i: pvalues[i])  # type: ignore[arg-type,return-value]
  m = len(order)
  out: list[float | None] = [None] * len(pvalues)
  running = 0.0
  for rank, i in enumerate(order):
    adj = min(1.0, (m - rank) * float(pvalues[i]))  # type: ignore[arg-type]
    running = max(running, adj)
    out[i] = running
  return out


def paired_axis(results: Sequence[EpisodeResult], *, run_id: str, policy_hash: str, task_name: str, axis: str,
                metric: str, replicates: int = 10_000, confidence: float = 0.95) -> PairedResult:
  """Axis effect: seed-cluster means of the perturbed axis minus the aligned baseline
  over common seed IDs (Mode C diagnostics)."""
  def means(ax):
    vals = [(r.seed, v) for r in results if r.policy_hash == policy_hash and r.experiment_axis == ax
            and r.outcome in VALID_OUTCOMES and (v := metric_value(r, metric)) is not None]
    return seed_means(vals)

  ma, mb = means(axis), means("")
  common = sorted(set(ma) & set(mb))
  union = set(ma) | set(mb)
  d = np.asarray([ma[s] - mb[s] for s in common], np.float64)
  ci = bootstrap_ci(d, replicates, confidence, bootstrap_seed(run_id, "axis", policy_hash, task_name, axis, metric))
  p = None
  if d.size >= 2:
    rng = np.random.Generator(np.random.Philox(bootstrap_seed(run_id, "axis-p", policy_hash, axis, metric)))
    boot = d[rng.integers(0, d.size, size=(replicates, d.size))].mean(axis=1)
    p = _bootstrap_p(boot)
  return PairedResult(policy_hash, policy_hash, task_name, metric, len(common), len(union),
                      len(common) / len(union) if union else 0.0, float(d.mean()) if d.size else None, ci, p,
                      axis=axis)
