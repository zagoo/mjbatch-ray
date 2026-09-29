# SPDX-License-Identifier: Apache-2.0
"""Metrics (design §11.1): ``raysimbatch_`` names, bounded labels, fixed buckets.

In any process connected to Ray (actors, tasks, and drivers such as the Sim2Sim
orchestrator or the CLI) metrics use ``ray.util.metrics``, exported by the node's Ray
metrics agent on ``metrics-export-port``; elsewhere a ``prometheus_client`` registry.
Unbounded identifiers (env/episode ids, checkpoint hashes) are rejected as labels.
Rates are derived from counters by Prometheus, never incremented as gauges.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)
ALLOWED_LABELS = frozenset({"mode", "model_family", "task_family", "node_pool", "outcome", "fault_code", "actor_role",
                            "stage", "actor", "node"})

CATALOG: dict[str, str] = {
  "physics_steps_total": "counter", "control_steps_total": "counter", "episodes_total": "counter",
  "aborted_episodes_total": "counter", "batch_step_seconds": "histogram", "control_interval_seconds": "histogram",
  "reset_seconds": "histogram", "set_const_seconds": "histogram", "observe_seconds": "histogram",
  "slots_active": "gauge", "slots_ready": "gauge", "slots_waiting": "gauge", "slots_quarantined": "gauge",
  "slot_health_failures_total": "counter", "warning_total": "counter", "worker_restarts_total": "counter",
  "worker_restart_exhausted_total": "counter", "worker_progress_age_seconds": "gauge",
  "heartbeat_age_seconds": "gauge", "worker_rss_bytes": "gauge", "worker_private_bytes": "gauge",
  "native_arena_peak_bytes": "gauge", "cpu_busy_ratio": "gauge", "numa_remote_ratio": "gauge",
  "affinity_violations_total": "counter", "queue_rows": "gauge", "queue_bytes": "gauge",
  "oldest_request_seconds": "gauge", "credits_available": "gauge", "credits_in_use": "gauge",
  "object_store_used_bytes": "gauge", "object_store_capacity_bytes": "gauge",
  "object_spilled_bytes_total": "counter", "object_restored_bytes_total": "counter",
  "publish_seconds": "histogram", "publish_bytes_total": "counter", "inference_seconds": "histogram",
  "inference_batch_rows": "histogram", "inference_deadlines_total": "counter", "inference_errors_total": "counter",
  "policy_lag_versions": "histogram", "policy_lag_seconds": "histogram", "fragments_committed_total": "counter",
  "fragments_rejected_total": "counter", "transitions_lost_upper_bound_total": "counter",
  "sink_ack_seconds": "histogram", "outbox_bytes": "gauge", "eval_units_pending": "gauge",
  "eval_units_running": "gauge", "eval_units_exhausted": "gauge", "eval_episodes_committed_total": "counter",
  "eval_coverage_ratio": "gauge", "eval_success_ratio": "gauge", "eval_return_mean": "gauge",
  "sim2sim_gate_failures_total": "counter", "sim2sim_validated_bundles_total": "counter",
  "sim2sim_transfer_ratio": "gauge", "sim2sim_paired_coverage_ratio": "gauge", "leases_expired_total": "counter",
  "stale_commands_total": "counter", "duplicate_commits_total": "counter", "scheduler_backlog_seconds": "gauge",
  "placement_ready_seconds": "histogram", "storage_errors_total": "counter", "storage_commit_seconds": "histogram",
  "circuit_state": "gauge", "healthy_capacity_ratio": "gauge", "log_dropped_total": "counter",
  "traces_dropped_total": "counter",
}


class MetricsExporter:
  def __init__(self, tags: Mapping[str, str], prefix: str = "raysimbatch"):
    bad = set(tags) - ALLOWED_LABELS
    if bad:
      raise ValueError(f"unbounded or unknown metric labels: {sorted(bad)}")
    self.tags = dict(tags)
    self.prefix = prefix
    self._metrics: dict[str, Any] = {}
    self._lock = threading.Lock()
    self._ray = _in_ray()

  def _get(self, metric: str, kind: str, label_keys: tuple[str, ...]):
    if metric not in CATALOG:
      raise KeyError(f"metric {metric!r} is not in the catalog")
    if CATALOG[metric] != kind:
      raise TypeError(f"{metric} is a {CATALOG[metric]}, not a {kind}")
    key = f"{metric}|{','.join(label_keys)}"
    with self._lock:
      m = self._metrics.get(key)
      if m is None:
        name = f"{self.prefix}_{metric}"
        keys = tuple(sorted(set(self.tags) | set(label_keys)))
        if self._ray:
          from ray.util import metrics as rm

          if kind == "counter":
            m = rm.Counter(name, tag_keys=keys)
          elif kind == "gauge":
            m = rm.Gauge(name, tag_keys=keys)
          else:
            m = rm.Histogram(name, boundaries=list(BUCKETS), tag_keys=keys)
        else:
          import prometheus_client as pc

          reg = _registry()
          if kind == "counter":
            m = pc.Counter(name.removesuffix("_total"), name, keys, registry=reg)
          elif kind == "gauge":
            m = pc.Gauge(name, name, keys, registry=reg)
          else:
            m = pc.Histogram(name, name, keys, buckets=BUCKETS, registry=reg)
        self._metrics[key] = (m, keys)
      return self._metrics[key]

  def _labels(self, keys: tuple[str, ...], extra: Mapping[str, str]) -> dict[str, str]:
    bad = set(extra) - ALLOWED_LABELS
    if bad:
      raise ValueError(f"unbounded labels {sorted(bad)}")
    d = {**self.tags, **{k: str(v) for k, v in extra.items()}}
    return {k: d.get(k, "") for k in keys}

  # Recording never raises into the data path (observed: Ray's Counter.inc(0) raises
  # ValueError, which failed a simulation command). Failures are counted and logged once
  # per metric; catalog conformance is enforced statically by tests.
  def count(self, metric: str, value: int = 1, **labels: str) -> None:
    if value <= 0:
      return
    self._record(metric, "counter", value, labels)

  def gauge(self, metric: str, value: float, **labels: str) -> None:
    self._record(metric, "gauge", value, labels)

  def observe(self, metric: str, seconds: float, **labels: str) -> None:
    self._record(metric, "histogram", seconds, labels)

  def _record(self, metric: str, kind: str, value: float, labels: Mapping[str, str]) -> None:
    try:
      m, keys = self._get(metric, kind, tuple(sorted(labels)))
      lab = self._labels(keys, labels)
      method = _METHOD[kind]
      if self._ray:
        getattr(m, method)(value, tags=lab)
      else:
        getattr(m.labels(**lab), method)(value)
    except Exception as exc:  # noqa: BLE001 - telemetry degradation, never a data-path failure
      global ERRORS
      ERRORS += 1
      if metric not in _REPORTED:
        _REPORTED.add(metric)
        import logging

        logging.getLogger("raysimbatch").error("metric %s dropped: %s", metric, exc, extra={"event": "metric_error"})

  def flush(self) -> None:
    """Ray/prometheus export asynchronously; nothing is buffered here."""


_METHOD = {"counter": "inc", "gauge": "set", "histogram": "observe"}
_REG = None
ERRORS = 0
_REPORTED: set[str] = set()


def _registry():
  global _REG
  if _REG is None:
    import prometheus_client as pc

    _REG = pc.CollectorRegistry()
  return _REG


def exposition() -> bytes:
  import prometheus_client as pc

  return pc.generate_latest(_registry())


def _in_ray() -> bool:
  try:
    import ray

    # Drivers too: Ray exports their metrics (observed on 2.58.0). Limiting this to worker
    # processes left the Sim2Sim orchestrator's metrics in an unscraped local registry.
    return ray.is_initialized()
  except Exception:  # noqa: BLE001
    return False
