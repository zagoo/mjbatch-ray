# SPDX-License-Identifier: Apache-2.0
"""Generate the Grafana dashboards in deploy/observability/dashboards/ (design §11.1-11.3,
production to-do 1.9). The panels are defined here, once; the JSON is generated.

  python tools/gen_dashboards.py              # write the dashboards
  python tools/gen_dashboards.py --check      # fail if the committed JSON differs
  python tools/gen_dashboards.py --rules-out r.yaml   # every panel query as a recording
                                                      # rule, for `promtool check rules`

Queries follow the alert rules' conventions (deploy/observability/alerts.yaml): catalog
metrics are exported by Ray as ray_raysimbatch_<name> with Ray's per-process labels,
which every query aggregates away (sum/max/min by bounded labels only); object-store
catalog metrics come from the raysimbatch:* recording rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "observability" / "dashboards"
DS = {"type": "prometheus", "uid": "${datasource}"}
W = "[5m]"


def m(name: str) -> str:
  return f"ray_raysimbatch_{name}"


def rate(name: str, by: str = "") -> str:
  return f"sum{f' by ({by})' if by else ''} (rate({m(name)}{W}))"


def q(name: str, quantile: float, by: str = "") -> str:
  return f"histogram_quantile({quantile}, sum by (le{', ' + by if by else ''}) (rate({m(name)}_bucket{W})))"


def panel(title: str, targets: list[tuple[str, str]], unit: str = "short", desc: str = "", kind: str = "timeseries",
          stack: bool = False) -> dict:
  p = {"type": kind, "title": title, "description": desc, "datasource": DS,
       "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": lg, "datasource": DS}
                   for i, (e, lg) in enumerate(targets)],
       "fieldConfig": {"defaults": {"unit": unit}, "overrides": []}, "options": {}}
  if stack:
    p["fieldConfig"]["defaults"]["custom"] = {"stacking": {"mode": "normal"}}
  return p


def dashboard(uid: str, title: str, desc: str, rows: list[tuple[str, list[dict]]]) -> dict:
  panels, pid, y = [], 1, 0
  for row_title, row_panels in rows:
    panels.append({"type": "row", "title": row_title, "id": pid, "collapsed": False,
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []})
    pid, y = pid + 1, y + 1
    for i, p in enumerate(row_panels):
      panels.append({**p, "id": pid, "gridPos": {"h": 8, "w": 12, "x": 12 * (i % 2), "y": y + 8 * (i // 2)}})
      pid += 1
    y += 8 * ((len(row_panels) + 1) // 2)
  return {
    "uid": uid, "title": title, "description": desc, "tags": ["raysimbatch"], "timezone": "utc",
    "schemaVersion": 39, "version": 1, "editable": True, "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"},
    "templating": {"list": [{"name": "datasource", "label": "Prometheus", "type": "datasource",
                             "query": "prometheus", "current": {}, "hide": 0}]},
    "annotations": {"list": []},
    "links": [{"title": "RaySimBatch dashboards", "type": "dashboards", "tags": ["raysimbatch"], "asDropdown": True}],
    "panels": panels,
  }


def overview() -> dict:
  return dashboard("rsb-overview", "RaySimBatch / Fleet overview",
                   "Throughput, outcomes, capacity and latency of all runs on this cluster.", [
    ("Throughput", [
      panel("Control steps per second", [(rate("control_steps_total"), "wrapper"),
                                         ("raysimbatch:calibrated_control_steps_per_second", "calibration (0 = unset)")],
            "ops", "Transitions per second. The calibration line is the throughput alert's baseline (runbook: "
                   "Throughput regression)."),
      panel("Physics steps per second", [(rate("physics_steps_total"), "substeps")], "ops"),
      panel("Episodes per second by outcome", [(rate("episodes_total", "outcome"), "{{outcome}}"),
                                               (rate("aborted_episodes_total"), "aborted (infrastructure/health)")],
            "ops", stack=True),
      panel("Slot health failures and engine warnings", [(rate("slot_health_failures_total"), "health failures"),
                                                         (rate("warning_total"), "MuJoCo warnings")], "ops"),
    ]),
    ("Capacity", [
      panel("Healthy capacity", [(f"min by (mode) ({m('healthy_capacity_ratio')})", "{{mode}}")], "percentunit",
            "Pages below 95% (runbook: Capacity degraded)."),
      panel("Slots by state", [(f"sum({m('slots_active')})", "active"), (f"sum({m('slots_ready')})", "ready"),
                               (f"sum({m('slots_waiting')})", "waiting"),
                               (f"sum({m('slots_quarantined')})", "quarantined (restart budget exhausted)")], "short"),
      panel("Circuit state by model family", [(f"max by (model_family) ({m('circuit_state')})", "{{model_family}}")],
            "short", "0 closed, 1 open, 2 half-open (runbook: Worker quarantined)."),
    ]),
    ("Latency", [
      panel("Native slice (batch_step_seconds)", [(q("batch_step_seconds", 0.5), "p50"),
                                                  (q("batch_step_seconds", 0.99), "p99")], "s"),
      panel("Command round trip (control_interval_seconds)",
            [(q("control_interval_seconds", 0.99, "stage"), "p99 {{stage}}")], "s"),
    ]),
  ])


def liveness() -> dict:
  return dashboard("rsb-liveness", "RaySimBatch / Liveness and recovery",
                   "Progress versus responsiveness, restarts, fencing and durable storage.", [
    ("Liveness", [
      panel("Active native call age", [(f"max({m('worker_progress_age_seconds')})", "oldest in-flight call")], "s",
            "Pages above 30 s, the maximum call deadline (runbook: Native hang)."),
      panel("Heartbeat age by node", [(f"max by (node) ({m('heartbeat_age_seconds')})", "{{node}}")], "s",
            "Oldest successful health reply among a node's watched workers (AM-16)."),
      panel("Worker restarts", [(rate("worker_restarts_total"), "restarts"),
                                (rate("worker_restart_exhausted_total"), "restart budget exhausted")], "ops"),
    ]),
    ("Fencing and deduplication", [
      panel("Stale commands and expired leases", [(rate("stale_commands_total"), "stale commands"),
                                                  (rate("leases_expired_total"), "leases expired")], "ops"),
      panel("Duplicate commits (deduplicated)", [(rate("duplicate_commits_total"), "duplicates")], "ops"),
    ]),
    ("Durable storage", [
      panel("Storage commit latency", [(q("storage_commit_seconds", 0.5), "p50"),
                                       (q("storage_commit_seconds", 0.99), "p99")], "s"),
      panel("Storage errors by outcome", [(rate("storage_errors_total", "outcome"), "{{outcome}}")], "ops",
            "digest_conflict pages (runbook: Digest conflict); unavailable: Database outage."),
    ]),
    ("Telemetry degradation", [
      panel("Dropped logs and spans", [(rate("log_dropped_total"), "log records"),
                                       (rate("traces_dropped_total"), "spans (export queue full)")], "ops"),
    ]),
  ])


def resources() -> dict:
  return dashboard("rsb-resources", "RaySimBatch / Resources and placement",
                   "Memory, CPU placement quality, object store, backpressure and control-plane delays.", [
    ("Memory", [
      panel("Worker memory (largest worker)", [(f"max({m('worker_rss_bytes')})", "RSS"),
                                               (f"max({m('worker_private_bytes')})", "private (Linux)")], "bytes"),
      panel("Object store by node", [("raysimbatch:object_store_used_bytes", "used {{NodeAddress}}"),
                                     ("raysimbatch:object_store_capacity_bytes", "capacity {{NodeAddress}}")], "bytes",
            "Occupancy above 85% pages (runbook: Object store pressure)."),
      panel("Spill and restore throughput", [("sum(rate(raysimbatch:object_spilled_bytes_total[5m]))", "spilled"),
                                             ("sum(rate(raysimbatch:object_restored_bytes_total[5m]))", "restored")],
            "Bps"),
    ]),
    ("CPU placement", [
      panel("CPU busy per leased thread", [(f"avg by (model_family) ({m('cpu_busy_ratio')})", "mean {{model_family}}"),
                                           (f"min({m('cpu_busy_ratio')})", "least busy worker")], "percentunit",
            "Process CPU time per wall second per leased thread; a busy simulation actor is near 1."),
      panel("NUMA remote memory and affinity violations",
            [(f"max({m('numa_remote_ratio')})", "worst remote-memory share (Linux)"),
             (rate("affinity_violations_total"), "affinity violations/s")], "short",
            "Runbook: NUMA and CPU imbalance."),
    ]),
    ("Backpressure", [
      panel("Credits by stage", [(f"sum by (stage) ({m('credits_in_use')})", "in use {{stage}}"),
                                 (f"sum by (stage) ({m('credits_available')})", "available {{stage}}")], "short"),
      panel("Queued rows and bytes by stage", [(f"sum by (stage) ({m('queue_rows')})", "rows {{stage}}"),
                                               (f"sum by (stage) ({m('queue_bytes')})", "bytes {{stage}}")], "short"),
      panel("Oldest queued request", [(f"max by (stage) ({m('oldest_request_seconds')})", "{{stage}}")], "s"),
      panel("Frame publication", [(q("publish_seconds", 0.99), "p99 latency"),
                                  (rate("publish_bytes_total"), "bytes/s")], "short"),
    ]),
    ("Native and adapter costs", [
      panel("Reset, set_const and observation (p99)", [(q("reset_seconds", 0.99), "reset"),
                                                       (q("set_const_seconds", 0.99), "set_const"),
                                                       (q("observe_seconds", 0.99), "observe (1 in 16 steps)")], "s"),
    ]),
    ("Control plane", [
      panel("Placement", [(f"max({m('scheduler_backlog_seconds')})", "waiting for placement"),
                          (q("placement_ready_seconds", 0.99), "p99 time to ready")], "s"),
    ]),
  ])


def workloads() -> dict:
  return dashboard("rsb-workloads", "RaySimBatch / Mode A, B and C",
                   "Rollout delivery and inference, evaluation progress, Sim2Sim transfer.", [
    ("Mode A: rollout delivery", [
      panel("Fragments", [(rate("fragments_committed_total"), "committed"),
                          (rate("fragments_rejected_total", "outcome"), "rejected {{outcome}}")], "ops"),
      panel("Transitions lost (upper bound)", [(rate("transitions_lost_upper_bound_total"), "per second")], "ops",
            "Conservative tail-loss bound from worker loss (gap records)."),
      panel("Policy lag", [(q("policy_lag_versions", 0.99), "p99 versions"),
                           (q("policy_lag_seconds", 0.99), "p99 seconds")], "short", "Runbook: Policy lag."),
      panel("Sink delivery", [(q("sink_ack_seconds", 0.99), "p99 ack latency (s)"),
                              (f"sum({m('outbox_bytes')})", "outbox bytes awaiting delivery")], "short"),
    ]),
    ("Mode A: inference", [
      panel("Inference latency", [(q("inference_seconds", 0.5), "p50"), (q("inference_seconds", 0.99), "p99")], "s"),
      panel("Inference batch rows", [(q("inference_batch_rows", 0.5), "p50"),
                                     (q("inference_batch_rows", 0.99), "p99")], "short"),
      panel("Inference failures", [(rate("inference_deadlines_total"), "deadline misses"),
                                   (rate("inference_errors_total"), "errors")], "ops",
            "Runbook: Inference overload."),
    ]),
    ("Mode B: evaluation", [
      panel("Units", [(f"sum({m('eval_units_pending')})", "pending"), (f"sum({m('eval_units_running')})", "running"),
                      (f"sum({m('eval_units_exhausted')})", "exhausted")], "short"),
      panel("Coverage and committed episodes", [(f"min({m('eval_coverage_ratio')})", "coverage"),
                                                (rate("eval_episodes_committed_total"), "episodes/s")], "short",
            "Runbook: Eval coverage."),
      panel("Suite results so far", [(f"avg({m('eval_success_ratio')})", "success among valid episodes"),
                                     (f"avg({m('eval_return_mean')})", "mean return")], "short",
            "Running summaries; the run's aggregate artifact is authoritative."),
    ]),
    ("Mode C: Sim2Sim", [
      panel("Gates", [(rate("sim2sim_gate_failures_total", "fault_code"), "failed {{fault_code}}"),
                      (rate("sim2sim_validated_bundles_total"), "validated bundles")], "ops",
            "Runbook: Sim2Sim score regression."),
      panel("Transfer", [(f"min({m('sim2sim_transfer_ratio')})", "conservative transfer ratio"),
                         (f"min({m('sim2sim_paired_coverage_ratio')})", "paired coverage")], "percentunit"),
    ]),
  ])


DASHBOARDS = {"overview": overview, "liveness": liveness, "resources": resources, "workloads": workloads}


def render() -> dict[str, str]:
  return {f"raysimbatch-{k}.json": json.dumps(f(), indent=1, ensure_ascii=False) + "\n" for k, f in DASHBOARDS.items()}


def expressions() -> list[tuple[str, str, str]]:
  """(dashboard, panel title, expr) for every query."""
  out = []
  for name, f in DASHBOARDS.items():
    for p in f()["panels"]:
      for t in p.get("targets", ()):
        out.append((name, p["title"], t["expr"]))
  return out


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--check", action="store_true")
  ap.add_argument("--rules-out", type=Path)
  args = ap.parse_args(argv)
  if args.rules_out:
    rules = [{"record": f"rsb_dashboard_query_{i}", "expr": e} for i, (_, _, e) in enumerate(expressions())]
    args.rules_out.write_text(json.dumps({"groups": [{"name": "dashboard-queries", "rules": rules}]}, indent=1))
    return 0
  files = render()
  if args.check:
    stale = [n for n, text in files.items() if not (OUT / n).is_file() or (OUT / n).read_text() != text]
    extra = sorted(p.name for p in OUT.glob("*.json") if p.name not in files) if OUT.is_dir() else []
    if stale or extra:
      print(f"dashboards out of date: {stale + extra}; run python tools/gen_dashboards.py")
      return 1
    print(f"{len(files)} dashboards up to date")
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  for n, text in files.items():
    (OUT / n).write_text(text)
  print(f"wrote {len(files)} dashboards to {OUT.relative_to(ROOT)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
