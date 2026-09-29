# SPDX-License-Identifier: Apache-2.0
"""Live check of the Grafana dashboards' queries against real telemetry (production to-do 1.9).

1. start a local Ray node on a known metrics port;
2. start the pinned Prometheus image, scraping that node every 2 s and evaluating
   deploy/observability/alerts.yaml (so the raysimbatch:* recording rules exist);
3. run the Mode C, Mode B, Mode A (local and central inference) and worker-loss
   end-to-end tests on that node, so every workload and recovery path exports its metrics;
4. query every dashboard panel's expression over the run window (query_range) and record
   which returned data;
5. load the dashboards into the pinned Grafana both ways the operations guide names (file
   provisioning, and the import API that a UI import uses), check that Grafana stores them
   unchanged with installed panel types, and run every stored panel query through Grafana:
   it must return the same series as Prometheus.

Panels over event counters (restarts, storage errors, drops, ...) show data only when the
event happens; they are listed as event-driven, not as failures. The script fails when a
panel that healthy runs of these workloads must fill stays empty.

  eval "$(tools/local_services.sh env)"
  python qualification/verify_dashboards.py --out evidence/observability/dashboards-live.json
  ... --hold 600   # keep Grafana (anonymous viewer) and the data up for a look before teardown
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_dashboards  # noqa: E402

IMG = "prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996"
GRAFANA = "grafana/grafana:13.2.2@sha256:ac461fb352abc50da10a51c7d02462e9c05488f11f53f14b3ad79a8145f638a0"
DS_UID = "rsb-prometheus"
# Mode C first: its scorecard gauges are recorded in this (driver) process at the end of
# evaluate, and exported only while the process stays connected (the other tests' time).
# Worker loss in Mode B and Mode A fills the recovery panels (restarts, circuit state,
# tail-loss bound).
TESTS = ["tests/e2e/test_mode_c.py", "tests/e2e/test_mode_b.py::test_mode_b_end_to_end",
         "tests/e2e/test_mode_a.py::test_heterogeneous_rollout", "tests/e2e/test_mode_b_chaos.py::test_worker_loss_mid_run",
         "tests/e2e/test_mode_a.py::test_worker_loss_records_gaps"]
# Series healthy runs of these workloads produce. A panel reading only other series is
# event-driven or platform-specific (listed in the output, not failed).
EXPECTED = {
  "control_steps_total", "physics_steps_total", "episodes_total", "healthy_capacity_ratio", "slots_active",
  "slots_ready", "slots_waiting", "slots_quarantined", "batch_step_seconds", "control_interval_seconds",
  "worker_progress_age_seconds", "heartbeat_age_seconds", "storage_commit_seconds", "worker_rss_bytes",
  "cpu_busy_ratio", "credits_in_use", "credits_available", "queue_rows", "queue_bytes", "oldest_request_seconds",
  "publish_seconds", "publish_bytes_total", "reset_seconds", "observe_seconds", "fragments_committed_total",
  "policy_lag_versions", "policy_lag_seconds", "sink_ack_seconds", "outbox_bytes", "inference_seconds",
  "inference_batch_rows", "eval_units_pending", "eval_units_running", "eval_units_exhausted", "eval_coverage_ratio",
  "eval_episodes_committed_total", "eval_success_ratio", "eval_return_mean", "sim2sim_validated_bundles_total",
  "sim2sim_transfer_ratio", "sim2sim_paired_coverage_ratio", "raysimbatch:object_store_used_bytes",
  "raysimbatch:object_store_capacity_bytes",
}
# Recovery events the worker-loss tests cause. Reported, not required: an event is exported
# only if its process lives one export interval after it (observed: a worker lost 0.3 s
# before its run completed, and the coordinator ended before exporting circuit_state).
EXERCISED = {"worker_restarts_total", "circuit_state", "transitions_lost_upper_bound_total"}


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def prom_query(base: str, expr: str, start: float, end: float) -> int:
  qs = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": "5"})
  data = json.loads(urllib.request.urlopen(f"{base}/api/v1/query_range?{qs}", timeout=30).read())
  if data["status"] != "success":
    raise RuntimeError(f"{expr}: {data}")
  return len(data["data"]["result"])  # series with samples in the window (a gauge at 0 is data)


def raw_expr(name: str) -> str:
  if name.startswith("raysimbatch:"):
    return name
  from raysimbatch.telemetry.metrics import CATALOG

  return f"ray_raysimbatch_{name}_bucket" if CATALOG.get(name) == "histogram" else f"ray_raysimbatch_{name}"


def series_of(expr: str) -> set[str]:
  names = {n for n, _ in re.findall(r"\bray_raysimbatch_([a-z0-9_]+?)(_bucket|_count|_sum)?\b", expr)}
  return names | set(re.findall(r"\b(raysimbatch:[a-z0-9_]+)", expr))


def grafana(base: str, auth: str, method: str, path: str, body: object = None,
            org: int | None = None) -> tuple[int, Any]:
  req = urllib.request.Request(f"{base}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                               headers={"Authorization": auth, "Content-Type": "application/json",
                                        **({"X-Grafana-Org-Id": str(org)} if org else {})})
  try:
    with urllib.request.urlopen(req, timeout=30) as r:
      return r.status, json.loads(r.read())
  except urllib.error.HTTPError as e:
    raw = e.read()
    try:
      return e.code, json.loads(raw)
    except ValueError:
      return e.code, {"body": raw[:300].decode(errors="replace")}


def panels_of(d: dict) -> list[tuple[str, str, list[str]]]:
  """What a viewer gets: each panel's type, title and queries, in order."""
  return [(p["type"], p["title"], [t["expr"] for t in p.get("targets", [])]) for p in d["panels"]]


def grafana_check(base: str, auth: str, datasource: dict, start: float, end: float, direct: dict[str, int]) -> dict:
  def call(method, path, body=None, org=None):
    code, data = grafana(base, auth, method, path, body, org)
    if code != 200:
      raise RuntimeError(f"{method} {path}: HTTP {code} {json.dumps(data)[:300]}")
    return data

  sources = {d["uid"]: d for d in (json.loads(f.read_text()) for f in sorted(gen_dashboards.OUT.glob("*.json")))}
  for _ in range(60):  # org 1: file provisioning, which runs after startup
    if len(call("GET", "/api/search?tag=raysimbatch")) >= len(sources):
      break
    time.sleep(1)
  org = call("POST", "/api/orgs", {"name": "rsb-import"})["orgId"]  # org 2: the import API
  call("POST", "/api/datasources", datasource, org=org)
  for d in sources.values():
    call("POST", "/api/dashboards/import", {"dashboard": d, "overwrite": False, "inputs": []}, org=org)
  installed = {p["id"] for p in call("GET", "/api/plugins?type=panel")} | {"row"}
  problems = []
  for o, how in ((1, "provisioned"), (org, "imported")):
    for uid, d in sources.items():
      code, got = grafana(base, auth, "GET", f"/api/dashboards/uid/{uid}", org=o)
      if code != 200:
        problems.append(f"{how} {uid}: not found (HTTP {code})")
        continue
      if panels_of(got["dashboard"]) != panels_of(d):
        problems.append(f"{how} {uid}: stored panels differ from the file")
      problems += [f"{how} {uid}: panel type {t} is not installed" for t, _, _ in panels_of(got["dashboard"])
                   if t not in installed]
  queries = same = 0
  for uid in sources:  # the stored queries, with ${datasource} resolved to the provisioned source
    for p in call("GET", f"/api/dashboards/uid/{uid}")["dashboard"]["panels"]:
      for t in p.get("targets", []):
        queries += 1
        code, res = grafana(base, auth, "POST", "/api/ds/query", {
          "from": str(int(start * 1000)), "to": str(int(end * 1000)),
          "queries": [{"refId": "A", "datasource": {"type": "prometheus", "uid": DS_UID}, "expr": t["expr"],
                       "range": True, "interval": "5s", "intervalMs": 5000, "maxDataPoints": 100000}]})
        a = res.get("results", {}).get("A", {})
        n = sum(1 for f in a.get("frames", []) if f.get("data", {}).get("values") and len(f["data"]["values"][0]))
        if code != 200 or a.get("error"):
          problems.append(f"query {uid} / {p['title']}: {t['expr']}: HTTP {code} {a.get('error', '')}")
        elif n != direct[t["expr"]]:
          problems.append(f"query {uid} / {p['title']}: {t['expr']}: {n} series via Grafana, "
                          f"{direct[t['expr']]} from Prometheus")
        else:
          same += 1
  return {"image": GRAFANA, "version": call("GET", "/api/health")["version"],
          "prometheus_plugin": call("GET", "/api/plugins/prometheus/settings")["info"]["version"],
          "dashboards": len(sources),
          "panel_types": sorted({t for d in sources.values() for t, _, _ in panels_of(d)}),
          "queries": queries, "same_series_as_prometheus": same, "problems": problems}


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path)
  ap.add_argument("--tests", nargs="*", default=TESTS, help="pytest node ids to run (default: all workloads)")
  ap.add_argument("--hold", type=int, default=0, help="seconds to keep Grafana and Prometheus up after the checks")
  args = ap.parse_args(argv)
  import ray

  metrics_port, prom_port, graf_port = free_port(), free_port(), free_port()
  ray.init(num_cpus=int(os.environ.get("RSB_TEST_RAY_CPUS", "16")), object_store_memory=1 << 30,
           include_dashboard=False, namespace="rsb-tests", log_to_driver=False, _metrics_export_port=metrics_port,
           _system_config={"metrics_report_interval_ms": 1000})
  tmp = Path(tempfile.mkdtemp())
  (tmp / "prometheus.yml").write_text(json.dumps({
    "global": {"scrape_interval": "1s", "evaluation_interval": "5s"},
    "rule_files": ["/rules/alerts.yaml"],
    "scrape_configs": [{"job_name": "ray", "static_configs": [{"targets": [f"host.docker.internal:{metrics_port}"]}]}],
  }))
  name = f"rsb-dashcheck-{os.getpid()}"
  datasource = {"name": "Prometheus", "type": "prometheus", "uid": DS_UID, "url": f"http://{name}:9090",
                "access": "proxy", "isDefault": True}
  for part, doc in (("datasources", {"apiVersion": 1, "datasources": [datasource]}),
                   ("dashboards", {"apiVersion": 1, "providers": [{"name": "raysimbatch", "folder": "RaySimBatch",
                                   "type": "file", "options": {"path": "/var/lib/grafana/dashboards/raysimbatch"}}]})):
    (tmp / "grafana" / part).mkdir(parents=True)
    (tmp / "grafana" / part / "raysimbatch.yaml").write_text(json.dumps(doc))
  password = secrets.token_urlsafe(24)  # throwaway admin of a loopback-only container; never printed
  auth = "Basic " + base64.b64encode(f"admin:{password}".encode()).decode()
  subprocess.run(["docker", "network", "create", name], check=True, capture_output=True)
  subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--network", name,
                  "--add-host", "host.docker.internal:host-gateway",
                  "-p", f"127.0.0.1:{prom_port}:9090", "-v", f"{tmp}:/etc/prometheus:ro",
                  "-v", f"{ROOT / 'deploy' / 'observability'}:/rules:ro", IMG,
                  "--config.file=/etc/prometheus/prometheus.yml"], check=True, capture_output=True)
  subprocess.run(["docker", "run", "-d", "--rm", "--name", f"{name}-grafana", "--network", name,
                  "-p", f"127.0.0.1:{graf_port}:3000", "-e", "GF_SECURITY_ADMIN_PASSWORD",
                  "-e", "GF_AUTH_ANONYMOUS_ENABLED=true", "-e", "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer",
                  "-e", "GF_ANALYTICS_REPORTING_ENABLED=false", "-e", "GF_ANALYTICS_CHECK_FOR_UPDATES=false",
                  "-e", "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false",
                  # Use the image's bundled plugins. By default Grafana downloads plugins in the
                  # background and replaced the bundled Prometheus data source mid-check (its
                  # queries failed "plugin not registered" while it was reinstalled).
                  "-e", "GF_PLUGINS_PREINSTALL_DISABLED=true",
                  "-v", f"{tmp / 'grafana'}:/etc/grafana/provisioning:ro",
                  "-v", f"{gen_dashboards.OUT}:/var/lib/grafana/dashboards/raysimbatch:ro", GRAFANA],
                 check=True, capture_output=True, env={**os.environ, "GF_SECURITY_ADMIN_PASSWORD": password})
  base, graf = f"http://127.0.0.1:{prom_port}", f"http://127.0.0.1:{graf_port}"
  try:
    # Start the tests once the Ray node has been scraped: the first scrape comes about 5 s
    # after Prometheus is ready, which is longer than a short test runs.
    qs = urllib.parse.urlencode({"query": 'up{job="ray"} == 1'})
    for _ in range(60):
      try:
        if json.loads(urllib.request.urlopen(f"{base}/api/v1/query?{qs}", timeout=2).read())["data"]["result"]:
          break
      except OSError:
        pass
      time.sleep(1)
    start = time.time()
    import pytest

    # tests/conftest.py ends the process after the session (an onnxruntime exit abort);
    # this script continues, and ends the same way itself (below).
    os.environ["RSB_TEST_NATURAL_EXIT"] = "1"
    rc = pytest.main(["-q", "-p", "no:cacheprovider", *[str(ROOT / t) for t in args.tests]])
    tests_end = time.time()  # the tests' Ray fixture shuts the shared node down when they end
    time.sleep(12)  # two more rule evaluations over the stored samples
    end = time.time()
    qs = urllib.parse.urlencode({"query": 'up{job="ray"}', "start": start, "end": tests_end, "step": "2"})
    ups = [float(v) for r in json.loads(urllib.request.urlopen(f"{base}/api/v1/query_range?{qs}", timeout=30)
                                        .read())["data"]["result"] for _, v in r["values"]]
    rows, missing, short = [], [], []
    for dash, title, expr in gen_dashboards.expressions():
      n = prom_query(base, expr, start, end)
      want = series_of(expr) & EXPECTED
      row = {"dashboard": dash, "panel": title, "expr": expr, "series": n, "expected_in_healthy_runs": bool(want)}
      if want and not n:
        # rate() needs two samples: a counter of a coordinator that ended seconds after
        # the event can have one. Then check that the raw series was exported at all.
        raw = {w: prom_query(base, raw_expr(w), start, end) for w in want}
        row["raw_series"] = raw
        if all(raw.values()):
          short.append(f"{dash} / {title}: {expr}")
        else:
          missing.append(f"{dash} / {title}: {expr}")
      rows.append(row)
    report = {"tests_exit": int(rc), "window_s": round(end - start, 1),
              "scrape_up_fraction": round(sum(ups) / len(ups), 3) if ups else 0.0,
              "queries": len(rows), "with_data": sum(1 for r in rows if r["series"]),
              "expected_with_data": sum(1 for r in rows if r["expected_in_healthy_runs"] and r["series"]),
              "expected": sum(1 for r in rows if r["expected_in_healthy_runs"]),
              "event_driven_or_platform_empty": [f"{r['dashboard']} / {r['panel']}: {r['expr']}" for r in rows
                                                 if not r["series"] and not r["expected_in_healthy_runs"]
                                                 and not series_of(r["expr"]) & EXERCISED],
              "exported_but_too_short_for_rate": short,
              "recovery_events_observed": {w: any(r["series"] for r in rows if w in series_of(r["expr"]))
                                           for w in sorted(EXERCISED)},
              "missing": missing, "rows": rows}
    try:
      report["grafana"] = grafana_check(graf, auth, datasource, start, end, {r["expr"]: r["series"] for r in rows})
    except Exception as e:  # noqa: BLE001  (recorded and failed below)
      report["grafana"] = {"image": GRAFANA, "problems": [f"check aborted: {e!r}"]}
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, indent=1))
    if args.out:
      args.out.parent.mkdir(parents=True, exist_ok=True)
      args.out.write_text(json.dumps(report, indent=1) + "\n")
    if args.hold:
      print(f"Grafana: {graf}/dashboards?tag=raysimbatch (up for {args.hold} s)", flush=True)
      time.sleep(args.hold)
  finally:
    subprocess.run(["docker", "rm", "-f", name, f"{name}-grafana"], capture_output=True)
    subprocess.run(["docker", "network", "rm", name], capture_output=True)
    ray.shutdown()
  ok = rc == 0 and not missing and report["scrape_up_fraction"] >= 0.9
  return 0 if ok and not report["grafana"]["problems"] else 1


if __name__ == "__main__":
  code = main()
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(code)  # skip third-party destructors (see tests/conftest.py pytest_unconfigure)
