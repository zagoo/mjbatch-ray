# Observability

RaySimBatch emits catalog metrics, structured JSON logs, sampled OpenTelemetry spans, a
durable audit trail, and a reproducibility manifest per run (design §11). Telemetry never
blocks or fails simulation: recording errors are absorbed and counted, and log queues
drop and count instead of stalling physics.

## Metrics

The catalog is `CATALOG` in `src/raysimbatch/telemetry/metrics.py` (design §11.1).
`tests/telemetry/test_catalog.py` statically checks every `count`/`gauge`/`observe` call
in the source against the catalog: name, kind, and bounded labels (`mode`,
`model_family`, `task_family`, `node_pool`, `outcome`, `fault_code`, `actor_role`,
`stage`, `actor`, `node`).

**Export path.** In every process connected to Ray (actors, tasks and drivers such as the
Sim2Sim orchestrator), metrics go through `ray.util.metrics` and the node's metrics
agent, on the `metrics-export-port` that the renderer sets from `telemetry.metrics_port`.
Observed on Ray 2.58.0:

- names are exported as **`ray_raysimbatch_<name>`**;
- Ray adds the per-process labels `WorkerId`, `Component`, `SessionName`,
  `NodeAddress` and `Version`, so aggregate them away (`sum by (...)`);
- export happens every `metrics_report_interval_ms` (10 s by default);
- a process's series disappear when it exits, so the last interval of a short-lived
  actor can be missing. End-of-run truth is the database and the run manifest.

Scrape each node's metrics port (the head and every worker pod). `tests/e2e/test_telemetry.py`
scrapes the real exporter during a Mode B run.

| Where | Metrics |
|---|---|
| Simulation workers | `control_steps_total`, `physics_steps_total`, `episodes_total{outcome}`, `aborted_episodes_total`, `slot_health_failures_total`, `warning_total`, `batch_step_seconds` (native call), `reset_seconds`, `set_const_seconds` (each reset that ran it), `observe_seconds` (**sampled: 1 in 16 control steps**; one metric call costs about 1.8 µs, 5–8% of a light model's serial work per step), `publish_seconds`, `publish_bytes_total`, `slots_active`/`slots_ready`/`slots_waiting`, `credits_in_use{stage}`, `credits_available{stage}`, `affinity_violations_total`. Recorded when the node watchdog polls `health` (every `failure.heartbeat_s`): `worker_rss_bytes`, `cpu_busy_ratio` (process CPU time per wall second per leased thread; near 1 for a busy actor, slightly above from Ray's own threads), `worker_private_bytes` (Linux, `/proc/self/smaps_rollup`, every 10 s), `numa_remote_ratio` (Linux, share of resident memory outside the NUMA nodes of the leased CPUs, from `/proc/self/numa_maps`, every 30 s), `log_dropped_total`, `traces_dropped_total` (spans the OpenTelemetry SDK dropped on a full export queue) |
| Coordinators | `control_interval_seconds{stage}` (command round trip), `worker_restarts_total`, `worker_restart_exhausted_total`, `stale_commands_total`, `leases_expired_total`, `healthy_capacity_ratio`, `slots_quarantined` (slots of workers whose restart budget is exhausted), `circuit_state{model_family}`, `worker_progress_age_seconds`, `eval_episodes_committed_total`, `duplicate_commits_total`, `transitions_lost_upper_bound_total` |
| Node watchdogs | `heartbeat_age_seconds` (oldest successful `health` reply among the node's watched workers) |
| Controllers | `placement_ready_seconds`, `scheduler_backlog_seconds` (time waiting for placement, 0 once placed), `eval_units_pending/running/exhausted`, `eval_coverage_ratio`, `eval_success_ratio` (successes among valid episodes, the aggregator's conditional rate) and `eval_return_mean` (over valid episodes): run-level, every 5 s and once more when the work is done |
| Assembler | `fragments_committed_total`, `fragments_rejected_total{outcome}`, `policy_lag_versions`, `policy_lag_seconds`, `sink_ack_seconds`, `duplicate_commits_total`, `queue_bytes{stage="assembler"}` (bytes held awaiting a sink ACK), `outbox_bytes` (written to the durable outbox, not yet delivered) |
| Inference service | `inference_seconds`, `inference_batch_rows`, `inference_deadlines_total`, `inference_errors_total`, `queue_rows`/`queue_bytes`/`oldest_request_seconds` with `stage="inference"` (the backlog left behind, once per batch) |
| Storage (any process) | `storage_commit_seconds`, `storage_errors_total{outcome=retry\|unavailable\|digest_conflict}` |
| Sim2Sim (orchestrator, a Ray driver) | `sim2sim_gate_failures_total{fault_code}`, `sim2sim_validated_bundles_total`, `sim2sim_transfer_ratio` (conservative) and `sim2sim_paired_coverage_ratio` (the coverage the release gate checks) from each scorecard. A one-shot process that exits right after the scorecard may not export them (see above); the scorecard artifact is authoritative |
| Prometheus recording rules (`alerts.yaml`) | `raysimbatch:object_store_used_bytes`, `raysimbatch:object_store_capacity_bytes`, `raysimbatch:object_spilled_bytes_total`, `raysimbatch:object_restored_bytes_total`: the catalog's object-store metrics, per node, derived from Ray's native series (design §11.1). Their Ray 2.58.0 semantics (cumulative spilled and restored bytes) are pinned by `tests/ray_contract/test_ray_metric_semantics.py` |

One catalog metric is **not emitted**, an accepted gap: `native_arena_peak_bytes`.
MJBatch's public API does not expose per-sim arena use (`bind("maxuse_arena")` raises
"unknown field", probed on the pinned revision), and RaySimBatch does not read MJBatch
internals (AM-0). `worker_rss_bytes` and `worker_private_bytes` bound a worker's memory.

## Dashboards

`deploy/observability/dashboards/` holds four Grafana dashboards (tag `raysimbatch`,
linked to each other): **Fleet overview** (throughput against the calibration,
outcomes, capacity, latency), **Liveness and recovery**, **Resources and placement**
(memory, CPU and NUMA placement, object store, backpressure, native costs, placement)
and **Mode A, B and C**. Import them in Grafana (or provision them from that directory)
and pick the Prometheus data source in the `datasource` variable. Descriptions name the
runbook for each paging condition. With Ray's Grafana integration (`RAY_GRAFANA_HOST`),
the Ray Dashboard links to the same Grafana.

The panels are defined once, in `tools/gen_dashboards.py`; never edit the JSON. Checks:

- `tests/telemetry/test_dashboards.py`: the JSON is what the generator produces; every
  query reads an existing series (a catalog metric as Ray exports it, a Ray native series
  from `ray-2.58.0-native-metrics.txt`, or a recording rule); counters only through
  `rate`, histograms only through `_bucket`, gauges never through `rate`; and every
  catalog metric is on a panel except the accepted gap above.
- `qualification/check_alerts.sh` also parses every panel query with the pinned promtool.
- `qualification/verify_dashboards.py` runs the Mode C, B, A (local and central
  inference) and worker-loss end-to-end tests on a local Ray node scraped by the pinned
  Prometheus, then queries every panel (`evidence/observability/dashboards-live.json`).
  Panels over event counters (restarts, storage errors, drops, rejections) show data only
  when the event happens; the Linux-only readings do not exist on a macOS host. It then
  loads the dashboards into the pinned Grafana 13.2.2 by provisioning and through the
  import API, checks that Grafana stores them unchanged with installed panel types, and
  runs every stored panel query through Grafana: it must return the same series as
  Prometheus. `--hold 600` keeps Grafana (anonymous viewer) up to look at the panels.

By default Grafana 12 and later download and update a set of plugins from grafana.com in
the background at startup (`[plugins] preinstall`, `preinstall_auto_update`). In the
check this replaced the bundled Prometheus data source while dashboards queried it, and
their queries failed for 16 s. In production, pin plugin versions or set
`[plugins] preinstall_disabled = true` and install plugins deliberately.

## Alerts

`deploy/observability/alerts.yaml` holds the Prometheus rules for every §11.3 page
condition, plus warnings, each linked to a [runbook](runbooks.md) section. Check them
with:

```bash
qualification/check_alerts.sh
```

It runs `promtool check rules` and `promtool test rules` (the unit tests in
`alerts.test.yaml`) in a digest-pinned `prom/prometheus` image.

- Ray native names used by the rules are recorded in
  `deploy/observability/ray-2.58.0-native-metrics.txt`. Re-verify them on every Ray
  upgrade with `python qualification/verify_ray_metrics.py --check`.
- `raysimbatch:calibrated_control_steps_per_second` is `0` (the throughput alert is
  disabled) until a qualification benchmark on the target hardware provides the value.
  A baseline is never guessed.

## Logs

`telemetry.logs` writes one JSON object per line to stderr: `timestamp` (UTC),
`severity`, `event`, `message`, and, where known, `run_id`, `job_epoch`,
`coordinator_id`, `actor_name`, `generation`, `unit_id`, `env_id`, `episode_id`,
`trace_id` and `error_code`. The handler has these properties:

- It writes on a background thread.
- It is byte-bounded (`telemetry.log_queue_bytes`): records beyond the bound are dropped
  and counted.
- Identical non-error messages are rate-limited per 10 s.

Every coordinator event (worker lost, suspect, restarted, quarantined, command rejected,
capacity degraded, ...) and every run state change is logged. With `log_to_driver`, Ray
forwards actor stderr to the driver; otherwise it is in the node's Ray log directory
(`worker-*.err`).

## Audit trail

Fences, commits, quarantine decisions, run state changes, janitor actions and
configuration changes are rows in the `audit` table (append-only), independent of log
delivery:

```sql
SELECT ts, event, payload FROM audit WHERE run_id = :rid ORDER BY id;
```

## Traces

`telemetry.tracing` records spans for admission, lease acquisition, native slices,
inference batches, durable commits, recovery and migration:

- Successful spans are sampled at `telemetry.trace_fraction` (default 0.1%). Failures are
  always recorded (`trace_errors`).
- With `telemetry.otel_endpoint` set, spans are exported over OTLP/HTTP; otherwise they
  stay in an in-memory exporter.
- Trace IDs travel explicitly in RSB/1 envelopes.

## Reproducibility manifest

Every run writes `run_manifest` (schema `rsb-run-manifest/1`), linked from the run's
results and recorded in the audit trail. It holds:

- the requested and resolved configuration, and its hash;
- the RaySimBatch and MJBatch revisions;
- the MJBatch binary and native-extension SHA-256s, and the engine ABI fingerprint;
- Python and package versions;
- the host: OS, CPU model, microcode, NUMA and SMT;
- BLAS and thread settings;
- artifact digests;
- seed derivation (`rsb-seed-v1`, Philox);
- counts, and failures (lost workers, fenced generations, lost coordinators, gaps);
- `deterministic_replay_certified: false`. Determinism is certified for the *build* by
  qualification (bitwise parity, chaos digests), not re-certified inside each run.
