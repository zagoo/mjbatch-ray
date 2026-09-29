# Operational runbooks

These runbooks cover the ten incident classes in design §11.6, plus sections for the
§11.3 alerts in `deploy/observability/alerts.yaml`. Every alert names one of the
sections below, and `tests/telemetry/test_alerts.py` fails if a section is missing.

Conventions used throughout:

- `NS` is the Kubernetes namespace of the RayCluster (for example `raysimbatch-qual`). `HEAD`
  is the head pod: `kubectl -n $NS get pod -l ray.io/node-type=head`.
- `CFG` is the run's configuration file. Every `raysimbatch` command reads the database
  and artifact store from it.
- Durable state is authoritative. Ray actor state, dashboards and metrics are views.
  Before you decide anything, read the run row, the unit counts and the `audit` table:

```sql
SELECT state, epoch, updated_at FROM runs WHERE run_id = :rid;
SELECT state, count(*) FROM units WHERE run_id = :rid GROUP BY state;
SELECT ts, event, payload FROM audit WHERE run_id = :rid ORDER BY id DESC LIMIT 50;
```

- Metrics are exported by Ray as `ray_raysimbatch_<name>`. Ray adds per-process labels
  (`WorkerId`, `Component`, `SessionName`), so aggregate them away. A process's series
  disappear when it exits, so the last export interval of an actor that ended can be
  missing. For end-of-run counts, use the run manifest (`results.run_manifest`) and
  the database.
- Never delete durable rows or blobs by hand while you contain an incident. Deletion is
  an audited administrative action (§11.5).

---

## Object store pressure

*Runbook 1 (object store full). Alerts: `RsbObjectStoreOccupancyHigh`, `RsbSpillPersisting`.*

**Diagnose**
1. Find where the pressure is: `raysimbatch:object_store_occupancy` by `NodeAddress`,
   and `ray_object_store_memory{Location="SPILLED"}`.
2. Find who owns the objects: `kubectl -n $NS exec $HEAD -c ray-head -- ray memory --stats-only`,
   then `ray memory --group-by STACK_TRACE --sort-by OBJECT_SIZE` for the top owners.
3. Check the credits. Frames and fragments are bounded by
   `flow.inflight_frames_per_worker` and `flow.fragment_slabs_per_worker`, so steady
   growth means a consumer is not acknowledging. Look at
   `ray_raysimbatch_credits_in_use{stage=...}` per worker and at `sink_ack_seconds`.
4. Check the spill disk: `kubectl -n $NS exec <pod> -- df -h /var/lib/ray-spill`.

**Contain and recover**
- A slow sink is the usual cause. Assembler byte credits make callers wait instead of
  dropping data. Fix the sink (see [Database outage](#database-outage)) before you add memory.
- Add object-store capacity (`cluster.object_store_gib`, which is a new-run change)
  only after you have ruled out an ownership leak, that is, refs held after their
  credit was acknowledged.

**Verify before closing**: occupancy below 60% for 30 s, no hot spilling, and
`credits_in_use` returning to the idle baseline between commands.

## Head or GCS failure

*Runbook 2.*

**Diagnose**: `kubectl -n $NS describe pod $HEAD` (look for restarts and OOMKilled),
`kubectl -n $NS logs $HEAD -c ray-head --previous`, and `ray status`.

**Contain and recover**
- Freeze admissions: stop submitting, and point the release at another color if one is
  healthy (`deploy/release.sh --context <ctx> --namespace <ns> rollback`; on kind `deploy/kind/release.sh rollback`).
- Without qualified Redis GCS fault tolerance (`cluster.gcs_ft=none`, the default), a
  head loss ends every actor. Rebuild the RayCluster, then resume each non-terminal
  run from durable state with `raysimbatch resume <run_id> --config $CFG`. Committed
  episodes are kept. Other ordinals restart from their original seeds.
- Redis-backed GCS FT is BLOCKED (not qualified; see the qualification report).

**Verify**: the resumed run's `epoch` increased, old generations are fenced
(`worker_fenced` and `coordinator_replaced` audit events), and expected unit counts are unchanged.

Observed in qualification: a node disk filled by repeated image loads crash-looped the
head with `ENOSPC` in `/tmp/ray`. Run `deploy/kind/release.sh prune-images` (kind), and in
production budget node ephemeral storage.

## Crashing model asset

*Runbook 3 (chronically crashing MJCF).*

**Diagnose**: in `audit`, count `worker_lost` by reason. Per-slot native failures appear
as `episodes.outcome = 'health_failure'` with a `fault_code`. The randomization hash of
each failed episode is in its row.

**Contain and recover**
- Quarantine the model hash or randomization bin (`quarantine` table). Workers keep
  running the other bins.
- Reproduce on the scalar lane with the sanitizer image (`deploy/docker/cpu.Dockerfile`,
  target `sanitize`, driven by `qualification/native_sanitizers.sh`).
- Roll the asset back by registering the previous content hash in the run configuration
  (a new run).

**Verify**: 100 canary episodes and the parity suite pass on the fixed asset, and the
exclusions are recorded in the run manifest (`failures`).

## Native hang

*Runbook 4. Alert: `RsbNativeProgressStalled`.*

**Diagnose**: `worker_progress_age_seconds` shows the active native call age per
coordinator; `heartbeat_age_seconds` shows a worker whose control group stopped
answering (e.g. a call holding the GIL). `worker_suspect` events name the worker, the
command kind and the trigger (`deadline`, or the node watchdog's fault code). On the node, run
`py-spy dump --pid <worker pid> --native` if it is available.

**Contain and recover** (automatic, §8.4): at the command deadline, or at once on a
node-watchdog fault, the coordinator cancels cooperatively, then
after `failure.cancel_grace_s` it fences the generation durably and kills the actor
with restart allowed. The worker is re-initialized under a new generation and its
in-flight units are re-leased from original seeds. You intervene only if restarts keep
recurring: see [Worker quarantined](#worker-quarantined).

**Verify**: the new generation advances (`worker_restarted`), and no stale output is
accepted (`stale_commands_total` is flat after the restart).

## NUMA and CPU imbalance

*Runbook 5. Alert: `RsbCpuUnderutilized`. The remote-NUMA alert is BLOCKED (see `alerts.yaml`).*

**Diagnose**
- The actual pod cpuset: `kubectl exec <worker pod> -- cat /sys/fs/cgroup/cpuset.cpus.effective`.
  With the static CPU Manager and Guaranteed pods this is an exclusive set.
- Affinity leases: each node's `rsb-affinity-<node>` actor. Its `status()` lists the pod
  CPUs, the reserved service cores and the live leases. `deployment/smoke.py` prints them.
- Ray's `ray_node_cpu_utilization` against the active slots.

**Contain and recover**: drain the affected actor, then fix the cpuset or the agent. The
agent persists leases under `/var/lib/raysimbatch` and reaps only confirmed-dead holders,
so an agent restart never double-allocates.

**Verify**: calibrated throughput is restored and leases don't overlap (every lease's
CPUs are disjoint and inside the pod cpuset).

## Inference overload

*Runbook 6 (GPU inference overload/OOM). CUDA qualification is BLOCKED; CPU/MPS central
inference is implemented and tested.*

**Diagnose**: `inference_seconds`, `inference_batch_rows`, `inference_deadlines_total`,
`inference_errors_total`, and the service's `status()` (queue rows, resident versions).

**Contain and recover**: rows past their deadline are rejected with
`INFERENCE_DEADLINE`, never substituted. Reduce `inference.max_batch_rows` to a certified
batch size, retire drained policy versions, or add replicas (`inference.gpu_replicas`, a
boundary change).

**Verify**: p99 is within the deadline and responses carry the requested policy hash.

## Zombie actors and placement groups

*Runbook 7.*

**Diagnose**: compare `ray list actors --detail --filter "state=ALIVE"` and
`ray list placement-groups --detail` with live leases (`SELECT * FROM leases WHERE expires_at > now()`).

**Contain and recover**: `raysimbatch janitor --config $CFG` kills controllers whose run
lease expired and removes `rsb-` placement groups that have no live run. It never
touches an actor whose lease is live.

**Verify**: resources are released (`ray status`) and no active lease was destroyed.

## Worker node OOM or loss

*Runbook 8.*

**Diagnose**: `kubectl describe node` / `kubectl get events` (OOMKilled, eviction),
`worker_rss_bytes`, and memory admission (`worker.memory_gib` × actors per node, which
validation checks against node memory minus the object store).

**Contain and recover**: KubeRay replaces the pod. Coordinators treat the lost workers
as worker faults, and evaluation units re-lease from original seeds. Rollouts record
explicit tail-loss `gaps` with conservative bounds. Reduce actors per node or raise the
memory request (a new run) if the OOM was genuine.

**Verify**: `gaps` rows exist for the lost tails, unit dedup holds (no episode ordinal
committed twice), and memory is below budget.

## Database outage

*Runbook 9 (replay/database/artifact outage). Alert: `RsbDatabaseCommitsFailing`.*

**Diagnose**: `storage_errors_total{outcome="retry"|"unavailable"}`,
`storage_commit_seconds`, the outbox depth
(`SELECT delivery_state, count(*) FROM outbox WHERE run_id=:rid GROUP BY 1`), and the
database or S3 health itself.

**Contain and recover**
- Transactions retry with bounded backoff (`storage.retry_attempts`), then surface
  `StorageUnavailable`. Leasing and episode starts stop because they need the
  database. Nothing is acknowledged before it is committed.
- Restore the service. Rollout fragments that were written to the outbox but not
  delivered are re-sent by `ExperienceAssembler.redeliver_pending()`, with digest dedup.
- The database must be UTF8. `migrate` refuses a SQL_ASCII database; that was observed
  when an image initialized the database under `LC_ALL=C` without `--encoding`.

**Verify**: outbox rows are `DELIVERED`, orphan blobs are reconciled, and replay dedup
passes.

## Sim2Sim score regression

*Runbook 10.*

**Diagnose**: gate reports (`sim2sim_gates` audit events and `sim2sim_gate_failures_total{fault_code}`),
the bundle and configuration hashes, and paired seed effects in the scorecard.

**Contain and recover**: block the release. Rerun the golden observation, actuator and
contact fixtures, and bisect asset, adapter and policy.

**Verify**: the gates and paired acceptance pass, and the scorecard explains any changed
assumptions.

---

## Capacity degraded

*Alert: `RsbHealthyCapacityLow`.* The healthy fraction of this coordinator's slots is
below 95%. Look at the `worker_quarantined` and `worker_lost` reasons in `audit`. A
deterministic worker error exhausts restarts. Units then retry up to
`evaluation.max_attempts` and end `EXHAUSTED`, and coordinator replacement is bounded by
`failure.max_restarts`, so the run terminates (FAILED, or COMPLETED with
`complete=false` under `allow_partial`) instead of looping. Continue with
[Worker quarantined](#worker-quarantined).

## Worker quarantined

*Alert: `RsbWorkerRestartExhausted`.* Read the last `worker_lost` reason for the worker.
If every worker in a model family fails the same way, it is an asset or code fault: see
[Crashing model asset](#crashing-model-asset). If it is one node, see
[Worker node OOM or loss](#worker-node-oom-or-loss).

## Digest conflict

*Alert: `RsbDigestConflict` (pages on a single occurrence).* The same fragment, episode
or commit key arrived with different content. Such a write is never retried and never
overwritten. Treat it as a determinism or fencing breach. Collect both artifacts (the
committed one from `commits`/`episodes` and the rejected one from the error), the
generations and epochs involved, and the image revisions. Stop the run and investigate
before you resume it.

## Control-plane partition

*Alert: `RsbCoordinatorLeasesLost`.* The coordinators lost their leases and no steps are
progressing. Check database reachability from the head and worker pods, and check the
clock (leases use database server time). A coordinator that loses its lease stops
itself (`coordinator_lease_lost`). The controller replaces it up to `failure.max_restarts`
times.

## Throughput regression

*Alerts: `RsbThroughputBelowCalibration` (inert until
`raysimbatch:calibrated_control_steps_per_second` is set from a qualification benchmark),
`RsbCpuUnderutilized`.* Compare `batch_step_seconds` (native) with
`control_interval_seconds` (end to end) and `publish_seconds`. If native time dominates,
check CPU placement ([NUMA and CPU imbalance](#numa-and-cpu-imbalance)). If publication
or credits dominate, check [Object store pressure](#object-store-pressure).

## Latency

*Alert: `RsbControlLatencyP99High`* (central inference, `stage="advance"`). Split the
round trip into inference (`inference_seconds`) and native (`batch_step_seconds`). Tune
`worker.slice_size` within `[slice_min, slice_max]` as §9.1 describes.

## Policy lag

*Alert: `RsbPolicyLagNearLimit`.* Fragments beyond `rollout.max_policy_lag_versions` are
rejected, never relabeled (`fragments_rejected_total{outcome="policy_lag"}`). Publish
the learner's policy (`RolloutOrchestrator.update_policy`) or slow the collection rate.
Workers restarted after an update load the current policy, as the Mode A tests verify.

## Eval coverage

*Alert: `RsbEvalCoverageMissing`.* No units are left, but coverage is below 99%. Find the
`EXHAUSTED` units and their fault codes:
`SELECT unit_id, attempt, fault FROM units WHERE run_id=:rid AND state='EXHAUSTED'`.
Resuming retries only the non-terminal units. Exhausted units need a new run or an
explicit operator decision.

## Head overload

*Alert: `RsbHeadCpuHigh`.* The head runs the controller and GCS. Check the actor count
(`ray list actors --limit 10000 | wc -l`) and the State API load. Coordinator fanout is
fixed at eight actors per coordinator (§7). Very large runs need the head sizing in the
deployment guide.
