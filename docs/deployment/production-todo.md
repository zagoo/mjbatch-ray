# Production deployment to-do

State on 2026-09-29 (reviewed at revision `8a8ed95`): **RaySimBatch is NOT production-qualified.**
This is everything between the current state and a production release, in the order it
has to happen. Each item names its source (requirement ID from the
[matrix](../qualification/requirements.md), `[VERIFY]` gate from the
[register](../qualification/verify-gates.md), or [report](../qualification/report.md)
section) and the check that closes it.

Nothing here authorizes a rollout. Phase 5 starts only after the go decision in phase 4,
and the infrastructure of phases 2–3 only after the authorization in phase 0.

| Phase | What | Can start |
|---|---|---|
| 0 | Owner decisions and authorization | now |
| 1 | Engineering work | now |
| 2 | Production platform | after 0.1 |
| 3 | Qualification on target hardware (the BLOCKED gates) | after 0.1 and 2 |
| 4 | Go / no-go review | after 1–3 |
| 5 | Rollout | after the go decision |

## Phase 0: owner decisions and authorization

- [ ] **0.1 Authorize the deployment and its qualification.** Target cluster and
  namespace, image registry, accounts, and budget. The budget includes the phase 3
  hardware: x86-64 reference nodes, CUDA nodes if used, and a scale cluster for at least
  24 hours. *Report §9 ("Production cluster deployment"), §11 item 4.*
- [x] **0.2 Liveness (AM-16).** Decided 2026-09-29: deploy the per-node watchdog (done
  in 1.8). *R-3.3.3.*
- [ ] **0.3 onnxruntime telemetry client** (bundled in both wheels): accept it, block pod
  egress, or build onnxruntime without it. *Report §11 item 3.*
- [ ] **0.4 Scope.** Each answer decides whether a phase 3 item is required:
  - central CUDA inference in Mode A → 1.3 and 3.7 (R-GPU);
  - Sim2Sim against Isaac Lab → 3.8 (R-ISAAC);
  - Redis GCS fault tolerance (`gcs_ft=redis_qualified`; the default is `off`) → 3.4
    (V-10);
  - SMT off (the node requirement) or SMT qualified (needs a qualification artifact);
  - the production models and their features (only cartpole, reacher and quadruped were
    measured) → 3.2.
- [x] **AM-2 light-model throughput:** the owner accepted the measured figures on
  2026-09-29 and closed the optimization ([AM-2](../design-amendments.md)). The ratio on
  the reference hardware is still measured in 3.1.

## Phase 1: engineering work (can start now)

- [x] **1.1 Rebuild the CPU image and requalify on kind.** Done 2026-09-29 at `6e0ad7e`
  (`deploy/kind/requalify.sh`, `evidence/deploy/requalify-6e0ad7e/`): smoke, routing
  through `raysimbatch-active`, admission, live watchdog metrics, and the rollback and
  janitor exercises all passed, each exercise identical to its reference run (report
  §6). Rerun `deploy/kind/requalify.sh` for every release candidate. The kind image is
  still `6e0ad7e`: the later pyright fixes and the 1.9 metrics have not run in it (1.10).
- [x] **1.2 x86-64 CPU image** (`linux/amd64`). Done 2026-09-29 at `050070e`, built on
  the arm64 workstation under Rosetta emulation (`deploy/docker/build.sh <tag> --platform
  linux/amd64`, no Dockerfile change; `evidence/deploy/amd64-image/`): both native
  modules are x86-64, `raysimbatch.cli compat` passes (qualified, all ten probes), and 126
  of the 130 Linux tests pass in the amd64 `test` image. The other 4
  (`tests/ray/test_node_helpers.py`) need Ray, which does not stay up under Rosetta
  (reproduced with stock Ray 2.58.0), so Ray has not yet run on x86-64; 3.1 checks it
  first. *V-1, R-X86.*
- [ ] **1.3 GPU image** (`deploy/docker/gpu.Dockerfile`), if 0.4 puts CUDA in scope.
  *R-GPU, R-3.5-OPS.*
- [ ] **1.4 Run CI on a real runner.** `.github/workflows/ci.yml` is written but has never
  run, and its service images are referenced by tag. Run it, pin the service images by
  digest. *R-3.5-OPS.*
- [x] **1.5 Triage the pyright errors.** All 424 at `e228b97` classified
  (`evidence/tests/pyright-triage-e228b97.csv`): 406 false positives, 16 on 3 real
  defects, 2 on 1 unsure issue that proved real. All four fixed with tests that fail on
  the old code, plus a lost-update race the triage pointed at (report §8). The
  remaining false positives were then removed (below).
  - [x] **CI gates on pyright** (owner decision 2026-09-29): `tools/pyright_ratchet.py`
    fails on any error not in `tools/pyright-baseline.json` (it started with the 409
    triaged false positives, per file, rule and message) and on a stale baseline. It runs in
    the `static` CI job; CI itself has not yet run on a real runner (1.4).
  - [x] Shrink the baseline: 409 → 0 on 2026-09-29, in 11 steps (report §2). `src` is
    pyright-clean, so any pyright error now fails CI.
- [ ] **1.6 Object-store-full / spill-stall fault injection.** Not implemented; credit
  bounds are tested. *R-3.3.4.*
- [x] **1.7 Production release tooling.** `deploy/release.sh --context <c> --namespace
  <ns>` (both required). `promote`/`rollback` now route admissions through the
  `raysimbatch-active` Service; before, they wrote a pointer nothing read. They refuse a
  cluster that is not ready, and `retire` refuses the active color.
  `deploy/kind/release.sh` wraps it for kind. Tested against a fake kubectl and on kind
  (1.1). Staging rehearsal remains 4.4.
- [x] **1.8 Watchdog deployment.** One watchdog per node; coordinators register every
  worker and act on its faults: a hung busy worker is recovered before its command
  deadline, an idle worker that died or hung without waiting for a command. Emits
  `heartbeat_age_seconds` (and, through its polling, `worker_rss_bytes`).
  *AM-16, R-3.3.3.*
- [x] **1.9 Observability gaps.** Done 2026-09-29. Every §11.1 metric is emitted except
  `native_arena_peak_bytes` (MJBatch's public API does not expose it; accepted gap), and
  four Grafana dashboards cover the rest, checked against a live Prometheus and in
  Grafana 13.2.2 ([observability.md](../operations/observability.md#dashboards)). *R-3.5-OBS.*
- [ ] **1.10 Requalify the current revision on kind.** Twelve commits to `src/` and
  `deploy/` since `6e0ad7e` (the pyright fixes and the 1.9 metrics) have run only in the
  local suite and the Linux container. Rebuild the image and rerun
  `deploy/kind/requalify.sh`. *R-3.5-OPS.*

## Phase 2: production platform (after 0.1)

Per [kuberay.md](kuberay.md):

- [ ] **2.1 KubeRay operator 1.7.0**; Ray 2.58.0, MuJoCo 3.13.0 and MJBatch `2f16914`
  in the images (`raysimbatch.cli compat` passes in each image). *V-1.*
- [ ] **2.2 Nodes.** Kubelet static CPU Manager with reserved system CPUs; SMT per 0.4;
  pools labelled `raysimbatch.io/pool: head` and `sim`; `/dev/shm` at least the object
  store; a local NVMe spill volume.
- [ ] **2.3 PostgreSQL 17** (UTF8), managed or replicated, with backups and a restore
  drill (design §11.5). Run `raysimbatch db migrate`.
- [ ] **2.4 S3-compatible blob store.** Secrets `rsb-db` (key `dsn`) and `rsb-s3`.
- [ ] **2.5 Head memory from measurement.** The 2 GiB kind head reached Ray's 95%
  threshold with two concurrent evaluations. Size it from 3.3. *Report §11 item 5.*
- [ ] **2.6 Alerting and dashboards.** Deploy `deploy/observability/alerts.yaml` (14
  alerts and 9 recording rules, promtool-tested) and route each alert to an on-call owner
  for its [runbook](../operations/runbooks.md). Import the four dashboards
  (`deploy/observability/dashboards/`) into the production Grafana, and pin its plugin
  versions or set `[plugins] preinstall_disabled = true`: by default Grafana replaces
  plugins from grafana.com at startup
  ([observability.md](../operations/observability.md#dashboards)). Set up the log and
  trace pipeline.
- [ ] **2.7 Registry and inventory.** Push the images, reference them by digest
  (`@sha256:`) in the platform inventory. `render --production` refuses tags, pools
  without node selectors and a missing database secret.

## Phase 3: qualification on target hardware (the BLOCKED gates)

Each gate's exact procedure is in [report §9](../qualification/report.md). Record every
result in the requirement matrix, the `[VERIFY]` register and the report.

- [ ] **3.1 Reference node** (x86-64 EPYC-class, 128 cores, SMT off, NUMA). *R-X86, V-2,
  V-3, V-8, R-3.1.4, R-3.3.5.*
  - 100,000-step bitwise parity (`RSB_PARITY_CONTROL_STEPS=100000`). It is verified on
    Apple arm64 only (R-6.2-PARITY).
  - The full suite and the Linux tests, including `tests/ray` in the amd64 image: Ray
    has never run on x86-64 (1.2).
  - `bench_throughput.py` (interleaved) and `memory_census.py`. This gives the §9.1 ratio
    on the reference hardware; if the light models fall materially below the accepted
    AM-2 figures, that goes back to the owner. It also gives the value for
    `raysimbatch:calibrated_control_steps_per_second`; the throughput alert is inert
    until it is set.
  - NUMA-local affinity leases (`smoke.py` affinity report and `numastat`), multi-NUMA
    admission, CPU busy ≥ 85%. `numa_remote_ratio` and `cpu_busy_ratio` (Resources
    dashboard) report these; their Linux readers were checked only in a single-node
    container.
- [ ] **3.2 Production models** (from 0.4): memory and throughput curves, and bitwise
  scalar-lane replay for any stateful plugin they use. *V-6, V-8.*
- [ ] **3.3 Scale.** Mode A and Mode B at 10k, 50k and 100k environments. Record GCS and
  head CPU and RSS, actor count, `placement_ready_seconds`, completion times and
  cross-node copy counts; compare with the design §9 model. *R-3.4, V-7, R-3.1.1.*
- [ ] **3.4 Failure under load.** Kill a worker node and the head during Mode A/B on a
  multi-node cluster; with Redis GCS FT if 0.4 chose it, recovery within the deadline.
  *R-8.1-NODE, V-10.*
- [ ] **3.5 24-hour soak** of Mode A with periodic Mode B and chaos (worker, coordinator
  and node kills). *Done when* there are no leaks (RSS, object store, actors), no stale
  ACKs, the exactly-once audits pass, and the recovery panels (worker restarts,
  transitions lost) show the injected kills. The short local runs could not show them
  reliably: a process's last export interval is lost when it exits
  ([observability.md](../operations/observability.md), "Export path"). *R-13.2-SOAK.*
- [ ] **3.6 Latency SLOs on target hardware:** p99 slice ≤ 20 ms and round trip ≤ 50 ms.
  *R-3.3.6.*
- [ ] **3.7 CUDA inference** (if in scope): p50/p99 against batch size, zero deadline
  misses at the planned load. *R-GPU, V-9.*
- [ ] **3.8 Isaac traces** (if in scope): an exporter producing `rsb-source-export/1`,
  `Sim2SimOrchestrator.validate` and `evaluate` on the golden tasks, design §13.4
  fidelity thresholds. *R-ISAAC, R-3.2.C-1.*

## Phase 4: go / no-go review

- [ ] **4.1** Every requirement VERIFIED, or its gap accepted in writing by the owner (as
  AM-2 was).
- [ ] **4.2** No BLOCKED gate left without a result or a written waiver.
- [ ] **4.3** Evidence at the release revision on the production image digests: suite,
  parity, smoke.
- [ ] **4.4** Promote and rollback rehearsed on staging with the production tooling (1.7).
- [ ] **4.5** An explicit go decision from the owner. The report's overall status changes
  only then.

## Phase 5: rollout (after the go decision)

1. Render with `raysimbatch render --production` and the digest-pinned inventory. Migrate
   the database.
2. Deploy the first color. Smoke it (Mode B, then Mode A), then promote.
3. Watch the alerts and throughput against the calibration from 3.1. The rollback path
   is rollback plus resume from seeds ([kuberay.md](kuberay.md), "Release").
4. Retire the old color after its runs drain.
