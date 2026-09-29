# Production qualification report

**Overall status: INCOMPLETE. RaySimBatch is NOT production-qualified.**

The implementation covers the design's normative scope, with the departures recorded in
[design-amendments.md](../design-amendments.md). Everything that could be qualified in
the available environment was run with real components; the results are below. Several
required gates cannot run here (§9) and one measured target is not met (§5; the owner
accepted the shortfall on 2026-09-29). What remains before production is listed in
[production-todo.md](../deployment/production-todo.md). Each
statement is labelled as an **observed** result, an **inference**, or **unverified**.

Revision under test: the final local suite ran at `5f4629f`. Other evidence names its
own revision (§10). MJBatch `2f169146b643adda65e195d8ad13d997abdd2a2e`
(v0.1.1), MuJoCo 3.13.0, Ray 2.58.0, KubeRay 1.7.0 (operator built from tag `v1.7.0`,
commit `59d663ae…`), PostgreSQL 17, MinIO. Design document SHA-256
`d90700723665bdbedad32d3c9856f733f8188b9428f160265f93811e02ac5462` (unchanged,
`tools/check_design_hash.py`).

The design's `[VERIFY]` release gates are enumerated with status and evidence in
[verify-gates.md](verify-gates.md): 5 VERIFIED on the available hardware (V-1, V-3 and V-7 with a
BLOCKED remainder on the reference hardware), 3 IMPLEMENTED (partly verified), 2 BLOCKED.

## 1. Environment (observed)

| Item | Value |
|---|---|
| Host | Apple M4 Max, arm64, 16 cores (no SMT), 64 GiB, macOS 26.6.2 |
| Linux | arm64 containers (Docker Desktop VM: 16 vCPU, 8 GiB, 59 GiB disk); linux/amd64 containers under Rosetta emulation (image build and the non-Ray tests only) |
| Kubernetes | kind v0.33.0, node image v1.37.0; 3 nodes (control plane, `head` pool, `sim` pool); kubelet static CPU Manager |
| GPU | Apple GPU (Metal/MPS) only. **No CUDA.** |
| Not available | x86-64 EPYC-class nodes, multi-socket NUMA, CUDA, Redis, Isaac Lab, cloud accounts, a production cluster |

Details are in [environment-inventory.md](../ledgers/environment-inventory.md).

## 2. Test suites (observed)

| Suite | Result | Notes |
|---|---|---|
| Full local suite (`pytest tests`, macOS, real Ray, PostgreSQL 17, MinIO, file blobs) | **300 passed, 7 skipped, exit 0** (373 s, `5f4629f`) | The skips are Linux-only (CPU affinity; the real /proc and /sys readers), run in the next row. `evidence/tests/pytest-final.log` |
| Linux-only (`tests/linux`, `tests/ray/test_node_helpers.py`) and `tests/core` in the `test` image | **130 passed, exit 0** (`5f4629f`, linux/aarch64) | CPU affinity pinning, affinity agent leases and admission against the real cpuset, the /proc and /sys readers of worker_private_bytes and numa_remote_ratio, and all of `tests/core` against the Linux build of the native module. `evidence/tests/linux-container.log` |
| The same set in the amd64 `test` image (linux/amd64 under Rosetta emulation) | **126 passed, exit 0** (`050070e`, linux/x86_64); the 4 in `tests/ray/test_node_helpers.py` not run | Ray 2.58.0 does not stay up under Rosetta: the raylet reports its agents dead and fate-shares with them (reproduced with stock Ray). Emulation, not x86-64 hardware. `evidence/deploy/amd64-image/` |
| MJBatch contract (`tests/upstream`) | 18 passed | pins the upstream semantics RaySimBatch relies on |
| Bitwise parity (`tests/core`), default 300 control steps per fixture and thread count | passed | the extended run is in §4 |
| Fused vs per-substep stepping (`tests/core/test_fused.py`), AM-2 | 15 passed | Every result field and per-slot state array is bitwise equal: terminations mid-interval and on the last substep, native failures on substep 0 and inside an interval, health failures, warnings on discarded substeps, saturation, position servos, snapshot/restore. Implementation mutants all fail. |
| Native bookkeeping vs the former NumPy code (`tests/core/test_book.py`, `test_substep_settle.py`) | 65 passed | The former implementations are kept as oracles and compared bitwise on randomized outcome mixes: signed zeros, time limits on the boundary, both health modes. Ten native mutants all fail. |
| Sanitizers (`qualification/native_sanitizers.sh`): ASan+UBSan and TSan builds of MJBatch and `_native` | exit 0 for both (`65e361d`; `native/` and the script are unchanged since) | `evidence/sanitizers/`. MuJoCo's shared library itself is not instrumented (prebuilt wheel). |
| Alert and recording rules, dashboard queries (`qualification/check_alerts.sh`): `promtool check` + unit tests | SUCCESS: 23 rules with their unit tests, and all 74 dashboard queries parse | `evidence/observability/promtool.log`. A mutated expectation fails. |
| Static analysis | `ruff check`: clean. `pyright` 1.1.414: **0 errors**, 1 warning in 74 files (`5cfea1b`) | `evidence/tests/static-analysis.log`. Every error at `e228b97` (424) was triaged (`evidence/tests/pyright-triage-e228b97.csv`): 406 false positives (155 from `mujoco` having no stubs, the rest Optional attributes set after construction, Ray and third-party stub gaps), and 4 real defects, all fixed with tests (§8). CI gates on pyright with a ratchet (`tools/pyright_ratchet.py`, owner decision 2026-09-29), in the `static` job. The baseline then shrank 409 → 0 the same day (`38e76f8`..`5cfea1b`): generated stubs for MuJoCo (155) and `_native` (5); `import ray.exceptions` (16); declared types for late-initialized and closed-only attributes; asserts for invariants that hold; a typed Ray boundary (`compat/ray_typing.py`); and local fixes. Some fixes turned `None` failures into explicit errors (a closed policy runtime, a missing base quaternion or artifact root). 9 targeted suppressions remain, each commented. |
| Examples (`tests/examples`) | all four run as documented | Modes A, B, C, and router stepping |

Mutation checks (the guarding test must fail with the fix reverted) were done case by case,
not systematically for every defect in §8. They were done for the controller-placement,
janitor-wait, fragment-flush, chaos-ack and alert-rule tests.

## 3. Correctness and recovery (observed)

| Property | Evidence |
|---|---|
| Bitwise parity with scalar MuJoCo (all fixtures, 1 and 3 threads) | `tests/core` |
| Exact sparse reset; other slots bitwise untouched; snapshot/restore across slots and processes | `tests/core` |
| Per-slot native and health failure isolation; failed slots keep pre-call state | `tests/core`, `tests/upstream`, runtime probes |
| Worker, coordinator and controller loss mid-run give per-episode results **identical** to an uninterrupted run, each committed exactly once | `tests/e2e/test_mode_b_chaos.py` |
| Orphaned runs (controller lost) are resumed by the janitor with identical results; resumes are bounded | `tests/e2e/test_mode_b_chaos.py` |
| A deterministic worker error terminates the run with its root cause (no crash loop) | `tests/e2e/test_mode_b_chaos.py` |
| Native hang (GIL released) and GIL-holding calls are detected and fenced, and the worker restarts | `tests/ray/test_watchdog.py` |
| Mode A exactly-once fragments, policy-lag rejection, updated policy survives worker restart, tail-loss gaps | `tests/e2e/test_mode_a.py` |
| Fragment-flush livelock: 1 stall in 8 loops before the fix, 0 in 25 after | `tests/ray/test_worker_actor.py`, loop evidence in §8 |
| Lease fencing: a non-owner never writes run state; a resume waits out a dying owner and refuses a live one | `tests/storage/test_storage.py` |
| Sim2Sim: 12 gates, each with an injected golden mismatch that must fail the specific gate | `tests/sim2sim/test_gates.py` |

## 4. Extended parity

`RSB_PARITY_CONTROL_STEPS=100000 python -m pytest -v tests/core -k parity`
(`evidence/tests/parity-100k.log`, revision `d3bd3a9`): **12 passed, exit 0, 575 s**.

- Each case is one of the three fixtures (cartpole, quadruped, reacher) at 1 or 3 MJBatch
  threads.
- Each case runs twice: with per-substep health, and with control-rate health.
  Control-rate health makes cartpole and reacher step fused (AM-2); quadruped (`pd`)
  stays per-substep. The first run of this check, at `ec33b47` (before AM-2), passed its
  6 per-substep cases.
- Each case steps 6 slots for 100,000 control steps with random actions, and compares
  the full MuJoCo state bitwise with scalar `mj_step` after every control step.
- Observed on Apple arm64 only. Parity on x86-64 (different compiler and FMA behavior) is
  **unverified**.

## 5. Performance (observed on this host only; not capacity claims)

`qualification/bench_throughput.py` at `d3bd3a9` (`evidence/perf/throughput-overhead-{64,256}.json`
and `.log`):

- **Method.** The upstream baselines start from the wrapper's exact post-reset state and
  control. The four modes are timed round-robin (7 rounds of 1 s windows), and ratios
  are medians of per-round ratios, so host drift affects both sides alike. Spreads are
  1–4%.
- **Host conditions.** The load average was 5–6 from other applications, and macOS
  schedules across performance and efficiency cores.
- **Modes compared.** *Per-substep* is the default configuration. *Fused* sets
  `health.check_each_substep=false`, with results identical to per-substep stepping
  (AM-2; `tests/core/test_fused.py`). Both run the native bookkeeping (AM-2).

| Model | Slots | Threads | Per-substep / upstream `nstep=d` | Fused / upstream `nstep=d` |
|---|---:|---:|---:|---:|
| cartpole (nv=2, torque, termination) | 64 | 1 / 2 / 4 | 0.75 / 0.60 / 0.45 | **0.91** / 0.86 / 0.78 |
| cartpole | 256 | 1 / 2 / 4 | 0.78 / 0.72 / 0.64 | **0.93** / **0.92** / 0.87 |
| reacher (torque, boundary forward) | 64 | 1 / 2 / 4 | 0.69 / 0.56 / 0.44 | 0.84 / 0.80 / 0.75 |
| reacher | 256 | 1 / 2 / 4 | 0.72 / 0.66 / 0.59 | 0.85 / 0.84 / 0.81 |
| quadruped (nv=18, contacts, pd: not fusable) | 64 | 1 / 2 / 4 | 1.17 / 1.08 / 0.97 | same path |
| quadruped | 256 | 1 / 2 / 4 | 1.20 / 1.15 / 1.08 | same path |

Earlier runs show the progression: `evidence/perf/throughput-fused-{64,256}.json`
(`4b6df90`, fused stepping before the native bookkeeping: fused 0.59–0.90, per-substep
0.35–0.75), and the sequential runs `evidence/perf/throughput-macos-arm64.json` and
`evidence/mjbatch-upgrade/throughput-*.json`.

**Serial overhead per call** (`qualification/bench_overhead.py`, one thread; wall time
minus time inside MJBatch; `evidence/perf/overhead-{before,after}.json`, `overhead.log`).
"Before" is `2454a01` built in a temporary worktree; "after" is `d3bd3a9`.

| Configuration | Per-substep before → after | Fused before → after |
|---|---|---|
| cartpole, 64 slots | 124 → 45 µs | 70 → 23 µs |
| cartpole, 64 slots, a termination inside every interval | 132 → 53 µs | 155 → 65 µs |
| cartpole, 256 slots | 169 → 79 µs | 109 → 41 µs |
| cartpole, 256 slots, a termination inside every interval | 185 → 88 µs | 202 → 105 µs |
| reacher, 64 slots | 214 → 89 µs | 88 → 33 µs |
| reacher, 256 slots | 327 → 168 µs | 156 → 75 µs |

**Target (design §9.1): wrapper ≥ 90% of equal-thread upstream MJBatch.**

- **Met** for the contact model at every measured configuration (0.97–1.20).
- **Met** for cartpole with fused stepping at 64 slots on 1 thread (0.91), and at 256
  slots on 1 and 2 threads (0.93, 0.92).
- **Not met** for cartpole otherwise (fused 0.78–0.87; per-substep 0.45–0.78), nor for
  reacher (fused 0.75–0.85).

**Owner decision (2026-09-29):** these figures are accepted and AM-2 optimization is
closed. The target stays unmet in the configurations above (R-3.3.6 remains
IMPLEMENTED), and it is unmeasured on the reference hardware (§9).

What remains, measured (not pursued):
- The fused MJBatch call is within about 4% of upstream's call on the same physics:
  history recording costs 3–4%, the extra bound fields about 1%.
- RaySimBatch's serial work is 23–41 µs per call (cartpole fused), about half of it native
  kernels (observation, action processing, controller). Its share grows with MJBatch
  threads and with smaller slices.
- A call that replays a mid-interval termination costs about 40–65 µs more.
- reacher runs a boundary `forward` (about 12% of its wall time) for its end-effector
  reward, which upstream's baseline does not compute.

The quadruped's wrapper throughput above upstream reflects different dynamics during the
window: the wrapper's PD control keeps it standing under less contact load.

**Memory** (`qualification/memory_census.py`, current RSS delta of building and warming
a shard, 4 threads): 2.3–4.7 MiB for 64–256 slots of these light models (10–60
KiB/slot). MuJoCo arenas (13–14 MiB per thread) are mostly untouched pages for
contact-light scenes. Heavy-contact worst cases are **unverified**.

**Inference runtime curves** (`qualification/bench_inference.py`,
`evidence/perf/inference-curves.{json,log}`, `8b3eb4b`):

- These are in-process `LocalPolicyRuntime` calls, 36→8, deterministic. The central
  service adds batching and RPC on top.
- The host was not idle (load average 4–6 from other applications).
- The ONNX policy is a Gaussian MLP; the TorchScript policy is a linear map. **Do not
  compare the two runtimes with each other.**

| Runtime / device | Policy | Batch 1 p50 / p99 (ms) | Batch 1024 p50 / p99 (ms) | Rows/s at 1024 |
|---|---|---:|---:|---:|
| ONNX Runtime 1.30 / CPU | MLP | 0.016 / 0.019 | 0.154 / 0.180 | 6.6 M |
| TorchScript (torch 2.14) / CPU | linear | 0.008 / 0.010 | 0.019 / 0.022 | 53.7 M |
| TorchScript / MPS | linear | 0.228 / 0.392 | 0.252 / 0.385 | 3.9 M |

MPS latency is flat at about 0.25 ms per call (dispatch overhead) up to batch 1024. CUDA
curves are **BLOCKED** (no CUDA device; §9).

## 6. Deployment, release and rollback (observed on kind)

- **Requalification of the current release image** (2026-09-29, `deploy/kind/requalify.sh`,
  `evidence/deploy/requalify-6e0ad7e/`). One image built at `6e0ad7e` (the code of
  `a9cdb7b`: watchdog deployed, triage fixes, release tool) served both colors:
  - Smoke passed on blue (Mode B) and green (Mode B, and Mode A with 5,626 transitions and
    0 gaps).
  - Routing: after `promote green`, the Service `raysimbatch-active` selected green's head
    and a Ray Client connection through it ran on `rsb-green-head-…`; after `rollback`,
    on `rsb-blue-head-…`. Retiring the active color was refused.
  - Admission: in a worker pod with cpuset `2-5`, the affinity agent accepted its own
    4-CPU request and refused an 8-CPU one.
  - The per-node watchdog ran on the worker node and exported `heartbeat_age_seconds`
    and both workers' `worker_rss_bytes` (`watchdog-live.txt`). No worker was reported
    lost in any of the runs.
  - Rollback exercise: green was force-deleted with 32 of 64 units committed; blue
    resumed the run, and its 128 episodes are identical to an uninterrupted reference
    run. It passed on the first attempt this time (the earlier head OOM did not recur).
  - Janitor exercise: blue was deleted mid-run and redeployed. Its janitor resumed the
    orphan on its first pass (8 s after start), the controller ran on a worker node, and
    the 128 episodes are identical to the reference run.
  The earlier items below were observed on older images.
- **Deployment.** Rendered RayClusters are validated against the KubeRay 1.7.0 CRD (no
  pruned fields), with head and worker groups, custom resources, and a spill volume. Pods
  are non-root with `fsGroup`, and secrets are passed by reference. Worker pods receive
  exclusive cpusets (`2-5`, `6-9`), and affinity leases fall inside them.
- **Smoke.** Mode B (COMPLETED, complete, coverage 1.0, LQR 1.0 / zero 0.0) and Mode A
  (≥5,000 transitions, 0 gaps) passed on blue and green, driven through the normal driver
  from inside the cluster (`evidence/deploy/*smoke*`).
- **Blue/green.**
  - Blue ran release N (`bf54767`) and green ran release N+1 (`5e01ac3`); both passed
    smoke and green was promoted (`evidence/deploy/05*`–`12*`).
  - Later iterations redeployed both colors from the same build to exercise rollback
    mechanics.
- **Autoscaling.** Up-scaling was observed. Down-scaling was observed only after AM-13:
  blue went from 2 workers to 1 without intervention (`21-blue-scale-down-watch.log`).
- **Rollback with durable resume** (`deploy/kind/rollback_exercise.sh`,
  `evidence/deploy/rollback-exercise/`):
  - Green was active, and a 64-unit evaluation started on it (32 seeds, 200,000-step
    episodes).
  - Green was force-deleted 6 s later, with 32 units committed and 32 in flight.
  - The pointer was rolled back to blue.
  - Blue's controller took the run lease after green's lapsed (46 s), re-leased the 32
    interrupted units (attempt 2, new lease epoch), and **completed the run: 64/64 units,
    run epoch 1**.
  - An uninterrupted reference run of the run's *stored* configuration (`4a80ec4c`, on
    blue) completed. Its per-episode semantic digests are **identical** to the resumed
    run's for all 128 episodes, with coverage 1.0 and LQR success 1.0 / zero 0.0 on both
    (`xresume-compare.json`, `three-way-compare.json`; image `ec33b47`).
- **Janitor orphan recovery on a real cluster** (found during the rollback exercise, then
  exercised with `deploy/kind/janitor_exercise.sh`):
  - The first reference attempt (`96d19e39`) was orphaned when its controller was
    OOM-killed on the head and blue was then replaced. The redeployed blue's janitor
    sidecar, connected through Ray Client, resumed it.
  - That replacement controller ran **on the head** and was OOM-killed 4 minutes later
    (head container 1.90/2.00 GiB). The janitor resumed it again, this time off-head, and
    the run completed. Its results are also **identical** to the reference
    (`rollback-exercise/audit-janitor-resumed.txt`, `controller-placement.txt`,
    `janitor.log`).
  - First diagnosis, **insufficient**: the janitor's first pass ran before any worker node
    had registered. Adding a bounded wait for a worker node (`--startup-wait-s`)
    did not fix it:
    - Exercise attempt 1 resumed on the second pass, when workers were long up, and
      didn't exercise the window.
    - In attempt 2 the run lease expired before the redeploy, so the first pass resumed
      the run, after a worker *had* registered. The controller **still** ran on the head.
  - Actual cause (GCS log, `janitor-exercise-attempt2-soft-affinity/gcs-scheduling.txt`):
    the controller's *soft* node affinity sent the lease to the worker raylet, registered
    4 s earlier. The raylet returned it at once, and the GCS placed the actor on the head.
  - Fix: **hard** affinity to the chosen worker node. Default placement is used only when
    no worker node is alive. The startup wait stays, so there is a worker node to choose.
  - Regression tests are in `tests/ray/test_controller_placement.py`. Mutants that make
    the affinity soft, drop the wait, or wait before every pass all fail.
  - Attempt 3 aborted on a full node disk (§8).
  - **Attempt 4** (`janitor-exercise/`, image `ebab31f`, hard affinity) passed:
    - The lease expired before the redeploy, the janitor waited for a worker node, and
      its **first** pass resumed the orphan (one resume).
    - The GCS sent the controller's lease to the worker node 3.9 s after that node
      registered, the same gap at which attempt 2's soft lease was handed back. The
      raylet kept it and started the controller 434 ms later, on a worker pod; no
      controller ran on the head.
    - All 128 episodes are **identical** to the uninterrupted reference.
    - This is one trial of each variant (soft vs hard in the same window), not a
      statistical result.
- **Earlier attempts and what they exposed** (all fixed; §8):
  - a fenced resume that wrote FAILED over a live run;
  - a driver that changed run state before holding the lease;
  - a lease wait shorter than the dying head's grace period;
  - controllers OOM-killed on the head;
  - an orphaned run with no janitor to resume it;
  - re-promotion erasing the rollback target.
- **Persistence.** PostgreSQL data survived pod deletion
  (`08-postgres-persistence-probe.log`) and a disk-full crash loop.

## 7. Observability (observed)

- Catalog metrics are exported through Ray's real metrics endpoint (scraped in
  `tests/e2e/test_telemetry.py`), with JSON logs showing the full run-state sequence, and
  a reproducibility manifest written and audited per run.
- Recording is fail-safe. A static test checks every recorded metric against the §11.1
  catalog.
- Alert rules cover every §11.3 page condition and pass `promtool` unit tests.
- Ray native metric names were verified on 2.58.0, and the spill/restore series'
  semantics are pinned by a Ray contract test.
- **Catalog complete except one accepted gap** (2026-09-29, production to-do 1.9): every
  §11.1 metric is emitted except `native_arena_peak_bytes`, which MJBatch's public API does
  not expose. The object-store metrics are recording rules over Ray's native series.
- **Dashboards:** four Grafana dashboards (`deploy/observability/dashboards/`, generated by
  `tools/gen_dashboards.py`) cover every emitted catalog metric. They are checked
  statically (every series exists and is read the right way) and by promtool (74 queries
  parse). Live, on a local Ray node scraped by the pinned Prometheus during the Mode C, B, A
  and worker-loss end-to-end tests (`evidence/observability/dashboards-live.json`), all 48
  panels that healthy runs fill had data, 55 of 74 overall. The rest read event counters
  (restarts, errors, drops, rejections) or Linux-only readings.
- **Dashboards in Grafana** (same run): in the pinned Grafana 13.2.2 (its bundled
  Prometheus data source, 13.1.7), the four dashboards load unchanged both by file
  provisioning and through the import API a UI import uses, and every stored panel query
  returns the same series through Grafana as from Prometheus (74 of 74, 55 with data). In
  Grafana's frontend, checked by hand in a browser, all 40 panels render, with "No data" on
  exactly the 9 whose queries returned nothing. Two findings, both fixed in the check:
  it started the tests before Prometheus's first scrape (about 5 s after it is ready), so
  a short run went unscraped; and Grafana by default downloads plugins from grafana.com at
  startup, and replaced its bundled Prometheus data source while the check queried it (all
  queries failed "plugin not registered" for 16 s). The check now disables that
  (operations note in [observability.md](../operations/observability.md#dashboards)).

## 8. Defects found by qualification (all fixed, with regression tests)

| Defect | Found by | Fix |
|---|---|---|
| Plain `docker build` produced the sanitizer stage | kind smoke | runtime stage last; `build.sh --target runtime`; `up.sh` guard |
| Secrets unreadable by uid 1000 | kind smoke | `fsGroup`, mode 0440 |
| SQL_ASCII database accepted | kind smoke | `migrate` refuses non-UTF8 |
| Unbounded plan (1,250 actors on a 4-actor cluster) | kind smoke | `CAPACITY` validation and `plan_capacity` bound |
| Coordinator crash when a worker died mid-ack (intermittent: 3 of 8 runs) | chaos loop | death-tolerant acks (0 of 10) |
| Restarted worker reverted to the original policy | code review of the restart path | policy updates persist in the group runtime |
| Fleet routes sized from the last member's envs | pyright triage | per-member sizing |
| Metrics dropped: dispatch bug; `inc(0)` rejected by Ray | telemetry integration test | fixed; recording never raises |
| `physics_steps` undercounted in `run_local` | telemetry test | accumulated per substep |
| Fragment-flush livelock | Mode A loop (1 in 8) | flush before the fullness check |
| Worker error crash-looped coordinators forever | telemetry wiring | worker fault; bounded replacement |
| Nodes never scaled down (live helpers and controllers) | kind | AM-13 |
| Controller fenced itself during a gang wait | kind | lease keeper thread |
| Resume after a force-deleted release: wrong waits and a non-owner write | kind rollback | grace-derived wait; owner-only writes |
| Orphaned runs never resumed; controllers OOM-killed on the head | kind rollback | janitor sidecar (Ray Client); off-head placement |
| Resumed controllers placed on the head during cluster start and OOM-killed there (soft affinity handed back by a just-registered worker raylet) | kind reference run; janitor exercise | hard node affinity to a worker node; janitor waits for one before its first pass |
| `release.sh prune-images` removed nothing (`ctr images rm` by ID; errors hidden), so kind import records and stale leases kept every old image and the node disk filled | janitor exercise attempt 3 (`No space left on device`; that run failed with the storage cause) | `prune-node.sh`: by ID, resolved import names, stale leases; reports counts |
| PostgreSQL readiness probe logged `database "rsb" does not exist` every 2 s | log review | probe names the database |
| `health.check_each_substep` was accepted by configuration but ignored by the core (health always ran per substep) | AM-2 fused-path work | control-rate semantics implemented and tested (`test_control_rate_health_runs_once_on_each_slots_last_substep`) |
| `tools/local_services.sh env` exported the S3 endpoint when MinIO was down, so the S3 test failed instead of skipping as the start message said | final suite with Docker stopped | the endpoint is exported only when it answers |
| Admission never compared a pod's real CPU set with its exclusive request (design line 1038 `[VERIFY]`); without the static CPU Manager, leases would have come from the shared pool. `NUMA_FIT` had no test. | `[VERIFY]` register review | the agent refuses the node on a mismatch (`RSB_POD_CPUS`); tests for both, mutation-checked; verified on kind (`evidence/deploy/admission/`) |
| YAML `1e-6` read as a string | kind (CLI) | YAML 1.2 floats; JSON loaded as JSON |
| Affinity verification crashed on vanished threads | kind | tolerant enumeration |
| Blanket image prune broke kind volume provisioning; ephemeral PostgreSQL lost state on OOM | kind | project-scoped pruning; PVCs and Guaranteed QoS |
| ONNX `device=mps` silently ran on CPU | review | explicit error |
| TorchScript path untested | review | CPU and MPS tests |
| Bootstrap p-value reported as 0.0 | example output | (k+1)/(B+1) |
| Upgrade tool: build-setting merge, fetch refspec, pristine scope, stored baseline, built version, promote check | upgrade exercise | fixed; exercised end to end |
| Pytest process aborted at exit (onnxruntime 1DS thread) | full suite | runner exits with pytest's status (**not** a fix of onnxruntime) |
| Documented `python …` commands resolved to the tool venv (uv only) after `source tools/env.sh` | following the docs verbatim | `tools/env.sh` puts the project `.venv/bin` first; candidate venvs still use the base interpreter (checked) |
| A restarted experience assembler (Ray reruns only the constructor) had no store or sink: every later `accept` failed, each failure leaked a worker credit, and the Mode A run stalled | pyright triage (2026-09-29) | set up again on first use and redeliver the outbox; the coordinator resends fragments whose `accept` died with the assembler (`test_assembler_restart_mid_run`; each half mutation-checked) |
| Retiring a policy version could close its runtime under a batch already taken off the queue; those requests failed with `INFERENCE_ERROR` | pyright triage | executing batches count as busy (`tests/policy/test_service.py`) |
| Policy and curriculum updates handed between threads by read-then-clear: an update landing in between was erased | pyright triage follow-up | read and clear under the owner's lock (`tests/control/test_update_handoff.py`, interleaving forced) |
| The smoke tool crashed with `TypeError` on a missing or unknown run id | pyright triage | required arguments per mode; unknown runs named (`tests/deployment/test_smoke.py`) |
| `mlp_onnx` with a normalization mean but no std exported a policy whose actions were all NaN | pyright triage | mean and std required together (`test_embedded_normalizer_needs_mean_and_std`) |
| Worker `health` was never polled in the run path, so `worker_rss_bytes` and `log_dropped_total` were never recorded | watchdog deployment (AM-16) | the node watchdog polls it every `heartbeat_s` |
| Metrics recorded in Ray driver processes (the Sim2Sim orchestrator) went to a local registry nothing scraped: the exporter used Ray only in worker processes | live dashboard check (1.9) | Ray exports driver metrics too (checked); the exporter uses Ray whenever it is initialized |
| Spans dropped by the OpenTelemetry export queue were not counted (the SDK only logs them) | catalog review (1.9) | counted through the SDK's drop accounting; offered = exported + queued + dropped is tested |
| A short evaluation run never exported its suite summary gauges (the first 5 s refresh came before any commit, and none followed) | e2e telemetry test (1.9) | one more refresh when the run's work is done (the controller lingers, so it is exported) |
| Release `promote`/`rollback` wrote a pointer nothing read (no admission routing), acted on the current kubectl context, and could route to a cluster that was not ready or retire the active one | production to-do 1.7 | `deploy/release.sh`: required context and namespace, the `raysimbatch-active` Service, readiness and retire guards (`tests/deployment/test_release_tool.py`, guards mutation-checked) |

## 9. BLOCKED gates (cannot be qualified here) and exact procedures

| Gate | Why blocked | Procedure |
|---|---|---|
| Reference hardware (§9.1): x86-64 EPYC, 128 cores, SMT off, NUMA | unavailable | Build the CPU image for linux/amd64 (`deploy/docker/build.sh <tag> --platform linux/amd64`) and run its Linux tests, including `tests/ray`: Ray has not yet run on x86-64 (to-do 1.2). Deploy per [kuberay.md](../deployment/kuberay.md) with CPU Manager static. Run `bench_throughput.py` and `memory_census.py` on a node. Verify affinity leases stay NUMA-local (`smoke.py` affinity report plus `numastat`). Record the calibrated throughput into the alert rules. |
| CUDA central inference and the GPU image | no CUDA | Build `gpu.Dockerfile`, run `bench_inference.py` with CUDA configs added, and run Mode A `placement=central, device=cuda` end to end. Record p50/p99 against batch size; deadline misses must be 0 at the planned load. |
| Scale: 10k / 50k / 100k environments; GCS and raylet budgets (§9.2–9.3) | no cluster | Mode A and Mode B at each size. Record GCS and head CPU and RSS, actor count, `placement_ready_seconds` and completion times; compare with the §9 model. |
| 24-hour soak (§11.5, §13.2) | duration and hardware | Run Mode A plus periodic Mode B for 24 h with chaos injection (worker, coordinator and node kills). Accept only with no leaks (RSS, object store, actors), no stale ACKs, and exactly-once audits. |
| Node loss under load; Redis GCS FT | single-VM kind; no Redis | Kill a worker node and the head during Mode A/B on a multi-node cluster; with `gcs_ft=redis_qualified`, verify recovery within the deadline. |
| Real Isaac traces (§6.7, §13.4) | no Isaac | Write an Isaac exporter producing `rsb-source-export/1`, run `Sim2SimOrchestrator.validate` and `evaluate` on the golden tasks, and meet the §13.4 fidelity thresholds. |
| Production cluster deployment | not authorized or available | Render with `--production` and a digest-pinned inventory, then follow the blue/green procedure. Do not roll out without authorization. |
| CI execution | no remote CI access | `.github/workflows/ci.yml` is authored but not executed. Its service images are tag-referenced, not digest-pinned. |

## 10. Final runs and evidence index

| Evidence | Revision | Result |
|---|---|---|
| `evidence/tests/pytest-final.log` (full local suite) | `5f4629f` | 300 passed, 7 skipped (Linux-only), exit 0 |
| `evidence/perf/throughput-fused-{64,256}.json`, `throughput-fused.log` | `4b6df90` | §5 (AM-2 fused stepping) |
| `evidence/perf/throughput-overhead-{64,256}.json`, `throughput-overhead.log` | `d3bd3a9` | §5 (after the native bookkeeping) |
| `evidence/perf/overhead-{before,after}.json`, `overhead.log` | `2454a01` → `d3bd3a9` | §5 serial overhead per call |
| `evidence/tests/linux-container.log` | `5f4629f` | 130 passed (Linux-only and `tests/core`), exit 0 |
| `evidence/deploy/amd64-image/` | `050070e` | amd64 image under Rosetta: compat passes; 126 passed (the same set without `tests/ray`), exit 0; Ray does not stay up |
| `evidence/deploy/admission/` | image `31d9431` | smoke: Mode B and Mode A passed; the agent accepted the exclusive cpuset `2-5` for a 4-CPU pod and refused a mismatched request inside a real worker pod |
| `evidence/tests/parity-100k.log` | `d3bd3a9` | 12 passed (100,000 control steps each; per-substep and fused), exit 0 |
| `evidence/tests/static-analysis.log` | `5cfea1b` | ruff clean; pyright 0 errors (baseline empty); ratchet exit 0 |
| `evidence/tests/pyright-triage-e228b97.csv` | `e228b97` | every one of 424 pyright errors classified: 406 false positive; 16 + 2 on 4 real defects, fixed in `a9cdb7b` |
| `evidence/deploy/requalify-6e0ad7e/` | image `6e0ad7e` | smoke, routing, admission, watchdog metrics, rollback and janitor exercises: all passed (§6) |
| `evidence/perf/inference-curves.{json,log}` | `8b3eb4b` | §5 |
| `evidence/perf/throughput-macos-arm64.json`, `evidence/mjbatch-upgrade/throughput-*.json` | §5 | light-model target not met (AM-2) |
| `evidence/perf/memory-census-macos-arm64.json` | §5 | 10–60 KiB per slot, light models |
| `evidence/deploy/rollback-exercise/` | image `ec33b47` | resumed run identical to the uninterrupted reference (128 episodes); `three-way-compare.json` also covers the janitor-resumed orphan |
| `evidence/deploy/janitor-exercise/` (attempt 4) | image `ebab31f` | first-pass resume, controller on a worker, identical to the reference |
| `evidence/deploy/janitor-exercise-attempt{1,2,3}-*` | `6c948b7`, `6c948b7`, `ebab31f` | second-pass resume (pass); controller on the head with soft affinity (defect); disk full (tooling defect) |
| `evidence/sanitizers/` | `65e361d` | ASan+UBSan and TSan exit 0 |
| `evidence/observability/promtool.log` | `5f4629f` | 23 rules and their unit tests SUCCESS; 74 dashboard queries parse |
| `evidence/observability/dashboards-live.json`, `dashboards-live.log` | `5f4629f` (working tree) | every panel queried against a live Prometheus: 48 of 48 healthy-run panels with data, 55 of 74 overall |
| `evidence/mjbatch-upgrade/` | see [mjbatch-noninvasiveness.md](mjbatch-noninvasiveness.md) | candidate `121e033` passed all gates; promote and rollback exercised in a clone |
| `evidence/deploy/*` (bring-up, smoke, blue/green, scale-down, persistence) | per file | §6 |

Events during qualification that were not product defects:

- **Test-harness race (2026-09-29).** The final suite at `3aac5d5` failed once in
  `tests/deployment/test_release_tool.py`: `release.sh` pipes `kubectl create` into
  `kubectl apply`, and the test's two fake `kubectl` processes read and rewrote one JSON
  state file at the same time. The fake now locks and replaces the file atomically
  (`437a3c0`); under load, 1 of 25 runs failed before and 0 of 25 after. The final suite
  was rerun at `437a3c0`.
- **Requalification restarted before any deployment (2026-09-29).** `requalify.sh`
  appended to the tracked `images.log` between its two builds, which would have stamped
  the second image `-dirty`. The run was stopped during the first build and restarted at
  `6e0ad7e` with the records written after both builds.

- **Local services stopped externally.** Between 04:33Z and 08:49Z the project's local
  PostgreSQL stopped without a shutdown record in its log, and Docker Desktop was stopped,
  taking MinIO and the kind cluster down. This session did neither. The first final-suite
  attempt failed on the missing services (15 failed, 18 errors). PostgreSQL was restarted
  and the suite rerun (§2). Docker was left as found.

- **Finder metadata.** macOS Finder wrote `.DS_Store` files into the repository at
  03:56:59Z when the folder was browsed. The copy inside `third_party/mjbatch` made the
  pristine-checkout gate fail, correctly. That one file was deleted, the gate passed
  again, and [troubleshooting.md](../troubleshooting.md) now covers it.
- **The user's MJBatch checkout** (`~/Workspace/Develop/GitHub/mjbatch`) was checked again at
  the end: `git status` is empty and it is on branch `main` at `2f16914`.

## 11. Open items requiring an owner decision

1. **Decided (2026-09-29).** AM-2: the owner accepted the measured light-model throughput
   (§5) and closed the optimization. The shortfall stays recorded; R-3.3.6 remains
   IMPLEMENTED.
2. **Decided (2026-09-29).** AM-16: the per-node watchdog is deployed in the run path.
3. onnxruntime's bundled telemetry client (present in both wheels): accept it, restrict
   pod egress, or build onnxruntime without it.
4. Production authorization and hardware for the BLOCKED gates. The ordered list of
   what remains is [production-todo.md](../deployment/production-todo.md).
5. Head memory sizing. The 2 GiB kind head reached Ray's 95% memory threshold with two
   concurrent evaluation runs and one driver process. Production head memory must be
   sized from measurements at the target scale (BLOCKED here).
