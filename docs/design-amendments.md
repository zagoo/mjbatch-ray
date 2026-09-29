# Design amendments

The technical design (`RaySimBatch-Technical-Design.md`, SHA-256 `d9070072…5462`, recorded in
[reading-ledger](ledgers/reading-ledger.md)) is the authority for behavior. Each entry
below records a place where the implementation departs from it: what the design said,
what was built, the cause (source evidence, a controlling assignment constraint, or an
observation in qualification), and how it is verified. MJBatch citations refer to revision
`2f169146b643adda65e195d8ad13d997abdd2a2e` (v0.1.1), with findings `M1`–`M20` from
[mjbatch-source-findings](ledgers/mjbatch-source-findings.md).

Status values: **ADOPTED** (implemented and verified), **ADOPTED-PARTIAL** (implemented
with a measured limitation), **OPEN** (needs an owner decision).

---

## AM-0: No maintained MJBatch extension

- **Design (§1, line 17):** "A small maintained MJBatch extension is required for
  structured per-slot errors, explicit reset/restore semantics, compiled substep
  controllers and task kernels, and CPU-affinity hooks."
- **Controlling constraint:** the implementation assignment requires that RaySimBatch
  never modify, patch, vendor-and-modify or monkey-patch upstream MJBatch, and never
  depend on private layouts.
- **Built:** no extension. Every capability the extension would have provided is
  composed from MJBatch's public API; see AM-1 through AM-7. `tools/check_mjbatch_clean.py`
  enforces the boundary, and `tests/tooling/test_noninvasive.py` shows that each class of
  violation is caught.
- **Cost (measured, AM-2):** per-substep composition costs throughput on very light
  models. Fused intervals and native bookkeeping recover most of it with identical
  results, but not all of it. See [qualification report §5](qualification/report.md).
- **Status:** ADOPTED-PARTIAL. The light-model throughput target is met in some measured
  configurations only; the owner accepted that shortfall on 2026-09-29 (see AM-2).

## AM-1: Per-slot call outcomes from a public-buffer sentinel

- **Design:** structured per-slot error status returned by the native call.
- **Source:** `batch.h` `Guarded`/`LogTrap` (M8). A call that fails in any sim raises one
  `RuntimeError` naming the *first failing sim in time*. Failed sims keep their pre-call
  state and pending writes; other sims complete. There is no per-sim status array.
- **Built:** `compat/mjbatch_adapter.py` `BatchHandle._sentinel_call`. Before a call, the
  world body's bound `xpos[:,0,0]` is set to a sentinel. Every completed sim's outputs are
  copied back, which overwrites the sentinel with MuJoCo's world-body position (0). Sims
  still holding the sentinel afterwards failed. The raised message attributes the first
  failure, and an error without failed sims (or the reverse) raises
  `MJBatchIncompatible` instead of guessing.
- **Verified:** `tests/upstream/test_mjbatch_contract.py`. The runtime probes
  `first_error_attributed`, `failed_slots_exact`, `failed_keep_state` and
  `others_advanced` run in every process (`raysimbatch compat`, and the smoke report
  `compat.probes`). `tests/core` covers per-slot native and health failures.

## AM-2: Per-substep stepping with compiled kernels between calls; fused intervals

- **Design (§6.2):** compiled substep controllers and task kernels run inside MJBatch's step.
- **Source:** k calls of `step(ids, nstep=1)` are bitwise equal to one
  `step(ids, nstep=k)` (M13). The pool is blocking and parks between calls (M5).
  `step(ids, nstep, history=H)` records every selected sim's integration state after each
  substep (M17). `warning` rows are input fields carried per sim.
- **Built, per substep:** `worker/core.py` `MJBatchCore._substeps`.
  - For each substep, the C++ controller (`native/src/kernels.h`) writes `ctrl` for the
    active slots, then `step(active, nstep=1)` runs.
  - Per-slot outcomes follow (AM-1). Then health and physics-rate termination remove
    slots from the active set.
  - `forward` runs once per control boundary.
- **Built, fused (owner decision, 2026-09-28):** `MJBatchCore._fused_interval`. It is used
  when three conditions hold:
  - `health.check_each_substep=false`;
  - the actuator is `torque` or `position` with no physics-clock delay (`Controller.fusable`);
  - `worker.fuse_substeps` is true (the default).

  How an interval runs:
  - One `step(ids, nstep=d, history=H)` call covers the interval.
  - The controller's per-substep bookkeeping (energy, saturation and eval counters,
    delay count) is completed from the history rows, in per-substep order.
  - Termination is a pure predicate on `qpos`, scanned over the rows.
  - A slot that terminated before the last substep is rewound and stepped again to its
    terminal substep k in one `step(nstep=k+1)` call per distinct k (`_replay_terminated`).
    The rewind writes its pre-interval state row, carrying the interval's `ctrl`, and its
    warning counters.
  - A slot that failed natively was not written back. Its pending `ctrl` is cancelled,
    and it is replayed by `_substeps` to find the failing substep.

  The failure granularity the owner accepted, a whole interval, is therefore not needed.
  The only semantic condition is control-rate health, which is what
  `check_each_substep=false` means. That flag was previously ignored and now has these
  semantics in both paths.
- **Verified:**
  - Bitwise parity with scalar MuJoCo (`tests/core`, all fixtures, 1 and 3 threads,
    per-substep and fused, `-ffp-contract=off`).
  - Fused and per-substep cores stay bitwise equal in every result field and every
    per-slot state array (`tests/core/test_fused.py`). The runs cover mid-interval and
    last-substep terminations, native failures on substep 0 and inside an interval,
    control-rate health failures, warnings raised on discarded substeps and by the
    rewinding forward, snapshot/restore, and truncation.
  - Nine implementation mutants all fail those tests.
  - Runtime probes `fused_substeps_equal` and `history_rows_are_states`, plus upstream
    contract tests.
- **Serial bookkeeping (native, 2026-09-29).** The work around the MJBatch calls,
  shared by both paths, runs as a few native passes (`native/src/kernels.h` `Book`):
  - slot checks (`ready`/`begin`), counters, time limit and return sums
    (`account`/`settle`);
  - the whole post-call pass of a fused interval (`fused_settle`);
  - the per-substep loop body (`substep_settle`).

  `Controller.FinishSlot` does a fused interval's constant work once. Contiguous slot runs
  are indexed as slices. The former NumPy implementations are kept in tests as oracles
  (`tests/core/test_book.py`, `test_substep_settle.py`) and compared bitwise.
- **Measured** (`qualification/bench_throughput.py`, modes interleaved per round, Apple
  M4 Max, host load average 5–6; `evidence/perf/throughput-overhead-{64,256}.json`,
  revision `d3bd3a9`). Wrapper vs upstream `nstep=d`:

  | Slots | Path | cartpole T=1 / 2 / 4 | reacher T=1 / 2 / 4 |
  |---|---|---|---|
  | 64 | per-substep | 0.75 / 0.60 / 0.45 | 0.69 / 0.56 / 0.44 |
  | 64 | fused | **0.91** / 0.86 / 0.78 | 0.84 / 0.80 / 0.75 |
  | 256 | per-substep | 0.78 / 0.72 / 0.64 | 0.72 / 0.66 / 0.59 |
  | 256 | fused | **0.93** / **0.92** / 0.87 | 0.85 / 0.84 / 0.81 |

  quadruped (`pd`) is not fusable and meets the target everywhere (0.97–1.20). Before the
  native bookkeeping (`evidence/perf/throughput-fused-{64,256}.json`, `4b6df90`), the
  fused figures were 0.59–0.90.

  Serial overhead per call (`qualification/bench_overhead.py`,
  `evidence/perf/overhead-{before,after}.json`, one thread), before → after:

  | Configuration | Per-substep | Fused |
  |---|---|---|
  | cartpole, 64 slots | 124 → 45 µs | 70 → 23 µs |
  | cartpole, 256 slots | 169 → 79 µs | 109 → 41 µs |
  | reacher, 64 slots | 214 → 89 µs | 88 → 33 µs |
  | reacher, 256 slots | 327 → 168 µs | 156 → 75 µs |

  With a termination inside every interval (cartpole): fused 155 → 65 µs at 64 slots and
  202 → 105 µs at 256; per-substep 132 → 53 and 185 → 88.
- **Status:** ADOPTED-PARTIAL; the shortfall is **accepted by the owner (2026-09-29)**.
  The §9.1 target (≥90% of equal-thread upstream) is met:
  - for the contact model at every measured configuration;
  - for cartpole with fused stepping at 64 slots on 1 thread and at 256 slots on 1–2
    threads.

  It is **not** met for cartpole otherwise (0.78–0.87 fused), nor for reacher (0.75–0.85).
- **Accepted baseline (owner decision, 2026-09-29).** AM-2 optimization is closed. The
  figures under *Measured* (revision `d3bd3a9`, Apple M4 Max) are the accepted
  light-model throughput. The levers below are recorded and not pursued. The acceptance:
  - does not make the target met: R-3.3.6 stays IMPLEMENTED, not VERIFIED;
  - covers this host only. The ratios on the reference x86-64 nodes are unmeasured
    (R-X86) and are measured before production
    ([production to-do](deployment/production-todo.md), item 3.1);
  - makes these figures the reference for regressions: a later change is compared with
    `evidence/perf/throughput-overhead-{64,256}.json` by the same interleaved benchmark.
- **Remaining gap (measured; not pursued):**
  - The fused MJBatch call is within about 4% of upstream's. History recording costs
    3–4%; the extra bound fields about 1% (same Batch, same physics).
  - Serial work of 23–41 µs per call remains (cartpole fused). About half of it is
    observation, action processing and controller kernels; the rest is Python dispatch
    and result allocation. It weighs most with 4 threads and at 64 slots.
  - A call that replays a termination still carries about 40–65 µs more (replay glue
    plus one extra MJBatch call). It is faster in wall time than per-substep stepping.
  - reacher runs one boundary `forward` per interval (about 12% of its wall time) for its
    end-effector reward, which upstream's baseline does not compute. Folding it into the
    step call would change which `qacc` the health check reads.

## AM-3: `set_const` as a pure function, re-run for isolation

- **Design (§6.2 lines 331–337):** refresh constants for selected slots, and prewarm the
  expansion closure on a disposable Batch.
- **Source:** `set_const` recomputes the selected sims, and may expand not-yet-expanded
  derived fields for *all* sims (closure growth, M14).
- **Built:** randomization validates the closure bytewise (`worker/randomization.py`
  `ClosureGuard`). Restore and reset re-run `set_const` on the affected slots, because
  the result is a pure function of the parameter rows.
- **Verified:** `tests/core` (sparse reset isolation, snapshot/restore across slots and
  processes), `tests/upstream` (closure semantics).

## AM-4: Pending-write neutralization

- **Design:** the native extension would guarantee that a failed slot leaves no pending
  write behind.
- **Source:** bound input fields are mirrored. Writes are applied on the next call, and
  a failed sim keeps its pending writes (M9).
- **Built:** the controller saves the `ctrl` rows it writes (`native/src/kernels.h`
  `Controller` `ctrl_saved`). For failed slots it restores them (`Neutralize`) before
  any later call, so a stale command is never applied.
- **Verified:** probe `neutralize_cancels_pending`, and `tests/core` per-slot failure
  tests.

## AM-5: CPU affinity without engine hooks

- **Design (§6.2, §11.4):** affinity hooks in the extension, and a node affinity agent.
- **Built:** `worker/affinity.py` and `worker/affinity_agent.py`.
  1. The lease is applied to the constructing thread *before* the Batch exists, so
     MJBatch's pool threads inherit it (Linux).
  2. Each new pool thread is then pinned to one core and verified from
     `/proc/self/task/*/status`.
  3. The per-node agent leases disjoint core sets from the pod's actual cpuset, which
     the kubelet static CPU manager assigns, never from Ray logical IDs. Leases are
     persisted and reaped only when their holder's process has died.
- **Verified:**
  - `tests/linux/test_affinity.py` and `tests/linux/test_affinity_agent.py`, run in the
    Linux test image.
  - In kind, worker pods received exclusive cpusets (`2-5` and `6-9`), and agent leases
    fell inside them (`evidence/deploy/02-smoke-blue-both.log`, `affinity`).
  - Found in kind and fixed: threads that exit while being inspected are skipped.
- **Not verified:** multi-socket NUMA (BLOCKED, see the qualification report).

## AM-6: Exact reset through integration-state rows

- **Source:** reset applies pending field writes (M10). The integration state includes
  `mjSTATE_WARMSTART`, and restoring a row replays it exactly (M12).
- **Built:** RaySimBatch never leaves a pending write (AM-4). The reset distribution is
  written through the authoritative state rows, then `forward` runs.
- **Verified:** `tests/core` reset and restore parity. The probe
  `reset_applies_pending_field_writes` pins the upstream behavior.

## AM-7: Jacobian and diagnostic queries on a private scalar model

- **Design:** batched Jacobians inside the extension.
- **Built:** `MJBatchCore.query_jacobians` evaluates the requested slots on a private
  scalar `MjModel`/`MjData` loaded from the slot's state row, because MJBatch exposes no
  per-thread `mjData`.
- **Verified:** `tests/core`.

## AM-8: Adapter contract fields for Sim2Sim ordering facts

- **Design (§B.3):** an adapter contract from which observations and actions can be assembled.
- **Built (`config/models.py`):**
  - `ObservationTerm.selectors`, `offsets`, `signs` and `period_s`;
  - observation sources `applied_action` and `base_quat`;
  - `ActuatorContract.actuator_names`;
  - `AdapterContract.history_order`, `term_op_order`, `normalization_clip` and `base_joint`.
- **Cause:** the Sim2Sim gates could not detect reversed signs or swapped history order
  unless the exporter attests these facts. A reversed sign was invisible until physical
  signs were derived from resolved properties.
- **Verified:** `tests/sim2sim/test_gates.py` (golden mismatch suite, 13 tests).

## AM-9: Sleep and stateful plugins run on the scalar lane

- **Source:** MJBatch rejects `sleep` (M20).
- **Built:** `worker/scalar.py` `ScalarHandle` gives the same handle interface for one
  environment. It captures and restores the complete `mjData`.
- **Verified:** `tests/core/test_scalar_lane.py`. Observed and recorded there: a restore
  of the integration state alone also replays sleeping scenes exactly.

## AM-10: `RunConfig.tasks`

The CLI submits runs from a configuration file, so the evaluation task suite is part of
`RunConfig`. The Python API still passes tasks explicitly.

## AM-11: `InferenceConfig.device`

The central service device (`cpu`, `mps` or `cuda`) is explicit. Requesting CUDA without
a CUDA provider is an error, never a silent CPU fallback (`policy/runtime.py`).

## AM-12: Compiled observation tables

Observation assembly uses compiled term tables in C++ (`native/src/kernels.h`
`Observation`: source, dimension, history, frame, index pools, sign pools), so no Python
marshalling happens at physics rate.

## AM-13: Per-node helpers and finished controllers exit when idle

- **Design (§7.3, §11.4):** long-lived per-node AssetCache and affinity agent actors, and
  a detached run controller.
- **Observed (kind, Ray 2.58.0 with the KubeRay 1.7.0 in-tree autoscaler):** a node that
  hosts any live actor, even one with no CPU, or that holds pinned objects, is never idle.
  Finished controllers, caches and agents therefore kept every worker node alive: 0/8
  CPU was in use for over an hour and 2 workers were kept.
- **Built:**
  - Controllers persist their results in the run row and exit after a linger.
  - Cache entries are held per run and released at cleanup (the janitor releases holds of
    runs that are no longer live); an empty, idle cache exits.
  - Agents exit after `idle_exit_s` without live leases.
- **Verified:**
  - `tests/ray/test_node_helpers.py` and `tests/e2e/test_mode_b.py::test_finished_run_controller_exits_and_status_stays_durable`.
  - In kind, blue scaled from 2 workers to 1 without intervention
    (`evidence/deploy/21-blue-scale-down-watch.log`).

## AM-14: KubeRay 1.7.0 `rayStartParams` quoting; `system-config` on the head only

- **Design (§11.4 CRD snippet):** unquoted JSON in `rayStartParams`, and `system-config`
  on every group.
- **Observed:**
  - KubeRay splices the values unquoted into `bash -c "ray start …"`, which mangled the JSON.
  - Ray 2.58 rejects `system-config` on worker nodes ("can only be set on the head node").
- **Built:** `deployment/render.py` `_shell_json` (compact JSON in single quotes).
  `system-config` is set on the head only.
- **Verified:** `tests/deployment/test_render.py` (bash splicing, and validation against
  the pinned KubeRay 1.7.0 CRD), and the kind deployment.

## AM-15: Ray metric export names and labels

- **Design (§11.1):** `raysimbatch_*` names with bounded labels.
- **Observed (Ray 2.58.0):**
  - Custom metrics are exported as `ray_raysimbatch_*`.
  - Ray adds per-process labels (`WorkerId`, `Component`, `SessionName`, `NodeAddress`,
    `Version`).
  - A process's series disappear when it exits.
  - Export happens every `metrics_report_interval_ms` (10 s).
- **Built:** alert rules aggregate the per-process labels away, and reference only
  catalog names or Ray native names verified on the pinned version
  (`deploy/observability/ray-2.58.0-native-metrics.txt`, from
  `qualification/verify_ray_metrics.py`). Run-level truth is the durable store and the
  run manifest.
- **Verified:** `tests/telemetry/*` (static checks), `qualification/check_alerts.sh`
  (`promtool check` and unit tests), and `tests/e2e/test_telemetry.py` (scrapes the real
  exporter).

## AM-16: Liveness: coordinator deadlines and the per-node watchdog

- **Design (§8.4):** a per-node Watchdog actor compares native progress and active-call
  deadlines.
- **Built (coordinator):** the coordinator implements the §8.4 sequence for every
  in-flight command: active-call deadline → cooperative cancel → `cancel_grace_s` →
  durable fence → kill with restart → re-initialization. Restart and circuit budgets are
  bounded.
- **Built (watchdog; owner decision 2026-09-29 to deploy it):** one `Watchdog` per node
  and namespace (`recovery/watchdog.py` `watchdog_for_node`: `rsb-watchdog-<node>`,
  pinned to its node, exits after 300 s with nothing registered). The first creator's
  failure settings apply to the whole node.
  - Each coordinator registers every worker after each (re)initialization, with its
    generation. It collects that node's faults for its own workers without blocking its
    loop (`inspect(names)`, at most one call per node in flight).
  - The watchdog polls every worker's control-group `health` each `heartbeat_s`,
    concurrently. Faults: `HEARTBEAT_TIMEOUT` (no answer for `suspect_s`, e.g. a
    GIL-holding call); `NATIVE_PROGRESS_STALLED` (active past `native_max_deadline_s`
    with no native completion for `suspect_s`); `WORKER_DIED`; `WORKER_RESTARTED` (back
    as a fresh, uninitialized process). Each fault names the generation it is about, and
    the coordinator ignores faults about a generation it has already replaced.
  - For a busy worker, a fault starts the cancel → fence → kill sequence at once instead
    of at the command deadline. An idle worker is recovered at once (fenced and killed
    first if it is hung). Before, an idle worker's death was found only when its next
    command failed.
  - A restarted watchdog reports the names it lost as `UNWATCHED`, and their owner
    registers them again. A watchdog that does not answer within `suspect_s` is replaced
    (event `watchdog_unavailable`); command deadlines still apply meanwhile.
  - It emits `heartbeat_age_seconds`. Its polling of `health` also records the workers'
    `worker_rss_bytes` and `log_dropped_total`, which no run path recorded before.
- **Verified:**
  - `tests/e2e/test_watchdog_liveness.py`: a worker holding the GIL inside a command is
    recovered by the watchdog well before its 60 s command deadline, and an idle worker
    killed mid-run is recovered without any command. Both runs give exactly the episodes
    of a clean run. Both tests fail when the coordinator's watchdog polling is removed.
  - `tests/ray/test_watchdog.py`: native hang and GIL hold are detected; a shared
    watchdog gives each owner only its own faults, reports names it does not watch, and
    stops watching a worker that is permanently gone.
- **Status:** ADOPTED.

## AM-17: The evaluation actor floor

- **Design (§10):** `evaluation.wave_actors` sizes an evaluation wave.
- **Built:** every model × policy runtime gets at least one actor, so the effective count
  is `max(wave_actors, models × policies)`. Validation (`CAPACITY`) and `plan_capacity`
  both reject plans above `max_cpu_nodes × actors_per_node`.
- **Cause (kind):** an unbounded plan started actors that could never be scheduled, and
  failed only at the worker-initialize deadline.
- **Verified:** `tests/config/test_config.py`.
