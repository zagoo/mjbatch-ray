# Architecture

RaySimBatch runs batched CPU MuJoCo simulation on a Ray cluster. Each simulation actor
owns one unmodified MJBatch `Batch` (a C++ pool that steps many MuJoCo instances), and
RaySimBatch adds what distributed training and evaluation need:

- compiled per-substep controllers, health checks and task kernels (C++);
- per-slot fault isolation, exact resets and snapshots;
- an immutable publication protocol (RSB/1) through Ray's object store;
- durable, exactly-once results in PostgreSQL plus a content-addressed blob store;
- three workload modes; and
- a KubeRay deployment.

The authority is the technical design. Where the implementation departs from it, the
change is recorded in [design-amendments.md](design-amendments.md). The boundary with
MJBatch is described in [mjbatch-integration.md](mjbatch-integration.md).

## Modes

| Mode | What it produces | Entry point |
|---|---|---|
| **A: rollout** | Transition fragments (heterogeneous model groups, local or central inference) in a durable replay sink or a learner queue, exactly-once by fragment digest; policy-lag bounded; tail losses recorded as explicit gaps | `RunRequest("rollout", ...)`, `RolloutOrchestrator` |
| **B: evaluation** | The checkpoint × task × seed matrix as durable units; per-episode results committed exactly once; seed-cluster bootstrap CIs, paired effects, Holm-adjusted comparisons; Parquet/CSV/JSON/Markdown exports and a manifest written last | `RunRequest("eval", ...)` |
| **C: Sim2Sim** | Gated transfer of a source-engine (e.g. Isaac) policy: 12 automated gates on a transfer bundle, then paired evaluation along diagnostic axes and a scorecard | `Sim2SimOrchestrator` |

## Runtime topology

```
client ─► ClusterDriver ──submit──► RunController (detached, 1 per run, run lease)
                                     │  admission: artifacts → compiled models/adapters/policies
                                     │  capacity plan → placement group (gang, eval waves)
                                     ▼
                         ShardCoordinator × ⌈actors / 8⌉  (coordinator lease; B.1 loop)
                           │ commands (RSB/1)   ▲ envelopes: status table + ObjectRefs
                           ▼                   │
                 SimulationWorker × N  (1 MJBatch Batch, T threads, E slots, credits)
                           │ frames/fragments via ray.put (immutable, credit-bounded)
                           ├──► PolicyInferenceService (central mode, batched, deadline)
                           └──► ExperienceAssembler ─► replay sink / learner queue (Mode A)
per node: AssetCache (shared immutable bytes), AffinityAgent (core leases), both exit when idle
head pod: cluster janitor sidecar (orphaned runs, zombies, placement groups, cache holds)
durable:  PostgreSQL (runs, units, episodes, outbox, leases, commits, gaps, audit) + S3 blobs
```

A worker step, for each control interval:

1. The C++ controller writes `ctrl` for each active slot.
2. MJBatch `step(active, nstep=1)` runs.
3. Per-slot outcomes are read from public buffers.
4. Health and task kernels run.
5. Failed and terminated slots leave the active set.
6. Steps 1–5 repeat for each substep. After the last one, a single `forward` produces
   observations through compiled term tables.

## Module map

| Area | Module | Responsibility |
|---|---|---|
| MJBatch boundary | `compat/mjbatch_adapter.py` | The only mjbatch import. `QUALIFIED` pins, behavior probes, `BatchHandle` (sentinel per-slot outcomes, `set_const` isolation), state layout. |
| Native kernels | `native/src/{kernels,arrays,philox}.h`, `module.cpp` | Controllers (torque/position/PD/DC motor, envelopes, delays), health, observation assembly, task kernels (balance, velocity tracking, reach), Philox RNG (bit-identical to numpy) |
| Worker | `worker/core.py` | `MJBatchCore`: the per-substep algorithm, exact reset, snapshot/restore, Jacobians on a private scalar model |
| | `worker/actor.py` | `SimulationWorker` actor: commands, credits, fencing, dedup cache, fragments, publication, local policy runs |
| | `worker/scalar.py` | Scalar lane for features that MJBatch rejects (sleep) or that need the complete mjData (stateful plugins) |
| | `worker/randomization.py`, `worker/model.py`, `worker/fragments.py` | Reset-time randomization over `expand` rows (closure guard); model compile and identity; columnar fragment building |
| | `worker/affinity.py`, `worker/affinity_agent.py` | Thread-inherited leases, per-TID pinning, NUMA locality; per-node core-lease agent |
| Adapters and tasks | `adapter.py`, `tasks.py` | Adapter contracts compiled into kernel tables; task bindings |
| Control plane | `control/driver.py` | `ClusterDriver` (submit/status/wait/cancel/resume/scale), `RaySimBatch` facade, `RolloutOrchestrator` |
| | `control/controller.py` | `RunController`: lease keeper, admission, placement, supervision, results, manifest, self-exit |
| | `control/coordinator.py` | `ShardCoordinator`: worker lifecycle, deadlines, cancel/fence/kill, restart and circuit budgets; `EvalProgram` and `RolloutProgram` |
| | `control/admission.py`, `control/placement.py`, `control/assets.py` | Artifact verification and compilation; capacity plans and gang reservation; the per-node asset cache |
| Evaluation | `evaluation/{scheduler,aggregator,stats,schema}.py` | Durable units and leasing; exactly-once commits and exports; bootstrap/paired/Holm statistics; closed result schemas |
| Experience | `experience/{assembler,sinks}.py` | Fragment acceptance (dedup, lag, outbox, byte credits); durable replay and learner-queue sinks |
| Policy | `policy/{bundle,runtime,service,export}.py` | Policy bundles with explicit tensor contracts; local ONNX/TorchScript runtime; central batched service |
| Routing | `routing/{router,fleet,groups}.py` | `HeteroBatchRouter`: explicit asynchronous or barrier stepping over heterogeneous groups, exact migration, DRR |
| Sim2Sim | `sim2sim/*` | Source export format, reference pipeline, 12 gates, alignment harness, orchestrator, scorecards |
| Storage | `storage/{db,blob,store,secrets}.py` | PostgreSQL schema and transactions (leases, fencing, dedup); S3 and filesystem blobs; secret references |
| Recovery | `recovery/{watchdog,circuit}.py` | Per-node Watchdog (every worker registered by its coordinator, AM-16) and RestartManager; circuit breaker; janitor pass |
| Telemetry | `telemetry/*` | Catalog metrics, JSON logs, spans, reproducibility manifest |
| Operations | `cli.py`, `deployment/{render,smoke}.py` | CLI; KubeRay manifests; in-cluster smoke and cross-cluster resume checks |
| Protocol | `protocol.py`, `seeds.py` | RSB/1 records (primitives plus C-contiguous LE arrays); HMAC seed derivation |

## Correctness mechanisms

| Concern | Mechanism | Evidence |
|---|---|---|
| Split-brain owners | PostgreSQL leases on database time. The run, coordinator and worker generations are fenced by epoch. A non-owner never writes run state. | `tests/storage`, `tests/e2e/test_mode_b_chaos.py`, kind rollback exercise |
| Duplicate or lost results | Episode commits keyed by unit, attempt and lease epoch, with digest dedup. Fragments keyed by content digest through an outbox. Blobs are written before the commit. | chaos tests: identical digests after worker, coordinator and controller loss |
| Hung native calls | Active-call deadline, or a node-watchdog fault → cooperative cancel → durable fence → kill with restart. Bounded restarts and circuit breaker. | `tests/ray/test_watchdog.py`, `tests/e2e/test_watchdog_liveness.py`, chaos tests |
| Unbounded memory | Output credits for frames and fragments, assembler byte credits, bounded queues, deadlines | `tests/ray/test_worker_actor.py` (credits), fragment livelock regression tests |
| Reproducibility | HMAC seed derivation; Philox counter RNG; bitwise parity with scalar MuJoCo (`-ffp-contract=off`); a reproducibility manifest per run | `tests/core`, run manifest |

## Where to go next

- [Installation and local development](installation.md)
- [Configuration reference](reference/configuration.md)
- Usage guides: [Mode A](usage/mode-a-rollout.md), [Mode B](usage/mode-b-evaluation.md), [Mode C](usage/mode-c-sim2sim.md), [heterogeneous batches and routing](usage/routing.md)
- [KubeRay deployment](deployment/kuberay.md), [production to-do](deployment/production-todo.md), [observability](operations/observability.md), [runbooks](operations/runbooks.md), [MJBatch upgrade](operations/mjbatch-upgrade.md)
- [Troubleshooting](troubleshooting.md), [extension points](extending.md)
- [Qualification report](qualification/report.md) and [requirement matrix](qualification/requirements.md)
