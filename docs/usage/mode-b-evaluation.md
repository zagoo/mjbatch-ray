# Mode B: evaluation

Mode B evaluates policy checkpoints on task suites. The checkpoint × task × seed matrix
becomes durable **units**. Each episode runs from a derived seed and is committed exactly
once, and the results are reported with seed-cluster statistics.

## Run the example

```bash
eval "$(tools/local_services.sh env)"
```

```bash
python examples/mode_b_evaluation.py
```

Output on the qualification workstation (a local Ray, local PostgreSQL, file blob store):

```
run 17e5fa0971b2490a8f46b67d5accd7a9: COMPLETED, complete=True, coverage=1.0
  zero  success 0.000  95% CI [0.000, 0.000]  episodes 32/32
  lqr   success 1.000  95% CI [1.000, 1.000]  episodes 32/32
  paired zero - lqr: -1.000 (95% CI [-1.000, -1.000], Holm p=0.0)
exports: ['aggregate_csv', 'aggregate_json', 'aggregate_parquet', 'episodes_csv', 'episodes_parquet', 'manifest', 'maps_csv', 'run_manifest', 'scorecard_markdown']
```

(That run predates the p-value correction. The p-value is now reported as at least
2/(B+1); see [Statistics](#statistics).)

## What you provide

| Input | How |
|---|---|
| Model groups | MJCF (plus assets), or a precompiled MJB, registered as content-addressed artifacts with an adapter contract, a task kernel binding, randomization rules and a health configuration (`ModelGroup`) |
| Policies | a policy bundle (ONNX or TorchScript executable plus explicit tensor contracts and certified batch sizes; `PolicySpec`) |
| Tasks | `TaskSpec`: kernel, parameters, model groups, success threshold |
| Evaluation plan | `EvalConfig`: `seeds`, `episodes_per_seed`, `max_attempts`, `lease_s`, `wave_actors`, `gang`, `allow_partial`, `bootstrap_replicates`, `confidence`, `task_weights`, `exports` |

`raysimbatch artifacts put <file> --root <artifact_root>` registers a file and prints
its `{uri, sha256}` reference for use in the YAML configuration. The examples use
`raysimbatch.testing.runs`, which registers the reference fixtures the same way.

## Submitting

Python:

```python
rsb = RaySimBatch(config, namespace="raysimbatch")
run_id = await rsb.evaluate(tasks, seeds=range(20))
status = await rsb.wait(run_id, timeout_s=3600)
```

CLI:

```bash
raysimbatch submit --config eval.yaml --key nightly-2026-09-28
```

```bash
raysimbatch status <run_id> --config eval.yaml --wait 3600
```

The mode comes from the configuration's `mode` field. Submission is idempotent per
`--key`: resubmitting returns the same run ID. `--options '{"actors": 4}'` passes run
options.

## Execution semantics

- **Units and leasing.** A unit is one (policy, task, seed, axis) cell with
  `episodes_per_seed` ordinals. Units are leased with `FOR UPDATE SKIP LOCKED`. The lease
  (`evaluation.lease_s`) is renewed while the unit runs.
- **Gang waves.** With `evaluation.gang` (default true), a wave's simulation and service
  resources are reserved atomically in one placement group, or the wave is deferred. The
  controller keeps its run lease alive while it waits.
- **Exactly once.** Episode rows are committed per (unit, ordinal) under the unit's
  attempt and lease epoch. A repeat delivery with the same digest is acknowledged; a
  different digest is refused and pages (`RsbDigestConflict`).
- **Failures.**
  - Worker loss: the worker restarts under a new generation, and its in-flight episodes
    restart from their *original seeds*.
  - Coordinator loss: a replacement coordinator is started, bounded by `failure.max_restarts`.
  - Run-controller loss: the lease expires, and the cluster janitor or `resume` starts a
    new controller.
  - Units retry up to `max_attempts`, then end `EXHAUSTED`.
  - The chaos tests show that results after each kind of loss are identical, per episode,
    to an uninterrupted run.
- **Completion.**
  - `complete=true` means every unit succeeded and every expected episode committed.
  - With `allow_partial` (the default), a run whose units exhausted still ends
    `COMPLETED`. The exports then carry `complete=false` and the coverage.
  - Without `allow_partial`, such a run ends `FAILED`.
- **Resume.** `raysimbatch resume <run_id> --config <cfg>` continues a non-terminal run
  from the durable store. Committed ordinals are skipped. A resume with any `new_run` or
  `forbidden` configuration change is refused.

## Statistics

- The **seed is the independent unit**. Each seed's episodes are averaged first, then
  percentile-bootstrap intervals are computed over seeds (`bootstrap_replicates`,
  default 10,000). The bootstrap seed derives from the run ID, so intervals are
  reproducible.
- Success rates are reported two ways:
  - *conditionally*, over valid completed episodes;
  - *conservatively*, over all scheduled episodes, with unresolved or model-failed
    episodes counted as failures.

  The infrastructure-failure fraction is reported separately.
- **Paired comparisons** between policies use common seeds only and report paired
  coverage. Two-sided bootstrap p-values are estimated as (k+1)/(B+1) per tail, then
  Holm-adjusted across the predeclared comparisons.
- `low_seed_count` flags aggregates with too few valid seeds for a stable interval.

## Outputs

The aggregator writes the episode rows and aggregates first. Its manifest is written
last, after the database commit.

| Artifact | Content |
|---|---|
| `episodes_parquet`, `episodes_csv` | `EpisodeResult` v1: identity, seed keys, outcome, success, return, control/physics steps, simulated seconds, tracking RMSE, energy, saturation, randomization hash, reward components, fault code |
| `aggregate_parquet`, `aggregate_csv`, `aggregate_json` | `AggregateResult` v1 per (policy, task, axis, metric), with `paired` rows and the weighted `suite` mean |
| `scorecard_markdown` | human-readable tables |
| `manifest` | the evaluation manifest (artifact digests, counts, completeness) |
| `run_manifest` | the reproducibility manifest (config, code and binary hashes, hardware, seeds, failures) |

## Sizing notes

`wave_actors` bounds concurrent simulation actors, but every model × policy runtime gets
at least one actor. Validation refuses plans larger than `max_cpu_nodes ×
(simulation_cpus // worker.num_cpus)` (`CAPACITY`).
