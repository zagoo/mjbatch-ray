# RaySimBatch

Distributed, batched CPU MuJoCo simulation on Ray, built on an **unmodified** upstream
[MJBatch](https://github.com/kevinzakka/mjbatch). It supports three workloads:

- **Mode A (rollouts):** heterogeneous model groups (different observation and action
  widths) under a curriculum, with local or central policy inference. Fragments are
  delivered exactly once to a durable replay sink or a learner queue.
- **Mode B (evaluation):** the checkpoint × task × seed matrix as durable units, with
  per-episode results committed exactly once. Statistics are seed-cluster bootstrap CIs
  with paired, Holm-adjusted comparisons. Exports are Parquet, CSV, JSON and Markdown.
- **Mode C (Sim2Sim):** twelve automated gates for policies transferred from another
  engine, then paired diagnostic evaluation and a transfer scorecard.

Deployment is KubeRay with blue/green releases. It includes Prometheus alerts, runbooks,
a reproducibility manifest per run, and a janitor that resumes orphaned runs from
durable state.

> **Status: implementation complete for the design's normative scope; production
> qualification INCOMPLETE.** Several gates cannot be run in the available environment:
> x86-64 128-core NUMA nodes, CUDA, 10k–100k-environment scale, 24-hour soak, real Isaac
> traces, a production cluster, and Redis GCS fault tolerance. The throughput target is
> met for the lightest models only in some configurations (AM-2; the owner accepted this
> shortfall). See the
> [qualification report](docs/qualification/report.md).

## Quick start (local)

```bash
git clone --recurse-submodules <repository-url> raysimbatch && cd raysimbatch
```

```bash
source tools/env.sh && uv sync --extra torch
```

```bash
tools/local_services.sh start && eval "$(tools/local_services.sh env)"
```

```bash
python examples/mode_b_evaluation.py
```

The last command evaluates an LQR policy against a do-nothing policy on cart-pole and
prints success rates with 95% seed-cluster intervals and the paired difference.

## Documentation

| Topic | Document |
|---|---|
| What it is and how it fits together | [docs/architecture.md](docs/architecture.md) |
| Install, build, test | [docs/installation.md](docs/installation.md) |
| Configuration (generated reference and JSON schema) | [docs/reference/configuration.md](docs/reference/configuration.md) |
| Modes A, B, C and explicit routing | [docs/usage/](docs/usage/) |
| KubeRay deployment, blue/green, rollback | [docs/deployment/kuberay.md](docs/deployment/kuberay.md) |
| Metrics, logs, traces, alerts | [docs/operations/observability.md](docs/operations/observability.md) |
| Incident runbooks | [docs/operations/runbooks.md](docs/operations/runbooks.md) |
| MJBatch boundary, and upgrade/rollback | [docs/mjbatch-integration.md](docs/mjbatch-integration.md), [docs/operations/mjbatch-upgrade.md](docs/operations/mjbatch-upgrade.md) |
| Troubleshooting (every entry observed) | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Extension points | [docs/extending.md](docs/extending.md) |
| Departures from the design | [docs/design-amendments.md](docs/design-amendments.md) |
| Qualification results, requirement matrix, `[VERIFY]` gates, MJBatch boundary | [docs/qualification/report.md](docs/qualification/report.md), [docs/qualification/requirements.md](docs/qualification/requirements.md), [docs/qualification/verify-gates.md](docs/qualification/verify-gates.md), [docs/qualification/mjbatch-noninvasiveness.md](docs/qualification/mjbatch-noninvasiveness.md) |

## Pinned stack

Python 3.12 · MuJoCo 3.14.0 · MJBatch 0.1.4 (`088e2d4c`) · Ray 2.58.0 · KubeRay 1.7.0 ·
PostgreSQL 17 · S3-compatible storage. Everything else is locked in `uv.lock`.
