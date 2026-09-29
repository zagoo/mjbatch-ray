# Implementation status and resume point

Overall: **INCOMPLETE: not production-qualified.** The implementation covers the design's
normative scope (with the amendments in `docs/design-amendments.md`). Several
qualification gates are BLOCKED in this environment, and the light-model throughput
target is not met. Details are in `docs/qualification/report.md` and the matrix in
`docs/qualification/requirements.md`.

## Where things are

| Area | State |
|---|---|
| Local test suite | 300 passed, 7 skipped (Linux-only; run in the container) at `5f4629f`: `evidence/tests/pytest-final.log` |
| Linux tests | 130 passed in the `test` image at `5f4629f` (Linux-only and `tests/core`): `evidence/tests/linux-container.log` |
| amd64 image | `raysimbatch-cpu:amd64` built at `050070e` under Rosetta (to-do 1.2): compat passes, 126 of the 130 Linux tests pass; the 4 Ray tests cannot run because Ray does not stay up under Rosetta (`evidence/deploy/amd64-image/`). Ray is unverified on x86-64 (3.1). The amd64 python base (pulled with approval 2026-09-29) is not project-built |
| kind KubeRay environment | `rsb-qual` is running (2026-09-29): `rsb-blue` active on image `6e0ad7e`, green deleted by the rollback exercise, PostgreSQL and MinIO on PVCs. Requalified at `6e0ad7e` (`evidence/deploy/requalify-6e0ad7e/`). If Docker was restarted, check `kind get clusters` and `kubectl -n raysimbatch-qual get pods`, and recreate with `deploy/kind/up.sh blue blue` if needed. |
| Release tooling | `deploy/release.sh --context <ctx> --namespace <ns>` (production), `deploy/kind/release.sh` (kind); requalify an image with `deploy/kind/requalify.sh <evidence-dir>` |
| Release rollback exercise | `deploy/kind/rollback_exercise.sh`; evidence in `evidence/deploy/requalify-6e0ad7e/`, `evidence/deploy/rollback-exercise*` |
| Janitor orphan exercise | `deploy/kind/janitor_exercise.sh`; evidence in `evidence/deploy/janitor-exercise*` |
| AM-2 throughput | Closed. The owner accepted the measured figures (`d3bd3a9`) on 2026-09-29; the target is not met for the lightest models in most configurations (`docs/design-amendments.md` AM-2) |
| MJBatch upgrade workflow | `tools/mjbatch_upgrade.py`; candidate `121e033` evidence in `evidence/mjbatch-upgrade/` |
| Docs | `docs/` (index: `docs/architecture.md`); qualification: `docs/qualification/report.md` |
| Observability | Every §11.1 metric emitted except `native_arena_peak_bytes` (accepted gap). Four dashboards, checked live against Prometheus and in the pinned Grafana 13.2.2 (`evidence/observability/dashboards-live.json`; rerun `qualification/verify_dashboards.py`, `--hold 600` to look at the panels). The Grafana image (~1.3 GB, pulled 2026-09-29) is not project-built, so the prune commands below keep it |
| Next | `docs/deployment/production-todo.md` (reviewed at `8a8ed95`): owner decisions 0.1, 0.3, 0.4; engineering that can start now: 1.10 (requalify the current revision on kind), 1.6, 1.4 (needs a real CI runner). `src/` is unchanged since the suite evidence at `5f4629f` |

## Resume commands

```bash
cd /Users/renshaobin/Workspace/Develop/GitHub/raysimbatch
```

```bash
source tools/env.sh && uv sync --extra torch
```

```bash
tools/local_services.sh start && eval "$(tools/local_services.sh env)"
```

```bash
python -m pytest -q -rs tests
```

```bash
python tools/check_mjbatch_clean.py && python tools/render_requirements.py --check && python tools/pyright_ratchet.py
```

Bring up the kind environment (requires Docker):

```bash
deploy/docker/build.sh blue && deploy/kind/up.sh blue blue && deploy/kind/release.sh smoke blue both
```

Free disk after image builds (project-scoped; check the printed counts):

```bash
deploy/docker/prune-build-cache.sh && deploy/kind/release.sh prune-images
```

Tear the qualification environment down (deletes the kind cluster and its volumes):

```bash
kind delete cluster --name rsb-qual
```

## Open items (owner decisions or unavailable resources)

The ordered production to-do list is `docs/deployment/production-todo.md`.

- Head memory sizing: the 2 GiB kind head reached Ray's 95% threshold under two runs.
- BLOCKED: x86-64 128-core NUMA nodes, CUDA inference and GPU image, 10k/50k/100k
  scale, 24-hour soak, real Isaac traces, production cluster, Redis GCS FT.
