# Implementation status and resume point

Overall: **INCOMPLETE: not production-qualified.** The implementation covers the design's
normative scope (with the amendments in `docs/design-amendments.md`). Several
qualification gates are BLOCKED in this environment, and the light-model throughput
target is not met. Details are in `docs/qualification/report.md` and the matrix in
`docs/qualification/requirements.md`.

## Where things are

| Area | State |
|---|---|
| Local test suite | On MJBatch 0.1.4 (port, 2026-10-09): 302 passed, 7 skipped (Linux-only), 1 failed because Ray refused to spill on a 98%-full disk; that test passed on rerun with Ray's disk threshold raised (`evidence/mjbatch-upgrade/088e2d4/pytest-full.log`, `pytest-spill-rerun.log`). Before, on 0.1.1: 300 passed at `5f4629f` |
| Linux tests | 153 passed in the `test` image on MJBatch 0.1.4 (Linux-only, `tests/core`, `tests/upstream`): `evidence/mjbatch-upgrade/088e2d4/linux-container.log` |
| amd64 image | `raysimbatch-cpu:amd64` and `:amd64-test` rebuilt on MJBatch 0.1.4 at `7f7d10a` under Rosetta (2026-10-09, `evidence/deploy/amd64-image-0.1.4/`): x86-64 native modules, compat passes, 149 of the 153 Linux-suite tests pass; the 4 Ray tests are unreliable because Ray stays up only seconds under Rosetta (passed once, died once). Ray is unverified on x86-64 hardware (3.1). The amd64 python base (pulled with approval 2026-09-29) is not project-built |
| kind KubeRay environment | `rsb-qual` is running (2026-10-09): `rsb-blue` active on image `8c46586` (MJBatch 0.1.4), green deleted by the rollback exercise, PostgreSQL and MinIO on PVCs. Requalified at `8c46586` (`evidence/deploy/requalify-8c46586/`). The kind config is `.tools/kubeconfig` (`source tools/env.sh`). If Docker was restarted, check `kubectl -n raysimbatch-qual get pods` for pods stuck Terminating (force-delete them), and recreate with `deploy/kind/up.sh blue blue` if needed. |
| Release tooling | `deploy/release.sh --context <ctx> --namespace <ns>` (production), `deploy/kind/release.sh` (kind); requalify an image with `deploy/kind/requalify.sh <evidence-dir>` |
| Release rollback exercise | `deploy/kind/rollback_exercise.sh`; evidence in `evidence/deploy/requalify-8c46586/`, `evidence/deploy/requalify-6e0ad7e/`, `evidence/deploy/rollback-exercise*` |
| Janitor orphan exercise | `deploy/kind/janitor_exercise.sh`; evidence in `evidence/deploy/janitor-exercise*` |
| AM-2 throughput | Closed. Accepted baseline and regression reference since 2026-10-09: the MJBatch 0.1.4 figures (`10fff3a`, `evidence/perf/mjbatch-0.1.4/throughput-overhead-{64,256}.json`), replacing the 0.1.1 figures accepted on 2026-09-29. The target is met for quadruped everywhere and for cartpole fused at 256 slots on 1–2 threads, not for the light models otherwise (`docs/design-amendments.md` AM-2) |
| MJBatch upgrade | Pinned to `088e2d4` (0.1.4, MuJoCo 3.14.0) since 2026-10-09; the port and what the gates missed (the thread pool change) are in `docs/qualification/mjbatch-noninvasiveness.md` §3. Tool: `tools/mjbatch_upgrade.py` (`source tools/env.sh` first); evidence in `evidence/mjbatch-upgrade/088e2d4/`. A passing Linux gate (`tools/mjbatch_upgrade.py linux`, runbook step 4a) is required for every pin; `tools/check_mjbatch_clean.py` enforces it in CI |
| Docs | `docs/` (index: `docs/architecture.md`); qualification: `docs/qualification/report.md` |
| Observability | Every §11.1 metric emitted except `native_arena_peak_bytes` (accepted gap). Four dashboards, checked live against Prometheus and in the pinned Grafana 13.2.2 (`evidence/observability/dashboards-live.json`; rerun `qualification/verify_dashboards.py`, `--hold 600` to look at the panels). The Grafana image (~1.3 GB, pulled 2026-09-29) is not project-built, so the prune commands below keep it |
| Next | `docs/deployment/production-todo.md` (reviewed at `8a8ed95`): owner decisions 0.1, 0.3, 0.4; engineering that can start now: 1.6, 1.4 (needs a real CI runner). The host disk was 98% full on 2026-10-09 (Ray refuses to spill above 95%). |

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
