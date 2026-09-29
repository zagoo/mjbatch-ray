# Installation and local development

## Supported platforms

| Platform | Status |
|---|---|
| Linux x86-64 and arm64 (container images, Kubernetes) | production target. Images are built for the build host's architecture; pass `--platform linux/amd64` (or `linux/arm64`) to `deploy/docker/build.sh` for the other. Ray does not stay up under Docker Desktop's Rosetta emulation, so test a cross-built image on its own architecture. |
| macOS arm64 | development and testing. No CPU affinity or NUMA (the Linux-only tests are skipped), and no CUDA. |
| Windows | not supported (MJBatch does not build there, M20) |

Pinned versions come from the release lock (`uv.lock`) and the submodule:

| Component | Version |
|---|---|
| Python | 3.12 |
| MuJoCo | 3.13.0 |
| MJBatch | 0.1.1 at `2f169146…` (submodule `third_party/mjbatch`) |
| Ray | 2.58.0 |
| KubeRay operator | 1.7.0 |
| PostgreSQL | 17 |
| Other dependencies | locked in `uv.lock` |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.12 or later (the repository uses 0.12.19).
- A C++17 compiler and CMake 3.15 or later (used by scikit-build-core to build MJBatch
  and `raysimbatch._native`).
- For the durable-storage tests: PostgreSQL 17 binaries on `PATH` (`initdb`, `pg_ctl`)
  and Docker (for MinIO).
- For the Kubernetes environment: Docker, `kubectl`, and `kind`/`helm` (installed
  project-locally under `.tools/bin`).

## Get the source

```bash
git clone --recurse-submodules <raysimbatch-repository-url> raysimbatch
```

```bash
cd raysimbatch && git submodule status
```

The submodule must show `2f169146b643adda65e195d8ad13d997abdd2a2e third_party/mjbatch`.

## Build and install

```bash
source tools/env.sh
```

```bash
uv sync --extra torch
```

`tools/env.sh` puts the project environment (`.venv/bin`, so `python` below is the one
`uv sync` creates) and the pinned tools on `PATH`, caps BLAS/OpenMP threads at 1, and keeps
Helm and kind state inside the repository. `uv sync` builds MJBatch from the pinned,
unmodified submodule, with its build directory outside the checkout at
`build/mjbatch/<wheel_tag>`. It also builds RaySimBatch's native kernels (`native/`,
compiled with `-ffp-contract=off` for bitwise parity). The `torch` extra enables
TorchScript policies; ONNX policies need only the base install.

Verify the installation:

```bash
python tools/check_mjbatch_clean.py
```

```bash
python -m raysimbatch.cli compat
```

`check_mjbatch_clean.py` must print `OK … pristine … boundary respected`. `compat` must
print `"qualified": true` and every probe `true`.

## Local services for the durable-storage tests

```bash
tools/local_services.sh start
```

```bash
eval "$(tools/local_services.sh env)"
```

`start` creates a project-local PostgreSQL 17 cluster in `var/pg` on port 55432
(initialized with `--locale=C --encoding=UTF8`) and a MinIO container on
`127.0.0.1:59000`. Credentials are generated into `var/services.env` and never printed.
`env` exports `RSB_TEST_DSN`, `RSB_TEST_S3_ENDPOINT` and the S3 credentials. Stop both
with `tools/local_services.sh stop`.

> PostgreSQL must run under a C or UTF-8 locale. On a workstation with a non-UTF-8
> default locale (observed with `zh_CN`), `postmaster` refused to start. The script forces
> `LC_ALL=C`. RaySimBatch refuses a non-UTF8 database (`migrate` raises).

## Run the tests

```bash
python -m pytest -q tests
```

| Suite | Needs | Covers |
|---|---|---|
| `tests/upstream` | nothing | MJBatch semantics RaySimBatch relies on |
| `tests/core` | nothing | bitwise parity, resets, snapshots, kernels, the scalar lane |
| `tests/config`, `tests/deployment`, `tests/telemetry`, `tests/tooling` | nothing | configuration, KubeRay rendering (CRD-validated), catalog and alert rules, non-invasiveness checks |
| `tests/storage` | `RSB_TEST_DSN` (+ `RSB_TEST_S3_ENDPOINT`) | leases, commits, outbox, blobs |
| `tests/ray`, `tests/ray_contract` | a local Ray (started by the fixture) | worker actors, credits, router, watchdog, node helpers, Ray semantics |
| `tests/e2e` | Ray + `RSB_TEST_DSN` | Modes A, B and C end to end, chaos, telemetry |
| `tests/linux` | Linux | CPU affinity and the affinity agent (run them in the test image, below) |

Tests that need a missing service are skipped with the reason shown (`-rs`). On this
repository's qualification workstation the full suite takes about three minutes. The
test runner ends the process with pytest's own status to avoid a third-party exit-time
abort (see [troubleshooting](troubleshooting.md#exit-code-134-after-all-tests-pass)).

Linux-only tests in a container:

```bash
docker build --target test -f deploy/docker/cpu.Dockerfile -t raysimbatch-cpu:test .
```

```bash
docker run --rm raysimbatch-cpu:test python -m pytest -q tests/linux tests/ray/test_node_helpers.py
```

## Static checks

```bash
ruff check src tests tools qualification
```

```bash
python tools/gen_config_reference.py --check
```

```bash
python tools/check_design_hash.py
```

```bash
python tools/pyright_ratchet.py
```

CI gates on pyright with this ratchet (pinned pyright, configured in `pyproject.toml` for
Linux on every host). MuJoCo ships no type stubs; `typings/mujoco` holds stubs generated
from its bindings by `tools/gen_mujoco_stubs.py` (never edited by hand).
`src/raysimbatch/_native.pyi` is likewise generated from the built native module by
`tools/gen_native_stub.py`; regenerate it after changing `native/src/module.cpp`
(`tests/tooling/test_native_stub.py` checks it).

`src` is pyright-clean: the baseline (`tools/pyright-baseline.json`) is empty, so any
pyright error fails CI. Fix the error. Only a false positive may be baselined
(`python tools/pyright_ratchet.py --update`), with the reason in the commit; the check
also fails on a baseline that lists errors that are gone.

## Images

```bash
deploy/docker/build.sh <tag>
```

This builds `deploy/docker/cpu.Dockerfile` with `--target runtime`, which is also the
Dockerfile's last stage. The runtime image runs as uid 1000, includes no test code, and
performs no runtime `pip` installs. The build stamps the RaySimBatch and MJBatch
revisions as labels and environment variables, and appends the image ID to
`var/images.log`. The same Dockerfile has a `test` stage and a `sanitize` stage
(ASan/UBSan or TSan builds of MJBatch and the kernels, set by build flags only; see
`qualification/native_sanitizers.sh`). The CUDA image (`deploy/docker/gpu.Dockerfile`)
has not been built or qualified here (no CUDA host).

## Next

- A local Kubernetes environment: [deployment/kuberay.md](deployment/kuberay.md#local-qualification-environment-kind)
- A first evaluation run: [usage/mode-b-evaluation.md](usage/mode-b-evaluation.md)
