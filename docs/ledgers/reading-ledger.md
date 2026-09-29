# Reading ledger

## Technical design

| Field | Value |
|---|---|
| Path | `/Users/renshaobin/Documents/ai infra/RaySimBatch-Technical-Design.md` |
| SHA-256 | `d90700723665bdbedad32d3c9856f733f8188b9428f160265f93811e02ac5462` |
| Bytes / lines | 161,655 / 1,288 (no line exceeds 1,500 characters) |
| Read on | 2026-09-28 |
| Contiguous ranges read | 1–220, 221–480, 481–740, 741–1020, 1021–1288 |
| Coverage | 1–1288, no gaps |
| Re-verification | `tools/check_design_hash.py` recomputes the hash; a mismatch requires reconciling the diff before dependent work continues |

The design is preserved unmodified at that path. Changes needed to meet it with an
unmodified MJBatch are recorded in [`../design-amendments.md`](../design-amendments.md).

## MJBatch (pinned dependency)

| Field | Value |
|---|---|
| User checkout | `/Users/renshaobin/Workspace/Develop/GitHub/mjbatch` (branch `main`, clean, not modified) |
| Session worktree | `.../mjbatch/.claude/worktrees/raysimbatch-implementation-7b14fb` (clean, not modified) |
| Remote | `https://github.com/kevinzakka/mjbatch.git` |
| Pinned revision | `2f169146b643adda65e195d8ad13d997abdd2a2e` ("Bump version to 0.1.1. (#8)", package 0.1.1) |
| Design's recorded revision | identical (`2f169146…`) |
| Upstream `main` at inspection | `2f169146…` (no newer commit) |
| Other upstream refs | `mujoco-versions` = `121e0331323eab76a30db34b5b9dae2111c837a8` (forks from `b84c0c2`, unmerged) |
| RaySimBatch copy | git submodule `third_party/mjbatch`, pinned to `2f169146…`, object store dissociated |

Files read completely (every line):

| File | Lines | Notes |
|---|---:|---|
| `src/mjbatch/csrc/batch.h` | 637 | execution engine, field tables, error trap |
| `src/mjbatch/csrc/bindings.cpp` | 84 | nanobind module, ABI check, docstring contract |
| `src/mjbatch/csrc/threadpool.h` | 86 | blocking sticky-slice pool |
| `src/mjbatch/__init__.py` | 83 | Python subclass, named views |
| `src/mjbatch/_bindings.pyi` | 85 | stub |
| `tests/test_batch.py` | 635 | upstream test suite |
| `CMakeLists.txt` | 43 | build, rpath, libmujoco discovery |
| `pyproject.toml` | 100 | scikit-build-core, pins `mujoco==3.13.0` |
| `Makefile`, `.github/workflows/ci.yml`, `.gitignore`, `.python-version`, `.clang-format` | — | build/CI policy |
| `README.md` | 79 | public contract |
| `benchmarks/scaling.py` | 81 | throughput methodology |
| `examples/hello.py` | 35 | |

Inspected, not read line by line: `examples/go1_joystick.py` (lines 1–120, reset/write
ordering and policy usage), the remaining examples (usage patterns only), `typings/`
(generated MuJoCo stubs), binary assets. No claim of complete reading is made for these.

## MuJoCo, Ray, KubeRay

Versions actually built and tested: MuJoCo 3.13.0 wheel (`mj_version()==3013000`), Ray
2.58.0, KubeRay operator per `deploy/kind/` (see the verification ledger). Behaviour the
implementation depends on is verified by executable tests rather than by source reading
alone: `tests/upstream/` (MJBatch + MuJoCo), `tests/ray_contract/` (Ray actor, cancel,
object and ownership semantics on 2.58.0).
