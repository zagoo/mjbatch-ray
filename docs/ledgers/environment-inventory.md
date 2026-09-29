# Environment inventory (qualification workstation)

Recorded 2026-09-28. This is the only hardware available to this implementation effort.

| Item | Observed |
|---|---|
| Host | Apple M4 Max MacBook Pro, `arm64`, 16 cores (hw.physicalcpu = hw.logicalcpu = 16; no SMT), 64 GiB RAM |
| OS | macOS 26.6.2 (Darwin 25.6.0) |
| GPU | Apple integrated GPU (Metal/MPS). **No CUDA device.** |
| NUMA | single memory domain |
| Compilers | Apple clang 21.0.0 |
| Python | CPython 3.12.8 (Homebrew) — design baseline 3.12 |
| Package manager | uv 0.12.19 (project-local, `.tools/venv`) |
| Container runtime | Docker Desktop 29.7.2 (linux/arm64 VM); linux/amd64 containers run under Rosetta emulation, where Ray does not stay up |
| Kubernetes tooling | kubectl present; kind/helm installed project-locally under `.tools/bin` when used |
| PostgreSQL | Homebrew `postgresql@17` binaries; project-local data dir under `var/` |
| S3-compatible store | MinIO container (Docker) when used |
| Redis | not installed (GCS FT profile not exercisable here) |
| Credentials / cloud accounts | none provided; no cloud resources are used |
| Isaac Lab / Isaac Sim | not available; no source-engine traces available |

## Consequences for qualification

- Linux x86-64 with 128-core EPYC nodes (design §9.1) is unavailable. Linux-only paths
  (CPU affinity, NUMA binding, CPU Manager) are exercised inside Linux arm64
  containers where possible and otherwise marked BLOCKED. The linux/amd64 image is built
  and its non-Ray tests run under Rosetta emulation (to-do 1.2); that is not x86-64
  hardware.
- CUDA GPU inference cannot be qualified. The central inference service is exercised on
  CPU and on the Apple GPU (torch MPS); CUDA/L40S service curves are BLOCKED.
- 10k/50k/100k-slot scale, 24-hour soak, node/head failure on real clusters, and GCS FT
  with HA Redis are BLOCKED; harnesses and exact procedures are provided.
- No Isaac source traces exist here; Sim2Sim gates run on golden fixtures produced by an
  independent reference implementation, and real Isaac-trace qualification is BLOCKED.
