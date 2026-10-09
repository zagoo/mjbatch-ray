#!/usr/bin/env bash
# The Linux-only test suite, run in a `test`-stage image: CPU affinity pinning, the
# affinity agent, the /proc and /sys readers, the native core and the MJBatch contract
# on Linux. macOS skips these. CI's image-and-linux job and the MJBatch upgrade's
# Linux gate (tools/mjbatch_upgrade.py linux) both run exactly this.
# Usage: tools/linux_suite.sh <image> [extra pytest args...]
set -euo pipefail
IMAGE="${1:?usage: linux_suite.sh <image>}"; shift
exec docker run --rm "$IMAGE" python -m pytest -q -rs \
  tests/linux tests/ray/test_node_helpers.py tests/telemetry/test_procfs.py tests/core tests/upstream "$@"
