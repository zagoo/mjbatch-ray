# SPDX-License-Identifier: Apache-2.0
import os

import pytest


@pytest.fixture(scope="session")
def ray_session():
  """A real local Ray runtime (not a mock): actors, object store, ownership."""
  import ray

  if not ray.is_initialized():
    ray.init(
      num_cpus=int(os.environ.get("RSB_TEST_RAY_CPUS", "16")),
      object_store_memory=int(os.environ.get("RSB_TEST_STORE_BYTES", str(1 << 30))),
      include_dashboard=False,
      namespace="rsb-tests",
      log_to_driver=False,
      # Export custom metrics every second (default 10 s) so telemetry tests observe
      # short runs; production keeps Ray's default.
      _system_config={"metrics_report_interval_ms": 1000},
    )
  yield ray
  ray.shutdown()


_EXIT: dict[str, int] = {}


def pytest_sessionfinish(session, exitstatus):
  _EXIT["status"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
  """Observed (onnxruntime 1.30, macOS and Linux wheels): ORT's bundled 1DS telemetry
  thread can abort the process during static destruction at interpreter exit (exit 134
  after every test passed). Once results are reported, exit with pytest's own status and
  skip third-party destructors. disable_telemetry_events() does not stop that thread."""
  import sys

  if "status" in _EXIT and os.environ.get("RSB_TEST_NATURAL_EXIT") != "1":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT["status"])
