# SPDX-License-Identifier: Apache-2.0
"""worker_private_bytes and numa_remote_ratio from the real kernel files (the parsers are
pinned on sample text in tests/telemetry/test_procfs.py): a Linux process has private
memory, and its remote-memory share is a fraction, 0 when the host has one NUMA node."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from raysimbatch.telemetry import procfs

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc and /sys")


def test_private_bytes_of_this_process():
  before = procfs.read_private_bytes()
  block = np.ones(64 * 2**20 // 8)  # 64 MiB, touched: private and resident
  after = procfs.read_private_bytes()
  assert before is not None and after is not None and after - before >= 48 * 2**20
  del block


def test_numa_remote_ratio_of_this_process():
  ratio = procfs.read_numa_remote_ratio(os.sched_getaffinity(0))
  nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
  if not Path("/proc/self/numa_maps").exists() or not nodes:
    assert ratio is None
    return
  assert ratio is not None and 0.0 <= ratio <= 1.0
  if len(nodes) == 1:
    assert ratio == 0.0
