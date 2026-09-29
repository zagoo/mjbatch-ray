# SPDX-License-Identifier: Apache-2.0
"""Fault-injection workers for liveness/chaos qualification (never used in production).

``hang_native`` makes one genuinely long MJBatch call with the GIL released (solver
iterations expanded to 1e9, tolerance 0): heartbeats keep answering, progress stops.
``hold_gil`` runs catastrophic regex backtracking inside CPython's C ``_sre`` module:
the GIL is held, so even the control group cannot answer.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import ray

from raysimbatch import protocol as P
from raysimbatch.compat.ray_typing import actor_class
from raysimbatch.worker.actor import SimulationWorker


class HangableWorker(SimulationWorker):
  @ray.method(concurrency_group="mutate")
  def hang_native(self) -> None:
    core = self.core
    assert core is not None  # used on initialized workers only
    core.h.expand("iterations")[:] = 10**9
    core.h.expand("tolerance")[:] = 0.0
    s = np.arange(core.N, dtype=np.int32)
    self._begin(P.new_id())
    core.advance_control(s, np.zeros((core.N, core.A), np.float32))

  @ray.method(concurrency_group="mutate")
  def hold_gil(self) -> None:
    self._begin(P.new_id())
    re.match(r"(a+)+$", "a" * 60 + "b")


RemoteHangable = actor_class(HangableWorker, concurrency_groups={"mutate": 1, "control": 1})


class GilHoldOnceWorker(SimulationWorker):
  """Each worker's first local-policy command holds the GIL until the process is killed;
  in the restarted process commands run normally. A marker file per run and actor makes
  "first" survive the restart (same host only)."""

  @ray.method(concurrency_group="mutate")
  def run_local(self, command):
    marker = Path(tempfile.gettempdir()) / f"rsb-gil-once-{self.spec.run_id}-{self.spec.actor_name}"
    if not marker.exists():
      marker.touch()
      self._begin(command["command_id"])
      re.match(r"(a+)+$", "a" * 60 + "b")
    # @ray.method leaves the function callable; Ray's stubs type it as a remote method only.
    return super().run_local(command)  # pyright: ignore[reportCallIssue]


RemoteGilHoldOnce = actor_class(GilHoldOnceWorker, concurrency_groups={"mutate": 1, "control": 1})


class FailingWorker(SimulationWorker):
  """Every local-policy command raises: a deterministic application error in a worker."""

  @ray.method(concurrency_group="mutate")
  def run_local(self, command):
    raise RuntimeError("injected worker fault")


RemoteFailingWorker = actor_class(FailingWorker, concurrency_groups={"mutate": 1, "control": 1})
