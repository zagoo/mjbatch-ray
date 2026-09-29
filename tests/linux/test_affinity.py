# SPDX-License-Identifier: Apache-2.0
"""Linux-only: CPU affinity without touching MJBatch (amendment AM-5). The calling
thread's lease is applied before the Batch exists so MJBatch's pool threads inherit
it; each physics thread is then pinned to one core and verified from /proc."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux CPU affinity")


def test_physics_threads_inherit_and_pin_the_lease():
  from raysimbatch.adapter import compile_adapter
  from raysimbatch.tasks import bind_task
  from raysimbatch.testing import fixtures as F
  from raysimbatch.worker.affinity import AffinityPlan, allowed_cpus, thread_ids
  from raysimbatch.worker.core import CoreSpec, MJBatchCore
  from raysimbatch.worker.model import ModelMetadata

  cpus = sorted(os.sched_getaffinity(0))
  if len(cpus) < 4:
    pytest.skip("needs >= 4 CPUs")
  lease = tuple(cpus[-2:])
  fx = F.quadruped()
  model = fx.model()
  meta = ModelMetadata.from_model(model)
  b = bind_task(fx.task_doc, meta)
  c = compile_adapter(fx.contract, meta, num_commands=b.num_commands)
  plan = AffinityPlan("numa_exclusive", lease, 2)
  plan.before_construction()
  core = MJBatchCore(model, meta, c, b, CoreSpec(num_slots=16, threads=2, decimation=4, timestep_ns=4_000_000,
                                                 keyframe=0))
  rep = plan.after_construction()
  assert not rep.violations, rep.violations
  assert sorted(rep.pinned.values()) == list(lease)
  for tid, cpu in rep.pinned.items():
    assert allowed_cpus(tid) == (cpu,)
  assert set(os.sched_getaffinity(0)) == set(lease)  # the calling (mutation) thread
  # The pinned threads really run the physics: stepping works and no new threads appear.
  before = thread_ids()
  s = np.arange(16, dtype=np.int32)
  core.reset_exact(s, s.astype(np.uint64), np.zeros(16, np.uint64), (s.astype(np.uint64), np.zeros(16, np.uint64)))
  core.advance_control(s, np.zeros((16, 8), np.float32))
  assert thread_ids() == before


def test_affinity_rejects_leases_outside_the_pod():
  from raysimbatch.worker.affinity import AffinityError, AffinityPlan

  with pytest.raises(AffinityError):
    AffinityPlan("numa_exclusive", (10_000, 10_001), 2).before_construction()


def test_threads_that_exit_during_verification_are_skipped(monkeypatch):
  """Found in kind: a transient runtime thread exited between listing /proc/self/task
  and reading its status, and worker initialization failed with FileNotFoundError."""
  import threading

  from raysimbatch.worker import affinity as A

  cpus = sorted(os.sched_getaffinity(0))
  plan = A.AffinityPlan("numa_exclusive", tuple(cpus[:1]), 1)
  plan.before_construction()
  stop = threading.Event()
  worker = threading.Thread(target=stop.wait)  # a runtime thread that stays alive
  worker.start()
  real = A.thread_ids
  monkeypatch.setattr(A, "thread_ids", lambda: real() | {2**22 + 7})  # listed, then gone
  try:
    report = plan.after_construction()
  finally:
    stop.set()
    worker.join()
    os.sched_setaffinity(0, set(cpus))
  assert 2**22 + 7 not in report.threads
  assert A.allowed_cpus(2**22 + 7) is None
