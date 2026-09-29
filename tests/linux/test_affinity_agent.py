# SPDX-License-Identifier: Apache-2.0
"""Affinity agent leasing semantics (disjoint, NUMA-local, persisted, reaped on death)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from raysimbatch.worker.affinity_agent import AffinityAgent


def test_disjoint_leases_service_reservation_and_persistence(tmp_path):
  path = tmp_path / "leases.json"
  a = AffinityAgent(service_cores=2, state_path=str(path), cpus=list(range(12)))
  x = a.lease("w0", 4, os.getpid())
  y = a.lease("w1", 4, os.getpid())
  assert set(x).isdisjoint(y) and not set(x) & {0, 1} and not set(y) & {0, 1}
  assert a.lease("w0", 4, os.getpid()) == x  # idempotent for a restarted holder
  with pytest.raises(RuntimeError, match="oversubscribe"):
    a.lease("w2", 4, os.getpid())  # only 2 cores left: never oversubscribed
  b = AffinityAgent(service_cores=2, state_path=str(path), cpus=list(range(12)))  # agent restart
  assert b.status()["leases"]["w1"]["cpus"] == y


def test_dead_holders_are_reaped(tmp_path):
  p = subprocess.Popen([sys.executable, "-c", "pass"])
  p.wait()
  a = AffinityAgent(service_cores=0, state_path=str(tmp_path / "l.json"), cpus=list(range(4)))
  a.lease("dead", 4, p.pid)
  assert a.lease("alive", 4, os.getpid())  # the dead holder's cores were reclaimed
  assert "dead" not in a.status()["leases"]


def test_status_reports_only_live_leases(tmp_path):
  """Observed in kind: status() listed a finished run's worker until the next lease."""
  p = subprocess.Popen([sys.executable, "-c", "pass"])
  p.wait()
  a = AffinityAgent(service_cores=0, cpus=list(range(4)))
  a.lease("gone", 2, p.pid)
  a.lease("here", 2, os.getpid())
  assert set(a.status()["leases"]) == {"here"}


def test_node_without_its_exclusive_allocation_is_refused_not_oversubscribed():
  """Design B.3 [VERIFY]: without the static CPU Manager the pod sees the node's shared
  pool (here 64 CPUs for a 4-CPU pod); leasing from it would oversubscribe the node."""
  with pytest.raises(RuntimeError, match="requested 4 exclusive CPUs"):
    AffinityAgent(cpus=list(range(64)), expected_cpus="4")
  with pytest.raises(RuntimeError, match="not a whole number"):
    AffinityAgent(cpus=list(range(4)), expected_cpus="3500m")
  a = AffinityAgent(cpus=[2, 3, 4, 5], expected_cpus="4")  # kind: exclusive cpuset 2-5
  assert a.lease("w0", 4, os.getpid()) == [2, 3, 4, 5]


def test_the_pod_request_comes_from_the_environment_only_for_the_real_cpu_set(monkeypatch):
  monkeypatch.setenv("RSB_POD_CPUS", "4")
  assert AffinityAgent(cpus=list(range(64))).all_cpus  # explicit sets (tests, tools) are not pod sets
  if not hasattr(os, "sched_getaffinity"):
    pytest.skip("Linux CPU affinity")
  real = len(os.sched_getaffinity(0))
  monkeypatch.setenv("RSB_POD_CPUS", str(real + 1))
  with pytest.raises(RuntimeError, match="refusing this node"):
    AffinityAgent()
  monkeypatch.setenv("RSB_POD_CPUS", str(real))
  assert len(AffinityAgent().all_cpus) == real
