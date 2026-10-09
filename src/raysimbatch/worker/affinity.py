# SPDX-License-Identifier: Apache-2.0
"""CPU affinity and NUMA placement without touching MJBatch (amendment AM-5).

Linux: the actor's calling thread is restricted to its leased CPU set *before* the
MJBatch Batch is constructed, so the pool threads MJBatch creates with
``std::thread`` (pthread_create) inherit that mask. After construction the new
thread IDs are discovered from ``/proc/self/task`` and each physics thread is pinned
to one core of the lease (since MJBatch 0.1.4 the calling thread is pool worker 0 and
takes the first core; the pool's other workers are the new threads), with memory policy set to the lease's NUMA node so
first-touch allocations stay local. The result is verified by reading every
thread's ``Cpus_allowed_list`` back. CPU IDs come from the node affinity agent's
lease, never from Ray logical resource IDs.

macOS has no CPU-affinity API (thread affinity tags are scheduler hints and are
ignored on Apple silicon); ``affinity=none`` is the only mode there, and production
validation rejects it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from dataclasses import dataclass, field

LINUX = sys.platform.startswith("linux")


class AffinityError(RuntimeError):
  pass


@dataclass
class AffinityReport:
  mode: str
  supported: bool
  requested: tuple[int, ...] = ()
  numa_node: int | None = None
  threads: dict[int, tuple[int, ...]] = field(default_factory=dict)  # tid -> allowed CPUs
  pinned: dict[int, int] = field(default_factory=dict)  # physics tid -> cpu
  violations: list[str] = field(default_factory=list)

  def as_dict(self) -> dict:
    return {
      "mode": self.mode,
      "supported": self.supported,
      "requested": list(self.requested),
      "numa_node": self.numa_node,
      "threads": {str(k): list(v) for k, v in self.threads.items()},
      "pinned": {str(k): v for k, v in self.pinned.items()},
      "violations": list(self.violations),
    }


def thread_ids() -> set[int]:
  if not LINUX:
    return set()
  return {int(t) for t in os.listdir("/proc/self/task")}


def allowed_cpus(tid: int) -> tuple[int, ...] | None:
  """The thread's CPU mask, or None when it exited after being listed (observed in kind:
  a transient runtime thread vanished between listing /proc/self/task and reading it)."""
  try:
    with open(f"/proc/self/task/{tid}/status") as f:
      for line in f:
        if line.startswith("Cpus_allowed_list:"):
          return tuple(_parse_list(line.split(":", 1)[1].strip()))
  except (FileNotFoundError, ProcessLookupError):
    return None
  return ()


def _parse_list(text: str) -> list[int]:
  out: list[int] = []
  for part in text.split(","):
    if not part:
      continue
    if "-" in part:
      a, b = part.split("-")
      out.extend(range(int(a), int(b) + 1))
    else:
      out.append(int(part))
  return out


def numa_node_of(cpu: int) -> int | None:
  base = f"/sys/devices/system/cpu/cpu{cpu}"
  try:
    for name in os.listdir(base):
      if name.startswith("node") and name[4:].isdigit():
        return int(name[4:])
  except OSError:
    return None
  return None


def _set_preferred_node(node: int) -> bool:
  """set_mempolicy(MPOL_PREFERRED, {node}) for the calling thread (first touch)."""
  libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
  MPOL_PREFERRED = 1
  mask = ctypes.c_ulong(1 << node)
  SYS_set_mempolicy = {"x86_64": 238, "aarch64": 237}.get(os.uname().machine)
  if SYS_set_mempolicy is None:
    return False
  rc = libc.syscall(SYS_set_mempolicy, MPOL_PREFERRED, ctypes.byref(mask), ctypes.c_ulong(64))
  return rc == 0


class AffinityPlan:
  """Two-phase application around Batch construction."""

  def __init__(self, mode: str, cpus: tuple[int, ...], threads: int):
    self.mode = mode
    self.cpus = tuple(cpus)
    self.threads = threads
    self._before: set[int] = set()
    self.report = AffinityReport(mode=mode, supported=LINUX, requested=self.cpus)

  def before_construction(self) -> None:
    if self.mode == "none":
      return
    if not LINUX:
      raise AffinityError("affinity=numa_exclusive requires Linux; this host cannot enforce CPU sets")
    if len(self.cpus) < self.threads:
      raise AffinityError(f"lease has {len(self.cpus)} CPUs for {self.threads} physics threads")
    online = os.sched_getaffinity(0)
    missing = set(self.cpus) - online
    if missing:
      raise AffinityError(f"leased CPUs {sorted(missing)} are not available to this pod")
    nodes = {numa_node_of(c) for c in self.cpus}
    if len(nodes) > 1:
      raise AffinityError(f"lease spans NUMA nodes {sorted(n for n in nodes if n is not None)}")
    node = nodes.pop()
    self.report.numa_node = node
    os.sched_setaffinity(0, set(self.cpus))  # inherited by MJBatch's pool threads
    if node is not None:
      _set_preferred_node(node)
    self._before = thread_ids()

  def after_construction(self) -> AffinityReport:
    if self.mode == "none" or not LINUX:
      return self.report
    new = sorted(thread_ids() - self._before)
    # MJBatch's pool has min(threads, num_sims) workers: worker 0 is the calling thread,
    # the others are new threads. More new threads may be runtime threads that started
    # concurrently; the pool's are the ones restricted to our mask.
    pool = [t for t in new if set(allowed_cpus(t) or ()) == set(self.cpus)][: self.threads - 1]
    os.sched_setaffinity(0, {self.cpus[0]})
    self.report.pinned[threading.get_native_id()] = self.cpus[0]
    for tid, cpu in zip(pool, self.cpus[1:], strict=False):
      try:
        os.sched_setaffinity(tid, {cpu})
      except ProcessLookupError:
        continue  # not a pool thread after all: it exited
      self.report.pinned[tid] = cpu
    for tid in thread_ids():
      mask = allowed_cpus(tid)
      if mask is not None:
        self.report.threads[tid] = mask
    if len(self.report.pinned) != self.threads:
      self.report.violations.append(f"found {len(self.report.pinned)} physics threads, expected {self.threads}")
    for tid, cpu in self.report.pinned.items():
      if self.report.threads.get(tid) != (cpu,):
        self.report.violations.append(f"thread {tid} not pinned to {cpu}")
    return self.report
