# SPDX-License-Identifier: Apache-2.0
"""Node affinity agent (design §11.4).

One named actor per node leases disjoint core sets to simulation workers from the
CPUs actually allocated to the pod (the kubelet static CPU Manager's cpuset, read from
the agent's own affinity mask), never from Ray logical resource IDs. The first
``service_cores`` CPUs are reserved for Ray/infrastructure threads. A lease is
NUMA-local; leases are persisted to ``state_path`` so an agent restart does not
double-allocate: a lease is released only when its holder releases it or its process
is confirmed dead. If the agent is unavailable, new actors cannot start (they fail
loudly), and existing leases stay valid.

An agent with no live lease for ``idle_exit_s`` exits so the node can become idle and
scale down (found in kind: any live actor keeps a Ray node busy); leases are persisted,
so the re-created agent resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import ray

from raysimbatch.compat.ray_typing import actor_class
from raysimbatch.worker.affinity import LINUX, numa_node_of


def check_exclusive_allocation(cpus: list[int], requested: str) -> None:
  """Design B.3 `[VERIFY]` admission: the pod's actual CPU set must be the exclusive
  allocation it requested. Otherwise the kubelet static CPU Manager is not in effect: the
  set is the shared pool (typically every CPU of the node), and leasing from it would
  oversubscribe. The node is refused instead; workers there fail to start, loudly."""
  if not requested.isdigit():
    raise RuntimeError(f"pod CPU request {requested!r} is not a whole number of CPUs, so the static CPU Manager "
                       "cannot assign exclusive cores; refusing this node")
  if len(cpus) != int(requested):
    raise RuntimeError(f"pod CPU set has {len(cpus)} CPUs but the pod requested {requested} exclusive CPUs: the "
                       "kubelet static CPU Manager did not assign them; refusing this node rather than oversubscribing")


class AffinityAgent:
  def __init__(self, service_cores: int = 0, state_path: str | None = None, cpus: list[int] | None = None,
               idle_exit_s: float | None = None, expected_cpus: str | None = None):
    if cpus is None:
      if not LINUX:
        raise RuntimeError("the affinity agent enforces Linux CPU sets; this host has no affinity API")
      cpus = sorted(os.sched_getaffinity(0))
      if expected_cpus is None:
        expected_cpus = os.environ.get("RSB_POD_CPUS")  # set by the renderer on worker pods
    if expected_cpus is not None:
      check_exclusive_allocation(cpus, expected_cpus)
    self.all_cpus = sorted(cpus)
    self.service = self.all_cpus[:service_cores]
    self.pool = self.all_cpus[service_cores:]
    self.path = Path(state_path) if state_path else None
    self.leases: dict[str, dict[str, Any]] = {}
    if self.path and self.path.exists():
      self.leases = json.loads(self.path.read_text())
      self._reap()
    self.idle_exit_s = idle_exit_s
    self.idle_since = time.monotonic()
    self._lock = threading.Lock()
    if idle_exit_s is not None:
      threading.Thread(target=self._idle_exit, daemon=True, name="rsb-affinity-idle").start()

  def idle(self, now: float | None = None) -> bool:
    """True once no live lease has existed for ``idle_exit_s``."""
    now = time.monotonic() if now is None else now
    with self._lock:
      self._reap()
      if self.leases:
        self.idle_since = now
        return False
      return self.idle_exit_s is not None and now - self.idle_since >= self.idle_exit_s

  def _idle_exit(self) -> None:
    while True:
      time.sleep(min(30.0, max(0.5, (self.idle_exit_s or 30.0) / 4)))
      if self.idle():
        ray.kill(ray.get_runtime_context().current_actor, no_restart=True)
        return

  def _save(self) -> None:
    if self.path:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      tmp = self.path.with_suffix(".tmp")
      tmp.write_text(json.dumps(self.leases, sort_keys=True))
      os.replace(tmp, self.path)

  def _reap(self) -> None:
    """Release leases whose holder process is confirmed dead."""
    for name, lease in list(self.leases.items()):
      pid = lease.get("pid")
      if pid and not _alive(int(pid)):
        del self.leases[name]
    self._save()

  def lease(self, holder: str, count: int, pid: int | None = None) -> list[int]:
    with self._lock:
      self.idle_since = time.monotonic()
      return self._lease(holder, count, pid)

  def _lease(self, holder: str, count: int, pid: int | None) -> list[int]:
    self._reap()
    if holder in self.leases:  # idempotent for a restarted holder
      self.leases[holder]["pid"] = pid
      self._save()
      return list(self.leases[holder]["cpus"])
    used = {c for v in self.leases.values() for c in v["cpus"]}
    free = [c for c in self.pool if c not in used]
    by_node: dict[int | None, list[int]] = {}
    for c in free:
      by_node.setdefault(numa_node_of(c), []).append(c)
    for node, cpus in sorted(by_node.items(), key=lambda kv: -len(kv[1])):
      if len(cpus) >= count:
        got = cpus[:count]
        self.leases[holder] = {"cpus": got, "numa_node": node, "pid": pid}
        self._save()
        return got
    raise RuntimeError(f"no NUMA-local set of {count} free CPUs (free={free}); refusing to oversubscribe")

  def release(self, holder: str) -> None:
    with self._lock:
      self.leases.pop(holder, None)
      self._save()

  def status(self) -> dict[str, Any]:
    with self._lock:
      self._reap()  # report live leases only
    return {"cpus": self.all_cpus, "service": self.service, "pool": self.pool, "leases": self.leases}


def _alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True


RemoteAffinityAgent = actor_class(AffinityAgent, num_cpus=0, max_restarts=-1, max_task_retries=0)


def agent_for_node(node_id: str, service_cores: int = 0, namespace: str | None = None,
                   state_dir: str = "/var/lib/raysimbatch", idle_exit_s: float = 300.0):
  from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

  name = f"rsb-affinity-{node_id[:16]}"
  try:
    return ray.get_actor(name, namespace=namespace)
  except ValueError:
    opts: dict[str, Any] = {"name": name, "get_if_exists": True, "lifetime": "detached",
                            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id, soft=False)}
    if namespace:
      opts["namespace"] = namespace
    path = f"{state_dir}/affinity-{node_id[:16]}.json" if sys.platform.startswith("linux") else None
    return RemoteAffinityAgent.options(**opts).remote(service_cores, path, idle_exit_s=idle_exit_s)
