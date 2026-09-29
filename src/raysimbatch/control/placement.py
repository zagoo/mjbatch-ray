# SPDX-License-Identifier: Apache-2.0
"""Capacity plans and gang reservation (design §5.3, §9.4, B.3).

A capacity plan assigns (actor_name, model_group, slot_count, CPU-set lease,
placement bundle index) and groups actors under coordinators (fanout <= 8). An
evaluation wave reserves all of its simulation, coordinator and inference resources
atomically in one placement group; if the gang cannot be placed within
``gang_timeout_s`` the wave is a visible deferred wave, never a silent partial gang.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from raysimbatch.config.models import RunConfig


class GangTimeout(RuntimeError):
  pass


class CapacityError(ValueError):
  """The plan exceeds the configured cluster maximum; no wave could ever be placed."""


@dataclass
class Assignment:
  actor_name: str
  group: str
  slots: int
  cpu_set: tuple[int, ...] = ()
  bundle_index: int = 0
  coordinator: int = 0
  resources: dict[str, float] = field(default_factory=dict)
  memory_bytes: int = 0


@dataclass
class CapacityPlan:
  routing_epoch: int
  assignments: list[Assignment]
  coordinators: int
  bundles: list[dict[str, float]]
  strategy: str

  def by_coordinator(self) -> dict[int, list[Assignment]]:
    out: dict[int, list[Assignment]] = {}
    for a in self.assignments:
      out.setdefault(a.coordinator, []).append(a)
    return out

  def to_dict(self) -> dict[str, Any]:
    return {"routing_epoch": self.routing_epoch, "coordinators": self.coordinators, "strategy": self.strategy,
            "assignments": [a.__dict__ for a in self.assignments], "bundles": self.bundles}


def plan_capacity(cfg: RunConfig, run_id: str, group_weights: dict[str, float], *, actors: int | None = None,
                  slots_per_actor: int | None = None, cluster_resources: Mapping[str, float] | None = None,
                  memory_bytes: int | None = None, routing_epoch: int = 0) -> CapacityPlan:
  w = cfg.worker
  E = slots_per_actor or w.envs
  if actors is None:
    actors = math.ceil(cfg.target_envs / E)
    if cfg.mode != "rollout":
      actors = min(actors, cfg.evaluation.wave_actors)
  groups = [g for g, wt in sorted(group_weights.items()) if wt > 0]
  if not groups:
    raise ValueError("no model group has positive weight")
  cap = cfg.cluster.max_cpu_nodes * (cfg.cluster.simulation_cpus // w.num_cpus)
  need = max(actors, len(groups))  # every positive-weight runtime gets at least one actor
  if need > cap:
    raise CapacityError(f"plan needs {need} simulation actors but the cluster maximum is {cap} "
                        f"({cfg.cluster.max_cpu_nodes} nodes x {cfg.cluster.simulation_cpus // w.num_cpus} actors)")
  total = sum(group_weights[g] for g in groups)
  # Largest-remainder apportionment; every positive-weight group gets >= 1 actor.
  quotas = {g: max(1.0, actors * group_weights[g] / total) for g in groups}
  alloc = {g: int(math.floor(q)) for g, q in quotas.items()}
  while sum(alloc.values()) < actors:
    g = max(groups, key=lambda x: quotas[x] - alloc[x])
    alloc[g] += 1
  res = cluster_resources or {}
  assignments = []
  k = 0
  mem = memory_bytes if memory_bytes is not None else w.memory_gib * 2**30
  for g in groups:
    pool = next((m.resource_pool for m in cfg.models if m.name == g), "sim_general")
    for _ in range(alloc[g]):
      r = {pool: 1.0} if pool in res else {}
      assignments.append(Assignment(f"sim-{run_id[:8]}-{k:04d}", g, E, (), k, k // cfg.cluster.coordinator_fanout,
                                    r, mem))
      k += 1
  coordinators = math.ceil(len(assignments) / cfg.cluster.coordinator_fanout)
  bundles = [{"CPU": float(w.num_cpus), "memory": float(mem), **a.resources} for a in assignments]
  bundles += [{"CPU": 1.0} for _ in range(coordinators)]
  return CapacityPlan(routing_epoch, assignments, coordinators, bundles, cfg.cluster.placement)


def reserve(plan: CapacityPlan, timeout_s: float, name: str, check=None, slice_s: float = 5.0):
  """Atomically reserve the whole gang; raises GangTimeout (deferred wave) otherwise.
  ``check()`` runs between wait slices so the caller can abort (e.g. on a lost lease)."""
  pg = placement_group(plan.bundles, strategy=plan.strategy, name=name)
  t0 = time.monotonic()
  try:
    while not pg.wait(timeout_seconds=max(0.1, min(slice_s, timeout_s - (time.monotonic() - t0)))):
      if check is not None:
        check()
      if time.monotonic() - t0 >= timeout_s:
        raise GangTimeout(f"gang {name} not schedulable within {timeout_s}s: deferred wave (no partial start)")
  except BaseException:
    remove_placement_group(pg)
    raise
  return pg, time.monotonic() - t0


def release(pg) -> None:
  if pg is not None:
    try:
      remove_placement_group(pg)
    except Exception:  # noqa: BLE001
      pass


def cluster_custom_resources() -> dict[str, float]:
  return {k: v for k, v in ray.cluster_resources().items() if k.startswith(("sim_", "policy_gpu"))}
