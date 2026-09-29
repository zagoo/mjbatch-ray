# SPDX-License-Identifier: Apache-2.0
"""Start SimulationWorker actors for explicit stepping through a HeteroBatchRouter.

Used by clients that drive environments themselves (``router.send``/``recv``) rather
than through a managed Mode A/B run, and by tests and examples.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray

from raysimbatch.config.models import FailureConfig, WorkerConfig
from raysimbatch.protocol import Route
from raysimbatch.routing.router import HeteroBatchRouter
from raysimbatch.worker.actor import RemoteSimulationWorker, WorkerSpec, worker_options


@dataclass
class FleetMember:
  spec: WorkerSpec
  mjb: np.ndarray
  count: int = 1
  policy: tuple[str, np.ndarray] | None = None  # (bundle json, executable) for local runs


def start_fleet(run_id: str, members: Sequence[FleetMember], *, namespace: str | None = None,
                memory_bytes: int = 256 * 2**20, generation: int = 1) -> tuple[HeteroBatchRouter, dict[str, Any]]:
  actors: dict[str, dict[str, Any]] = {}
  routes: dict[int, Route] = {}
  handles = []
  env = 0
  for m in members:
    for k in range(m.count):
      name = f"{m.spec.actor_name or 'fleet'}-{k}-{uuid.uuid4().hex[:6]}"
      spec = WorkerSpec.from_dict({**m.spec.to_dict(), "actor_name": name, "run_id": run_id})
      wc = WorkerConfig.from_obj(spec.worker)
      opts = worker_options(wc, FailureConfig.from_obj(spec.failure), name=name, namespace=namespace,
                            memory_bytes=memory_bytes)
      h = RemoteSimulationWorker.options(**opts).remote(spec.to_dict(), f"fleet://{run_id}/{name}")
      lease = {"owner": "fleet", "epoch": generation, "expires_unix_ns": time.time_ns() + int(3600e9)}
      handles.append((name, h, m, spec, lease))
  readies = ray.get([h.initialize.remote(ray.put(m.mjb), ray.put(m.policy[1]) if m.policy else None, lease,
                                         m.policy[0] if m.policy else None)
                     for name, h, m, spec, lease in handles])
  for (name, h, _m, spec, _lease), ready in zip(handles, readies, strict=True):
    actors[name] = {"handle": h, "generation": ready["generation"], "group_hash": spec.group_hash,
                    "schema_hash": ready["schema_hash"], "obs_dim": ready["obs_dim"],
                    "action_dim": ready["action_dim"]}
    n = spec.num_slots or WorkerConfig.from_obj(spec.worker).envs  # this member's envs, not the last one's
    for s in range(n):
      routes[env] = Route(name, 0, s, ready["generation"], 0)
      env += 1
  return HeteroBatchRouter(run_id, actors, routes), {"readies": readies, "actors": list(actors)}


def stop_fleet(router: HeteroBatchRouter) -> None:
  for a in router.actors.values():
    try:
      ray.kill(a.handle, no_restart=True)
    except Exception:  # noqa: BLE001
      pass
