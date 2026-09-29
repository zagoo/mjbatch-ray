# SPDX-License-Identifier: Apache-2.0
"""Model-group identity (design §6.3).

A group key is SHA-256 over the compiled model hash, task-kernel hash, observation and
action schema hashes, timestep, decimation, controller hash, solver configuration and
randomization schema. Timestep is grouped explicitly even though upstream options can
vary per slot: homogeneous schedules keep timing contracts auditable.
"""

from __future__ import annotations

import hashlib
import json

from raysimbatch.adapter import CompiledAdapter
from raysimbatch.config.models import ModelGroup
from raysimbatch.tasks import TaskBinding


def group_key(model_hash: str, task: TaskBinding, adapter: CompiledAdapter, group: ModelGroup) -> str:
  doc = {
    "model": model_hash,
    "task": task.task_hash,
    "observation_schema": adapter.adapter_hash,
    "action_schema": hashlib.sha256(
      json.dumps({"actions": adapter.num_actions, "cfg": adapter.action_cfg}, sort_keys=True).encode()
    ).hexdigest(),
    "timestep_ns": group.timestep_ns,
    "decimation": group.decimation,
    "controller": hashlib.sha256(json.dumps(adapter.controller_cfg, sort_keys=True).encode()).hexdigest(),
    "solver": [group.solver, group.integrator, group.cone, group.iterations, group.tolerance],
    "randomization": [r.model_dump(mode="json") for r in group.randomization],
  }
  return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
