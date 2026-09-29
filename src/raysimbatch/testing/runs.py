# SPDX-License-Identifier: Apache-2.0
"""Build complete, artifact-backed RunConfigs from the reference fixtures.

Every model, contract, task kernel and policy is registered content-addressed in the
run's artifact store, exactly as a production submission would reference them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from raysimbatch.config.models import ArtifactRef, ModelGroup, PolicySpec, RunConfig, TaskScoring, TaskSpec
from raysimbatch.policy.bundle import PolicyBundle
from raysimbatch.storage.blob import open_blob_store
from raysimbatch.testing.fixtures import Fixture


def register(root: str, payload: bytes, media: str) -> ArtifactRef:
  a = open_blob_store(root).put(payload, media)
  return ArtifactRef(uri=a.uri, sha256=a.sha256)


def group_from_fixture(root: str, fx: Fixture, **overrides: Any) -> ModelGroup:
  kernel = {**fx.task_doc, "reset_distribution": fx.reset_distribution}
  return fx.group.model_copy(update={
    "mjcf": register(root, fx.xml, "application/xml"),
    "adapter": register(root, fx.contract.model_dump_json().encode(), "application/json"),
    "task_kernel": register(root, json.dumps(kernel, sort_keys=True).encode(), "application/json"),
    "randomization": fx.randomization,
    "initial_keyframe": fx.keyframe,
    **overrides,
  })


def policy_spec(root: str, bundle: PolicyBundle, exe: bytes, name: str | None = None) -> PolicySpec:
  return PolicySpec(
    name=name or bundle.name,
    bundle=register(root, bundle.to_json(), "application/json"),
    executable=register(root, exe, "application/octet-stream"),
    version=bundle.version,
    deterministic=bundle.deterministic_inference,
    normalization=bundle.normalization,
    distribution=bundle.distribution,
    certified_batch_sizes=bundle.certified_batch_sizes,
  )


def task_spec(root: str, name: str, groups: Sequence[ModelGroup], success_threshold: float = 0.8) -> TaskSpec:
  g = groups[0]
  return TaskSpec(name=name, kernel=g.task_kernel, model_groups=tuple(x.name for x in groups),
                  scoring=TaskScoring(success_threshold=success_threshold))


def run_config(root: str, dsn_ref: str, groups: Sequence[ModelGroup], policies: Sequence[PolicySpec],
               tasks: Sequence[TaskSpec] = (), **kw: Any) -> RunConfig:
  worker = kw.pop("worker", {})
  base = {
    "artifact_root": root,
    "database_secret": dsn_ref,
    "models": [g.model_dump(mode="json") for g in groups],
    "policies": [p.model_dump(mode="json") for p in policies],
    "tasks": [t.model_dump(mode="json") for t in tasks],
    "worker": {"envs": 16, "threads": 2, "num_cpus": 2, "slice_size": 16, "slice_min": 16, "slice_max": 128,
               "memory_gib": 1, "affinity": "none", **worker},
    "cluster": {"coordinator_fanout": 8, "gang_timeout_s": 60.0, **kw.pop("cluster", {})},
  }
  for k, v in kw.items():
    base[k] = v.model_dump(mode="json") if hasattr(v, "model_dump") else v
  return RunConfig.from_obj(base)
