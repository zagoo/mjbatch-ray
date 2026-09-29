# SPDX-License-Identifier: Apache-2.0
"""Admission: resolve content-addressed artifacts, compile each model once, and build
per-group runtimes (worker spec template + MJB/policy bytes) for coordinators.

Everything a worker loads is verified by digest here and again where it is read. An
artifact that is missing, corrupt or inconsistent with its contract fails admission
with a precise error listing every problem; nothing is guessed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from raysimbatch.adapter import CompiledAdapter, Normalizer, compile_adapter
from raysimbatch.config.models import (
  AdapterContract,
  ArtifactRef,
  ModelGroup,
  PolicySpec,
  RunConfig,
  TaskSpec,
)
from raysimbatch.policy.bundle import PolicyBundle, verify_executable
from raysimbatch.routing.groups import group_key
from raysimbatch.storage.store import read_ref
from raysimbatch.tasks import TaskBinding, bind_task
from raysimbatch.worker.model import ModelMetadata, compile_model, model_hash, to_mjb


class AdmissionError(ValueError):
  def __init__(self, problems: list[str]):
    self.problems = problems
    super().__init__("admission failed:\n  " + "\n  ".join(problems))


@dataclass
class GroupRuntime:
  name: str
  group: ModelGroup
  mjb: bytes
  model_hash: str
  meta: ModelMetadata
  binding: TaskBinding
  compiled: CompiledAdapter
  contract: AdapterContract
  group_hash: str
  reset_distribution: dict[str, Any] | None
  envelope: dict[str, Any] | None
  normalizer: dict[str, list[float]] | None
  policies: dict[str, tuple[PolicyBundle, bytes]] = field(default_factory=dict)

  @property
  def mjb_sha(self) -> str:
    return hashlib.sha256(self.mjb).hexdigest()


def _json(ref: ArtifactRef, what: str, problems: list[str]) -> Any:
  if ref.empty:
    problems.append(f"{what}: artifact not populated")
    return None
  try:
    return json.loads(read_ref(ref))
  except Exception as exc:  # noqa: BLE001
    problems.append(f"{what}: {exc}")
    return None


def _bytes(ref: ArtifactRef, what: str, problems: list[str]) -> bytes | None:
  if ref.empty:
    problems.append(f"{what}: artifact not populated")
    return None
  try:
    return read_ref(ref)
  except Exception as exc:  # noqa: BLE001
    problems.append(f"{what}: {exc}")
    return None


def load_policy(spec: PolicySpec, problems: list[str]) -> tuple[PolicyBundle, bytes] | None:
  doc = _bytes(spec.bundle, f"policies[{spec.name}].bundle", problems)
  exe = _bytes(spec.executable, f"policies[{spec.name}].executable", problems)
  if doc is None or exe is None:
    return None
  try:
    b = PolicyBundle.from_json(doc)
    verify_executable(b, exe)
  except Exception as exc:  # noqa: BLE001
    problems.append(f"policies[{spec.name}]: {exc}")
    return None
  if spec.normalization != b.normalization:
    problems.append(f"policies[{spec.name}]: normalization {spec.normalization} != bundle {b.normalization}")
  return b, exe


def admit(cfg: RunConfig, tasks: tuple[TaskSpec, ...] = ()) -> tuple[dict[str, GroupRuntime], dict[str, list[str]]]:
  """Returns group runtimes and the task->groups binding. Raises AdmissionError."""
  problems: list[str] = []
  policies = {}
  for p in cfg.policies:
    lp = load_policy(p, problems)
    if lp is not None:
      policies[p.name] = lp
  groups: dict[str, GroupRuntime] = {}
  for g in cfg.models:
    xml = None if g.mjcf.empty else _bytes(g.mjcf, f"models[{g.name}].mjcf", problems)
    assets = None if g.assets.empty else _bytes(g.assets, f"models[{g.name}].assets", problems)
    kernel = _json(g.task_kernel, f"models[{g.name}].task_kernel", problems)
    contract_doc = _json(g.adapter, f"models[{g.name}].adapter", problems)
    if xml is None or kernel is None or contract_doc is None:
      continue
    try:
      model = compile_model(xml, assets, g)
    except Exception as exc:  # noqa: BLE001
      problems.append(f"models[{g.name}]: {exc}")
      continue
    if model.opt.enableflags & 1 << 5 and cfg.worker.native_backend != "scalar":  # mjENBL_SLEEP
      problems.append(f"models[{g.name}]: sleep requires the scalar lane")
    meta = ModelMetadata.from_model(model)
    try:
      binding = bind_task(kernel, meta)
      contract = AdapterContract.from_obj(contract_doc)
      norm = None
      if contract.normalization == "external":
        nd = _json(contract.normalizer, f"models[{g.name}].adapter.normalizer", problems)
        norm = None if nd is None else {"mean": list(map(float, nd["mean"])), "var": list(map(float, nd["var"]))}
      env = None
      if contract.actuator.kind == "dc_motor":
        env = _json(contract.actuator.envelope, f"models[{g.name}].adapter.actuator.envelope", problems)
      compiled = compile_adapter(contract, meta, num_commands=binding.num_commands,
                                 normalizer=None if norm is None else Normalizer(np.asarray(norm["mean"]),
                                                                                  np.asarray(norm["var"])),
                                 envelope=env)
    except Exception as exc:  # noqa: BLE001
      problems.append(f"models[{g.name}]: {exc}")
      continue
    mh = model_hash(xml, assets, g)
    rt = GroupRuntime(
      name=g.name, group=g, mjb=to_mjb(model), model_hash=mh, meta=meta, binding=binding, compiled=compiled,
      contract=contract, group_hash=group_key(mh, binding, compiled, g),
      reset_distribution=kernel.get("reset_distribution"), envelope=env, normalizer=norm,
    )
    for name, (b, exe) in policies.items():
      if b.obs_dim == compiled.obs_dim and b.action_dim == compiled.num_actions:
        rt.policies[name] = (b, exe)
    if not rt.policies:
      problems.append(f"models[{g.name}]: no policy matches obs_dim={compiled.obs_dim} action_dim={compiled.num_actions}")
    groups[g.name] = rt
  task_groups: dict[str, list[str]] = {}
  for t in tasks or cfg.tasks:
    if not t.name:
      problems.append("tasks: every task needs a name")
      continue
    gs = list(t.model_groups) or [g for g in groups]
    for gname in gs:
      if gname not in groups:
        problems.append(f"tasks[{t.name}]: unknown model group {gname}")
        continue
      if not t.kernel.empty and t.kernel.sha256 != cfg_group(cfg, gname).task_kernel.sha256:
        problems.append(f"tasks[{t.name}]: kernel artifact differs from model group {gname}'s task kernel")
    task_groups[t.name] = [g for g in gs if g in groups]
  if problems:
    raise AdmissionError(problems)
  return groups, task_groups


def cfg_group(cfg: RunConfig, name: str) -> ModelGroup:
  return next(g for g in cfg.models if g.name == name)
