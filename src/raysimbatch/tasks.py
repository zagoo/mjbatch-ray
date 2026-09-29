# SPDX-License-Identifier: Apache-2.0
"""Task kernel bindings: a signed kernel entrypoint plus ordered numeric parameters.

A ModelGroup's ``task_kernel`` artifact is a JSON document::

    {"name": "velocity_tracking", "version": 1, "spec": {...named parameters...}}

``bind_task`` resolves names against ModelMetadata into the native kernel's ordered
``params``/``iparams``. Free-form executable strings are not accepted: the entrypoint
must be compiled into ``raysimbatch._native`` (design B.2).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from raysimbatch import _native
from raysimbatch.worker.model import ModelError, ModelMetadata


@dataclass(frozen=True)
class TaskBinding:
  name: str
  version: int
  params: np.ndarray
  iparams: np.ndarray
  num_commands: int
  num_state: int
  num_components: int
  component_names: tuple[str, ...]
  needs_site_xpos: bool
  task_hash: str
  spec: Mapping[str, Any]


def _f(spec: Mapping[str, Any], key: str) -> float:
  if key not in spec:
    raise ModelError(f"task spec missing {key!r}")
  v = float(spec[key])
  if not np.isfinite(v):
    raise ModelError(f"task spec {key!r} must be finite")
  return v


def _vec3(spec: Mapping[str, Any], key: str) -> list[float]:
  v = [float(x) for x in spec.get(key, ())]
  if len(v) != 3:
    raise ModelError(f"task spec {key!r} needs three values")
  return v


def bind_task(doc: Mapping[str, Any], meta: ModelMetadata) -> TaskBinding:
  name, version, spec = doc.get("name"), int(doc.get("version", 0)), doc.get("spec", {})
  if not isinstance(name, str):
    raise ModelError("task kernel document needs a name")
  ncmd, nstate, ncomp = _native.task_spec(name, version)
  site = False
  if name == "balance":
    angle = meta.joint(spec["angle_joint"]).qposadr
    slider = meta.joint(spec["slider_joint"]).qposadr if spec.get("slider_joint") else -1
    params = [_f(spec, "angle_limit"), float(spec.get("slider_limit", np.inf))]
    iparams = [angle, slider]
    comps: tuple[str, ...] = ("alive",)
  elif name == "velocity_tracking":
    base = meta.joint(spec["base_joint"])
    if base.type != 0:
      raise ModelError("velocity_tracking needs a free base joint")
    lo, hi = _vec3(spec, "command_low"), _vec3(spec, "command_high")
    w = spec.get("weights", {})
    params = [
      _f(spec, "sigma_lin"), _f(spec, "sigma_ang"), _f(spec, "min_height"), _f(spec, "min_up_z"),
      lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
      float(w.get("lin", 1.0)), float(w.get("ang", 0.5)), float(w.get("alive", 0.0)),
    ]
    iparams = [base.qposadr, base.dofadr, int(spec.get("resample_steps", 0))]
    comps = ("lin_vel", "ang_vel", "alive")
  elif name == "reach":
    sid = meta.sites.get(spec["site"])
    if sid is None:
      raise ModelError(f"no site {spec['site']!r}")
    lo, hi = _vec3(spec, "target_low"), _vec3(spec, "target_high")
    params = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], _f(spec, "success_tol")]
    iparams = [sid]
    comps = ("distance",)
    site = True
  else:
    raise ModelError(f"unknown task kernel {name!r}")
  canon = json.dumps({"name": name, "version": version, "spec": spec}, sort_keys=True)
  return TaskBinding(
    name=name,
    version=version,
    params=np.asarray(params, np.float64),
    iparams=np.asarray(iparams, np.int32),
    num_commands=ncmd,
    num_state=nstate,
    num_components=ncomp,
    component_names=comps,
    needs_site_xpos=site,
    task_hash=hashlib.sha256(canon.encode()).hexdigest(),
    spec=spec,
  )
