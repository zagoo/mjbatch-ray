# SPDX-License-Identifier: Apache-2.0
"""Model compilation, content identity and name resolution (ModelMetadata).

Compiled models are process-local: workers receive immutable XML/asset/MJB bytes
(ObjectRefs of uint8 arrays) and reconstruct the native model themselves. No
``mjModel`` pointer ever crosses a process boundary (design §1, §7.3).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from raysimbatch.config.models import ModelGroup

_SOLVER = {"PGS": mujoco.mjtSolver.mjSOL_PGS, "CG": mujoco.mjtSolver.mjSOL_CG, "Newton": mujoco.mjtSolver.mjSOL_NEWTON}
_INTEGRATOR = {
  "Euler": mujoco.mjtIntegrator.mjINT_EULER,
  "RK4": mujoco.mjtIntegrator.mjINT_RK4,
  "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_CONE = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}
_QW = {0: 7, 1: 4, 2: 1, 3: 1}  # qpos width by mjtJoint: free, ball, slide, hinge
_DW = {0: 6, 1: 3, 2: 1, 3: 1}


class ModelError(ValueError):
  pass


@dataclass(frozen=True)
class JointInfo:
  name: str
  id: int
  type: int
  qposadr: int
  dofadr: int
  nq: int
  nv: int
  limited: bool
  range: tuple[float, float]
  qpos0: tuple[float, ...]


@dataclass(frozen=True)
class ActuatorInfo:
  name: str
  id: int
  trntype: int
  trnid: int
  gear: float
  gaintype: int
  biastype: int
  gainprm0: float
  biasprm: tuple[float, float, float]
  ctrlrange: tuple[float, float]
  ctrllimited: bool
  forcerange: tuple[float, float]


@dataclass(frozen=True)
class SensorInfo:
  name: str
  id: int
  type: int
  adr: int
  dim: int


@dataclass(frozen=True)
class ModelMetadata:
  nq: int
  nv: int
  nu: int
  na: int
  nbody: int
  nsite: int
  nsensordata: int
  nkey: int
  timestep: float
  joints: dict[str, JointInfo] = field(default_factory=dict)
  actuators: dict[str, ActuatorInfo] = field(default_factory=dict)
  sensors: dict[str, SensorInfo] = field(default_factory=dict)
  bodies: dict[str, int] = field(default_factory=dict)
  sites: dict[str, int] = field(default_factory=dict)
  quat_adrs: tuple[int, ...] = ()
  free_joints: tuple[str, ...] = ()

  @classmethod
  def from_model(cls, m: mujoco.MjModel) -> ModelMetadata:
    joints: dict[str, JointInfo] = {}
    quat: list[int] = []
    free: list[str] = []
    for j in range(m.njnt):
      jt = int(m.jnt_type[j])
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
      qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
      joints[name] = JointInfo(
        name, j, jt, qa, da, _QW[jt], _DW[jt], bool(m.jnt_limited[j]),
        (float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])),
        tuple(float(x) for x in m.qpos0[qa : qa + _QW[jt]]),
      )
      if jt == 0:
        quat.append(qa + 3)
        free.append(name)
      elif jt == 1:
        quat.append(qa)
    actuators = {}
    for a in range(m.nu):
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}"
      actuators[name] = ActuatorInfo(
        name, a, int(m.actuator_trntype[a]), int(m.actuator_trnid[a, 0]), float(m.actuator_gear[a, 0]),
        int(m.actuator_gaintype[a]), int(m.actuator_biastype[a]), float(m.actuator_gainprm[a, 0]),
        (float(m.actuator_biasprm[a, 0]), float(m.actuator_biasprm[a, 1]), float(m.actuator_biasprm[a, 2])),
        (float(m.actuator_ctrlrange[a, 0]), float(m.actuator_ctrlrange[a, 1])), bool(m.actuator_ctrllimited[a]),
        (float(m.actuator_forcerange[a, 0]), float(m.actuator_forcerange[a, 1])),
      )
    sensors = {}
    for s in range(m.nsensor):
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor{s}"
      sensors[name] = SensorInfo(name, s, int(m.sensor_type[s]), int(m.sensor_adr[s]), int(m.sensor_dim[s]))
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}": b for b in range(m.nbody)}
    sites = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or f"site{s}": s for s in range(m.nsite)}
    return cls(
      int(m.nq), int(m.nv), int(m.nu), int(m.na), int(m.nbody), int(m.nsite), int(m.nsensordata), int(m.nkey),
      float(m.opt.timestep), joints, actuators, sensors, bodies, sites, tuple(quat), tuple(free),
    )

  def joint(self, name: str) -> JointInfo:
    try:
      return self.joints[name]
    except KeyError:
      raise ModelError(f"model has no joint named {name!r}") from None

  def actuator_for_joint(self, joint: str) -> ActuatorInfo:
    j = self.joint(joint)
    hits = [a for a in self.actuators.values() if a.trntype == int(mujoco.mjtTrn.mjTRN_JOINT) and a.trnid == j.id]
    if len(hits) != 1:
      raise ModelError(f"joint {joint!r} must be driven by exactly one joint actuator, found {len(hits)}")
    return hits[0]


def apply_group_options(model: mujoco.MjModel, group: ModelGroup) -> None:
  """The ModelGroup's solver configuration is authoritative (it is part of the group key)."""
  model.opt.timestep = group.timestep_ns * 1e-9
  model.opt.solver = _SOLVER[group.solver]
  model.opt.integrator = _INTEGRATOR[group.integrator]
  model.opt.cone = _CONE[group.cone]
  model.opt.iterations = group.iterations
  model.opt.tolerance = group.tolerance
  if group.sleep:
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)


def group_options(group: ModelGroup) -> dict[str, Any]:
  return {
    "timestep_ns": group.timestep_ns,
    "solver": group.solver,
    "integrator": group.integrator,
    "cone": group.cone,
    "iterations": group.iterations,
    "tolerance": group.tolerance,
    "sleep": group.sleep,
  }


def unpack_assets(archive: bytes | None) -> dict[str, bytes]:
  """Asset archives are tar (optionally gzip) files of mesh/texture/hfield sources."""
  if not archive:
    return {}
  out: dict[str, bytes] = {}
  with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tf:
    for member in tf.getmembers():
      if member.isfile():
        f = tf.extractfile(member)
        if f is None:
          continue
        name = os.path.normpath(member.name)
        if name.startswith("..") or os.path.isabs(name):
          raise ModelError(f"asset path escapes the archive: {member.name}")
        out[name] = f.read()
  return out


def compile_model(xml: bytes, assets: bytes | None, group: ModelGroup) -> mujoco.MjModel:
  try:
    model = mujoco.MjModel.from_xml_string(xml.decode(), unpack_assets(assets))
  except ValueError as exc:
    raise ModelError(f"XML_ABI_MISMATCH: {exc}") from exc
  apply_group_options(model, group)
  return model


def to_mjb(model: mujoco.MjModel) -> bytes:
  buf = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
  mujoco.mj_saveModel(model, None, buf)
  return buf.tobytes()


def from_mjb(blob: bytes | np.ndarray) -> mujoco.MjModel:
  data = blob.tobytes() if isinstance(blob, np.ndarray) else blob
  fd, path = tempfile.mkstemp(suffix=".mjb")
  try:
    with os.fdopen(fd, "wb") as f:
      f.write(data)
    return mujoco.MjModel.from_binary_path(path)
  finally:
    os.unlink(path)


def model_hash(xml: bytes, assets: bytes | None, group: ModelGroup) -> str:
  """Compiled-model content identity: sources, applied options and engine version."""
  h = hashlib.sha256()
  h.update(hashlib.sha256(xml).digest())
  h.update(hashlib.sha256(assets or b"").digest())
  h.update(json.dumps(group_options(group), sort_keys=True).encode())
  h.update(f"mujoco={mujoco.__version__}".encode())
  return h.hexdigest()
