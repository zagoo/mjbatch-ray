# SPDX-License-Identifier: Apache-2.0
"""Adapter contracts compiled into native kernels (design §6.7 ordering, steps 1-8).

``compile_adapter`` validates an ``AdapterContract`` against ``ModelMetadata`` and
produces the tables the native ``ActionProcessor``/``Controller``/``Observation``
kernels execute. Nothing is inferred from robot names or XML layout: joint order,
signs, offsets, history layout, normalization location and action transforms must be
attested by the contract; missing facts are admission errors listing every field.

``Sim2SimAdapter`` owns the kernels and every per-slot adapter state array (history,
filters, delay queues, previous actions), so save/restore is a row copy. It binds its
source views (MJBatch bind() arrays) once at construction; per-call arguments are only
slot indices, so no Python marshalling happens at physics rate (AM-12).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from raysimbatch import _native
from raysimbatch.config.models import AdapterContract, ObservationTerm
from raysimbatch.worker.model import ModelMetadata

SOURCE_CODES = {
  "joint_pos": 0,
  "joint_vel": 1,
  "base_lin_vel": 2,
  "base_ang_vel": 3,
  "gravity": 4,
  "command": 5,
  "previous_action": 6,
  "phase": 7,
  "rot6d": 8,
  "sensor": 9,
  "applied_action": 10,
  "base_quat": 11,
}
KIND_CODES = {"torque": 0, "position": 1, "pd": 2, "dc_motor": 3}
_FIXED_DIM = {"base_lin_vel": 3, "base_ang_vel": 3, "gravity": 3, "rot6d": 6, "base_quat": 4, "phase": 2}
_SENSOR_TERMS = {"sensor"}


class AdapterError(ValueError):
  """Admission failure; ``codes`` lists (field path, gate code, message)."""

  def __init__(self, issues: list[tuple[str, str, str]]):
    self.issues = issues
    super().__init__("; ".join(f"{p}: [{c}] {m}" for p, c, m in issues))


@dataclass(frozen=True)
class Envelope:
  """Signed piecewise torque-speed envelope, per joint (design §6.7 DC motor)."""

  speed: np.ndarray  # [A, K] ascending, rad/s
  motor: np.ndarray  # [A, K] Nm, motoring quadrant
  brake: np.ndarray  # [A, K] Nm, braking quadrant

  @classmethod
  def from_json(cls, doc: Mapping[str, Any], num_actions: int) -> Envelope:
    arrs = [np.asarray(doc[k], np.float64) for k in ("speed", "motor", "brake")]
    for a in arrs:
      if a.ndim != 2 or a.shape[0] != num_actions or a.shape[1] < 1:
        raise AdapterError([("actuator.envelope", "ACTUATOR", "envelope arrays must be [num_actions, K]")])
    if np.any(np.diff(arrs[0], axis=1) <= 0) or np.any(arrs[1] < 0) or np.any(arrs[2] < 0):
      raise AdapterError([("actuator.envelope", "ACTUATOR", "speeds must ascend and torque bounds be >= 0")])
    return cls(*arrs)


@dataclass(frozen=True)
class Normalizer:
  mean: np.ndarray
  var: np.ndarray


@dataclass
class CompiledAdapter:
  contract: AdapterContract
  num_actions: int
  obs_dim: int
  history_width: int
  needs_forward: bool
  adapter_hash: str
  action_cfg: dict[str, Any]
  action_arrays: dict[str, np.ndarray]
  controller_cfg: dict[str, Any]
  controller_arrays: dict[str, np.ndarray]
  obs_cfg: dict[str, Any]
  obs_arrays: dict[str, np.ndarray]
  joint_qpos_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
  ctrl_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))


def _vec(name: str, v: tuple[float, ...], n: int, issues: list, *, allow_one: bool, required: bool) -> np.ndarray:
  if not v:
    if required:
      issues.append((name, "CONTRACT", f"{name} must be attested (empty lists are not zero)"))
    return np.zeros(n)
  if len(v) == n:
    return np.asarray(v, np.float64)
  if allow_one and len(v) == 1:
    return np.full(n, float(v[0]))
  issues.append((name, "CONTRACT", f"{name} needs {n} entries{' or exactly one' if allow_one else ''}"))
  return np.zeros(n)


def compile_adapter(
  contract: AdapterContract,
  meta: ModelMetadata,
  *,
  num_commands: int = 0,
  normalizer: Normalizer | None = None,
  envelope: Mapping[str, Any] | None = None,
) -> CompiledAdapter:
  issues: list[tuple[str, str, str]] = []
  c = contract
  A = len(c.joint_names)
  if meta.nu > 0 and A == 0:
    issues.append(("joint_names", "JOINT_MAPPING", "actuated robot needs an explicit joint order"))
  if len(set(c.joint_names)) != A:
    issues.append(("joint_names", "JOINT_MAPPING", "joint names must be unique (bijective map)"))
  if len(c.joint_signs) != A:
    issues.append(("joint_signs", "JOINT_MAPPING", "one explicit sign per joint required"))
  qadr, dadr = np.zeros(A, np.int32), np.zeros(A, np.int32)
  for i, name in enumerate(c.joint_names):
    j = meta.joints.get(name)
    if j is None:
      issues.append((f"joint_names.{i}", "JOINT_MAPPING", f"model has no joint {name!r}"))
      continue
    if j.nv != 1:
      issues.append((f"joint_names.{i}", "JOINT_MAPPING", f"{name!r} is not a 1-DoF joint"))
      continue
    qadr[i], dadr[i] = j.qposadr, j.dofadr
  sign = np.asarray(c.joint_signs if len(c.joint_signs) == A else [1] * A, np.float64)

  # Actuators: explicit names, else the unique joint actuator of each joint.
  act = c.actuator
  ctrl_idx = np.zeros(A, np.int32)
  model_gear = np.ones(A)
  infos = []
  if act.actuator_names:
    if len(act.actuator_names) != A:
      issues.append(("actuator.actuator_names", "JOINT_MAPPING", "one actuator per joint"))
    for i, an in enumerate(act.actuator_names[:A]):
      info = meta.actuators.get(an)
      if info is None:
        issues.append((f"actuator.actuator_names.{i}", "JOINT_MAPPING", f"no actuator {an!r}"))
        continue
      if i < len(c.joint_names) and c.joint_names[i] in meta.joints:
        if info.trntype != int(mujoco.mjtTrn.mjTRN_JOINT) or info.trnid != meta.joints[c.joint_names[i]].id:
          issues.append((f"actuator.actuator_names.{i}", "JOINT_MAPPING", f"{an!r} does not drive {c.joint_names[i]!r}"))
      infos.append(info)
  else:
    for i, jn in enumerate(c.joint_names):
      if jn not in meta.joints:
        continue
      try:
        infos.append(meta.actuator_for_joint(jn))
      except ValueError as exc:
        issues.append((f"joint_names.{i}", "JOINT_MAPPING", str(exc)))
  if len(infos) == A:
    ctrl_idx = np.asarray([a.id for a in infos], np.int32)
    model_gear = np.asarray([a.gear for a in infos], np.float64)
    if len(set(ctrl_idx.tolist())) != A:
      issues.append(("actuator", "JOINT_MAPPING", "two joints map to one actuator"))
  kind = KIND_CODES[act.kind]
  need_gains = act.kind in ("pd", "dc_motor")
  kp = _vec("actuator.kp", act.kp, A, issues, allow_one=False, required=need_gains)
  kd = _vec("actuator.kd", act.kd, A, issues, allow_one=False, required=need_gains)
  elim = _vec("actuator.effort_limit", act.effort_limit, A, issues, allow_one=False, required=need_gains)
  gear = _vec("actuator.gear", act.gear, A, issues, allow_one=False, required=False) if act.gear else model_gear
  if np.any(gear == 0):
    issues.append(("actuator.gear", "ACTUATOR", "gear must be nonzero"))
  for i, info in enumerate(infos):
    servo = info.biastype == int(mujoco.mjtBias.mjBIAS_AFFINE) and info.biasprm[1] != 0.0
    if need_gains and servo:
      issues.append((f"actuator.{i}", "ACTUATOR", "explicit PD over a position servo applies stiffness twice"))
    if act.kind == "position" and not servo:
      issues.append((f"actuator.{i}", "ACTUATOR", "position mode needs a native position actuator"))
    if act.kind == "position" and act.kp and abs(act.kp[i] - info.gainprm0) > 1e-9 * max(1.0, abs(info.gainprm0)):
      issues.append((f"actuator.kp.{i}", "ACTUATOR", "contract kp differs from the compiled position gain"))
  env_arrays: dict[str, np.ndarray] = {}
  if act.kind == "dc_motor":
    if envelope is None:
      issues.append(("actuator.envelope", "ACTUATOR", "dc_motor requires a signed torque-speed envelope"))
    else:
      try:
        env = Envelope.from_json(envelope, A)
        env_arrays = {"envelope_speed": env.speed, "envelope_motor": env.motor, "envelope_brake": env.brake}
      except AdapterError as exc:
        issues.extend(exc.issues)

  scale = _vec("action_scale", c.action_scale, A, issues, allow_one=True, required=A > 0)
  offset = _vec("action_offset", c.action_offset, A, issues, allow_one=True, required=A > 0)

  # Observation terms.
  base_q = base_d = -1
  base_name = c.base_joint
  if base_name is None and len(meta.free_joints) == 1:
    base_name = meta.free_joints[0]
  if base_name is not None:
    bj = meta.joints.get(base_name)
    if bj is None or bj.type != 0:
      issues.append(("base_joint", "OBS_CONTRACT", f"{base_name!r} is not a free joint"))
    else:
      base_q, base_d = bj.qposadr, bj.dofadr
  sign_of = dict(zip(c.joint_names, sign.tolist(), strict=False))
  table, params, idx_pool, off_pool, sgn_pool = [], [], [], [], []
  width = 0
  needs_forward = False
  for t, term in enumerate(c.observation_terms):
    p = f"observation_terms.{t}"
    rows = _term(term, p, meta, sign_of, A, num_commands, c, base_q, issues)
    if rows is None:
      continue
    dim, frame, idx, off, sg = rows
    table.append([SOURCE_CODES[term.source], dim, term.history, frame, len(idx_pool), len(idx), width])
    params.append([term.scale, term.clip or 0.0, term.noise_std, term.period_s])
    idx_pool.extend(idx)
    off_pool.extend(off)
    sgn_pool.extend(sg)
    width += dim * term.history
    needs_forward |= term.source in _SENSOR_TERMS
  if c.history_layout == "time_major" and len({t.history for t in c.observation_terms}) > 1:
    issues.append(("history_layout", "OBS_CONTRACT", "time_major requires equal term histories"))
  if not c.observation_terms:
    issues.append(("observation_terms", "OBS_CONTRACT", "observation terms must be attested"))
  norm_mean = norm_inv = None
  if c.normalization == "external":
    if normalizer is None:
      issues.append(("normalizer", "NORMALIZATION", "external normalization requires the normalizer artifact"))
    else:
      mean, var = np.asarray(normalizer.mean, np.float64), np.asarray(normalizer.var, np.float64)
      if mean.shape != (width,) or var.shape != (width,):
        issues.append(("normalizer", "NORMALIZATION", f"normalizer must have obs_dim={width} entries"))
      elif not (np.isfinite(mean).all() and np.isfinite(var).all()) or np.any(var < 0):
        issues.append(("normalizer", "NORMALIZATION", "means/variances must be finite and variances >= 0"))
      else:
        norm_mean = mean
        norm_inv = 1.0 / np.sqrt(var + c.normalization_epsilon)
  elif normalizer is not None:
    issues.append(("normalization", "NORMALIZATION", "a normalizer was supplied but the contract declares none: one normalization only"))
  if issues:
    raise AdapterError(issues)

  action_cfg = {
    "num_actions": A,
    "order": 0 if c.action_order == "clip_scale_offset" else 1,
    "clip": float(c.action_clip),
    "ema_alpha": float(c.ema_alpha),
    "ema_initial": float(c.ema_initial),
    "policy_delay_ticks": c.delay_ticks if c.delay_clock == "policy" else 0,
  }
  controller_cfg = {
    "num_actions": A,
    "kind": kind,
    "feedforward": float(act.feedforward),
    "physics_delay_ticks": c.delay_ticks if c.delay_clock == "physics" else 0,
  }
  obs_cfg = {
    "history_layout": 0 if c.history_layout == "term_major" else 1,
    "history_reset": 0 if c.history_reset == "zeros" else 1,
    "history_order": 0 if c.history_order == "oldest_first" else 1,
    "term_op_order": 0 if c.term_op_order == "noise_clip_scale" else 1,
    "quaternion_xyzw": 1 if c.quaternion == "xyzw" else 0,
    "base_qpos_adr": base_q,
    "base_dof_adr": base_d,
    "obs_dim": width,
    "history_width": width,
    "normalization_clip": float(c.normalization_clip),
  }
  obs_arrays = {
    "term_table": np.asarray(table, np.int32).reshape(-1, 7),
    "term_params": np.asarray(params, np.float64).reshape(-1, 4),
    "index_pool": np.asarray(idx_pool, np.int32),
    "offset_pool": np.asarray(off_pool, np.float64),
    "sign_pool": np.asarray(sgn_pool, np.float64),
  }
  if norm_mean is not None:
    obs_arrays["norm_mean"] = norm_mean
    obs_arrays["norm_inv_std"] = norm_inv
  controller_arrays = {
    "qpos_adr": qadr, "dof_adr": dadr, "ctrl_idx": ctrl_idx, "gear": gear, "kp": kp, "kd": kd, "effort_limit": elim,
    **env_arrays,
  }
  action_arrays = {"scale": scale, "offset": offset, "sign": sign}
  h = hashlib.sha256()
  h.update(c.model_dump_json().encode())
  for d in (action_cfg, controller_cfg, obs_cfg):
    h.update(json.dumps(d, sort_keys=True).encode())
  for d in (action_arrays, controller_arrays, obs_arrays):
    for k in sorted(d):
      h.update(k.encode())
      h.update(np.ascontiguousarray(d[k]).tobytes())
  return CompiledAdapter(
    contract=c, num_actions=A, obs_dim=width, history_width=width, needs_forward=needs_forward,
    adapter_hash=h.hexdigest(), action_cfg=action_cfg, action_arrays=action_arrays,
    controller_cfg=controller_cfg, controller_arrays=controller_arrays, obs_cfg=obs_cfg, obs_arrays=obs_arrays,
    joint_qpos_adr=qadr, ctrl_indices=ctrl_idx,
  )


def _term(
  term: ObservationTerm,
  p: str,
  meta: ModelMetadata,
  sign_of: Mapping[str, float],
  A: int,
  num_commands: int,
  c: AdapterContract,
  base_q: int,
  issues: list,
) -> tuple[int, int, list[int], list[float], list[float]] | None:
  if not term.name or term.dimension <= 0:
    issues.append((p, "OBS_CONTRACT", "term name and dimension are required"))
    return None
  src = term.source
  if src == "custom":
    issues.append((f"{p}.source", "OBS_CONTRACT", "custom terms need a registered signed kernel (none in v1)"))
    return None
  dim = term.dimension
  frame = 0
  idx: list[int] = []
  off: list[float] = [0.0] * dim
  sg: list[float] = [1.0] * dim
  if src in _FIXED_DIM:
    if dim != _FIXED_DIM[src]:
      issues.append((f"{p}.dimension", "OBS_CONTRACT", f"{src} has dimension {_FIXED_DIM[src]}"))
      return None
    if src != "phase" and base_q < 0:
      issues.append((f"{p}.source", "OBS_CONTRACT", f"{src} needs a floating base joint"))
      return None
    if src in ("base_lin_vel", "base_ang_vel"):
      if term.frame != c.velocity_frame:
        issues.append((f"{p}.frame", "OBS_CONTRACT", "velocity term frame must match velocity_frame"))
      frame = 0 if c.velocity_frame == "body" else 1
    if src == "rot6d":
      if term.reference_frame != "world":
        issues.append((f"{p}.reference_frame", "OBS_CONTRACT", "rot6d is defined against the world frame in v1"))
      frame = 0 if term.rot6d_layout == "columns" else 1
    if src == "phase" and term.period_s <= 0:
      issues.append((f"{p}.period_s", "OBS_CONTRACT", "phase needs a positive period"))
    if term.offsets or term.signs or term.selectors:
      issues.append((f"{p}", "OBS_CONTRACT", f"{src} takes no selectors, offsets or signs"))
    return dim, frame, [], [], []
  if src in ("joint_pos", "joint_vel"):
    if len(term.selectors) != dim:
      issues.append((f"{p}.selectors", "OBS_CONTRACT", "one joint name per element"))
      return None
    for k, jn in enumerate(term.selectors):
      j = meta.joints.get(jn)
      if j is None or j.nv != 1:
        issues.append((f"{p}.selectors.{k}", "JOINT_MAPPING", f"{jn!r} is not a 1-DoF model joint"))
        return None
      if term.signs:
        if len(term.signs) != dim:
          issues.append((f"{p}.signs", "JOINT_MAPPING", "one sign per selected joint"))
          return None
        sg[k] = float(term.signs[k])
        if jn in sign_of and sign_of[jn] != sg[k]:
          issues.append((f"{p}.signs.{k}", "JOINT_MAPPING", f"{jn!r} sign contradicts joint_signs"))
          return None
      elif jn in sign_of:
        sg[k] = sign_of[jn]
      else:
        issues.append((f"{p}.selectors.{k}", "JOINT_MAPPING", f"{jn!r} has no attested sign"))
        return None
      idx.append(j.qposadr if src == "joint_pos" else j.dofadr)
    if src == "joint_pos" and len(term.offsets) != dim:
      issues.append((f"{p}.offsets", "OBS_CONTRACT", "joint_pos needs explicit default-position offsets"))
      return None
  elif src == "sensor":
    if len(term.selectors) != 1 or term.selectors[0] not in meta.sensors:
      issues.append((f"{p}.selectors", "OBS_CONTRACT", "sensor terms name exactly one model sensor"))
      return None
    s = meta.sensors[term.selectors[0]]
    if s.dim != dim:
      issues.append((f"{p}.dimension", "OBS_CONTRACT", f"sensor {s.name!r} has dimension {s.dim}"))
      return None
    idx = list(range(s.adr, s.adr + s.dim))
  elif src in ("command", "previous_action", "applied_action"):
    limit = num_commands if src == "command" else A
    sel = [int(x) for x in term.selectors] if term.selectors else list(range(dim))
    if len(sel) != dim or any(not 0 <= x < limit for x in sel):
      issues.append((f"{p}.selectors", "OBS_CONTRACT", f"{src} indices must be {dim} values in [0, {limit})"))
      return None
    idx = sel
  if term.offsets:
    if len(term.offsets) != dim:
      issues.append((f"{p}.offsets", "OBS_CONTRACT", "offsets need one value per element"))
      return None
    off = list(term.offsets)
  return dim, frame, idx, off, sg


# -------------------------------------------------------------------- runtime


class Sim2SimAdapter:
  """Adapter runtime over N slots: action processing, substep control, observation.

  ``views`` are the MJBatch bind() arrays (qpos, qvel, ctrl, sensordata) and the
  core's shared per-slot arrays (command, sim_time_ns, control_seq, noise_keys).
  """

  def __init__(self, compiled: CompiledAdapter, views: Mapping[str, np.ndarray], num_slots: int):
    self.compiled = compiled
    N, A, W = num_slots, compiled.num_actions, compiled.history_width
    ca, cc = compiled.action_cfg, compiled.controller_cfg
    nu = views["ctrl"].shape[1]
    self.state: dict[str, np.ndarray] = {
      "prev_action": np.zeros((N, A), np.float32),
      "ema": np.zeros((N, A), np.float64),
      "delay_buffer": np.zeros((N, (ca["policy_delay_ticks"] + 1) * A), np.float64),
      "delay_count": np.zeros((N, 1), np.int64),
      "targets": np.zeros((N, A), np.float64),
      "physics_delay_buffer": np.zeros((N, (cc["physics_delay_ticks"] + 1) * A), np.float64),
      "physics_delay_count": np.zeros((N, 1), np.int64),
      "effort": np.zeros((N, A), np.float64),
      "ctrl_saved": np.zeros((N, nu), np.float64),
      "saturated": np.zeros((N, 1), np.int64),
      "effort_evals": np.zeros((N, 1), np.int64),
      "energy": np.zeros((N, 1), np.float64),
      "history": np.zeros((N, W), np.float32),
      "history_filled": np.zeros((N, 1), np.uint8),
    }
    s = self.state
    self.actions = _native.ActionProcessor(
      ca,
      {**compiled.action_arrays, "prev_action": s["prev_action"], "ema": s["ema"], "delay_buffer": s["delay_buffer"],
       "delay_count": s["delay_count"], "targets": s["targets"]},
    )
    self.controller = _native.Controller(
      cc,
      {**compiled.controller_arrays, "qpos": views["qpos"], "qvel": views["qvel"], "ctrl": views["ctrl"],
       "targets": s["targets"], "physics_delay_buffer": s["physics_delay_buffer"],
       "physics_delay_count": s["physics_delay_count"], "effort": s["effort"], "ctrl_saved": s["ctrl_saved"],
       "saturated": s["saturated"], "effort_evals": s["effort_evals"], "energy": s["energy"]},
    )
    obs_arrays = {
      **compiled.obs_arrays,
      "qpos": views["qpos"],
      "qvel": views["qvel"],
      "sim_time_ns": views["sim_time_ns"],
      "noise_keys": views["noise_keys"],
      "control_seq": views["control_seq"],
      "prev_action": s["prev_action"],
      "targets": s["targets"],
      "history": s["history"],
      "history_filled": s["history_filled"],
    }
    for opt in ("sensordata", "command"):
      if opt in views and views[opt] is not None and views[opt].shape[1] > 0:
        obs_arrays[opt] = views[opt]
    self.observation = _native.Observation(compiled.obs_cfg, obs_arrays)

  @property
  def obs_dim(self) -> int:
    return self.compiled.obs_dim

  def reset(self, slots: np.ndarray) -> None:
    self.actions.reset(slots)
    self.controller.reset(slots)
    self.observation.reset(slots)

  def process_action(self, slots: np.ndarray, policy_action: np.ndarray) -> np.ndarray:
    self.actions.process(slots, np.ascontiguousarray(policy_action, np.float32))
    return self.state["targets"][slots]

  def before_substep(self, slots: np.ndarray, dt: float) -> None:
    self.controller.before_substep(slots, dt)

  def neutralize(self, slots: np.ndarray) -> None:
    self.controller.neutralize(slots)

  @property
  def fusable(self) -> bool:
    """ctrl is constant over a policy interval (torque/position, no physics-clock delay)."""
    return bool(self.controller.fusable)

  def prepare_fused(self, slots: np.ndarray) -> None:
    self.controller.prepare_fused(slots)

  def finish_fused(self, slots: np.ndarray, rows: np.ndarray, history: np.ndarray, qpos_offset: int,
                   qvel_offset: int, dt: float) -> None:
    self.controller.finish_fused(slots, rows, history, qpos_offset, qvel_offset, dt)

  def observe(self, slots: np.ndarray, output: np.ndarray) -> None:
    self.observation.observe(slots, output)

  def save(self, slots: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[slots].copy() for k, v in self.state.items()}

  def restore(self, slots: np.ndarray, state: Mapping[str, np.ndarray]) -> None:
    for k, v in self.state.items():
      v[slots] = state[k]
