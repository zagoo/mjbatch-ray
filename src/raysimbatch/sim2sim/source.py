# SPDX-License-Identifier: Apache-2.0
"""Source-engine export format ("rsb-source-export/1") and a reference exporter.

A real Isaac Lab/PhysX export must provide the same package (see docs/sim2sim.md):

* ``properties``: the resolved-properties table (design B.2), SI, source frames.
* ``conventions``: joint order (policy order), per-joint axis sign relative to the
  canonical asset, quaternion order, velocity frames, decimation, dt.
* arrays: observation fixtures (named state + per-term values + flattened policy
  input), history sequences with resets, policy input/output pairs, actuator sweeps,
  filter impulse/delay responses, a timing trace, kinematic poses, per-joint
  perturbation responses, contact fixture curves and the initial state.

``export_reference`` produces this package by simulating a *source model* that
differs from the canonical MJCF in its conventions (joint order, reversed axes,
quaternion order), computing observations with the independent numpy pipeline in
``sim2sim.reference``. No Isaac Lab installation exists on the qualification host,
so this exporter is what exercises the gates end to end; qualification against real
Isaac traces is recorded as BLOCKED until such traces are supplied.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from raysimbatch.sim2sim import reference as ref

FORMAT = "rsb-source-export/1"


@dataclass
class SourceExport:
  manifest: dict[str, Any]
  arrays: dict[str, np.ndarray]

  def to_bytes(self) -> bytes:
    buf = io.BytesIO()
    arrays: dict[str, Any] = self.arrays  # as savez keywords (the stub matches them against allow_pickle)
    np.savez_compressed(buf, __manifest__=np.frombuffer(json.dumps(self.manifest, sort_keys=True).encode(), np.uint8),
                        **arrays)
    return buf.getvalue()

  @classmethod
  def from_bytes(cls, blob: bytes) -> SourceExport:
    with np.load(io.BytesIO(blob)) as z:
      manifest = json.loads(z["__manifest__"].tobytes())
      arrays = {k: z[k] for k in z.files if k != "__manifest__"}
    if manifest.get("format") != FORMAT:
      raise ValueError(f"not a {FORMAT} package")
    return cls(manifest, arrays)

  @property
  def digest(self) -> str:
    return hashlib.sha256(self.to_bytes()).hexdigest()


# --------------------------------------------------------------- property tables


def resolved_properties(m: mujoco.MjModel) -> dict[str, Any]:
  """Named SI tables of bodies/joints/geoms/actuators (design B.2). Rows are keyed by
  unique names: an unnamed body or joint fails ingestion."""
  def name(obj, i):
    n = mujoco.mj_id2name(m, obj, i)
    if not n:
      if obj in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT):
        raise ValueError(f"unnamed {obj.name} {i}: resolved-property rows need unique names")
      return f"{obj.name}{i}"
    return n

  bodies = []
  for b in range(1, m.nbody):
    inertia = m.body_inertia[b]
    bodies.append({
      "name": name(mujoco.mjtObj.mjOBJ_BODY, b), "parent": name(mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[b])),
      "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(), "mass": float(m.body_mass[b]),
      "ipos": m.body_ipos[b].tolist(), "iquat": m.body_iquat[b].tolist(),
      # symmetric inertia [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] in the body frame
      "inertia": _principal_to_full(inertia, m.body_iquat[b]),
    })
  joints = []
  for j in range(m.njnt):
    d = int(m.jnt_dofadr[j])
    joints.append({
      "name": name(mujoco.mjtObj.mjOBJ_JOINT, j), "type": int(m.jnt_type[j]), "axis": m.jnt_axis[j].tolist(),
      "range": m.jnt_range[j].tolist() if m.jnt_limited[j] else None, "damping": float(m.dof_damping[d]),
      "frictionloss": float(m.dof_frictionloss[d]), "armature": float(m.dof_armature[d]),
      "body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])),
    })
  geoms = []
  for g in range(m.ngeom):
    geoms.append({
      "name": name(mujoco.mjtObj.mjOBJ_GEOM, g), "body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
      "type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(),
      "quat": m.geom_quat[g].tolist(), "friction": m.geom_friction[g].tolist(), "solref": m.geom_solref[g].tolist(),
      "solimp": m.geom_solimp[g].tolist(), "condim": int(m.geom_condim[g]),
      "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g]),
    })
  acts = []
  for a in range(m.nu):
    acts.append({
      "name": name(mujoco.mjtObj.mjOBJ_ACTUATOR, a), "joint": name(mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])),
      "gear": float(m.actuator_gear[a, 0]), "gainprm": m.actuator_gainprm[a, :3].tolist(),
      "biasprm": m.actuator_biasprm[a, :3].tolist(), "forcerange": m.actuator_forcerange[a].tolist(),
      "ctrlrange": m.actuator_ctrlrange[a].tolist(),
    })
  return {"bodies": bodies, "joints": joints, "geoms": geoms, "actuators": acts,
          "option": {"timestep": float(m.opt.timestep), "gravity": m.opt.gravity.tolist(),
                     "iterations": int(m.opt.iterations), "solver": int(m.opt.solver),
                     "integrator": int(m.opt.integrator), "cone": int(m.opt.cone)}}


def _principal_to_full(principal: np.ndarray, iquat: np.ndarray) -> list[float]:
  R = ref.quat_wxyz_to_mat(np.asarray(iquat))
  full = R @ np.diag(principal) @ R.T
  return [full[0, 0], full[1, 1], full[2, 2], full[0, 1], full[0, 2], full[1, 2]]


# ------------------------------------------------------------ reference source


@dataclass
class SourceConventions:
  """How the source engine names, orders and signs things relative to the canonical
  asset. ``joint_sign[j]`` is +1 when the source joint axis equals the canonical one."""

  joint_order: tuple[str, ...]
  joint_sign: dict[str, int]
  quat_order: str = "xyzw"
  default_pos: dict[str, float] = field(default_factory=dict)  # source convention
  decimation: int = 4
  terms: tuple[dict[str, Any], ...] = ()
  history_reset: str = "repeat_initial"
  history_layout: str = "term_major"
  history_order: str = "oldest_first"
  action_clip: float = 4.0
  action_scale: float = 0.5
  ema_alpha: float = 1.0
  kp: float = 20.0
  kd: float = 0.5
  effort_limit: float = 12.0
  base_joint: str = "root"
  base_body: str = "torso"
  contact_geom: str = "fl_foot"


def source_model_xml(canonical_xml: bytes, conv: SourceConventions) -> bytes:
  """The source engine's model: canonical geometry with its own joint axis signs."""
  canon = mujoco.MjModel.from_xml_string(canonical_xml.decode())
  flip = [int(canon.jnt_qposadr[mujoco.mj_name2id(canon, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n, s_ in conv.joint_sign.items() if s_ == -1]
  spec = mujoco.MjSpec.from_string(canonical_xml.decode())
  for j in spec.joints:
    if conv.joint_sign.get(j.name, 1) == -1:
      j.axis = [-a for a in j.axis]
      lo, hi = j.range
      j.range = [-hi, -lo]
  for key in spec.keys:  # keyframes are expressed in the source convention too
    q = list(key.qpos)
    for a in flip:
      if a < len(q):
        q[a] = -q[a]
    key.qpos = q
  return spec.to_xml().encode()


def export_reference(canonical_xml: bytes, conv: SourceConventions, policy_layers, *, n_fixtures: int = 1000,
                     seed: int = 0, dt: float = 0.004, contact_scale: float = 1.0,
                     solver: dict[str, Any] | None = None) -> SourceExport:
  """``solver`` is the declared target solver configuration (iterations, tolerance,
  solver/integrator names); the source export records what it simulated with."""
  rng = np.random.default_rng(seed)
  src_xml = source_model_xml(canonical_xml, conv)
  m = mujoco.MjModel.from_xml_string(src_xml.decode())
  m.opt.timestep = dt
  sv = solver or {}
  if "iterations" in sv:
    m.opt.iterations = int(sv["iterations"])
  if "tolerance" in sv:
    m.opt.tolerance = float(sv["tolerance"])
  if contact_scale != 1.0:  # a deliberately different source contact response (tests)
    m.geom_friction[:, 0] *= contact_scale
  d = mujoco.MjData(m)
  J = len(conv.joint_order)
  jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in conv.joint_order]
  qadr = np.asarray([m.jnt_qposadr[j] for j in jid])
  default = np.asarray([conv.default_pos.get(n, 0.0) for n in conv.joint_order])
  arrays: dict[str, np.ndarray] = {}

  # -- observation fixtures (independent states; history filled by repeat/zeros rule)
  quats = rng.normal(size=(n_fixtures, 4))
  quats /= np.linalg.norm(quats, axis=1, keepdims=True)
  quats[quats[:, 0] < 0] *= -1
  state = {
    "base_quat_wxyz": quats,
    "base_lin_vel_w": rng.uniform(-1.5, 1.5, (n_fixtures, 3)),
    "base_ang_vel_b": rng.uniform(-2, 2, (n_fixtures, 3)),
    "joint_pos": default + rng.uniform(-0.4, 0.4, (n_fixtures, J)),
    "joint_vel": rng.uniform(-4, 4, (n_fixtures, J)),
    "command": rng.uniform(-1, 1, (n_fixtures, 3)),
    "prev_action": rng.uniform(-1, 1, (n_fixtures, J)).astype(np.float32),
    "time_s": rng.uniform(0, 10, n_fixtures),
  }
  quat_src = ref.wxyz_to_xyzw(quats) if conv.quat_order == "xyzw" else quats
  arrays.update({"obs/base_quat": quat_src, "obs/base_lin_vel_w": state["base_lin_vel_w"],
                 "obs/base_ang_vel_b": state["base_ang_vel_b"], "obs/joint_pos": state["joint_pos"],
                 "obs/joint_vel": state["joint_vel"], "obs/command": state["command"],
                 "obs/prev_action": state["prev_action"], "obs/time_s": state["time_s"]})
  terms = [dict(t) for t in conv.terms]
  for t in terms:
    t.setdefault("index", list(range(J)) if t["source"] in ("joint_pos", "joint_vel", "previous_action") else
                 list(range(3)))
    if t["source"] == "joint_pos":
      t.setdefault("offsets", default.tolist())
  flat = []
  for k in range(n_fixtures):
    st = {kk: v[k] for kk, v in state.items()}
    vals = [ref.process_term(ref.term_value(t, st), t) for t in terms]
    hist = ref.History(terms, conv.history_reset)
    flat.append(ref.flatten(hist.push(vals), conv.history_layout, conv.history_order))
  for t in terms:
    arrays[f"terms/{t['name']}"] = np.stack([
      ref.process_term(ref.term_value(t, {kk: v[k] for kk, v in state.items()}), t) for k in range(n_fixtures)
    ])
  obs_flat = np.asarray(flat, np.float32)
  arrays["obs/flat"] = obs_flat
  # -- history sequences with a mid-sequence reset of every other env
  S, T = 8, 12
  seq_states = {k: np.stack([v[rng.integers(0, n_fixtures, T)] for _ in range(S)]) for k, v in state.items()}
  seq_obs = np.zeros((S, T, obs_flat.shape[1]), np.float32)
  resets = np.zeros((S, T), bool)
  resets[::2, T // 2] = True
  for s_ in range(S):
    hist = ref.History(terms, conv.history_reset)
    for t_ in range(T):
      if resets[s_, t_]:
        hist.reset()
      st = {kk: v[s_, t_] for kk, v in seq_states.items()}
      seq_obs[s_, t_] = ref.flatten(hist.push([ref.process_term(ref.term_value(t, st), t) for t in terms]),
                                    conv.history_layout, conv.history_order)
  for k, v in seq_states.items():
    vv = ref.wxyz_to_xyzw(v) if (k == "base_quat_wxyz" and conv.quat_order == "xyzw") else v
    arrays[f"seq/{'base_quat' if k == 'base_quat_wxyz' else k}"] = vv
  arrays["seq/obs"] = seq_obs
  arrays["seq/reset"] = resets
  # -- policy outputs (float64 source forward)
  arrays["policy/obs"] = obs_flat
  arrays["policy/action"] = ref.mlp_forward(policy_layers, obs_flat.astype(np.float64), clip=conv.action_clip)
  # -- actuator sweep (source convention) and filter impulse
  M = 2000
  tgt = default + rng.uniform(-1.0, 1.0, (M, J))
  q = default + rng.uniform(-1.0, 1.0, (M, J))
  dq = rng.uniform(-20, 20, (M, J))
  arrays["act/target"], arrays["act/q"], arrays["act/dq"] = tgt, q, dq
  arrays["act/effort"] = ref.pd_effort(tgt, q, dq, conv.kp, conv.kd, np.full(J, conv.effort_limit))
  impulse = np.zeros((20, J))
  impulse[1] = 1.0
  filt, prev = [], np.zeros(J)
  for a in impulse:
    prev = ref.ema(ref.process_action(a, clip=conv.action_clip, scale=np.full(J, conv.action_scale),
                                      offset=default, sign=np.ones(J)), prev, conv.ema_alpha)
    filt.append(prev.copy())
  arrays["act/impulse_action"] = impulse.astype(np.float32)
  arrays["act/impulse_target"] = np.asarray(filt)
  # -- timing: which physics ticks start a new policy step
  arrays["timing/policy_tick"] = (np.arange(400) % conv.decimation == 0)
  # -- kinematics and joint perturbations (base at identity)
  K = 50
  kin_q = default + rng.uniform(-0.6, 0.6, (K, J))
  bnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(1, m.nbody)]
  pos = np.zeros((K, m.nbody - 1, 3))
  quat = np.zeros((K, m.nbody - 1, 4))
  base = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, conv.base_joint)]
  for k in range(K):
    mujoco.mj_resetData(m, d)
    d.qpos[base : base + 7] = [0, 0, 0.5, 1, 0, 0, 0]
    d.qpos[qadr] = kin_q[k]
    mujoco.mj_kinematics(m, d)
    pos[k], quat[k] = d.xpos[1:], d.xquat[1:]
  arrays["kin/joint_pos"], arrays["kin/body_pos"], arrays["kin/body_quat"] = kin_q, pos, quat
  delta = 0.05
  pert = np.zeros((J, m.nbody - 1, 3))
  mujoco.mj_resetData(m, d)
  d.qpos[base : base + 7] = [0, 0, 0.5, 1, 0, 0, 0]
  d.qpos[qadr] = default
  mujoco.mj_kinematics(m, d)
  nominal = d.xpos[1:].copy()
  for j in range(J):
    d.qpos[qadr] = default
    d.qpos[qadr[j]] += delta
    mujoco.mj_kinematics(m, d)
    pert[j] = d.xpos[1:] - nominal
  arrays["perturb/delta_pos"] = pert
  # -- contact fixtures on the declared geom's material
  geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, conv.contact_geom)
  arrays.update(contact_fixtures(m.geom_friction[geom].copy(), m.geom_solref[geom].copy(),
                                 m.geom_solimp[geom].copy(), dt))
  # -- initial state (reset keyframe) in source convention
  mujoco.mj_resetDataKeyframe(m, d, 0)
  arrays["init/qpos_joints"] = d.qpos[qadr].copy()
  arrays["init/base_qpos"] = d.qpos[base : base + 7].copy()
  manifest = {
    "format": FORMAT, "source_engine": "rsb-reference-mujoco", "source_version": mujoco.__version__,
    "conventions": {"joint_order": list(conv.joint_order), "joint_sign": conv.joint_sign,
                    "quat_order": conv.quat_order, "decimation": conv.decimation, "dt": dt,
                    "history_reset": conv.history_reset, "history_layout": conv.history_layout,
                    "history_order": conv.history_order, "base_joint": conv.base_joint,
                    "base_body": conv.base_body, "contact_geom": conv.contact_geom,
                    "ema_alpha": conv.ema_alpha, "action_clip": conv.action_clip,
                    "action_scale": conv.action_scale, "default_pos": list(default),
                    "actuator": {"kind": "pd", "kp": conv.kp, "kd": conv.kd, "effort_limit": conv.effort_limit}},
    "terms": terms,
    "body_names": bnames,
    "properties": resolved_properties(m),
  }
  return SourceExport(manifest, arrays)


def contact_fixtures(friction: np.ndarray, solref: np.ndarray, solimp: np.ndarray, dt: float) -> dict[str, np.ndarray]:
  """Drop, rest and friction-ramp fixtures for one contact material."""
  f = " ".join(map(str, friction))
  sr = " ".join(map(str, solref))
  si = " ".join(map(str, solimp))
  xml = f"""
  <mujoco><option timestep="{dt}"/>
    <worldbody><geom type="plane" size="5 5 .1" friction="{f}" solref="{sr}" solimp="{si}"/>
      <body name="drop" pos="0 0 .3"><freejoint/><geom type="sphere" size=".03" mass=".2" friction="{f}"
        solref="{sr}" solimp="{si}"/></body>
      <body name="slide" pos="1 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1" friction="{f}"
        solref="{sr}" solimp="{si}"/></body>
    </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)
  steps = int(1.0 / dt)
  z = np.zeros(steps)
  x = np.zeros(steps)
  slide_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "slide")
  onset = np.nan
  for k in range(steps):
    # Horizontal force ramp on the box: 0 -> 2 * mu * m * g over one second.
    force = 2.0 * friction[0] * 9.81 * (k / steps)
    d.xfrc_applied[slide_body, 0] = force
    mujoco.mj_step(m, d)
    z[k] = d.qpos[2]
    x[k] = d.qpos[7]
    if np.isnan(onset) and d.qvel[6] > 0.05:
      onset = force
  return {"contact/drop_z": z, "contact/slide_x": x, "contact/slide_onset_force": np.asarray([onset]),
          "contact/rest_penetration": np.asarray([0.03 - z[-1]])}


def quadruped_conventions(history: int = 3) -> SourceConventions:
  """Isaac-style conventions for the quadruped fixture: hips then knees, right-side hip
  axes reversed, xyzw root quaternion, 3-step term history, repeat-initial reset."""
  order = ("fl_hip", "fr_hip", "hl_hip", "hr_hip", "fl_knee", "fr_knee", "hl_knee", "hr_knee")
  sign = {"fr_hip": -1, "hr_hip": -1}
  default = {"fl_hip": 0.5, "fr_hip": -0.5, "hl_hip": 0.5, "hr_hip": -0.5,
             "fl_knee": -1.0, "fr_knee": -1.0, "hl_knee": -1.0, "hr_knee": -1.0}
  terms = (
    {"name": "gravity", "source": "gravity", "dim": 3, "history": history},
    {"name": "ang_vel", "source": "base_ang_vel", "dim": 3, "scale": 0.25, "history": history},
    {"name": "lin_vel", "source": "base_lin_vel", "dim": 3, "scale": 2.0, "history": history},
    {"name": "quat", "source": "base_quat", "dim": 4, "quaternion": "xyzw", "history": 1},
    {"name": "rot6d", "source": "rot6d", "dim": 6, "layout": "columns", "history": 1},
    {"name": "command", "source": "command", "dim": 3, "history": 1},
    {"name": "joint_pos", "source": "joint_pos", "dim": 8, "history": history},
    {"name": "joint_vel", "source": "joint_vel", "dim": 8, "scale": 0.05, "clip": 10.0, "history": history},
    {"name": "last_action", "source": "previous_action", "dim": 8, "history": 1},
  )
  return SourceConventions(order, sign, "xyzw", default, 4, terms)


def perturb_conventions(conv: SourceConventions, **changes: Any) -> SourceConventions:
  c = copy.deepcopy(conv)
  for k, v in changes.items():
    setattr(c, k, v)
  return c


def simulate_source(canonical_xml: bytes, conv: SourceConventions, policy_layers, *, seeds, episodes: int,
                    steps: int, dt: float, min_height: float, min_up_z: float, reset_noise: float = 0.05,
                    solver: dict[str, Any] | None = None) -> dict[str, Any]:
  """Source-side outcomes: the reference source engine runs the float64 source policy
  with the numpy observation/action pipeline and PD at physics rate. Success means the
  episode reached its time limit without falling (the fixture task's criterion)."""
  m = mujoco.MjModel.from_xml_string(source_model_xml(canonical_xml, conv).decode())
  m.opt.timestep = dt
  if solver and "iterations" in solver:
    m.opt.iterations = int(solver["iterations"])
  d = mujoco.MjData(m)
  J = len(conv.joint_order)
  jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in conv.joint_order]
  qadr = np.asarray([m.jnt_qposadr[j] for j in jid])
  dadr = np.asarray([m.jnt_dofadr[j] for j in jid])
  aid = np.asarray([next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j) for j in jid])
  gear = m.actuator_gear[aid, 0]
  base = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, conv.base_joint)]
  bdof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, conv.base_joint)]
  default = np.asarray([conv.default_pos.get(n, 0.0) for n in conv.joint_order])
  terms = [dict(t) for t in conv.terms]
  for t in terms:
    t.setdefault("index", list(range(J)) if t["source"] in ("joint_pos", "joint_vel", "previous_action") else
                 list(range(3)))
    if t["source"] == "joint_pos":
      t.setdefault("offsets", default.tolist())
  by_seed: dict[int, list[bool]] = {}
  for seed in seeds:
    rng = np.random.default_rng(1_000_003 * int(seed) + 17)
    for _ in range(episodes):
      mujoco.mj_resetDataKeyframe(m, d, 0)
      d.qpos[qadr] += rng.uniform(-reset_noise, reset_noise, J)
      mujoco.mj_forward(m, d)
      hist = ref.History(terms, conv.history_reset)
      prev = np.zeros(J)
      ema_state = np.zeros(J)
      fell = False
      for _k in range(steps):
        quat = d.qpos[base + 3 : base + 7].copy()
        R = ref.quat_wxyz_to_mat(quat)
        st = {"base_quat_wxyz": quat, "base_lin_vel_w": d.qvel[bdof : bdof + 3].copy(),
              "base_ang_vel_b": d.qvel[bdof + 3 : bdof + 6].copy(), "joint_pos": d.qpos[qadr].copy(),
              "joint_vel": d.qvel[dadr].copy(), "command": np.zeros(3), "prev_action": prev, "time_s": d.time}
        obs = ref.flatten(hist.push([ref.process_term(ref.term_value(t, st), t) for t in terms]),
                          conv.history_layout, conv.history_order)
        a = ref.mlp_forward(policy_layers, obs[None].astype(np.float32).astype(np.float64), clip=conv.action_clip)[0]
        prev = a.astype(np.float32)
        tgt = ref.process_action(a, clip=conv.action_clip, scale=np.full(J, conv.action_scale), offset=default,
                                 sign=np.ones(J))
        ema_state = ref.ema(tgt, ema_state, conv.ema_alpha) if conv.ema_alpha < 1 else tgt
        for _s in range(conv.decimation):
          tau = ref.pd_effort(ema_state, d.qpos[qadr], d.qvel[dadr], conv.kp, conv.kd, np.full(J, conv.effort_limit))
          d.ctrl[aid] = tau / gear
          mujoco.mj_step(m, d)
          R = ref.quat_wxyz_to_mat(d.qpos[base + 3 : base + 7])
          if d.qpos[base + 2] < min_height or R[2, 2] < min_up_z:
            fell = True
            break
        if fell:
          break
      by_seed.setdefault(int(seed), []).append(not fell)
  per_seed = {s: float(np.mean(v)) for s, v in by_seed.items()}
  return {"rate": float(np.mean([x for v in by_seed.values() for x in v])), "per_seed": per_seed,
          "episodes": int(sum(len(v) for v in by_seed.values())), "engine": "rsb-reference-mujoco"}
