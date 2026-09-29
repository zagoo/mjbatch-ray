# SPDX-License-Identifier: Apache-2.0
"""Automated transfer gates (design §6.7 table). Each gate returns a GateResult with its
failure code and metrics. Gates are blocking for production scoring; tolerances come
from the versioned ParityConfig and are never changed during a run."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from raysimbatch.adapter import AdapterError, Normalizer, compile_adapter
from raysimbatch.config.models import ParityConfig
from raysimbatch.policy.runtime import LocalPolicyRuntime
from raysimbatch.sim2sim import reference as ref
from raysimbatch.sim2sim.bundle import GateReport, GateResult, TransferBundle
from raysimbatch.sim2sim.source import contact_fixtures
from raysimbatch.tasks import bind_task
from raysimbatch.worker.core import CoreSpec, MJBatchCore
from raysimbatch.worker.model import ModelMetadata, compile_model


class Rig:
  """Target-side execution for gates: the canonical model through the real compiled
  adapter and native kernels (MJBatchCore)."""

  def __init__(self, b: TransferBundle, n: int, decimation: int | None = None):
    self.b = b
    self.model = compile_model(b.canonical_xml, None, b.group)
    self.meta = ModelMetadata.from_model(self.model)
    doc = dict(b.task_doc)
    spec = dict(doc.get("spec", {}))
    if doc.get("name") == "velocity_tracking":
      spec.update(min_height=-1e9, min_up_z=-2.0)  # gates must not terminate fixtures
    self.binding = bind_task({**doc, "spec": spec}, self.meta)
    norm = None
    if b.normalizer:
      norm = Normalizer(np.asarray(b.normalizer["mean"]), np.asarray(b.normalizer["var"]))
    self.compiled = compile_adapter(b.contract, self.meta, num_commands=self.binding.num_commands, normalizer=norm,
                                    envelope=b.envelope)
    self.core = MJBatchCore(self.model, self.meta, self.compiled, self.binding, CoreSpec(
      num_slots=n, threads=1, decimation=decimation or b.group.decimation, timestep_ns=b.group.timestep_ns,
      keyframe=-1, max_control_steps=10**9, forward_at_control_boundary=True, health=b.group.health))
    s = np.arange(n, dtype=np.int32)
    self.core.reset_exact(s, s.astype(np.uint64), np.zeros(n, np.uint64), (s.astype(np.uint64), np.zeros(n, np.uint64)))
    conv = b.source.manifest["conventions"]
    self.src_order = list(conv["joint_order"])
    self.src_quat = conv["quat_order"]
    self.base = self.meta.joint(conv["base_joint"])
    c = b.contract
    # The attested signs are what the gates test; fixture states are mapped with the
    # physical truth from the resolved property tables (source vs canonical joint axes).
    self.contract_sign = {n: float(sg) for n, sg in zip(c.joint_names, c.joint_signs, strict=False)}
    self.sign = physical_signs(b.source.manifest["properties"], self.model, self.meta)

  def set_state(self, slots: np.ndarray, q_src: np.ndarray, dq_src: np.ndarray, base_quat_src: np.ndarray | None = None,
                lin_w: np.ndarray | None = None, ang_b: np.ndarray | None = None, base_pos=(0.0, 0.0, 0.5),
                signs: dict[str, float] | None = None) -> None:
    """Harness step 1: named source state -> target integration-state rows, using the
    physical axis signs unless ``signs`` overrides them."""
    sign_map = self.sign if signs is None else signs
    core = self.core
    lay = core.layout
    qo, vo = lay["qpos"][0], lay["qvel"][0]
    st = core.h.state
    n = slots.size
    quat = np.tile([1.0, 0, 0, 0], (n, 1)) if base_quat_src is None else (
      ref.xyzw_to_wxyz(base_quat_src) if self.src_quat == "xyzw" else base_quat_src)
    st[np.ix_(slots, qo + self.base.qposadr + np.arange(3))] = np.asarray(base_pos)
    st[np.ix_(slots, qo + self.base.qposadr + 3 + np.arange(4))] = quat
    st[np.ix_(slots, vo + self.base.dofadr + np.arange(6))] = 0.0
    if lin_w is not None:
      st[np.ix_(slots, vo + self.base.dofadr + np.arange(3))] = lin_w
    if ang_b is not None:
      st[np.ix_(slots, vo + self.base.dofadr + 3 + np.arange(3))] = ang_b
    for i, name in enumerate(self.src_order):
      j = self.meta.joints.get(name)
      if j is None:
        continue
      sg = sign_map.get(name, 1.0)
      st[slots, qo + j.qposadr] = sg * q_src[:, i]
      st[slots, vo + j.dofadr] = sg * dq_src[:, i]
    out = core.h.forward(slots)
    if not out.ok:
      raise RuntimeError(f"forward failed while loading fixture states: {out.message}")


def _err(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> tuple[float, bool]:
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  diff = np.abs(a - b)
  ok = bool(np.all(diff <= atol + rtol * np.abs(b)))
  return float(diff.max()) if diff.size else 0.0, ok


# ------------------------------------------------------------------------ gates


def gate_xml_abi(b: TransferBundle, p: ParityConfig) -> GateResult:
  problems = []
  try:
    m = compile_model(b.canonical_xml, None, b.group)
  except Exception as exc:  # noqa: BLE001
    return GateResult("XML_ABI_MISMATCH", False, {}, f"compile failed: {exc}")
  if b.asset_contract.target_version != mujoco.__version__:
    problems.append(f"target_version {b.asset_contract.target_version} != engine {mujoco.__version__}")
  if m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
    problems.append("sleep is not supported by the Batch lane")
  if m.nplugin:
    problems.append("plugins require the certified scalar lane")
  names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)}
  props = b.source.manifest["properties"]
  missing = [j["name"] for j in props["joints"] if j["name"] not in names]
  if missing:
    problems.append(f"source joints missing in target: {missing}")
  bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
  missing_b = [x["name"] for x in props["bodies"] if x["name"] not in bodies]
  if missing_b:
    problems.append(f"source bodies missing in target: {missing_b}")
  return GateResult("XML_ABI_MISMATCH", not problems, {"problems": float(len(problems))}, "; ".join(problems))


def gate_joint_map(rig: Rig, p: ParityConfig) -> GateResult:
  b = rig.b
  c = b.contract
  src = rig.src_order
  msgs = []
  if list(c.joint_names) != src:
    msgs.append(f"policy joint order {list(c.joint_names)} != source order {src}")
  if len(set(c.joint_names)) != len(c.joint_names):
    msgs.append("joint map is not bijective")
  # Single-joint positive perturbation responses must agree in direction.
  arr = b.source.arrays
  names = b.source.manifest["body_names"]
  default_src = np.asarray(b.source.manifest["conventions"]["default_pos"])
  s = np.array([0], np.int32)
  wrong = [n for n in src if n in rig.contract_sign and rig.contract_sign[n] != rig.sign.get(n, 1.0)]
  if wrong:
    msgs.append(f"attested signs contradict the resolved joint axes for {wrong}")
  rig.set_state(s, default_src[None], np.zeros((1, len(src))), signs=rig.contract_sign)
  nominal = _xpos(rig)
  bad = []
  for j in range(len(src)):
    q = default_src.copy()
    q[j] += 0.05
    rig.set_state(s, q[None], np.zeros((1, len(src))), signs=rig.contract_sign)
    delta = _xpos(rig) - nominal
    want = arr["perturb/delta_pos"][j]
    moved = np.linalg.norm(want, axis=1) > 1e-6
    if not moved.any():
      continue
    dots = np.sum(delta[moved] * want[moved], axis=1)
    cos = dots / (np.linalg.norm(delta[moved], axis=1) * np.linalg.norm(want[moved], axis=1) + 1e-12)
    if np.any(cos < 0.99):
      bad.append(src[j])
  del names
  if bad:
    msgs.append(f"pose response sign/direction differs for joints {bad}")
  # (the contract mapping was used for the perturbations above on purpose)
  return GateResult("JOINT_MAPPING", not msgs, {"joints_with_wrong_response": float(len(bad))}, "; ".join(msgs))


def physical_signs(props: dict, model: mujoco.MjModel, meta: ModelMetadata) -> dict[str, float]:
  """+1/-1 per hinge/slide joint: whether the source joint axis (resolved property table,
  source body frame) equals the canonical model's axis. Body frames are compared by the
  ASSET_FRAME gate; a sign here is only meaningful when those agree."""
  out = {}
  for j in props["joints"]:
    tj = meta.joints.get(j["name"])
    if tj is None or tj.nv != 1:
      continue
    d = float(np.dot(np.asarray(j["axis"]), model.jnt_axis[tj.id]))
    out[j["name"]] = 1.0 if d >= 0 else -1.0
  return out


def _xpos(rig: Rig) -> np.ndarray:
  m, d = rig.model, mujoco.MjData(rig.model)
  mujoco.mj_setState(m, d, rig.core.h.state[0], int(mujoco.mjtState.mjSTATE_INTEGRATION))
  mujoco.mj_kinematics(m, d)
  names = rig.b.source.manifest["body_names"]
  return np.stack([d.xpos[rig.meta.bodies[n]] for n in names])


def _load_obs_fixtures(rig: Rig, arr: dict[str, np.ndarray], idx: np.ndarray, slots: np.ndarray) -> None:
  core = rig.core
  rig.set_state(slots, arr["obs/joint_pos"][idx], arr["obs/joint_vel"][idx], arr["obs/base_quat"][idx],
                arr["obs/base_lin_vel_w"][idx], arr["obs/base_ang_vel_b"][idx])
  if core.command.shape[1]:
    core.command[slots] = arr["obs/command"][idx][:, : core.command.shape[1]]
  core.adapter.state["prev_action"][slots] = arr["obs/prev_action"][idx]
  core.book["sim_time_ns"][slots, 0] = np.round(arr["obs/time_s"][idx] * 1e9).astype(np.int64)
  core.adapter.observation.reset(slots)


def term_slices(rig: Rig) -> dict[str, tuple[int, int, int]]:
  """name -> (offset of the term's history block, dim, history) in term-major layout."""
  out, off = {}, 0
  for t in rig.b.contract.observation_terms:
    out[t.name] = (off, t.dimension, t.history)
    off += t.dimension * t.history
  return out


def gate_observation(rig: Rig, p: ParityConfig) -> GateResult:
  arr = rig.b.source.arrays
  N = arr["obs/flat"].shape[0]
  n = rig.core.N
  worst: dict[str, float] = {}
  ok = True
  sl = term_slices(rig)
  flat_err = 0.0
  for a in range(0, N, n):
    idx = np.arange(a, min(N, a + n))
    slots = np.arange(idx.size, dtype=np.int32)
    _load_obs_fixtures(rig, arr, idx, slots)
    out = np.zeros((idx.size, rig.compiled.obs_dim), np.float32)
    rig.core.adapter.observe(slots, out)
    for name, (off, dim, H) in sl.items():
      key = f"terms/{name}"
      if key not in arr:
        ok = False
        worst[f"missing:{name}"] = 1.0
        continue
      latest = out[:, off + (H - 1) * dim : off + H * dim] if rig.b.contract.history_order == "oldest_first" else (
        out[:, off : off + dim])
      e, good = _err(latest, arr[key][idx], p.observation_atol, p.observation_rtol)
      worst[name] = max(worst.get(name, 0.0), e)
      ok &= good
    if out.shape == arr["obs/flat"][idx].shape:
      e, good = _err(out, arr["obs/flat"][idx], p.observation_atol, p.observation_rtol)
      flat_err = max(flat_err, e)
      ok &= good
    else:
      ok = False
      flat_err = math.inf
  seq_err, seq_ok = _sequence_error(rig, p)  # frames/history exact over time
  ok &= seq_ok
  bad = [k for k, v in worst.items() if v > p.observation_atol * 10 or k.startswith("missing")]
  why = bad or ("history frames" if not seq_ok else "flattened layout")
  return GateResult("OBS_CONTRACT", ok, {"fixtures": float(N), "flat_max_abs_error": flat_err,
                                         "history_sequence_max_abs_error": seq_err,
                                         **{f"term:{k}": v for k, v in worst.items()}},
                    "" if ok else f"out of tolerance: {why}")


def _sequence_error(rig: Rig, p: ParityConfig) -> tuple[float, bool]:
  """Replay interleaved multi-step sequences (with resets of every other env) through the
  compiled observation kernel; compare with the source's per-env sequence outputs."""
  arr = rig.b.source.arrays
  S, T = arr["seq/reset"].shape
  slots = np.arange(S, dtype=np.int32)
  worst = 0.0
  ok = True
  rig.core.adapter.observation.reset(slots)
  for t in range(T):
    r = np.flatnonzero(arr["seq/reset"][:, t]).astype(np.int32)
    if r.size:
      rig.core.adapter.observation.reset(r)
    rig.set_state(slots, arr["seq/joint_pos"][:, t], arr["seq/joint_vel"][:, t], arr["seq/base_quat"][:, t],
                  arr["seq/base_lin_vel_w"][:, t], arr["seq/base_ang_vel_b"][:, t])
    if rig.core.command.shape[1]:
      rig.core.command[slots] = arr["seq/command"][:, t][:, : rig.core.command.shape[1]]
    rig.core.adapter.state["prev_action"][slots] = arr["seq/prev_action"][:, t]
    rig.core.book["sim_time_ns"][slots, 0] = np.round(arr["seq/time_s"][:, t] * 1e9).astype(np.int64)
    out = np.zeros((S, rig.compiled.obs_dim), np.float32)
    rig.core.adapter.observe(slots, out)
    if out.shape != arr["seq/obs"][:, t].shape:
      return math.inf, False
    e, good = _err(out, arr["seq/obs"][:, t], p.observation_atol, p.observation_rtol)
    worst, ok = max(worst, e), ok and good
  return worst, ok


def gate_history_state_leak(rig: Rig, p: ParityConfig) -> GateResult:
  """Interleaved envs with resets vs isolated per-env references (history + RNN state)."""
  worst, ok = _sequence_error(rig, p)
  arr = rig.b.source.arrays
  S, T = arr["seq/reset"].shape
  # Recurrent policies: interleaved batched inference must equal isolated inference.
  rnn_err = 0.0
  if rig.b.policy.recurrent:
    rt = LocalPolicyRuntime(rig.b.policy, rig.b.executable)
    obs = arr["seq/obs"]
    shp = {s.name: s.shape for s in rig.b.policy.recurrent}
    hb = {k: np.zeros((S, *v), np.float32) for k, v in shp.items()}
    hi = [{k: np.zeros((1, *v), np.float32) for k, v in shp.items()} for _ in range(S)]
    for t in range(T):
      for s in np.flatnonzero(arr["seq/reset"][:, t]):
        for k in hb:
          hb[k][s] = 0.0
          hi[s][k][:] = 0.0
      ob = rt.infer(obs[:, t], hb)
      hb = ob.recurrent
      for s in range(S):
        oi = rt.infer(obs[s : s + 1, t], hi[s])
        hi[s] = oi.recurrent
        rnn_err = max(rnn_err, float(np.abs(oi.action - ob.action[s : s + 1]).max()))
    ok &= rnn_err <= p.action_atol
  return GateResult("STATE_LEAK", ok, {"history_max_abs_error": worst, "recurrent_max_abs_error": rnn_err},
                    "" if ok else "history/recurrent state differs from isolated references")


def gate_normalization(rig: Rig, p: ParityConfig) -> GateResult:
  b = rig.b
  c = b.contract
  msgs = []
  embedded_graph = _graph_embeds_normalization(b.executable) if b.policy.runtime == "onnx" else (
    b.policy.normalization == "embedded")
  if c.normalization == "external" and (embedded_graph or b.policy.normalization == "embedded"):
    msgs.append("double normalization: external normalizer plus a normalizing policy graph")
  if c.normalization == "embedded" and not embedded_graph and b.policy.normalization != "embedded":
    msgs.append("contract says embedded but the exported graph has no normalization")
  if c.normalization in ("none",) and embedded_graph:
    msgs.append("graph normalizes but the contract declares none")
  err = 0.0
  if c.normalization == "external":
    if not b.normalizer:
      msgs.append("external normalization without a normalizer artifact")
    else:
      mean, var = np.asarray(b.normalizer["mean"]), np.asarray(b.normalizer["var"])
      if not (np.isfinite(mean).all() and np.isfinite(var).all()) or np.any(var < 0):
        msgs.append("normalizer means/variances must be finite and variances nonnegative")
      src_eps = b.source.manifest.get("normalization", {}).get("epsilon")
      if src_eps is not None and float(src_eps) != c.normalization_epsilon:
        msgs.append(f"epsilon {c.normalization_epsilon} != source {src_eps}")
      arr = b.source.arrays
      if "norm/raw" in arr:
        got = (arr["norm/raw"] - mean) / np.sqrt(var + c.normalization_epsilon)
        err, good = _err(got, arr["norm/normalized"], p.observation_atol, p.observation_rtol)
        if not good:
          msgs.append("golden raw->normalized inputs differ")
  return GateResult("NORMALIZATION", not msgs, {"golden_max_abs_error": err, "graph_embeds": float(embedded_graph)},
                    "; ".join(msgs))


def _graph_embeds_normalization(exe: bytes) -> bool:
  """ONNX graphs whose observation input first feeds Sub then Div by constants."""
  import onnx

  g = onnx.load_from_string(exe).graph
  inp = g.input[0].name
  consts = {i.name for i in g.initializer}
  first = [n for n in g.node if inp in n.input]
  for n in first:
    if n.op_type == "Sub" and any(x in consts for x in n.input):
      nxt = [m for m in g.node if n.output[0] in m.input]
      if any(m.op_type == "Div" and any(x in consts for x in m.input) for m in nxt):
        return True
  return False


def gate_policy_export(rig: Rig, p: ParityConfig) -> GateResult:
  b = rig.b
  arr = b.source.arrays
  rt = LocalPolicyRuntime(b.policy, b.executable)
  rec = {s.name: np.zeros((arr["policy/obs"].shape[0], *s.shape), np.float32) for s in b.policy.recurrent}
  out = rt.infer(arr["policy/obs"], rec, deterministic=True)
  err, ok = _err(out.action, arr["policy/action"], p.action_atol, 0.0)
  shapes_ok = True
  for size in b.policy.certified_batch_sizes:
    o = np.zeros((size, b.policy.obs_dim), np.float32)
    r = {s.name: np.zeros((size, *s.shape), np.float32) for s in b.policy.recurrent}
    try:
      if rt.infer(o, r, deterministic=True).action.shape != (size, b.policy.action_dim):
        shapes_ok = False
    except Exception:  # noqa: BLE001
      shapes_ok = False
  return GateResult("POLICY_EXPORT", ok and shapes_ok, {"max_action_error": err, "samples": float(len(out.action)),
                                                        "batch_shapes_ok": float(shapes_ok)},
                    "" if ok and shapes_ok else "exported runtime differs from source policy or batch shapes fail")


def gate_control_timing(b: TransferBundle, p: ParityConfig) -> GateResult:
  conv = b.source.manifest["conventions"]
  msgs = []
  if int(conv["decimation"]) != b.group.decimation:
    msgs.append(f"decimation {b.group.decimation} != source {conv['decimation']}")
  if abs(float(conv["dt"]) - b.group.timestep_ns * 1e-9) > 1e-12:
    msgs.append(f"timestep {b.group.timestep_ns} ns != source {conv['dt']} s")
  rig = Rig(b, 2)
  s = np.arange(2, dtype=np.int32)
  A = rig.compiled.num_actions
  drift = 0
  evals_before = rig.core.adapter.state["effort_evals"][s, 0].copy()
  for k in range(100):
    out = rig.core.advance_control(s, np.zeros((2, A), np.float32))
    if not (out.completed_substeps == b.group.decimation).all():
      msgs.append("a policy step ran a different number of substeps")
      break
    want = (k + 1) * b.group.timestep_ns * b.group.decimation
    drift = max(drift, int(np.abs(out.sim_time_ns - want).max()))
  evals = rig.core.adapter.state["effort_evals"][s, 0] - evals_before
  if rig.compiled.controller_cfg["kind"] in (2, 3) and not (evals == 100 * b.group.decimation * A).all():
    msgs.append("controller was not evaluated at every physics substep")
  ticks = b.source.arrays["timing/policy_tick"]
  want_ticks = (np.arange(ticks.size) % b.group.decimation == 0)
  if not np.array_equal(ticks, want_ticks):
    msgs.append("policy update schedule differs from the source")
  if drift:
    msgs.append(f"phase drift {drift} ns")
  return GateResult("CONTROL_RATE", not msgs, {"drift_ns": float(drift), "policy_updates": 100.0}, "; ".join(msgs))


def gate_actuator(rig: Rig, p: ParityConfig) -> GateResult:
  arr = rig.b.source.arrays
  conv = rig.b.source.manifest["conventions"]
  c = rig.b.contract
  msgs = []
  if conv["actuator"]["kind"] != c.actuator.kind:
    msgs.append(f"actuator kind {c.actuator.kind} != source {conv['actuator']['kind']}")
  core = rig.core
  n = core.N
  M = arr["act/target"].shape[0]
  sign = np.asarray([rig.sign.get(nm, 1.0) for nm in rig.src_order])
  worst, rng_ = 0.0, np.abs(arr["act/effort"]).max(axis=0)
  tol = np.maximum(p.effort_atol_nm, p.effort_rtol * rng_)
  bad = 0
  for a in range(0, M, n):
    idx = np.arange(a, min(M, a + n))
    slots = np.arange(idx.size, dtype=np.int32)
    rig.set_state(slots, arr["act/q"][idx], arr["act/dq"][idx])
    core.adapter.state["targets"][slots] = sign * arr["act/target"][idx]
    core.adapter.state["physics_delay_count"][slots] = 0
    core.adapter.controller.before_substep(slots, rig.b.group.timestep_ns * 1e-9)
    eff = core.adapter.state["effort"][slots] if c.actuator.kind != "position" else np.full((idx.size, len(sign)), np.nan)
    got = sign * eff  # back to the source convention
    diff = np.abs(got - arr["act/effort"][idx])
    diff = np.where(np.isfinite(diff), diff, np.inf)
    worst = max(worst, float(diff.max()))
    bad += int(np.sum(diff > tol))
  if bad:
    msgs.append(f"{bad} effort samples outside max({p.effort_atol_nm} Nm, {p.effort_rtol:.0%} of range)")
  # EMA/filter impulse response through the compiled action processor.
  imp = arr["act/impulse_action"]
  core.adapter.actions.reset(np.array([0], np.int32))
  core.adapter.state["ema"][0] = 0.0
  ferr = 0.0
  for t in range(imp.shape[0]):
    core.adapter.actions.process(np.array([0], np.int32), imp[t : t + 1])
    got = sign * core.adapter.state["targets"][0]
    ferr = max(ferr, float(np.abs(got - arr["act/impulse_target"][t]).max()))
  if ferr > p.filter_atol:
    msgs.append(f"filter impulse error {ferr:.3g} > {p.filter_atol}")
  return GateResult("ACTUATOR", not msgs, {"max_effort_error": worst, "filter_max_error": ferr,
                                           "samples": float(M)}, "; ".join(msgs))


def gate_inertial(rig: Rig, p: ParityConfig) -> GateResult:
  m = rig.model
  props = rig.b.source.manifest["properties"]
  mass_e = com_e = inertia_e = 0.0
  bad = []
  for body in props["bodies"]:
    bid = rig.meta.bodies.get(body["name"])
    if bid is None:
      bad.append(body["name"])
      continue
    mass_e = max(mass_e, abs(m.body_mass[bid] - body["mass"]) / max(body["mass"], 1e-12))
    com_e = max(com_e, float(np.linalg.norm(m.body_ipos[bid] - np.asarray(body["ipos"]))))
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = body["inertia"]
    full = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    ev_src = np.sort(np.linalg.eigvalsh(full))
    if np.any(ev_src <= 0) or ev_src[0] + ev_src[1] < ev_src[2] * (1 - 1e-9):
      bad.append(f"{body['name']}:not SPD/triangle")
    ev_tgt = np.sort(m.body_inertia[bid])
    inertia_e = max(inertia_e, float(np.max(np.abs(ev_tgt - ev_src) / np.maximum(ev_src, 1e-12))))
  ok = not bad and mass_e <= p.mass_rtol and com_e <= p.com_atol_m and inertia_e <= p.inertia_rtol
  return GateResult("INERTIAL", ok, {"mass_rel": mass_e, "com_m": com_e, "inertia_eig_rel": inertia_e},
                    "" if ok else f"inertial tables differ (bad={bad})")


def gate_kinematics(rig: Rig, p: ParityConfig) -> GateResult:
  arr = rig.b.source.arrays
  names = rig.b.source.manifest["body_names"]
  m = rig.model
  d = mujoco.MjData(m)
  tmax = amax = 0.0
  for k in range(arr["kin/joint_pos"].shape[0]):
    mujoco.mj_resetData(m, d)
    d.qpos[rig.base.qposadr : rig.base.qposadr + 7] = [0, 0, 0.5, 1, 0, 0, 0]
    for i, name in enumerate(rig.src_order):
      j = rig.meta.joints[name]
      d.qpos[j.qposadr] = rig.sign.get(name, 1.0) * arr["kin/joint_pos"][k, i]
    mujoco.mj_kinematics(m, d)
    for bi, bn in enumerate(names):
      b = rig.meta.bodies[bn]
      tmax = max(tmax, float(np.linalg.norm(d.xpos[b] - arr["kin/body_pos"][k, bi])))
      q1, q2 = d.xquat[b], arr["kin/body_quat"][k, bi]
      ang = 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(q1, q2))))))
      amax = max(amax, ang)
  ok = tmax <= p.pose_atol_m and amax <= p.angle_atol_deg
  return GateResult("ASSET_FRAME", ok, {"max_translation_m": tmax, "max_rotation_deg": amax},
                    "" if ok else "link poses differ beyond tolerance")


def gate_contact(rig: Rig, p: ParityConfig) -> GateResult:
  conv = rig.b.source.manifest["conventions"]
  m = rig.model
  g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, conv["contact_geom"])
  if g < 0:
    return GateResult("CONTACT", False, {}, f"contact geom {conv['contact_geom']} missing")
  tgt = contact_fixtures(m.geom_friction[g].copy(), m.geom_solref[g].copy(), m.geom_solimp[g].copy(),
                         rig.b.group.timestep_ns * 1e-9)
  arr = rig.b.source.arrays
  drop = float(np.max(np.abs(tgt["contact/drop_z"] - arr["contact/drop_z"])) / 0.27)
  so, st = float(arr["contact/slide_onset_force"][0]), float(tgt["contact/slide_onset_force"][0])
  slide = abs(st - so) / max(abs(so), 1e-9) if math.isfinite(so) and math.isfinite(st) else math.inf
  pen_s, pen_t = float(arr["contact/rest_penetration"][0]), float(tgt["contact/rest_penetration"][0])
  rest = abs(pen_t - pen_s) / max(abs(pen_s), 1e-6)
  ok = drop <= p.contact_curve_rtol and slide <= p.slide_rtol and rest <= p.contact_curve_rtol
  return GateResult("CONTACT", ok, {"drop_curve_norm_error": drop, "slide_onset_rel": slide,
                                    "rest_penetration_rel": rest},
                    "" if ok else "contact response differs from the source fixtures")


def gate_physics_config(rig: Rig, p: ParityConfig) -> GateResult:
  m = rig.model
  props = rig.b.source.manifest["properties"]
  opt = props["option"]
  msgs = []
  if abs(m.opt.timestep - opt["timestep"]) > 1e-12:
    msgs.append("timestep differs")
  if not np.allclose(m.opt.gravity, opt["gravity"], atol=0, rtol=0):
    msgs.append("gravity differs")
  if int(m.opt.iterations) != int(opt["iterations"]):
    msgs.append("solver iteration budget differs")
  for j in props["joints"]:
    tj = rig.meta.joints.get(j["name"])
    if tj is None or j["type"] != tj.type:
      msgs.append(f"joint {j['name']} missing or of another type")
      continue
    d = tj.dofadr
    sg = rig.sign.get(j["name"], 1.0)
    for key, val in (("damping", m.dof_damping[d]), ("armature", m.dof_armature[d]),
                     ("frictionloss", m.dof_frictionloss[d])):
      if abs(val - j[key]) > 1e-12 * max(1.0, abs(j[key])):
        msgs.append(f"{j['name']}.{key} differs")
    if j["range"] is not None:
      lo, hi = sorted(sg * np.asarray(j["range"]))
      if abs(lo - m.jnt_range[tj.id, 0]) > 1e-12 or abs(hi - m.jnt_range[tj.id, 1]) > 1e-12:
        msgs.append(f"{j['name']} limits differ")
  # Initialization replay: the target reset state expressed in source conventions.
  arr = rig.b.source.arrays
  d_ = mujoco.MjData(m)
  kf = rig.b.group.initial_keyframe
  if kf >= 0:
    mujoco.mj_resetDataKeyframe(m, d_, kf)
  else:
    mujoco.mj_resetData(m, d_)
  q = np.asarray([rig.sign.get(n, 1.0) * d_.qpos[rig.meta.joints[n].qposadr] for n in rig.src_order])
  init_err = float(np.max(np.abs(q - arr["init/qpos_joints"])))
  base_err = float(np.max(np.abs(d_.qpos[rig.base.qposadr : rig.base.qposadr + 7] - arr["init/base_qpos"])))
  if max(init_err, base_err) > p.initial_state_atol:
    msgs.append(f"initial state differs ({max(init_err, base_err):.3g})")
  return GateResult("PHYSICS_CONFIG", not msgs, {"init_max_abs_error": max(init_err, base_err)}, "; ".join(msgs))


def run_gates(b: TransferBundle, p: ParityConfig | None = None, rig_slots: int = 128,
              static: bool = True, dynamics: bool = True) -> GateReport:
  p = p or ParityConfig()
  results: list[GateResult] = []
  results.append(gate_xml_abi(b, p))
  try:
    rig = Rig(b, rig_slots)
  except (AdapterError, Exception) as exc:  # noqa: BLE001 - admission failure blocks every model gate
    msg = f"target rig could not be built: {exc}"
    for code in ("JOINT_MAPPING", "OBS_CONTRACT", "NORMALIZATION", "STATE_LEAK", "POLICY_EXPORT", "CONTROL_RATE",
                 "ACTUATOR", "INERTIAL", "ASSET_FRAME", "CONTACT", "PHYSICS_CONFIG"):
      results.append(GateResult(code, False, {}, msg))
    return GateReport(b.digest, results, p.model_dump(mode="json"))
  if static:
    for fn in (gate_joint_map, gate_observation, gate_normalization, gate_history_state_leak, gate_policy_export):
      results.append(_safe(fn, rig, p))
    results.append(_safe_b(gate_control_timing, b, p))
    results.append(_safe(gate_actuator, rig, p))
  if dynamics:
    for fn in (gate_inertial, gate_kinematics, gate_contact, gate_physics_config):
      results.append(_safe(fn, rig, p))
  return GateReport(b.digest, results, p.model_dump(mode="json"))


def _safe(fn, rig: Rig, p: ParityConfig) -> GateResult:
  code = {"gate_joint_map": "JOINT_MAPPING", "gate_observation": "OBS_CONTRACT", "gate_normalization": "NORMALIZATION",
          "gate_history_state_leak": "STATE_LEAK", "gate_policy_export": "POLICY_EXPORT", "gate_actuator": "ACTUATOR",
          "gate_inertial": "INERTIAL", "gate_kinematics": "ASSET_FRAME", "gate_contact": "CONTACT",
          "gate_physics_config": "PHYSICS_CONFIG"}[fn.__name__]
  try:
    return fn(rig, p)
  except Exception as exc:  # noqa: BLE001 - an exception is a failed gate, never a pass
    return GateResult(code, False, {}, f"gate error: {type(exc).__name__}: {exc}")


def _safe_b(fn, b: TransferBundle, p: ParityConfig) -> GateResult:
  try:
    return fn(b, p)
  except Exception as exc:  # noqa: BLE001
    return GateResult("CONTROL_RATE", False, {}, f"gate error: {type(exc).__name__}: {exc}")


def bundle_summary(r: GateReport) -> dict[str, Any]:
  return {x.code: {"passed": x.passed, **x.metrics} for x in r.results}
