# SPDX-License-Identifier: Apache-2.0
"""MJBatchCore semantics against scalar MuJoCo references (design §6.2, §13.1).

Parity tests are bitwise: the core composes on unmodified MJBatch and must reproduce
exactly what mj_step on a scalar MjData does under the same inputs.
"""

from __future__ import annotations

import copy
import os

import mujoco
import numpy as np
import pytest

from raysimbatch import seeds
from raysimbatch.protocol import FaultCode, Flag, SlotState, SlotStatus
from raysimbatch.testing import fixtures as F
from raysimbatch.worker.core import CoreError
from raysimbatch.worker.randomization import ClosureGuard, compile_rules

PARITY_STEPS = int(os.environ.get("RSB_PARITY_CONTROL_STEPS", "300"))
FULL = int(mujoco.mjtState.mjSTATE_INTEGRATION)

EQ_CARTPOLE = F.asset("cartpole.xml").replace(
  b"</worldbody>",
  b'<body name="ball" pos="1 1 1"><freejoint/><geom type="sphere" size=".05" mass=".2" contype="0" '
  b'conaffinity="0"/></body></worldbody><equality><connect body1="ball" body2="world" anchor="1 1 1.2"/></equality>',
)


def start(core, n=None, episode=0):
  n = core.N if n is None else n
  s = np.arange(n, dtype=np.int32)
  ids = s.astype(np.uint64)
  ep = np.full(n, episode, np.uint64)
  r = core.reset_exact(s, ids, ep, (ids, ep))
  assert r.ok.all(), r.message
  return s, r


def reference_models(core):
  """One scalar model per slot carrying that slot's expanded parameter rows."""
  models = []
  for i in range(core.N):
    m = copy.copy(core.model)
    for f in core.expanded_fields:
      row = core.h.expand(f)[i]
      target = getattr(m.opt, f) if hasattr(m.opt, f) else getattr(m, f)
      target[...] = row.reshape(target.shape)
    if core.randomizer.needs_refresh:
      mujoco.mj_setConst(m, mujoco.MjData(m))
    models.append(m)
  return models


def pd_ctrl(core, q, dq, target):
  """numpy reference of the native PD/torque controller for these fixtures."""
  arr = core.compiled.controller_arrays
  kind = core.compiled.controller_cfg["kind"]
  qa, da, ci, gear = arr["qpos_adr"], arr["dof_adr"], arr["ctrl_idx"], arr["gear"]
  lim = np.where(arr["effort_limit"] > 0, arr["effort_limit"], np.inf)
  if kind == 0:
    tau = target
  else:
    tau = arr["kp"] * (target - q[qa]) - arr["kd"] * dq[da] + 0.0
  ctrl = np.zeros(core.meta.nu)
  ctrl[ci] = np.clip(tau, -lim, lim) / gear
  return ctrl


def action_targets(core, a):
  c = core.compiled
  sc, off, sg = c.action_arrays["scale"], c.action_arrays["offset"], c.action_arrays["sign"]
  clip = c.action_cfg["clip"]
  x = np.clip(a.astype(np.float64), -clip, clip) * sc + off
  return x * sg


@pytest.mark.parametrize("name", ["cartpole", "quadruped", "reacher"])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("each_substep", [True, False])
def test_bitwise_parity_with_scalar_mujoco(name, threads, each_substep):
  """With control-rate health, torque fixtures step fused (AM-2): still mj_step exactly."""
  fx = F.no_termination(F.FIXTURES[name](max_control_steps=10**9))
  health = fx.group.health.model_copy(update={"check_each_substep": each_substep})
  core = fx.core(num_slots=6, threads=threads, health=health)
  assert core.stepping == ("fused" if not each_substep and name != "quadruped" else "per_substep")
  s, _ = start(core)
  models = reference_models(core)
  datas = [mujoco.MjData(m) for m in models]
  for i, d in enumerate(datas):
    mujoco.mj_setState(models[i], d, core.h.state[i], FULL)
  rng = np.random.default_rng(0)
  for step in range(PARITY_STEPS):
    act = rng.uniform(-0.3, 0.3, (6, core.A)).astype(np.float32)
    out = core.advance_control(s, act)
    assert (out.status == SlotStatus.OK).all(), (step, out.status, out.fault)
    for i, (m, d) in enumerate(zip(models, datas, strict=True)):
      target = action_targets(core, act[i])
      for _ in range(fx.group.decimation):
        d.ctrl[:] = pd_ctrl(core, d.qpos, d.qvel, target)
        mujoco.mj_step(m, d)
    ref = np.stack([_state(models[i], datas[i]) for i in range(6)])
    np.testing.assert_array_equal(core.h.state, ref, err_msg=f"step {step}")


def _state(m, d):
  out = np.empty(mujoco.mj_stateSize(m, FULL))
  mujoco.mj_getState(m, d, out, FULL)
  return out


def test_sparse_reset_leaves_other_slots_bitwise_untouched():
  fx = F.quadruped(max_control_steps=10**9)
  a, b = fx.core(num_slots=8, threads=3), fx.core(num_slots=8, threads=3)
  sa, _ = start(a)
  start(b)
  rng = np.random.default_rng(1)
  params_before = {f: b.h.expand(f).copy() for f in b.expanded_fields}
  for k in range(40):
    act = rng.uniform(-0.5, 0.5, (8, a.A)).astype(np.float32)
    a.advance_control(sa, act)
    b.advance_control(sa, act)
    if k == 20:
      reset = np.array([3, 6], np.int32)
      ep = np.array([1, 1], np.uint64)
      r = a.reset_exact(reset, reset.astype(np.uint64), ep, (reset.astype(np.uint64), ep))
      assert r.ok.all()
      keep = np.setdiff1d(np.arange(8), reset)
      np.testing.assert_array_equal(a.h.state[keep], b.h.state[keep])
      # Only the reset slots' randomized/derived parameter rows changed.
      for f in a.expanded_fields:
        np.testing.assert_array_equal(a.h.expand(f)[keep], params_before[f][keep], err_msg=f)
        assert not np.array_equal(a.h.expand(f)[reset], params_before[f][reset]) or f == "geom_friction" and False
  keep = np.setdiff1d(np.arange(8), [3, 6])
  np.testing.assert_array_equal(a.h.state[keep], b.h.state[keep])


def eq_core(n=8, threads=3):
  fx = F.cartpole(max_control_steps=10**9)
  fx = F.Fixture("cartpole_eq", EQ_CARTPOLE, fx.group, fx.contract, fx.task_doc, fx.reset_distribution)
  return fx.core(num_slots=n, threads=threads)


@pytest.mark.parametrize("threads", [1, 3])
def test_native_failures_are_per_slot_and_never_replay_successes(threads):
  a, b = eq_core(threads=threads), eq_core(threads=threads)
  s, _ = start(a)
  start(b)
  rng = np.random.default_rng(2)
  for _ in range(5):
    act = rng.uniform(-1, 1, (8, 1)).astype(np.float32)
    a.advance_control(s, act)
    b.advance_control(s, act)
  before = a.h.state.copy()
  ctrl_before = a.views["ctrl"].copy()
  a.h.expand("eq_type")[[1, 4, 6]] = 99  # fault injection on three slots
  act = rng.uniform(-1, 1, (8, 1)).astype(np.float32)
  out = a.advance_control(s, act)
  b.advance_control(s, act)
  bad = np.array([1, 4, 6])
  good = np.setdiff1d(s, bad)
  assert list(np.flatnonzero(out.status == SlotStatus.NATIVE_FAILURE)) == [1, 4, 6]
  assert (out.fault[bad] == FaultCode.MUJOCO_ERROR).all()
  assert (out.completed_substeps[bad] == 0).all() and (out.failed_substep[bad] == 0).all()
  assert (out.flags[bad] & Flag.ABORTED).all() and not (out.flags[bad] & Flag.VALID).any()
  assert (out.status[good] == SlotStatus.OK).all()
  assert (out.completed_substeps[good] == 5).all()
  np.testing.assert_array_equal(a.h.state[bad], before[bad])  # failed slots kept their state
  np.testing.assert_array_equal(a.h.state[good], b.h.state[good])  # others equal a clean run
  np.testing.assert_array_equal(a.views["ctrl"][bad], ctrl_before[bad])  # pending ctrl cancelled
  assert (a.book["slot_state"][bad] == SlotState.FAILED).all()
  with pytest.raises(CoreError):
    a.advance_control(bad.astype(np.int32), np.zeros((3, 1), np.float32))
  # Repair and reset: the reset is exact (no stale ctrl lands after mj_resetData).
  a.h.expand("eq_type")[bad] = a.model.eq_type
  ep = np.ones(3, np.uint64)
  r = a.reset_exact(bad.astype(np.int32), bad.astype(np.uint64), ep, (bad.astype(np.uint64), ep))
  assert r.ok.all()
  fresh = eq_core(threads=threads)
  rr = fresh.reset_exact(bad.astype(np.int32), bad.astype(np.uint64), ep, (bad.astype(np.uint64), ep))
  np.testing.assert_array_equal(a.h.state[bad], fresh.h.state[bad])
  np.testing.assert_array_equal(r.observation, rr.observation)


def test_health_failures_abort_only_the_affected_slot():
  core = F.cartpole(max_control_steps=10**9).core(num_slots=4, threads=2)
  s, _ = start(core)
  off, _ = core.layout["qvel"]
  core.h.state[1, off] = 1e4  # beyond max_abs_qvel
  core.h.state[2, off + 1] = np.nan  # MuJoCo auto-resets and raises BADQVEL
  out = core.advance_control(s, np.zeros((4, 1), np.float32))
  assert out.status[1] == SlotStatus.HEALTH_FAILURE and out.fault[1] == FaultCode.VELOCITY_BOUND
  assert out.status[2] == SlotStatus.HEALTH_FAILURE and out.fault[2] == FaultCode.FATAL_WARNING
  assert out.failed_substep[1] == 0 and out.completed_substeps[1] == 1
  assert (out.status[[0, 3]] == SlotStatus.OK).all()
  assert (out.flags[[1, 2]] & Flag.ABORTED).all()


def test_terminal_observation_is_captured_before_reset():
  core = F.cartpole().core(num_slots=4, threads=2)
  s, r = start(core)
  obs = r.observation
  for _ in range(400):
    out = core.advance_control(s, np.full((4, 1), 0.8, np.float32))
    if (out.status == SlotStatus.TERMINATED).any():
      break
  term = np.flatnonzero(out.status == SlotStatus.TERMINATED)
  assert term.size
  j = term[0]
  q, dq = core.views["qpos"][j], core.views["qvel"][j]
  np.testing.assert_array_equal(out.observation[j], np.array([q[0], q[1], dq[0], dq[1]], np.float32))
  assert abs(q[1]) > 0.6 or abs(q[0]) > 2.4
  assert out.flags[j] & Flag.TERMINATED and out.flags[j] & Flag.VALID
  assert out.metrics[j, 0] == 1.0  # fell
  assert core.book["slot_state"][j] == SlotState.TERMINAL
  ep = np.array([1], np.uint64)
  rr = core.reset_exact(np.array([j], np.int32), np.array([j], np.uint64), ep, (np.array([j], np.uint64), ep))
  assert not np.array_equal(rr.observation[0], out.observation[j])
  assert core.book["control_seq"][j, 0] == 0 and core.book["episode_id"][j] == 1
  del obs


def test_time_limit_truncates_with_bootstrap_eligibility():
  core = F.cartpole(max_control_steps=7).core(num_slots=3, threads=1)
  s, r = start(core)
  obs = r.observation
  for k in range(7):  # noqa: B007 - k is checked after the loop
    out = core.advance_control(s, F.cartpole_policy(obs))
    obs = out.observation
  assert (out.status == SlotStatus.TRUNCATED).all() and k == 6
  assert (out.flags & Flag.TRUNCATED).all() and not (out.flags & Flag.TERMINATED).any()
  assert (out.control_seq == 7).all()


@pytest.mark.parametrize("target", ["same_core", "other_slot", "other_core"])
def test_snapshot_restore_replays_bitwise_including_warmstart(target):
  fx = F.quadruped(history=3, max_control_steps=10**9)
  core = fx.core(num_slots=10, threads=3)
  s, _ = start(core)
  rng = np.random.default_rng(3)
  for _ in range(25):
    core.advance_control(s, rng.uniform(-1, 1, (10, core.A)).astype(np.float32))
  src = np.array([2, 5], np.int32)
  snap = core.save_state(src)
  acts = [rng.uniform(-1, 1, (2, core.A)).astype(np.float32) for _ in range(15)]
  first = []
  for a in acts:
    out = core.advance_control(src, a)
    first.append((out.observation.copy(), out.reward.copy(), core.h.state[src].copy()))
  if target == "same_core":
    dst, dcore = src, core
  elif target == "other_slot":
    dst, dcore = np.array([7, 9], np.int32), core
  else:
    dcore = fx.core(num_slots=10, threads=2)
    start(dcore)
    dst = np.array([0, 8], np.int32)
  assert dcore.load_state(dst, snap).all()
  for (o, rw, st), a in zip(first, acts, strict=True):
    out = dcore.advance_control(dst, a)
    np.testing.assert_array_equal(out.observation, o)
    np.testing.assert_array_equal(out.reward, rw)
    np.testing.assert_array_equal(dcore.h.state[dst], st)


def test_jacobian_queries_do_not_perturb_the_trajectory():
  fx = F.quadruped(max_control_steps=10**9)
  a, b = fx.core(num_slots=4, threads=2), fx.core(num_slots=4, threads=2)
  s, _ = start(a)
  start(b)
  site = np.array([a.meta.sites["imu"]], np.int32)
  rng = np.random.default_rng(4)
  for _ in range(30):
    act = rng.uniform(-1, 1, (4, a.A)).astype(np.float32)
    a.advance_control(s, act)
    b.advance_control(s, act)
    jacp, jacr = a.query_jacobians(s, site)
  np.testing.assert_array_equal(a.h.state, b.h.state)
  m = reference_models(a)[1]
  d = mujoco.MjData(m)
  mujoco.mj_setState(m, d, a.h.state[1], FULL)
  mujoco.mj_forward(m, d)
  ref_p, ref_r = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  mujoco.mj_jacSite(m, d, ref_p, ref_r, int(site[0]))
  np.testing.assert_allclose(jacp[1, 0], ref_p, atol=1e-12)
  np.testing.assert_allclose(jacr[1, 0], ref_r, atol=1e-12)


def test_randomization_is_keyed_deterministic_and_never_compounds():
  fx = F.quadruped()
  a, b = fx.core(num_slots=4, threads=1), fx.core(num_slots=4, threads=1)
  start(a)
  start(b)
  mass0 = a.randomizer.baseline["body_mass"]
  np.testing.assert_array_equal(a.h.expand("body_mass"), b.h.expand("body_mass"))
  slot = np.array([2], np.int32)
  seen = []
  for ep in (1, 2, 1):
    e = np.array([ep], np.uint64)
    r = a.reset_exact(slot, np.array([2], np.uint64), e, (np.array([2], np.uint64), e))
    row = a.h.expand("body_mass")[2].copy()
    ratio = row[1:] / mass0[1:]
    assert np.all((ratio >= 0.8) & (ratio < 1.2))  # always baseline * U[0.8, 1.2)
    seen.append((row, r.randomization_hash[0].copy()))
  np.testing.assert_array_equal(seen[0][0], seen[2][0])  # same key -> same draw
  np.testing.assert_array_equal(seen[0][1], seen[2][1])
  assert not np.array_equal(seen[0][0], seen[1][0])
  # Derived constants were refreshed for the slot only and match mj_setConst.
  m = copy.copy(a.model)
  m.body_mass[:] = seen[2][0]
  m.geom_friction[:] = a.h.expand("geom_friction")[2]
  mujoco.mj_setConst(m, mujoco.MjData(m))
  np.testing.assert_array_equal(a.h.expand("body_subtreemass")[2], m.body_subtreemass)


def test_closure_guard_refuses_derived_field_growth():
  fx = F.quadruped()
  model = fx.model()
  rules = compile_rules(fx.randomization, model)
  guard = ClosureGuard(model, rules)
  mass = model.body_mass.copy()
  mass[1:] *= 1.1
  changed = guard.changed({"body_mass": mass, "geom_friction": model.geom_friction.copy()})
  assert "body_subtreemass" in changed
  assert guard.closure == frozenset()  # nothing prewarmed: this draw would grow it
  core = fx.core(num_slots=2, threads=1)
  assert changed <= core.randomizer.closure
  core.randomizer.closure = frozenset()  # simulate an incomplete prewarm
  e = np.zeros(2, np.uint64)
  ids = np.arange(2, dtype=np.int32)
  r = core.reset_exact(ids, ids.astype(np.uint64), e, (ids.astype(np.uint64), e))
  assert not r.ok.any() and (r.fault == FaultCode.RANDOMIZATION_FAILED).all()


def test_reset_draws_depend_only_on_seed_identity():
  fx = F.reacher()
  a, b = fx.core(num_slots=4, threads=1), fx.core(num_slots=6, threads=2)
  ids_a = np.array([0, 1], np.int32)
  ids_b = np.array([4, 5], np.int32)  # different placement, same logical identities
  key_a = (np.array([11, 12], np.uint64), np.array([3, 3], np.uint64))
  ra = a.reset_exact(ids_a, key_a[0], key_a[1], key_a)
  rb = b.reset_exact(ids_b, key_a[0], key_a[1], key_a)
  np.testing.assert_array_equal(ra.observation, rb.observation)
  np.testing.assert_array_equal(a.h.state[ids_a], b.h.state[ids_b])


def test_invalid_commands_are_rejected_before_mutation():
  core = F.cartpole().core(num_slots=4, threads=1)
  s, _ = start(core)
  before = core.h.state.copy()
  with pytest.raises(CoreError):
    core.advance_control(s, np.full((4, 1), np.nan, np.float32))
  with pytest.raises(CoreError):
    core.advance_control(s, np.zeros((4, 2), np.float32))
  for bad in ([2, 1], [1, 1], [-1, 2]):  # unsorted, duplicate, negative
    with pytest.raises(CoreError, match="sorted, unique"):
      core.advance_control(np.array(bad, np.int32), np.zeros((2, 1), np.float32))
  np.testing.assert_array_equal(core.h.state, before)
  assert (core.book["slot_state"] == SlotState.READY).all()


def test_seed_keys_match_numpy_philox():
  key = seeds.derive_key(123, "ab" * 32, 7, 9, "domain")
  keys = np.array([key], np.uint64)
  u = seeds.uniform(keys, 3, 4, "domain")
  for i in range(4):
    v0 = seeds.numpy_block(key, 3, i, "domain")[0]
    assert u[0, i] == (v0 >> 11) * 2.0**-53
