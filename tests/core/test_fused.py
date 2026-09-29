# SPDX-License-Identifier: Apache-2.0
"""Fused stepping (AM-2) must be indistinguishable from substep-by-substep stepping.

Two cores with the same configuration (health checked at control rate) step the same
slots with the same actions, one per substep and one fused. Every result field and every
piece of per-slot state (MJBatch state rows and input fields, controller counters and
energy, health baselines, task state, bookkeeping) must stay bitwise equal: episodes
that terminate mid-interval and natively failing slots are replayed from their
pre-interval state, so they report the substep, state and results the per-substep path
would. If this ever diverges, a run's results depend on an implementation detail.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from raysimbatch.adapter import compile_adapter
from raysimbatch.config.models import HealthConfig
from raysimbatch.protocol import SlotState, SlotStatus
from raysimbatch.testing import fixtures as F

CONTROL_RATE = {"check_each_substep": False}
RESULT_FIELDS = ("status", "fault", "completed_substeps", "failed_substep", "warning_delta", "reward", "components",
                 "flags", "observation", "action", "applied_action", "control_seq", "sim_time_ns", "metrics")
INPUT_VIEWS = ("qpos", "qvel", "ctrl", "time", "warning")


def pair(fx, n, threads, **spec):
  health = fx.group.health.model_copy(update=CONTROL_RATE)
  a = fx.core(num_slots=n, threads=threads, health=health, fuse_substeps=False, **spec)
  b = fx.core(num_slots=n, threads=threads, health=health, fuse_substeps=True, **spec)
  assert (a.stepping, b.stepping) == ("per_substep", "fused")
  return a, b


def reset(cores, slots, episode):
  ids = slots.astype(np.uint64)
  ep = np.full(slots.size, episode, np.uint64)
  outs = [c.reset_exact(slots, ids, ep, (ids, ep)) for c in cores]
  np.testing.assert_array_equal(outs[0].observation, outs[1].observation)
  return outs[0]


def assert_same(a, b, oa, ob):
  for f in RESULT_FIELDS:
    np.testing.assert_array_equal(getattr(oa, f), getattr(ob, f), err_msg=f)
  np.testing.assert_array_equal(a.h.state, b.h.state, err_msg="state rows")
  for v in INPUT_VIEWS:
    np.testing.assert_array_equal(a.views[v], b.views[v], err_msg=v)
  live = oa.slots[oa.status != SlotStatus.NATIVE_FAILURE]  # derived fields of a failed slot are not read
  for v in ("qacc", "xpos"):
    np.testing.assert_array_equal(a.h.bind(v)[live], b.h.bind(v)[live], err_msg=v)
  for k in a.adapter.state:
    np.testing.assert_array_equal(a.adapter.state[k], b.adapter.state[k], err_msg=f"adapter/{k}")
  for k in a.health_state:
    np.testing.assert_array_equal(a.health_state[k], b.health_state[k], err_msg=f"health/{k}")
  np.testing.assert_array_equal(a.task_state["state"], b.task_state["state"], err_msg="task state")
  np.testing.assert_array_equal(a.command, b.command, err_msg="command")
  for k in a.book:
    if k != "episode_started_ns":  # wall clock at reset
      np.testing.assert_array_equal(a.book[k], b.book[k], err_msg=f"book/{k}")


def drive(a, b, steps, rng, scale=1.0, on_step=None):
  """Step both cores with the same random actions, resetting ended slots in both."""
  s = np.arange(a.N, dtype=np.int32)
  reset((a, b), s, 0)
  episode = 1
  outs = []
  for t in range(steps):
    act = rng.uniform(-scale, scale, (s.size, a.A)).astype(np.float32)
    oa, ob = a.advance_control(s, act), b.advance_control(s, act)
    assert_same(a, b, oa, ob)
    outs.append(oa)
    ended = s[(oa.flags != 0) & (oa.status != SlotStatus.OK)]
    if ended.size:
      reset((a, b), ended, episode)
      episode += 1
    if on_step:
      on_step(t)
  return outs


@pytest.mark.parametrize("threads", [1, 3])
def test_cartpole_terminations_mid_interval_are_replayed_exactly(threads):
  a, b = pair(F.cartpole(), 16, threads)
  d = a.spec.decimation
  outs = drive(a, b, 250, np.random.default_rng(7))
  done = [(o.completed_substeps[o.status == SlotStatus.TERMINATED]) for o in outs]
  done = np.concatenate(done)
  # Both fused outcomes are exercised: terminations before the last substep (replayed)
  # and on it (finished from the history rows).
  assert (done < d).sum() > 10 and (done == d).sum() > 0, np.bincount(done)
  assert b.replayed_slots == (done < d).sum() and b.fused_intervals == 250


@pytest.mark.parametrize("threads", [1, 3])
def test_reacher_needs_no_replay_and_matches_including_energy(threads):
  a, b = pair(F.reacher(), 12, threads)
  drive(a, b, 450, np.random.default_rng(3))  # crosses max_control_steps: truncations too
  assert b.replayed_slots == 0 and b.fused_intervals == 450
  assert (b.adapter.state["energy"] > 0).all() and (b.adapter.state["effort_evals"] > 0).all()



def test_saturating_torques_match():
  """Efforts clipped at the limit: saturation counters and clipped energy terms."""
  fx = F.reacher()
  act = fx.contract.actuator.model_copy(update={"effort_limit": (0.4, 0.4, 0.4)})
  fx = F.Fixture(fx.name, fx.xml, fx.group, fx.contract.model_copy(update={"actuator": act}), fx.task_doc,
                 fx.reset_distribution)
  a, b = pair(fx, 8, 2)
  drive(a, b, 120, np.random.default_rng(4))
  sat, evals = b.adapter.state["saturated"], b.adapter.state["effort_evals"]
  assert (sat > 0).all() and (sat < evals).all()


def test_position_servos_match():
  """Targets written to native position actuators: no effort terms, constant ctrl."""
  fx = F.reacher()
  xml = fx.xml
  for j in ("j1", "j2", "j3"):
    xml = xml.replace(b'<motor name="%s" joint="%s" gear="1"/>' % (j.encode(), j.encode()),
                      b'<position name="%s" joint="%s" kp="5"/>' % (j.encode(), j.encode()))
  act = fx.contract.actuator.model_copy(update={"kind": "position", "kp": (5.0, 5.0, 5.0), "effort_limit": ()})
  fx = F.Fixture(fx.name, xml, fx.group, fx.contract.model_copy(update={"actuator": act}), fx.task_doc,
                 fx.reset_distribution)
  a, b = pair(fx, 8, 2)
  drive(a, b, 120, np.random.default_rng(6))
  assert np.isnan(b.adapter.state["effort"]).all() and (b.adapter.state["effort_evals"] == 0).all()

def test_snapshot_restore_is_path_independent():
  a, b = pair(F.cartpole(max_control_steps=10**9), 8, 2)
  s = np.arange(8, dtype=np.int32)
  snaps = {}

  def on_step(t):
    if t == 20:
      snaps["a"], snaps["b"] = a.save_state(s), b.save_state(s)
      for k in set(snaps["a"]) - {"book/episode_started_ns"}:  # wall clock at reset
        np.testing.assert_array_equal(snaps["a"][k], snaps["b"][k], err_msg=k)
    if t == 40:  # rewind both to the fused core's snapshot
      assert a.load_state(s, snaps["b"]).all() and b.load_state(s, snaps["b"]).all()

  drive(a, b, 80, np.random.default_rng(11), scale=0.2, on_step=on_step)


BOXES = b"".join(
  b'<body pos="%.1f 1 .5"><freejoint/><geom type="box" size=".1 .1 .1" mass="1" condim="6"/></body>' % (0.3 * i)
  for i in range(6)
)
# The arena holds the 24 contacts, but the Newton solver's stack overflows when they
# appear: a deterministic native error at touchdown, a few substeps into an interval.
BOX_XML = F.asset("cartpole.xml").replace(b"</worldbody>", BOXES + b"</worldbody>").replace(
  b'<option timestep="0.002"/>', b'<size memory="160000"/><option timestep="0.002"/>')


def box_fixture():
  fx = F.cartpole(max_control_steps=10**9)
  return F.Fixture("cartpole_boxes", BOX_XML, fx.group, fx.contract, F.no_termination(fx).task_doc,
                   fx.reset_distribution)


def drop_boxes(cores, slots, heights):
  """Box bottoms just above the floor, per slot: touchdown on a slot-dependent substep."""
  for c in cores:
    off = c.layout["qpos"][0] + 2  # slider, hinge, then six free joints
    for j, z in zip(slots, heights, strict=True):
      c.h.state[j, off + 2 : off + 42 : 7] = z
    assert c.h.forward(slots).ok


def test_native_failures_are_replayed_to_their_exact_substep():
  fx = box_fixture()
  m = fx.model()
  d0 = mujoco.MjData(m)
  d0.qpos[2 + 2 :: 7] = 0.10005
  steps = 0
  with pytest.raises(mujoco.FatalError, match="stack"):  # the premise: an error on touchdown ...
    while steps < 40:
      mujoco.mj_step(m, d0)
      steps += 1
  assert 0 < steps < 4, steps  # ... inside the first policy interval (retune `memory` if MuJoCo changes)
  a, b = pair(fx, 8, 2)
  d = a.spec.decimation
  s = np.arange(8, dtype=np.int32)
  reset((a, b), s, 0)
  heights = np.array([0.1000005, 0.10001, 0.10005, 0.10008, 0.1003, 0.3, 0.3, 0.3])
  drop_boxes((a, b), s, heights)
  seen = set()
  rng = np.random.default_rng(5)
  for _ in range(6):
    act = rng.uniform(-0.2, 0.2, (8, 1)).astype(np.float32)
    live = s[a.book["slot_state"][s] != SlotState.FAILED]
    oa, ob = a.advance_control(live, act[live]), b.advance_control(live, act[live])
    assert_same(a, b, oa, ob)
    seen |= set(oa.failed_substep[oa.status == SlotStatus.NATIVE_FAILURE].tolist())
  assert any(0 < k < d - 1 for k in seen), seen  # failures inside an interval, not only at its ends
  assert b.replayed_slots >= len(seen)




def test_native_failures_on_the_first_substep_leave_no_pending_ctrl():
  """A slot failing on substep 0 carries the fused call's ctrl as a pending write; it is
  cancelled before the replay, so the slot keeps its pre-interval ctrl as it would have."""
  fx0 = F.cartpole(max_control_steps=10**9)
  xml = F.asset("cartpole.xml").replace(
    b"</worldbody>",
    b'<body name="ball" pos="1 1 1"><freejoint/><geom type="sphere" size=".05" mass=".2" contype="0" '
    b'conaffinity="0"/></body></worldbody><equality><connect body1="ball" body2="world" anchor="1 1 1.2"/>'
    b"</equality>")
  fx = F.Fixture("cartpole_eq", xml, fx0.group, fx0.contract, fx0.task_doc, fx0.reset_distribution)
  a, b = pair(fx, 8, 2)
  s = np.arange(8, dtype=np.int32)
  reset((a, b), s, 0)
  rng = np.random.default_rng(2)
  for t in range(8):
    if t == 4:
      for c in (a, b):
        c.h.expand("eq_type")[[1, 4, 6]] = 99  # fault injection: fails on the first substep
    act = rng.uniform(-1, 1, (8, 1)).astype(np.float32)
    live = s[a.book["slot_state"][s] != SlotState.FAILED]
    oa, ob = a.advance_control(live, act[live]), b.advance_control(live, act[live])
    assert_same(a, b, oa, ob)
  assert (a.book["slot_state"][[1, 4, 6]] == SlotState.FAILED).all() and b.replayed_slots >= 3

def test_replayed_slots_do_not_inherit_warnings_of_discarded_substeps():
  """A slot that terminates on substep 0 is replayed; the fused call ran on and raised
  fatal CNSTRFULL warnings on later substeps that per-substep stepping never runs. Its
  warning counters come back with its pre-interval state, so it terminates cleanly."""
  fx0 = F.cartpole(max_control_steps=10**9)
  xml = BOX_XML.replace(b'memory="160000"', b'memory="100000"')  # contacts overflow the arena: warnings
  fx = F.Fixture("cartpole_boxes_warn", xml, fx0.group, fx0.contract, fx0.task_doc, fx0.reset_distribution)
  a, b = pair(fx, 2, 1)
  s = np.arange(2, dtype=np.int32)
  reset((a, b), s, 0)
  drop_boxes((a, b), s, np.array([0.10008, 0.3]))  # touchdown on substep 2 for slot 0
  for c in (a, b):
    q, v = c.layout["qpos"][0], c.layout["qvel"][0]
    c.h.state[0, q + 1], c.h.state[0, v + 1] = 0.595, 3.0  # the pole passes angle_limit on substep 0
    assert c.h.forward(s).ok
  oa, ob = a.advance_control(s, np.zeros((2, 1), np.float32)), b.advance_control(s, np.zeros((2, 1), np.float32))
  assert_same(a, b, oa, ob)
  assert oa.status[0] == SlotStatus.TERMINATED and oa.completed_substeps[0] == 1 and oa.warning_delta[0] == 0
  assert b.replayed_slots == 1


def test_rewinding_forward_adds_no_warnings_to_the_replay():
  """Boxes already in (arena-overflowing) contact warn on every MuJoCo call, including
  the forward that rewinds a replayed slot; that forward's warnings are not the
  interval's (CNSTRFULL is made non-fatal so the counts are observable)."""
  fx0 = F.cartpole(max_control_steps=10**9)
  health = fx0.group.health.model_copy(update={"fatal_warning_codes": ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL")})
  group = fx0.group.model_copy(update={"health": health})
  xml = BOX_XML.replace(b'memory="160000"', b'memory="100000"')
  fx = F.Fixture("cartpole_boxes_warn", xml, group, fx0.contract, fx0.task_doc, fx0.reset_distribution)
  a, b = pair(fx, 2, 1)
  s = np.arange(2, dtype=np.int32)
  reset((a, b), s, 0)
  drop_boxes((a, b), s, np.array([0.0999, 0.0999]))  # already touching: warnings from substep 0
  for c in (a, b):
    q, v = c.layout["qpos"][0], c.layout["qvel"][0]
    c.h.state[0, q + 1], c.h.state[0, v + 1] = 0.595, 1.0  # passes angle_limit a few substeps in
    assert c.h.forward(s).ok
  for c in (a, b):
    c.health.rebase(s)  # the drops above warned; start counting from here
  oa, ob = a.advance_control(s, np.zeros((2, 1), np.float32)), b.advance_control(s, np.zeros((2, 1), np.float32))
  assert_same(a, b, oa, ob)
  assert oa.status[0] == SlotStatus.TERMINATED and 1 < oa.completed_substeps[0] < a.spec.decimation
  assert oa.warning_delta[0] == oa.completed_substeps[0] and b.replayed_slots == 1

def test_health_failures_at_control_rate_match():
  a, b = pair(F.no_termination(F.cartpole(max_control_steps=10**9)), 6, 2)
  d = a.spec.decimation
  s = np.arange(6, dtype=np.int32)
  reset((a, b), s, 0)
  for c in (a, b):
    off = c.layout["qvel"][0]
    c.h.state[1, off] = 1e4  # beyond max_abs_qvel
    c.h.state[2, off + 1] = np.nan  # MuJoCo auto-resets and raises BADQVEL
  oa, ob = a.advance_control(s, np.zeros((6, 1), np.float32)), b.advance_control(s, np.zeros((6, 1), np.float32))
  assert_same(a, b, oa, ob)
  for j in (1, 2):
    assert oa.status[j] == SlotStatus.HEALTH_FAILURE and oa.failed_substep[j] == d - 1
    assert oa.completed_substeps[j] == d


def test_control_rate_health_runs_once_on_each_slots_last_substep():
  """check_each_substep=false: one check per interval on the slot's final state, before
  its termination counts; a slot that is unhealthy where it terminates is aborted, not
  terminated, and its task never records the fall."""
  fx = F.cartpole(max_control_steps=10**9)
  per = fx.core(num_slots=4, threads=1)  # check_each_substep=true
  ctl = fx.core(num_slots=4, threads=1, health=fx.group.health.model_copy(update=CONTROL_RATE),
                fuse_substeps=False)
  d = ctl.spec.decimation
  s = np.arange(4, dtype=np.int32)
  for c in (per, ctl):
    reset((c, c), s, 0)
    q, v = c.layout["qpos"][0], c.layout["qvel"][0]
    c.h.state[1, v + 1] = np.nan  # MuJoCo auto-resets to qpos0 and warns: unhealthy, not terminated
    c.h.state[2, q + 1] = 1.0  # beyond angle_limit: terminates at substep 0
    c.h.state[2, v] = 1e4  # ... and unhealthy there
    c.h.state[3, q + 1] = 1.0  # terminates at substep 0, healthy
  op, oc = per.advance_control(s, np.zeros((4, 1), np.float32)), ctl.advance_control(s, np.zeros((4, 1), np.float32))
  assert op.failed_substep[1] == 0 and op.completed_substeps[1] == 1
  assert oc.status[1] == SlotStatus.HEALTH_FAILURE and oc.failed_substep[1] == d - 1 and oc.completed_substeps[1] == d
  for o in (op, oc):
    assert o.status[2] == SlotStatus.HEALTH_FAILURE and o.failed_substep[2] == 0
    assert o.metrics[2, 0] == 0.0  # the fall is not recorded for an aborted episode
    assert o.status[3] == SlotStatus.TERMINATED and o.completed_substeps[3] == 1 and o.metrics[3, 0] == 1.0


def test_fused_only_where_it_is_identical():
  health = {"health": F.cartpole().group.health.model_copy(update=CONTROL_RATE)}
  assert F.cartpole().core(num_slots=2, threads=1, **health).stepping == "fused"
  assert F.cartpole().core(num_slots=2, threads=1).stepping == "per_substep"  # per-substep health
  assert F.cartpole().core(num_slots=2, threads=1, fuse_substeps=False, **health).stepping == "per_substep"
  fx = F.cartpole()
  one = F.Fixture(fx.name, fx.xml, fx.group.model_copy(update={"decimation": 1}), fx.contract, fx.task_doc,
                  fx.reset_distribution)
  assert one.core(num_slots=2, threads=1, **health).stepping == "per_substep"
  quad = F.quadruped()  # pd reads the state every substep
  assert quad.core(num_slots=2, threads=1, health=quad.group.health.model_copy(update=CONTROL_RATE)).stepping == (
    "per_substep")
  fx = F.cartpole()
  delayed = fx.contract.model_copy(update={"delay_clock": "physics", "delay_ticks": 1})
  meta = fx.meta()
  assert not compile_adapter(delayed, meta).controller_cfg["physics_delay_ticks"] == 0
  fx2 = F.Fixture(fx.name, fx.xml, fx.group, delayed, fx.task_doc, fx.reset_distribution)
  assert fx2.core(num_slots=2, threads=1, **health).stepping == "per_substep"


def test_fusable_matches_the_controller_kinds_that_hold_ctrl():
  from raysimbatch.worker.scalar import ScalarHandle

  assert not hasattr(ScalarHandle, "stepn")  # the certified scalar lane keeps its own stepping
  assert HealthConfig().check_each_substep  # the default configuration is unchanged
