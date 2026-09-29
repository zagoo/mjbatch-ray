# SPDX-License-Identifier: Apache-2.0
"""The scalar lane: same kernels and semantics as the Batch lane; complete feature
state (sleep) snapshots replay bit for bit (replay certification)."""

from __future__ import annotations

import numpy as np
import pytest

from raysimbatch.testing import fixtures as F
from raysimbatch.worker.core import CoreError, CoreSpec, MJBatchCore
from raysimbatch.worker.model import ModelMetadata, compile_model
from raysimbatch.worker.scalar import ScalarHandle

SLEEPY = F.asset("quadruped.xml").replace(b'<option timestep="0.004"/>',
                                          b'<option timestep="0.004"><flag sleep="enable"/></option>')


def scalar_core(fx, xml=None):
  from raysimbatch.adapter import compile_adapter
  from raysimbatch.tasks import bind_task

  model = compile_model(xml or fx.xml, None, fx.group)
  meta = ModelMetadata.from_model(model)
  b = bind_task(fx.task_doc, meta)
  c = compile_adapter(fx.contract, meta, num_commands=b.num_commands)
  spec = CoreSpec(num_slots=1, threads=1, decimation=fx.group.decimation, timestep_ns=fx.group.timestep_ns,
                  keyframe=fx.keyframe, max_control_steps=10**9, randomization=fx.randomization,
                  reset_distribution=fx.reset_distribution)
  return MJBatchCore(model, meta, c, b, spec, handle_factory=ScalarHandle)


def start(core):
  s = np.zeros(1, np.int32)
  r = core.reset_exact(s, np.zeros(1, np.uint64), np.zeros(1, np.uint64), (np.zeros(1, np.uint64), np.zeros(1, np.uint64)))
  assert r.ok.all(), r.message
  return s


def test_scalar_lane_matches_batch_lane_bitwise():
  fx = F.quadruped(max_control_steps=10**9)
  a = scalar_core(fx)
  b = fx.core(num_slots=1, threads=1)
  sa, sb = start(a), start(b.__class__.__mro__[0] and b) if False else (start(a), None)
  s = np.zeros(1, np.int32)
  b.reset_exact(s, np.zeros(1, np.uint64), np.zeros(1, np.uint64), (np.zeros(1, np.uint64), np.zeros(1, np.uint64)))
  rng = np.random.default_rng(0)
  for _ in range(50):
    act = rng.uniform(-1, 1, (1, 8)).astype(np.float32)
    oa, ob = a.advance_control(s, act), b.advance_control(s, act)
    np.testing.assert_array_equal(oa.observation, ob.observation)
    np.testing.assert_array_equal(a.h.state, b.h.state)
  del sa, sb


def test_batch_lane_rejects_sleep():
  fx = F.quadruped()
  with pytest.raises(Exception, match="sleep"):
    fx.core(num_slots=2, threads=1) if False else _batch_with(SLEEPY, fx)


def _batch_with(xml, fx):
  from raysimbatch.adapter import compile_adapter
  from raysimbatch.tasks import bind_task

  model = compile_model(xml, None, fx.group)
  meta = ModelMetadata.from_model(model)
  b = bind_task(fx.task_doc, meta)
  c = compile_adapter(fx.contract, meta, num_commands=b.num_commands)
  return MJBatchCore(model, meta, c, b, CoreSpec(num_slots=2, threads=1, decimation=4, timestep_ns=4_000_000))


SLEEP_SCENE = F.asset("cartpole.xml").replace(
  b'<option timestep="0.002"/>', b'<option timestep="0.002"><flag sleep="enable"/></option>'
).replace(b"</worldbody>", b"".join(
  b'<body name="box%d" pos="%f 1 0.2"><freejoint/><geom type="box" size=".05 .05 .05" mass=".5"/></body>'
  % (i, 0.5 + 0.3 * i) for i in range(3)) + b"</worldbody>")


def test_sleep_feature_state_snapshot_replays_bitwise():
  """Replay certification for the scalar lane: snapshot while passive bodies are still
  settling, continue until they fall asleep; the complete feature-state snapshot
  (sleep/island arrays included) replays bit for bit across the transition.

  Observed on MuJoCo 3.13.0: in this scene an integration-state-only restore also
  replays identically (awake status is re-derived from the restored velocities). The
  scalar lane still serializes the full feature state because that equivalence is not
  a documented guarantee for every sleep/plugin configuration."""
  fx = F.cartpole(max_control_steps=10**9)
  core = scalar_core(fx, SLEEP_SCENE)
  s = start(core)
  pol = F.cartpole_policy
  obs = core.adapter.state  # noqa: F841
  last = np.zeros((1, 4), np.float32)
  for _ in range(10):  # boxes are mid-air: awake with a sleep countdown
    last = core.advance_control(s, pol(last)).observation
  assert core.h.d.ntree_awake == core.h.m.ntree
  snap = core.save_state(s)

  def run(obs0):
    o, out = obs0, []
    for _ in range(80):
      o = core.advance_control(s, pol(o)).observation
      out.append(core.h.state.copy())
    return out

  first = run(last)
  assert core.h.d.ntree_awake < core.h.m.ntree  # the boxes fell asleep meanwhile
  assert core.load_state(s, snap).all()
  np.testing.assert_array_equal(run(last), first)  # full feature state: exact replay
  assert "mj/extra/tree_asleep" in snap and "mj/extra/body_awake" in snap


def test_scalar_failure_keeps_pre_call_state():
  xml = F.asset("cartpole.xml").replace(
    b"</worldbody>",
    b'<body name="ball" pos="1 1 1"><freejoint/><geom type="sphere" size=".05" mass=".2" contype="0" '
    b'conaffinity="0"/></body></worldbody><equality><connect body1="ball" body2="world" anchor="1 1 1.2"/></equality>',
  )
  fx = F.cartpole(max_control_steps=10**9)
  core = scalar_core(fx, xml)
  s = start(core)
  core.advance_control(s, np.zeros((1, 1), np.float32))
  before = core.h.state.copy()
  core.h.m.eq_type[0] = 99
  out = core.advance_control(s, np.zeros((1, 1), np.float32))
  assert out.status[0] == 3 and out.completed_substeps[0] == 0
  np.testing.assert_array_equal(core.h.state, before)
  with pytest.raises(CoreError):
    core.advance_control(s, np.zeros((1, 1), np.float32))  # FAILED slots need a reset
