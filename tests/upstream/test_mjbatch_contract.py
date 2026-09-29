# SPDX-License-Identifier: Apache-2.0
"""Executable record of the upstream MJBatch behaviour RaySimBatch builds on.

Each test pins one fact that docs/design-amendments.md cites. They run against the
pinned revision in CI and against every upgrade candidate (tools/mjbatch_upgrade.py);
a failure here means the compatibility adapter must change before the pin moves.
These tests deliberately use mjbatch directly: they qualify the dependency itself.
"""

import gc
import threading
import time

import mujoco
import numpy as np
import pytest
from mjbatch import Batch

CART = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 .05">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
      </body>
    </body>
    <body name="ball" pos="0 1 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass=".2"/>
    </body>
  </worldbody>
  <equality><connect body1="ball" body2="world" anchor="0 1 .8"/></equality>
  <actuator><motor joint="slide" gear="5"/><motor joint="hinge" gear="1"/></actuator>
  <keyframe><key qpos="0.3 0.5 0 1 .5 1 0 0 0" ctrl="0.1 0.2"/></keyframe>
</mujoco>
"""

# Contact-rich scene where the Newton solver's warmstart matters.
BOXES = """
<mujoco>
  <option timestep="0.004" solver="CG" iterations="3" tolerance="0"/>
  <worldbody>
    <geom type="plane" size="3 3 .1"/>
    <body pos="0 0 .3"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    <body pos=".05 0 .6"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    <body pos="-.05 .02 .9"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
</mujoco>
"""
N = 8


def cart():
  return mujoco.MjModel.from_xml_string(CART)


def make_error(batch, sims):
  """An invalid equality type makes mj_step raise a MuJoCo error for these sims only."""
  eq_type = batch.expand("eq_type")
  for i in sims:
    eq_type[i] = 99
  return eq_type


@pytest.mark.parametrize("threads", [1, 3])
def test_failed_sims_keep_pre_call_state_and_are_identifiable(threads):
  """Per-slot status can be derived without a fork: a failing sim's time and warning
  rows are not copied out, while every successful sim's time advances."""
  model = cart()
  batch = Batch(model, N, num_threads=threads)
  time_v, warn = batch.bind("time"), batch.bind("warning")
  state = batch.bind("state")
  batch.step(np.arange(N, dtype=np.int32))
  before_state, before_time = state.copy(), time_v.copy()
  make_error(batch, [2, 5])
  with pytest.raises(RuntimeError, match=r"^sim (2|5): "):
    batch.step(nstep=3)
  changed = (time_v != before_time) | np.any(warn[:, :, 1] != 0, axis=1)
  assert list(np.flatnonzero(~changed)) == [2, 5]
  np.testing.assert_array_equal(state[[2, 5]], before_state[[2, 5]])
  np.testing.assert_allclose(time_v[[0, 1, 3, 4, 6, 7]], before_time[[0, 1, 3, 4, 6, 7]] + 3 * 0.002)


def test_error_message_names_only_the_first_failure():
  batch = Batch(cart(), N, num_threads=1)
  make_error(batch, [3, 6])
  with pytest.raises(RuntimeError) as err:
    batch.step()
  assert str(err.value).startswith("sim 3: ")
  assert "sim 6" not in str(err.value)


def test_with_several_threads_the_named_sim_is_first_in_time_not_lowest_index():
  """error_ is set by whichever worker thread fails first: attribution of *which*
  sims failed must therefore never be derived from the message alone."""
  seen = set()
  for _ in range(40):
    batch = Batch(cart(), N, num_threads=4)
    make_error(batch, [1, 7])
    with pytest.raises(RuntimeError) as err:
      batch.step()
    seen.add(str(err.value).split(":")[0])
  assert seen <= {"sim 1", "sim 7"}


def test_failed_sim_keeps_pending_write_and_restoring_the_old_value_cancels_it():
  """Input fields are copied in element-wise where they differ from what MJBatch last
  copied out. Writing back the pre-write value therefore cancels a pending write:
  RaySimBatch's neutralization of failed slots relies on exactly this."""
  model = cart()
  batch = Batch(model, N, num_threads=2)
  ctrl, qpos = batch.bind("ctrl"), batch.bind("qpos")
  eq_type = make_error(batch, [4])
  saved = ctrl.copy()
  ctrl[:, 0] = 1.0
  with pytest.raises(RuntimeError, match="sim 4"):
    batch.step()
  eq_type[:] = model.eq_type
  assert ctrl[4, 0] == 1.0  # still pending
  ctrl[4] = saved[4]  # neutralize
  batch.step(np.array([4]))
  ref = mujoco.MjData(model)
  mujoco.mj_step(model, ref)  # ctrl stays zero
  np.testing.assert_array_equal(qpos[4], ref.qpos)


def test_pending_field_write_survives_reset_but_state_row_write_does_not():
  model = cart()
  batch = Batch(model, N)
  qpos, state, ctrl = batch.bind("qpos"), batch.bind("state"), batch.bind("ctrl")
  batch.step(nstep=5)
  ctrl[1, 0] = 0.7  # pending field write
  state[2] = state[0] * 0 + 1.0  # pending state row write
  batch.reset(np.array([1, 2]))
  assert ctrl[1, 0] == 0.7  # the field write landed after mj_resetData
  np.testing.assert_array_equal(qpos[2], model.qpos0)  # the row write was discarded
  np.testing.assert_array_equal(ctrl[2], 0.0)


def test_reset_keyframe_then_forward_matches_mujoco():
  model = cart()
  batch = Batch(model, N)
  xpos, qpos = batch.bind("xpos"), batch.bind("qpos")
  batch.reset(np.array([3]), keyframe=0)
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  mujoco.mj_forward(model, d)
  np.testing.assert_array_equal(qpos[3], d.qpos)
  np.testing.assert_array_equal(xpos[3], d.xpos)


@pytest.mark.parametrize("threads", [1, 4])
def test_forward_leaves_integration_state_and_trajectory_bitwise_unchanged(threads):
  """forward() is a pure query of the integration state (including qacc_warmstart):
  RaySimBatch calls it at control boundaries and for diagnostics."""
  model = mujoco.MjModel.from_xml_string(BOXES)
  a, b = Batch(model, N, num_threads=threads), Batch(model, N, num_threads=threads)
  sa, sb = a.bind("state"), b.bind("state")
  for _ in range(60):
    a.step()
    b.step()
    before = sb.copy()
    b.forward(np.arange(0, N, 2))
    np.testing.assert_array_equal(sb, before)
  np.testing.assert_array_equal(sa, sb)


def test_state_rows_include_warmstart_and_restore_replays_bitwise():
  model = mujoco.MjModel.from_xml_string(BOXES)
  full = mujoco.mjtState.mjSTATE_INTEGRATION
  warm = mujoco.mjtState.mjSTATE_WARMSTART
  assert int(full) & int(warm)
  batch = Batch(model, N, num_threads=3)
  state = batch.bind("state")
  batch.step(nstep=200)  # boxes are stacked and in contact by now
  snap = state.copy()
  batch.step(nstep=25)
  once = state.copy()
  state[:] = snap
  batch.step(nstep=25)
  np.testing.assert_array_equal(state, once)
  # Zeroing only the warmstart component changes the continuation: it is state.
  offset = sum(
    mujoco.mj_stateSize(model, int(bit))
    for bit in mujoco.mjtState.__members__.values()
    if int(bit) & int(full) and int(bit) < int(warm) and bin(int(bit)).count("1") == 1
  )
  nwarm = mujoco.mj_stateSize(model, int(warm))
  ref = mujoco.MjData(model)
  mujoco.mj_setState(model, ref, snap[0], full)
  np.testing.assert_array_equal(snap[0, offset : offset + nwarm], ref.qacc_warmstart)
  state[:] = snap
  state[:, offset : offset + nwarm] = 0.0
  batch.step(nstep=25)
  assert not np.array_equal(state, once)


def test_single_substep_calls_equal_one_multi_substep_call():
  """Per-substep execution (controller between calls) is semantically identical to
  nstep=k when the inputs are unchanged, so RaySimBatch's substep loop adds no drift."""
  model = mujoco.MjModel.from_xml_string(BOXES)
  a, b = Batch(model, N, num_threads=2), Batch(model, N, num_threads=2)
  a.step(nstep=8)
  for _ in range(8):
    b.step()
  np.testing.assert_array_equal(a.bind("state"), b.bind("state"))



def test_history_rows_are_the_state_after_each_substep():
  """Fused intervals (AM-2) read each substep's state from history (M17): row k of
  selected sim j equals its state row after k+1 single steps."""
  model = mujoco.MjModel.from_xml_string(BOXES)
  a, b = Batch(model, N, num_threads=2), Batch(model, N, num_threads=2)
  ids = np.array([1, 3, 4], np.int32)
  hist = np.zeros((ids.size, 5, a.nstate))
  a.step(ids, nstep=5, history=hist)
  for k in range(5):
    b.step(ids)
    np.testing.assert_array_equal(hist[:, k], b.bind("state")[ids])
  np.testing.assert_array_equal(a.bind("state"), b.bind("state"))


def test_written_warning_counters_are_applied_at_the_next_call():
  """A replayed slot's warning counters are rewound by writing its bound rows (warning is
  an input field, carried across calls like the state)."""
  batch = Batch(cart(), N, num_threads=1)
  warn = batch.bind("warning")
  warn[2, 3, 1] = 7
  batch.step(np.array([2], np.int32))
  assert warn[2, 3, 1] == 7 and warn[1, 3, 1] == 0

def test_set_const_subset_semantics():
  """The first time a set_const output differs from the template it is expanded and
  recomputed for every sim; afterwards set_const(ids) recomputes only the selected
  sims. RaySimBatch prewarms that closure at construction so a reset touches only its
  own rows."""
  model = cart()
  pole = model.body("pole").id
  batch = Batch(model, N)
  mass = batch.expand("body_mass")
  mass[:, pole] = np.linspace(0.1, 0.5, N)
  batch.set_const(np.array([5]))  # closure expansion recomputes all sims
  sub = batch.expand("body_subtreemass")
  np.testing.assert_allclose(sub[:, pole], mass[:, pole])
  before = sub.copy()
  mass[[2, 4], pole] = 2.0
  batch.set_const(np.array([2]))  # closure already expanded: only sim 2 changes
  assert sub[2, pole] == 2.0
  np.testing.assert_array_equal(np.delete(sub, 2, axis=0), np.delete(before, 2, axis=0))


def test_bound_views_keep_the_batch_alive():
  batch = Batch(cart(), N)
  qpos = batch.bind("qpos")
  batch.step()
  del batch
  gc.collect()
  assert qpos.shape == (N, cart().nq)
  assert np.isfinite(qpos).all()


def test_threads_are_clamped_to_num_sims():
  assert Batch(cart(), 3, num_threads=8).num_threads == 3
  assert Batch(cart(), 16, num_threads=4).num_threads == 4


def test_step_releases_the_gil():
  model = mujoco.MjModel.from_xml_string(BOXES)
  batch = Batch(model, 64, num_threads=2)
  ticks = []
  stop = threading.Event()

  def spin():
    while not stop.is_set():
      ticks.append(time.perf_counter())

  t = threading.Thread(target=spin)
  t.start()
  t0 = time.perf_counter()
  batch.step(nstep=400)
  t1 = time.perf_counter()
  stop.set()
  t.join()
  assert sum(t0 < x < t1 for x in ticks) > 100


def test_ids_contract():
  batch = Batch(cart(), N)
  for bad in ([3, 1], [1, 1], [N], [-1]):
    with pytest.raises(ValueError):
      batch.step(np.array(bad, np.int32))
  batch.step(np.array([], np.int32))  # an empty selection is a no-op


def test_sleep_is_rejected():
  xml = CART.replace("<option", '<option><flag sleep="enable"/></option><option')
  with pytest.raises(ValueError, match="sleep"):
    Batch(mujoco.MjModel.from_xml_string(xml), N)


def test_asset_fields_cannot_be_expanded():
  xml = CART.replace(
    '<geom type="box" size=".1 .1 .05" mass="1"/>',
    '<geom type="box" size=".1 .1 .05" mass="1"/><geom type="mesh" mesh="m" contype="0" conaffinity="0"/>',
  ).replace("<worldbody>", '<asset><mesh name="m" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"/></asset><worldbody>', 1)
  batch = Batch(mujoco.MjModel.from_xml_string(xml), N)
  with pytest.raises(ValueError, match="asset"):
    batch.expand("mesh_vert")
