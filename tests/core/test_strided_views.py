# SPDX-License-Identifier: Apache-2.0
"""MJBatch 0.1.4 binds qpos, qvel and ctrl as views into each sim's state row: a row is
contiguous, but rows are ``nstate`` elements apart. The native kernels read and write
those views in place, so on row-strided views they must give exactly the results they
give on contiguous arrays, write ctrl into the viewed buffer (the batch) rather than a
copy, and never touch the memory between rows. The views are built here with NumPy, so
this holds on any MJBatch version."""

from __future__ import annotations

import numpy as np
import pytest

from raysimbatch.adapter import Sim2SimAdapter
from raysimbatch.testing import fixtures as F

N = 8
PAD = 5
SENTINEL = 12345.0
BATCH_FIELDS = ("qpos", "qvel", "ctrl", "act", "qacc", "sensordata", "site_xpos")


def row_strided(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """A copy of ``a`` whose rows sit ``PAD`` elements apart inside a wider buffer."""
  wide = np.full((a.shape[0], a.shape[1] + PAD), SENTINEL, a.dtype)
  view = wide[:, 2 : 2 + a.shape[1]]
  view[...] = a
  return wide, view


def setup():
  fx = F.cartpole()
  core = fx.core(num_slots=N, threads=1)
  s = np.arange(N, dtype=np.int32)
  ids = s.astype(np.uint64)
  assert core.reset_exact(s, ids, np.zeros(N, np.uint64), (ids, np.zeros(N, np.uint64))).ok.all()
  b = core.book
  shared = {**core.views, "command": core.command, "sim_time_ns": b["sim_time_ns"],
            "noise_keys": b["noise_keys"], "control_seq": b["control_seq"]}
  rng = np.random.default_rng(7)
  shared["qpos"][...] += rng.uniform(-0.2, 0.2, shared["qpos"].shape)
  shared["qvel"][...] = rng.uniform(-1.0, 1.0, shared["qvel"].shape)
  return core, shared, s, rng


def test_kernels_on_row_strided_batch_views_match_contiguous_and_write_in_place():
  core, shared, s, rng = setup()
  dense = {k: v.copy() for k, v in shared.items()}
  strided, wides = {}, {}
  for k, v in shared.items():
    if k in BATCH_FIELDS and v.ndim == 2 and v.shape[1] > 0:
      wides[k], strided[k] = row_strided(v)
      assert not strided[k].flags["C_CONTIGUOUS"]
    else:
      strided[k] = v.copy()
  a_dense = Sim2SimAdapter(core.compiled, dense, N)
  a_strided = Sim2SimAdapter(core.compiled, strided, N)
  for a in (a_dense, a_strided):
    a.reset(s)
  for _ in range(3):
    action = rng.uniform(-1.0, 1.0, (N, core.A)).astype(np.float32)
    targets = [a.process_action(s, action) for a in (a_dense, a_strided)]
    np.testing.assert_array_equal(targets[1], targets[0])
    for a in (a_dense, a_strided):
      a.before_substep(s, core.dt)
    # The controller wrote ctrl through the view, i.e. into the wide (batch) buffer.
    assert np.any(dense["ctrl"] != 0)
    np.testing.assert_array_equal(wides["ctrl"][:, 2 : 2 + dense["ctrl"].shape[1]], dense["ctrl"])
  obs = [np.zeros((N, core.O), np.float32) for _ in range(2)]
  a_dense.observe(s, obs[0])
  a_strided.observe(s, obs[1])
  np.testing.assert_array_equal(obs[1], obs[0])
  for k, wide in wides.items():
    w = shared[k].shape[1]
    assert (wide[:, :2] == SENTINEL).all() and (wide[:, 2 + w :] == SENTINEL).all(), k


def test_rows_that_are_not_contiguous_are_still_refused():
  core, shared, _, _ = setup()
  views = {k: v.copy() for k, v in shared.items()}
  assert views["qpos"].shape[1] > 1  # a width-1 row is contiguous at any stride
  wide = np.zeros((N, 2 * views["qpos"].shape[1]), np.float64)
  views["qpos"] = wide[:, ::2]  # every other element: a row is not contiguous
  with pytest.raises(ValueError, match="qpos: each row must be contiguous"):
    Sim2SimAdapter(core.compiled, views, N)


def test_health_reads_a_row_strided_one_dimensional_time_view():
  """``time`` holds one value per sim, and since 0.1.4 it too is a view into the state
  rows: a 1-D array whose elements are nstate apart."""
  import mujoco

  from raysimbatch import _native
  from raysimbatch.worker.model import ModelMetadata

  fx = F.cartpole()
  core, shared, s, _ = setup()
  hc = core.spec.health
  meta = ModelMetadata.from_model(fx.model())
  fatal = [int(getattr(mujoco.mjtWarning, "mjWARN_" + c)) for c in hc.fatal_warning_codes]
  cfg = {"max_abs_qvel": hc.max_abs_qvel, "max_abs_qacc": hc.max_abs_qacc,
         "quaternion_norm_tolerance": hc.quaternion_norm_tolerance, "time_rtol": 1e-9,
         "num_warnings": shared["warning"].shape[1]}
  dense = {k: shared[k].copy() for k in ("qpos", "qvel", "qacc", "warning", "time")}
  strided = {"warning": shared["warning"].copy()}
  for k in ("qpos", "qvel", "qacc"):
    _, strided[k] = row_strided(shared[k])
  time_wide = np.full((N, 1 + PAD), SENTINEL)
  strided["time"] = time_wide[:, 2]
  strided["time"][...] = shared["time"]
  assert strided["time"].ndim == 1 and not strided["time"].flags["C_CONTIGUOUS"]
  faults = []
  for views in (dense, strided):
    h = _native.Health(cfg, {"quat_adr": np.asarray(meta.quat_adrs, np.int32),
                             "fatal_warnings": np.asarray(fatal, np.int32), "act": None,
                             **views, **{k: v.copy() for k, v in core.health_state.items()}})
    h.rebase(s)
    views["time"][...] += core.dt  # one substep...
    views["time"][5] += 1.0  # ...except slot 5, whose clock jumped
    views["qvel"][3, 0] = 700.0  # beyond max_abs_qvel
    fault, warn = np.zeros(N, np.uint16), np.zeros(N, np.int32)
    h.check(s, core.dt, fault, warn)
    faults.append(fault)
  np.testing.assert_array_equal(faults[1], faults[0])
  assert faults[0][3] and faults[0][5] and not faults[0][[0, 1, 2, 4, 6, 7]].any()
  assert (time_wide[:, :2] == SENTINEL).all() and (time_wide[:, 3:] == SENTINEL).all()
