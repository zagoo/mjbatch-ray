# SPDX-License-Identifier: Apache-2.0
"""``Book.substep_settle`` replaces the NumPy body of the per-substep loop (health,
termination, statuses, compaction of the active slots). It must do exactly what that
body did: the reference below is the former implementation, kept as the oracle, with the
task predicate evaluated through the kernel on each slot's state row."""

from __future__ import annotations

import numpy as np
import pytest

from raysimbatch.protocol import SlotStatus
from raysimbatch.testing import fixtures as F
from raysimbatch.worker.core import _Interval


def core_pair(each):
  fx = F.cartpole(max_control_steps=10**9)
  health = fx.group.health.model_copy(update={"check_each_substep": each})
  cores = [fx.core(num_slots=24, threads=1, health=health, fuse_substeps=False) for _ in range(2)]
  s = np.arange(24, dtype=np.int32)
  ids = s.astype(np.uint64)
  ep = np.zeros(24, np.uint64)
  for c in cores:
    assert c.reset_exact(s, ids, ep, (ids, ep)).ok.all()
  return cores


def perturb(cores, rng):
  """Unhealthy, terminating, both, or neither; identical in both cores."""
  kind = rng.integers(0, 5, 24)
  for c in cores:
    q, v = c.layout["qpos"][0], c.layout["qvel"][0]
    for j, kd in enumerate(kind):
      if kd == 1:
        c.h.state[j, v] = 700.0  # beyond max_abs_qvel (500)
      elif kd == 2:
        c.h.state[j, q + 1] = 0.7  # beyond angle_limit: terminates
      elif kd == 3:
        c.h.state[j, q + 1], c.h.state[j, v] = 0.7, 700.0  # both
      elif kd == 4:
        c.h.state[j, v + 1] = np.nan  # auto-reset and a fatal warning
    assert c.h.forward(np.arange(24, dtype=np.int32)).ok


def terminated(core, active):
  hist = np.ascontiguousarray(core.h.state[active][:, None, :])
  out = np.zeros(active.size, np.int32)
  core.task.first_termination(np.arange(active.size, dtype=np.int32), hist, core.layout["qpos"][0], out)
  return (out == 0).astype(np.uint8)


def reference_settle(core, active, pos, k, r):
  """The former per-substep loop body after a successful step (both health modes)."""
  b = core.book
  each, last = core.spec.health.check_each_substep, core.spec.decimation - 1
  r.completed[pos] += 1
  b["sim_time_ns"][active, 0] += core.spec.timestep_ns
  if each:
    core._check_health(active, pos, np.ones(active.size, bool), core.dt, k, r)
    keep = r.status[pos] == SlotStatus.OK
    active, pos = active[keep], pos[keep]
    term = terminated(core, active)
    if term.any():
      core.task.mark_terminated(active[term != 0])
  else:
    term = terminated(core, active)
    checked = (term != 0) | (k == last)
    core._check_health(active, pos, checked, core.dt * (k + 1), k, r)
    healthy = r.status[pos] == SlotStatus.OK
    term &= healthy
    keep = healthy & ~checked
    done = term != 0
    if done.any():
      core.task.mark_terminated(active[done])
    active, pos, term = active[keep | done], pos[keep | done], term[keep | done]
  done = term != 0
  if done.any():
    r.status[pos[done]] = SlotStatus.TERMINATED
    r.ended[pos[done]] = True
    active, pos = active[~done], pos[~done]
  return active, pos


@pytest.mark.parametrize("each", [True, False])
@pytest.mark.parametrize("k", [0, 2, 4])
@pytest.mark.parametrize("seed", range(4))
def test_native_substep_settle_equals_the_former_numpy_body(each, k, seed):
  ref, nat = core_pair(each)
  perturb((ref, nat), np.random.default_rng(seed))
  rng = np.random.default_rng(100 + seed)
  active0 = np.sort(rng.choice(24, 18, replace=False)).astype(np.int32)
  pos0 = np.sort(rng.choice(24, 18, replace=False)).astype(np.int32)  # any distinct result positions
  results = []
  # Health last ran at the reset: per-substep health checks after one substep, control-
  # rate health on substep k after k+1 (earlier substeps only step).
  for c in (ref, nat):
    act = active0.copy()
    for _ in range(1 if each else k + 1):
      c.adapter.before_substep(act, c.dt)
      assert c.h.step1(act).ok
    results.append(_Interval(24))
  a_ref, p_ref = reference_settle(ref, active0.copy(), pos0.copy(), k, results[0])
  act, pos = active0.copy(), pos0.copy()
  kept = nat._book.substep_settle(nat.health, nat.task, act, pos, k, nat.spec.decimation - 1,
                                  nat.spec.health.check_each_substep, nat.dt, nat.spec.timestep_ns,
                                  results[1].status, results[1].fault, results[1].completed, results[1].failed_sub,
                                  results[1].warn, results[1].ended)
  np.testing.assert_array_equal(act[:kept], a_ref, err_msg="active")
  np.testing.assert_array_equal(pos[:kept], p_ref, err_msg="pos")
  for f in ("status", "fault", "completed", "failed_sub", "warn", "ended"):
    np.testing.assert_array_equal(getattr(results[1], f), getattr(results[0], f), err_msg=f)
  for name, x, y in (("task", nat.task_state["state"], ref.task_state["state"]),
                     ("sim_time", nat.book["sim_time_ns"], ref.book["sim_time_ns"]),
                     *((f"health/{h}", nat.health_state[h], ref.health_state[h]) for h in nat.health_state)):
    np.testing.assert_array_equal(x, y, err_msg=name)
  statuses = set(results[0].status[pos0].tolist())
  assert {SlotStatus.HEALTH_FAILURE, SlotStatus.TERMINATED} <= statuses, statuses  # both outcomes exercised
