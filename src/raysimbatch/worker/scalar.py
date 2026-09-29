# SPDX-License-Identifier: Apache-2.0
"""The isolated scalar lane (design §6.2): one environment, one MjModel/MjData, one CPU.

``ScalarHandle`` implements the BatchHandle interface over a single ``MjData`` so the
MJBatchCore kernels, reset/restore algorithm and per-slot semantics are reused
unchanged. It covers features the Batch lane rejects (sleep) or cannot certify
(stateful plugins): ``extra_state`` serializes the *complete* mjData arrays (sleep and
island bookkeeping included), and a failed call restores the exact pre-call mjData,
matching MJBatch's failed-sim contract. Capacity of this lane is benchmarked
separately and excluded from the default capacity forecast.
"""

from __future__ import annotations

import mujoco
import numpy as np

from raysimbatch.compat.mjbatch_adapter import CallOutcome, _as_ids

_SKIP = {"contact", "warning", "timer", "solver", "solver_niter", "solver_nnz", "solver_fwdinv", "energy"}


class ScalarHandle:
  def __init__(self, model: mujoco.MjModel, num_sims: int = 1, num_threads: int = 1):
    if num_sims != 1 or num_threads != 1:
      raise ValueError("the scalar lane is exactly one environment on one thread")
    self.model = model
    self.m = model.__copy__()
    self.d = mujoco.MjData(self.m)
    self._spare = mujoco.MjData(self.m)
    self._full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    self._nstate = int(mujoco.mj_stateSize(self.m, self._full))
    self._state = np.zeros((1, self._nstate))
    self._synced = np.zeros((1, self._nstate))
    self._time = np.zeros(1)
    self._warning = np.zeros((1, int(mujoco.mjtWarning.mjNWARNING), 2), np.int32)
    self._opt_scalars: dict[str, np.ndarray] = {}
    mujoco.mj_forward(self.m, self.d)
    self._sync_out()

  num_sims = 1
  num_threads = 1

  @property
  def nstate(self) -> int:
    return self._nstate

  @property
  def state(self) -> np.ndarray:
    return self._state

  def bind(self, name: str) -> np.ndarray:
    if name == "state":
      return self._state
    if name == "time":
      return self._time
    if name == "warning":
      return self._warning
    arr = getattr(self.d, name)
    return arr[None]  # a live view into this mjData

  def expand(self, name: str) -> np.ndarray:
    if hasattr(self.m.opt, name):
      v = getattr(self.m.opt, name)
      if isinstance(v, np.ndarray):
        return v[None]
      if name not in self._opt_scalars:
        self._opt_scalars[name] = np.asarray([v], dtype=np.float64 if isinstance(v, float) else np.int32)
      return self._opt_scalars[name]
    return getattr(self.m, name)[None]

  # ---------------------------------------------------------------- calls
  def _pre(self) -> None:
    for name, arr in self._opt_scalars.items():
      setattr(self.m.opt, name, arr[0].item())
    if not np.array_equal(self._state, self._synced):
      mujoco.mj_setState(self.m, self.d, self._state[0], self._full)
    mujoco.mj_copyData(self._spare, self.m, self.d)

  def _sync_out(self) -> None:
    mujoco.mj_getState(self.m, self.d, self._state[0], self._full)
    self._synced[:] = self._state
    self._time[0] = self.d.time
    for i in range(self._warning.shape[1]):
      w = self.d.warning[i]
      self._warning[0, i, 0], self._warning[0, i, 1] = w.lastinfo, w.number

  def _call(self, ids, fn) -> CallOutcome:
    ids = _as_ids(ids)
    if ids.size == 0:
      return CallOutcome(ids, np.zeros(0, bool))
    if ids.tolist() != [0]:
      raise ValueError("scalar lane has exactly slot 0")
    self._pre()
    try:
      fn()
    except Exception as exc:  # noqa: BLE001 - MuJoCo errors become per-slot failures
      mujoco.mj_copyData(self.d, self.m, self._spare)  # the failed sim keeps its pre-call state
      self._sync_out()
      return CallOutcome(ids, np.ones(1, bool), str(exc), 0)
    self._sync_out()
    return CallOutcome(ids, np.zeros(1, bool))

  def step1(self, ids) -> CallOutcome:
    return self._call(ids, lambda: mujoco.mj_step(self.m, self.d))

  def forward(self, ids) -> CallOutcome:
    return self._call(ids, lambda: mujoco.mj_forward(self.m, self.d))

  def reset(self, ids, keyframe: int = -1) -> CallOutcome:
    def fn():
      if keyframe >= 0:
        mujoco.mj_resetDataKeyframe(self.m, self.d, keyframe)
      else:
        mujoco.mj_resetData(self.m, self.d)
      mujoco.mj_forward(self.m, self.d)
    return self._call(ids, fn)

  def set_const(self, ids) -> CallOutcome:
    return self._call(ids, lambda: mujoco.mj_setConst(self.m, self.d))

  # ---------------------------------------------------- feature state (sleep etc.)
  def extra_state(self) -> dict[str, np.ndarray]:
    out = {}
    for name in dir(self.d):
      if name.startswith("_") or name in _SKIP:
        continue
      try:
        v = getattr(self.d, name)
      except Exception:  # noqa: BLE001
        continue
      if isinstance(v, np.ndarray) and v.dtype != object:
        out[name] = v.copy()
      elif isinstance(v, int | float) and not isinstance(v, bool):
        out[name] = np.asarray([v])
    return out

  def restore_extra(self, snap: dict[str, np.ndarray]) -> None:
    for name, v in snap.items():
      cur = getattr(self.d, name)
      if isinstance(cur, np.ndarray):
        cur[...] = v
      else:
        try:
          setattr(self.d, name, type(cur)(v[0]))
        except (AttributeError, TypeError):
          pass  # read-only derived counters are recomputed by the next call
    self._sync_out()
