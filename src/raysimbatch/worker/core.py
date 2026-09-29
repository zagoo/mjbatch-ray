# SPDX-License-Identifier: Apache-2.0
"""MJBatchCore: one MJBatch instance plus RaySimBatch's compiled kernels (BatchCore).

Control-step algorithm (design §6.2, amended AM-2/AM-4): for each physics substep,
the compiled controller writes ctrl for the still-active slots, MJBatch steps exactly
those slots once (``step(ids, nstep=1)``), the per-slot outcome is read from public
buffers, failed slots have their pending ctrl write cancelled and leave the active
set, the health kernel checks every stepped slot, and the task kernel's physics-rate
termination removes terminated slots immediately. Successful slots are never replayed.
``forward`` runs once at the control boundary when observations need derived fields.
With ``health.check_each_substep=false`` health runs once per interval, on each slot's
last substep. When, in addition, ctrl is constant over the interval, the interval is one
``step(ids, nstep=decimation)`` call whose recorded state rows let every slot finish, or
be replayed substep by substep, exactly as above (fused stepping, AM-2).

Exact sparse reset: baseline parameter rows are restored, the reset draw is applied
and validated (closure guard), ``set_const`` refreshes only the selected slots,
``reset`` runs with no pending field writes (RaySimBatch never leaves one), the
reset-state distribution is written through the integration-state rows (which MJBatch
treats as authoritative, finding M9/M10), then ``forward``; every kernel's per-slot
state is cleared and the initial observation is produced at sequence 0.

Snapshot/restore copies integration-state rows (including warmstart), randomized
parameter rows, every kernel state row and the slot bookkeeping. Restoring rewrites
the parameter rows, refreshes derived constants with ``set_const`` (a pure function of
those rows), rewrites the state rows and forwards.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from raysimbatch import _native, seeds
from raysimbatch.adapter import CompiledAdapter, Sim2SimAdapter
from raysimbatch.compat.mjbatch_adapter import BatchHandle, check_environment, state_layout
from raysimbatch.config.models import HealthConfig, RandomizationRule
from raysimbatch.protocol import FaultCode, Flag, SlotState, SlotStatus
from raysimbatch.tasks import TaskBinding
from raysimbatch.worker.model import ModelMetadata
from raysimbatch.worker.randomization import Randomizer

TASK_METRICS = int(_native.TASK_METRICS)


class CoreError(RuntimeError):
  pass


@dataclass(frozen=True)
class CoreSpec:
  num_slots: int
  threads: int
  decimation: int
  timestep_ns: int
  keyframe: int = -1
  max_control_steps: int = 1000
  forward_at_control_boundary: bool = True
  fuse_substeps: bool = True
  health: HealthConfig = HealthConfig()
  run_seed: int = 0
  randomization: tuple[RandomizationRule, ...] = ()
  reset_distribution: Mapping[str, Any] | None = None


@dataclass
class AdvanceResult:
  slots: np.ndarray
  status: np.ndarray  # uint8 SlotStatus
  fault: np.ndarray  # uint16 FaultCode
  completed_substeps: np.ndarray  # uint32
  failed_substep: np.ndarray  # int32, -1 if none
  warning_delta: np.ndarray  # int32
  reward: np.ndarray  # float32
  components: np.ndarray  # float32 [n, K]
  flags: np.ndarray  # uint8 Flag
  observation: np.ndarray  # float32 [n, O]: next obs, or terminal obs for ended episodes
  action: np.ndarray  # float32 [n, A] raw policy action
  applied_action: np.ndarray  # float32 [n, A] adapter output (joint targets / efforts)
  control_seq: np.ndarray  # uint64 sequence after this step
  sim_time_ns: np.ndarray  # int64
  metrics: np.ndarray  # float64 [n, TASK_METRICS] (valid for ended episodes)
  native_seconds: float = 0.0
  message: str | None = None


@dataclass
class ResetResult:
  slots: np.ndarray
  ok: np.ndarray  # bool
  fault: np.ndarray  # uint16
  observation: np.ndarray  # float32 [n, O]
  randomization_hash: np.ndarray  # uint8 [n, 32]
  draws: dict[str, np.ndarray] = field(default_factory=dict)
  message: str | None = None


class _Interval:
  """Per-slot outcome arrays of one policy interval, filled by the stepping paths."""

  def __init__(self, n: int):
    self.status = np.zeros(n, np.uint8)
    self.fault = np.zeros(n, np.uint16)
    self.completed = np.zeros(n, np.uint32)
    self.failed_sub = np.full(n, -1, np.int32)
    self.warn = np.zeros(n, np.int32)
    self.ended = np.zeros(n, bool)
    self.message: str | None = None


def _span(s: np.ndarray) -> slice | np.ndarray:
  """A slice when the (sorted, unique) slots are one contiguous run, the common case:
  indexing with it is a plain row copy instead of a gather."""
  if s.size and int(s[-1]) - int(s[0]) + 1 == s.size:
    return slice(int(s[0]), int(s[-1]) + 1)
  return s


def _ids(slots: np.ndarray) -> np.ndarray:
  s = np.ascontiguousarray(slots, dtype=np.int32)
  if s.ndim != 1:
    raise CoreError("slots must be one-dimensional")
  if s.size and (np.any(np.diff(s) <= 0) or s[0] < 0):
    raise CoreError("slots must be sorted, unique and non-negative")
  return s


class MJBatchCore:
  """BatchCore over one MJBatch ``Batch`` (design §6.2 ``BatchCore`` protocol)."""

  _diag_model: mujoco.MjModel | None = None  # private copy for diagnostic queries, made on first use (AM-7)

  def __init__(
    self,
    model: mujoco.MjModel,
    meta: ModelMetadata,
    adapter: CompiledAdapter,
    task: TaskBinding,
    spec: CoreSpec,
    handle_factory=None,
  ):
    self.compat = check_environment()
    if abs(model.opt.timestep - spec.timestep_ns * 1e-9) > 1e-15:
      raise CoreError("model timestep must equal spec.timestep_ns")
    self.model, self.meta, self.spec, self.binding = model, meta, spec, task
    N = spec.num_slots
    self.N = N
    self.dt = spec.timestep_ns * 1e-9
    if handle_factory is None:
      if model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
        raise CoreError("sleep-enabled models run only on the certified scalar lane")
      handle_factory = BatchHandle
    # Declared non-Optional: close() drops it (to None) only when the whole core is discarded.
    # The scalar lane passes its own handle with the same interface.
    self.h: BatchHandle = handle_factory(model, N, spec.threads)
    self.layout = state_layout(model)
    v = self.views = {
      "qpos": self.h.bind("qpos"),
      "qvel": self.h.bind("qvel"),
      "ctrl": self.h.bind("ctrl"),
      "time": self.h.bind("time"),
      "warning": self.h.bind("warning"),
      "qacc": self.h.bind("qacc"),
    }
    if meta.na:
      v["act"] = self.h.bind("act")
    if meta.nsensordata:
      v["sensordata"] = self.h.bind("sensordata")
    if task.needs_site_xpos:
      v["site_xpos"] = self.h.bind("site_xpos")
    self.needs_forward = spec.forward_at_control_boundary and (adapter.needs_forward or task.needs_site_xpos)
    # Slot bookkeeping (part of every snapshot).
    self.book: dict[str, np.ndarray] = {
      "slot_state": np.full(N, SlotState.UNINITIALIZED, np.uint8),
      "env_id": np.zeros(N, np.uint64),
      "episode_id": np.zeros(N, np.uint64),
      "control_seq": np.zeros((N, 1), np.uint64),
      "sim_time_ns": np.zeros((N, 1), np.int64),
      "physics_steps": np.zeros(N, np.uint64),
      "return_sum": np.zeros(N, np.float64),
      "noise_keys": np.zeros((N, 2), np.uint64),
      "task_keys": np.zeros((N, 2), np.uint64),
      "randomization_hash": np.zeros((N, 32), np.uint8),
      "episode_started_ns": np.zeros(N, np.int64),
      "warning_total": np.zeros(N, np.int64),
      "component_sums": np.zeros((N, max(task.num_components, 1)), np.float64),
    }
    b = self.book
    shared = {
      **v,
      "command": np.zeros((N, task.num_commands), np.float64),
      "sim_time_ns": b["sim_time_ns"],
      "noise_keys": b["noise_keys"],
      "control_seq": b["control_seq"],
    }
    self.command = shared["command"]
    self.adapter = Sim2SimAdapter(adapter, shared, N)
    self.compiled = adapter
    self.A, self.O = adapter.num_actions, adapter.obs_dim
    hc = spec.health
    nwarn = v["warning"].shape[1]
    fatal = [int(getattr(mujoco.mjtWarning, "mjWARN_" + c)) for c in hc.fatal_warning_codes]
    self.health_state = {
      "warning_base": np.zeros((N, nwarn), np.int32),
      "expected_time": np.zeros((N, 1), np.float64),
    }
    self.health = _native.Health(
      {
        "max_abs_qvel": hc.max_abs_qvel,
        "max_abs_qacc": hc.max_abs_qacc,
        "quaternion_norm_tolerance": hc.quaternion_norm_tolerance,
        "time_rtol": 1e-9,
        "num_warnings": nwarn,
      },
      {
        "quat_adr": np.asarray(meta.quat_adrs, np.int32),
        "fatal_warnings": np.asarray(fatal, np.int32),
        "qpos": v["qpos"],
        "qvel": v["qvel"],
        "act": v.get("act"),
        "qacc": v["qacc"],
        "warning": v["warning"],
        "time": v["time"],
        **self.health_state,
      },
    )
    self.task_state = {"state": np.zeros((N, task.num_state), np.float64)}
    targs = {
      "params": task.params,
      "iparams": task.iparams,
      "qpos": v["qpos"],
      "qvel": v["qvel"],
      "state": self.task_state["state"],
      "keys": b["task_keys"],
    }
    if task.num_commands:
      targs["command"] = self.command
    if task.needs_site_xpos:
      targs["site_xpos"] = v["site_xpos"]
    self.task = _native.TaskKernel(task.name, task.version, targs)
    self.K = task.num_components
    self.randomizer = Randomizer(self.h, model, spec.randomization, task.task_hash, spec.run_seed)
    self.expanded_fields = sorted(set(self.randomizer.fields) | set(self.randomizer.closure))
    # Preallocated per-call buffers.
    self._fault = np.zeros(N, np.uint16)
    self._warn = np.zeros(N, np.int32)
    self._reset_dist = _ResetDistribution(spec.reset_distribution, meta, self.layout) if spec.reset_distribution else None
    # Fused stepping (AM-2) when it is identical to per-substep stepping: see _fused_interval.
    self.fused = bool(
      spec.fuse_substeps and not spec.health.check_each_substep and spec.decimation > 1
      and self.adapter.fusable and hasattr(self.h, "stepn")
    )
    self.stepping = "fused" if self.fused else "per_substep"
    self.fused_intervals = self.replayed_slots = 0
    if self.fused:
      self._history = np.zeros((N, spec.decimation, self.h.nstate), np.float64)
      self._replay_rows = np.zeros(N * spec.decimation * self.h.nstate, np.float64)  # reshaped per replay
      self._qpos_off = self.layout["qpos"][0]
      self._qvel_off = self.layout["qvel"][0]
      self._ctrl_cols = slice(self.layout["ctrl"][0], self.layout["ctrl"][0] + self.layout["ctrl"][1])
    self._book = _native.Book(
      {"max_control_steps": spec.max_control_steps, "num_components": self.K,
       "status": {k: int(SlotStatus[k]) for k in ("OK", "TERMINATED", "TRUNCATED", "NATIVE_FAILURE",
                                                   "HEALTH_FAILURE")},
       "state": {k: int(SlotState[k]) for k in ("READY", "WAITING_ACTION", "STEPPING", "TERMINAL", "FAILED")},
       "flag": {k: int(Flag[k]) for k in ("VALID", "TERMINATED", "TRUNCATED", "ABORTED")}},
      {k: b[k] for k in ("slot_state", "physics_steps", "warning_total", "control_seq", "sim_time_ns",
                         "return_sum", "component_sums")},
    )
    self.progress = 0  # monotonic native-completion counter (read by the control thread)
    self.observe_s = 0.0  # the last control step's observation time (observe_seconds)
    self.set_const_s: float | None = None  # the last reset's set_const time, None if it ran none
    self.last_native_ns = 0

  # ----------------------------------------------------------------------- reset

  def reset_exact(
    self,
    slots: np.ndarray,
    env_id: np.ndarray,
    episode_id: np.ndarray,
    key_ids: tuple[np.ndarray, np.ndarray],
    parameters: Mapping[str, np.ndarray] | None = None,
    states: np.ndarray | None = None,
    keyframe: int | None = None,
  ) -> ResetResult:
    """Exact sparse reset of ``slots`` into new episodes.

    ``key_ids`` are the placement-independent seed identities (env_id, episode_id)
    for rollouts or (unit_seed, ordinal) for evaluation. ``parameters`` overrides the
    randomization draw with explicit rows (Mode C experiment axes); ``states``
    restores explicit integration-state rows instead of the reset distribution.
    """
    s = _ids(slots)
    n = s.size
    b = self.book
    if np.any(b["slot_state"][s] == SlotState.STEPPING):
      raise CoreError("cannot reset a stepping slot")
    self.set_const_s = None
    b["slot_state"][s] = SlotState.RESETTING
    fault = np.zeros(n, np.uint16)
    th = self.binding.task_hash
    ka, kb = key_ids
    domain = seeds.derive_keys(self.spec.run_seed, th, ka, kb, "domain")
    reset_keys = seeds.derive_keys(self.spec.run_seed, th, ka, kb, "reset")
    b["noise_keys"][s] = seeds.derive_keys(self.spec.run_seed, th, ka, kb, "obs_noise")
    b["task_keys"][s] = seeds.derive_keys(self.spec.run_seed, th, ka, kb, "task")
    # 1. parameters: baseline, draw (or explicit rows), validate, write, refresh.
    rz = self.randomizer
    draws: dict[str, np.ndarray] = {}
    rhash = np.zeros((n, 32), np.uint8)
    message = None
    if rz.fields or parameters:
      rz.restore_baseline(s)
      if parameters:
        values = {f: np.asarray(parameters[f], np.float64) for f in rz.fields if f in parameters}
        extra = set(parameters) - set(rz.fields)
        if extra:
          raise CoreError(f"parameter rows for non-randomized fields {sorted(extra)}")
        for f in rz.fields:
          values.setdefault(f, np.broadcast_to(rz.baseline[f], (n, *rz.baseline[f].shape)).copy())
        for j in range(n):
          rhash[j] = np.frombuffer(
            hashlib.sha256(b"".join(values[f][j].tobytes() for f in rz.fields)).digest(), np.uint8
          )
      else:
        values, rhash, draws = rz.draw(domain)
      ok = rz.validate(values)
      fault[~ok] = FaultCode.RANDOMIZATION_FAILED
      rz.write(s[ok], {f: v[ok] for f, v in values.items()})
      if rz.needs_refresh and ok.any():
        t0 = time.perf_counter()
        out = self.h.set_const(s[ok])
        if not out.ok:
          bad = out.failed_ids
          rz.restore_baseline(bad)
          self.h.set_const(bad)  # baseline rows cannot fail: the template compiled
          fault[np.isin(s, bad)] = FaultCode.SET_CONST_FAILED
          message = out.message
        self.set_const_s = time.perf_counter() - t0
    live = s[fault == 0]
    # 2. reset (no pending field writes exist by construction).
    kf = self.spec.keyframe if keyframe is None else keyframe
    if live.size:
      out = self.h.reset(live, kf)
      if not out.ok:
        fault[np.isin(s, out.failed_ids)] = FaultCode.RESET_FAILED
        message = message or out.message
        live = out.completed_ids
    # 3. initial state: explicit rows, or the reset distribution, via state rows.
    if live.size and (states is not None or self._reset_dist is not None):
      if states is not None:
        pos = np.searchsorted(s, live)
        self.h.state[live] = np.asarray(states)[pos]
      else:
        assert self._reset_dist is not None  # no explicit states: the distribution applies
        rk = reset_keys[np.searchsorted(s, live)]
        self._reset_dist.apply(self.h.state, live, rk)
      out = self.h.forward(live)
      if not out.ok:
        fault[np.isin(s, out.failed_ids)] = FaultCode.RESET_FAILED
        message = message or out.message
        live = out.completed_ids
    # 4. kernel and bookkeeping state.
    ok = fault == 0
    self.adapter.reset(s)
    self.task.reset(s)
    self.health.rebase(s)
    b["env_id"][s] = env_id
    b["episode_id"][s] = episode_id
    b["control_seq"][s] = 0
    b["sim_time_ns"][s] = 0
    b["physics_steps"][s] = 0
    b["return_sum"][s] = 0.0
    b["component_sums"][s] = 0.0
    b["warning_total"][s] = 0
    b["randomization_hash"][s] = rhash
    b["episode_started_ns"][s] = time.time_ns()
    b["slot_state"][s] = np.where(ok, SlotState.READY, SlotState.FAILED)
    obs = np.zeros((n, self.O), np.float32)
    if live.size:
      live_obs = np.zeros((live.size, self.O), np.float32)
      self.adapter.observe(live, live_obs)
      obs[np.searchsorted(s, live)] = live_obs
    self.progress += 1
    return ResetResult(s, ok, fault, obs, rhash, draws, message)

  # --------------------------------------------------------------------- advance

  def advance_control(self, slots: np.ndarray, actions: np.ndarray) -> AdvanceResult:
    """One policy interval (``decimation`` physics substeps) for ``slots``."""
    s = np.ascontiguousarray(slots, dtype=np.int32)
    if s.ndim != 1:
      raise CoreError("slots must be one-dimensional")
    n = s.size
    b = self.book
    bad = self._book.ready(s)  # the checks of _ids, plus the slot states, in one pass
    if bad == 1:
      raise CoreError("slots must be sorted, unique and non-negative")
    if bad == 2:
      raise CoreError("advance requires READY/WAITING_ACTION slots")
    actions = np.ascontiguousarray(actions, np.float32)
    if actions.shape != (n, self.A):
      raise CoreError(f"actions must have shape ({n}, {self.A})")
    if not np.isfinite(actions).all():
      raise CoreError("nonfinite actions must be rejected before mutation")
    self._book.begin(s)
    out = _Interval(n)
    self.adapter.process_action(s, actions)
    applied = self.adapter.state["targets"][s].astype(np.float32)
    t0 = time.perf_counter_ns()
    if self.fused:
      self._fused_interval(s, out)
    else:
      self._substeps(s.copy(), np.arange(n, dtype=np.int32), out)
    native_ns = time.perf_counter_ns() - t0
    status, fault, completed, failed_sub, warn, ended, message = (
      out.status, out.fault, out.completed, out.failed_sub, out.warn, out.ended, out.message)
    # Counters, the control sequence and the time limit (OK slots at max_control_steps
    # are truncated); `good` slots stepped or terminated cleanly.
    seq = np.empty(n, np.uint64)
    good = np.empty(n, bool)
    self._book.account(s, status, completed, warn, ended, seq, good)
    good_ids = s[good]
    if self.needs_forward and good_ids.size:
      out = self.h.forward(good_ids)
      if not out.ok:
        bad_pos = np.flatnonzero(good)[out.failed]
        status[bad_pos] = SlotStatus.NATIVE_FAILURE
        fault[bad_pos] = FaultCode.MUJOCO_ERROR
        message = message or out.message
        good[bad_pos] = False
        good_ids = s[good]
    reward = np.zeros(n, np.float32)
    comps = np.zeros((n, self.K), np.float32)
    obs = np.zeros((n, self.O), np.float32)
    if good_ids.size == n:  # terminal observations are captured here, pre-reset
      self.task.control(s, reward, comps)
      t0 = time.perf_counter()
      self.adapter.observe(s, obs)
      self.observe_s = time.perf_counter() - t0
    elif good_ids.size:
      r = np.zeros(good_ids.size, np.float32)
      c = np.zeros((good_ids.size, self.K), np.float32)
      self.task.control(good_ids, r, c)
      reward[good], comps[good] = r, c
      o = np.zeros((good_ids.size, self.O), np.float32)
      t0 = time.perf_counter()
      self.adapter.observe(good_ids, o)
      self.observe_s = time.perf_counter() - t0
      obs[good] = o
    # Return sums, flags, aborted slots ended, and each slot's next state.
    flags = np.empty(n, np.uint8)
    self._book.settle(s, status, good, reward, comps, ended, flags)
    metrics = np.full((n, TASK_METRICS), np.nan)
    if ended.any():
      m = np.zeros((int(ended.sum()), TASK_METRICS))
      self.task.metrics(s[ended], m)
      metrics[ended] = m
    self.progress += 1
    self.last_native_ns = time.monotonic_ns()
    return AdvanceResult(
      slots=s,
      status=status,
      fault=fault,
      completed_substeps=completed,
      failed_substep=failed_sub,
      warning_delta=warn,
      reward=reward,
      components=comps,
      flags=flags,
      observation=obs,
      action=actions,
      applied_action=applied,
      control_seq=seq,
      sim_time_ns=b["sim_time_ns"][s, 0].copy(),
      metrics=metrics,
      native_seconds=native_ns * 1e-9,
      message=message,
    )

  def _substeps(self, active: np.ndarray, pos: np.ndarray, r: _Interval) -> None:
    """The policy interval substep by substep for ``active`` (positions ``pos`` in the
    result), from substep 0. Health runs after every substep, or, with
    ``health.check_each_substep=false``, once per interval on each slot's last substep
    (the substep it terminated on, or the final one), before its termination counts
    (``Book.substep_settle``). ``active`` and ``pos`` are int32 arrays owned here: they
    are compacted in place."""
    each = self.spec.health.check_each_substep
    last = self.spec.decimation - 1
    for k in range(self.spec.decimation):
      if active.size == 0:
        break
      self.adapter.before_substep(active, self.dt)
      out = self.h.step1(active)
      if not out.ok:
        bad = out.failed
        self.adapter.neutralize(active[bad])
        r.status[pos[bad]] = SlotStatus.NATIVE_FAILURE
        r.fault[pos[bad]] = FaultCode.MUJOCO_ERROR
        r.failed_sub[pos[bad]] = k
        r.message = r.message or out.message
        active, pos = active[~bad], pos[~bad]
        if active.size == 0:
          break
      kept = self._book.substep_settle(self.health, self.task, active, pos, k, last, each, self.dt,
                                       self.spec.timestep_ns, r.status, r.fault, r.completed, r.failed_sub,
                                       r.warn, r.ended)
      active, pos = active[:kept], pos[:kept]

  def _check_health(self, active: np.ndarray, pos: np.ndarray, sel: np.ndarray, dt: float, k: int,
                    r: _Interval) -> None:
    """Health of ``active[sel]`` over the last ``dt`` seconds; failures are recorded on ``r``."""
    idx = np.flatnonzero(sel)
    if idx.size == 0:
      return
    ids, p = active[idx], pos[idx]
    f = self._fault[: ids.size]
    w = self._warn[: ids.size]
    self.health.check(ids, dt, f, w)
    r.warn[p] += w
    bad = f != 0
    if bad.any():
      r.status[p[bad]] = SlotStatus.HEALTH_FAILURE
      r.fault[p[bad]] = f[bad]
      r.failed_sub[p[bad]] = k

  def _fused_interval(self, s: np.ndarray, r: _Interval) -> None:
    """One MJBatch call for the whole interval (AM-2), identical to ``_substeps``.

    ctrl is constant over the interval (``Controller.fusable``) and health is checked at
    control rate, so the only per-substep work left is the controller's bookkeeping and
    termination, both functions of the recorded state (history rows, M17). Slots that
    completed the interval without terminating before its last substep finish from the
    history. A slot that terminated on an earlier substep k is rewound and stepped again
    to k in one call (``_replay_terminated``); a natively failed slot is replayed by
    ``_substeps`` to find its failing substep. Either way it reports the substep, state
    and results per-substep stepping would.
    """
    n, d = s.size, self.spec.decimation
    ix = _span(s)
    pre_state = self.h.state[ix].copy()
    pre_warn = self.views["warning"][ix].copy()
    self.adapter.prepare_fused(s)
    hist = self._history[:n]
    out = self.h.stepn(s, d, hist)
    # One native pass: finish every slot that completed the interval without terminating
    # before its last substep; mark the others for replay.
    first = np.empty(n, np.int32)
    replay = np.empty(n, bool)
    self._book.fused_settle(self.adapter.controller, self.health, self.task, s, out.failed, hist, self._qpos_off,
                            self._qvel_off, self.dt, self.spec.timestep_ns, r.status, r.fault, r.completed,
                            r.failed_sub, r.warn, r.ended, first, replay)
    self.fused_intervals += 1
    if replay.any():
      again = np.flatnonzero(replay).astype(np.int32)
      self.replayed_slots += again.size
      failed = out.failed[again]
      term = again[~failed]
      if term.size:
        self._replay_terminated(s, term, first[term], pre_state, pre_warn, r)
      bad = again[failed]
      if bad.size:
        # A failed sim was not written back (its state row, warning counters and views
        # are still the pre-interval ones); only the call's ctrl is pending: cancel it.
        self.adapter.neutralize(s[bad])
        self._substeps(s[bad], bad, r)
    r.message = r.message or out.message

  def _replay_terminated(self, s: np.ndarray, rows: np.ndarray, first: np.ndarray, pre_state: np.ndarray,
                         pre_warn: np.ndarray, r: _Interval) -> None:
    """Slots ``s[rows]`` terminated on substep ``first`` < decimation-1 of a fused call that
    succeeded. Their pre-interval state rows (carrying the interval's constant ctrl) and
    warning counters are written back, and each group sharing a terminal substep k is
    stepped k+1 substeps in one call: the same trajectory (M13), stopped where
    per-substep stepping stops it. Bookkeeping then follows as for a clean slot."""
    nstate = self.h.nstate
    for k in np.unique(first):
      g = rows[first == k]
      gs, m, steps = s[g], g.size, int(k) + 1
      back = pre_state[g]
      back[:, self._ctrl_cols] = self.views["ctrl"][gs]  # the call's copied-out constant ctrl
      self.h.state[gs] = back
      self.views["warning"][gs] = pre_warn[g]  # applied when the replay call copies inputs in
      hist = self._replay_rows[: m * steps * nstate].reshape(m, steps, nstate)
      out = self.h.stepn(gs, steps, hist)
      local = np.arange(m, dtype=np.int32)
      again = np.zeros(m, np.int32)
      if out.ok:
        self.task.first_termination(local, hist, self._qpos_off, again)
      if not out.ok or np.any(again != k):
        raise CoreError(f"replaying slots {gs.tolist()} to substep {k} did not reproduce the fused call")
      self.adapter.finish_fused(gs, local, hist, self._qpos_off, self._qvel_off, self.dt)
      r.completed[g] = steps
      self.book["sim_time_ns"][gs, 0] += steps * self.spec.timestep_ns
      self._check_health(gs, g, np.ones(m, bool), self.dt * steps, int(k), r)
      done = g[r.status[g] == SlotStatus.OK]
      if done.size:
        self.task.mark_terminated(s[done])
        r.status[done] = SlotStatus.TERMINATED
        r.ended[done] = True

  def observe(self, slots: np.ndarray, output: np.ndarray) -> None:
    """Recompute the current observation without advancing history (diagnostics)."""
    raise CoreError("observations are produced by reset/advance; history must not be pushed twice")

  # ------------------------------------------------------------------ snapshots

  def save_state(self, slots: np.ndarray) -> dict[str, np.ndarray]:
    s = _ids(slots)
    b = self.book
    if np.any(b["slot_state"][s] == SlotState.STEPPING):
      raise CoreError("cannot snapshot a stepping slot")
    snap: dict[str, np.ndarray] = {"mj/state": self.h.state[s].copy()}
    for f in self.expanded_fields:
      snap[f"param/{f}"] = self.h.expand(f)[s].copy()
    for k, a in self.adapter.save(s).items():
      snap[f"adapter/{k}"] = a
    for k, a in self.health_state.items():
      snap[f"health/{k}"] = a[s].copy()
    snap["task/state"] = self.task_state["state"][s].copy()
    snap["task/command"] = self.command[s].copy()
    for k, a in b.items():
      snap[f"book/{k}"] = a[s].copy()
    extra_state = getattr(self.h, "extra_state", None)
    if extra_state is not None:  # scalar lane: complete feature state (sleep, islands)
      for k, v in extra_state().items():
        snap[f"mj/extra/{k}"] = v
    return snap

  def load_state(self, slots: np.ndarray, snap: Mapping[str, np.ndarray]) -> np.ndarray:
    """Restore rows; returns per-slot success. Derived constants are recomputed and
    must match the snapshot bit for bit (the snapshot's engine/extension hashes are
    checked by the caller before this is called)."""
    s = _ids(slots)
    b = self.book
    if np.any(b["slot_state"][s] == SlotState.STEPPING):
      raise CoreError("cannot restore into a stepping slot")
    ok = np.ones(s.size, bool)
    for f in self.randomizer.fields:
      self.h.expand(f)[s] = snap[f"param/{f}"]
    if self.randomizer.needs_refresh:
      out = self.h.set_const(s)
      ok &= ~out.failed
      for f in self.expanded_fields:
        ok &= np.all(self.h.expand(f)[s].reshape(s.size, -1) == snap[f"param/{f}"].reshape(s.size, -1), axis=1)
    self.h.state[s] = snap["mj/state"]
    out = self.h.forward(s)
    ok &= ~out.failed
    restore_extra = getattr(self.h, "restore_extra", None)
    if restore_extra is not None:
      restore_extra({k.removeprefix("mj/extra/"): v for k, v in snap.items() if k.startswith("mj/extra/")})
    self.adapter.restore(s, {k.removeprefix("adapter/"): v for k, v in snap.items() if k.startswith("adapter/")})
    for k, a in self.health_state.items():
      a[s] = snap[f"health/{k}"]
    self.task_state["state"][s] = snap["task/state"]
    self.command[s] = snap["task/command"]
    for k, a in b.items():
      a[s] = snap[f"book/{k}"]
    b["slot_state"][s] = np.where(ok, snap["book/slot_state"], SlotState.FAILED)
    self.progress += 1
    return ok

  # ---------------------------------------------------------------- diagnostics

  def query_jacobians(self, slots: np.ndarray, site_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Site Jacobians computed on a private scalar model/data; the Batch is not touched,
    so the trajectory (including warmstart) cannot change (verified bitwise in tests)."""
    s = _ids(slots)
    site_ids = np.asarray(site_ids, np.int32)
    m = _private_model(self)
    d = mujoco.MjData(m)
    nv = self.meta.nv
    jacp = np.zeros((s.size, site_ids.size, 3, nv))
    jacr = np.zeros((s.size, site_ids.size, 3, nv))
    full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    for j, slot in enumerate(s):
      for f in self.expanded_fields:
        _assign_field(m, f, self.h.expand(f)[slot])
      mujoco.mj_setState(m, d, self.h.state[slot], full)
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      for k, sid in enumerate(site_ids):
        mujoco.mj_jacSite(m, d, jacp[j, k], jacr[j, k], int(sid))
    return jacp, jacr

  def memory_census(self) -> dict[str, int]:
    arrays = [*self.book.values(), *self.adapter.state.values(), *self.health_state.values(),
              *self.task_state.values(), self.command]
    return {
      "kernel_state_bytes": int(sum(a.nbytes for a in arrays)),
      "state_bytes": int(self.h.state.nbytes),
      "bound_bytes": int(sum(a.nbytes for a in self.views.values())),
      "expanded_bytes": int(sum(self.h.expand(f).nbytes for f in self.expanded_fields)),
    }

  def close(self) -> None:
    self.views.clear()
    self.h = None  # type: ignore[assignment]


def _private_model(core: MJBatchCore) -> mujoco.MjModel:
  m = getattr(core, "_diag_model", None)
  if m is None:
    import copy

    m = copy.copy(core.model)
    core._diag_model = m
  return m


def _assign_field(m: mujoco.MjModel, name: str, row: np.ndarray) -> None:
  if hasattr(m.opt, name):
    target = getattr(m.opt, name)
    if isinstance(target, np.ndarray):
      target[...] = row
    else:
      setattr(m.opt, name, row.item() if hasattr(row, "item") else row)
  else:
    getattr(m, name)[...] = row.reshape(getattr(m, name).shape)


class _ResetDistribution:
  """Initial-state distribution applied through integration-state rows.

  Document: {"qpos": [{"joints": [...], "low": a, "high": b, "operation": "add"|"replace"}],
             "qvel": [...]}; draws use the ``reset`` stream.
  """

  def __init__(self, doc: Mapping[str, Any], meta: ModelMetadata, layout: Mapping[str, tuple[int, int]]):
    self.terms: list[tuple[str, np.ndarray, float, float, str]] = []
    for comp in ("qpos", "qvel"):
      off = layout[comp][0]
      for t in doc.get(comp, ()):
        adrs = []
        for jn in t["joints"]:
          j = meta.joint(jn)
          if j.nv != 1:
            raise CoreError(f"reset distribution joint {jn!r} must be 1-DoF")
          adrs.append(off + (j.qposadr if comp == "qpos" else j.dofadr))
        op = t.get("operation", "add")
        if op not in ("add", "replace"):
          raise CoreError("reset distribution operation must be add|replace")
        self.terms.append((comp, np.asarray(adrs, np.int64), float(t["low"]), float(t["high"]), op))

  def apply(self, state: np.ndarray, slots: np.ndarray, keys: np.ndarray) -> None:
    for i, (_, adrs, lo, hi, op) in enumerate(self.terms):
      u = seeds.uniform(keys, i, adrs.size, "reset")
      v = lo + (hi - lo) * u
      if op == "add":
        state[np.ix_(slots, adrs)] += v
      else:
        state[np.ix_(slots, adrs)] = v
