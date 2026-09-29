# SPDX-License-Identifier: Apache-2.0
"""The MJBatch compatibility adapter: the only module in RaySimBatch that imports mjbatch.

Everything version-specific about the upstream dependency lives here:

* which mjbatch/MuJoCo versions are qualified (``QUALIFIED``),
* runtime behaviour probes that re-establish, on the running binary, the upstream
  semantics RaySimBatch composes on (docs/ledgers/mjbatch-source-findings.md M8-M14),
* per-slot call outcomes derived from public buffers (no fork, no private layout),
* the ``mjSTATE_INTEGRATION`` component layout (MuJoCo's documented bit order).

Unsupported versions fail with an actionable ``MJBatchIncompatible`` rather than
silently changing behaviour. ``RSB_MJBATCH_CANDIDATE=1`` admits an unqualified version
*only* when every behaviour probe passes; it exists for the upgrade workflow
(tools/mjbatch_upgrade.py) and is rejected by production configuration validation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

try:
  import mjbatch as _mjbatch
  from mjbatch import Batch as _Batch
except ImportError as exc:  # pragma: no cover - exercised by packaging checks
  raise ImportError(
    "raysimbatch requires the pinned upstream mjbatch (third_party/mjbatch). "
    "Install it with `uv sync` from the repository root; see docs/operations/installation.md."
  ) from exc

# Qualified pairs. A new entry is added only by tools/mjbatch_upgrade.py after the
# candidate passes the full gate list in docs/operations/mjbatch-upgrade.md.
QUALIFIED: dict[str, dict[str, str]] = {
  "0.1.1": {"mujoco": "3.13.0", "revision": "2f169146b643adda65e195d8ad13d997abdd2a2e"},
}
CANDIDATE_ENV = "RSB_MJBATCH_CANDIDATE"
_FIRST_ERROR = re.compile(r"^sim (\d+): (.*)$", re.S)


class MJBatchIncompatible(RuntimeError):
  """The installed mjbatch/MuJoCo pair is not qualified or violates a probed contract."""


@dataclass(frozen=True)
class CompatReport:
  mjbatch_version: str
  mujoco_version: str
  mj_version: int
  qualified: bool
  candidate_mode: bool
  binary_sha256: str
  probes: dict[str, bool] = field(default_factory=dict)

  @property
  def abi_fingerprint(self) -> str:
    text = f"mjbatch={self.mjbatch_version};mujoco={self.mujoco_version};bin={self.binary_sha256}"
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class CallOutcome:
  """Per-slot result of one MJBatch call over ``ids`` (sorted int32 slot indices)."""

  ids: np.ndarray
  failed: np.ndarray  # bool per id
  message: str | None = None  # MuJoCo's message for the first failing sim only
  first_failed: int | None = None

  @property
  def ok(self) -> bool:
    return not bool(self.failed.any())

  @property
  def completed_ids(self) -> np.ndarray:
    return self.ids[~self.failed]

  @property
  def failed_ids(self) -> np.ndarray:
    return self.ids[self.failed]


def binary_sha256() -> str:
  """Digest of the installed mjbatch extension module (release-lock identity)."""
  pkg = Path(_mjbatch.__file__).parent
  libs = sorted(pkg.glob("_bindings*"))
  libs = [p for p in libs if p.suffix in (".so", ".pyd", ".dylib")]
  if not libs:
    raise MJBatchIncompatible(f"no compiled mjbatch._bindings found in {pkg}")
  h = hashlib.sha256()
  for p in libs:
    h.update(p.read_bytes())
  return h.hexdigest()


def state_layout(model: mujoco.MjModel) -> dict[str, tuple[int, int]]:
  """(offset, size) of each mjSTATE_INTEGRATION component in a bind("state") row.

  MuJoCo concatenates components in mjtState bit order (mj_getState documentation).
  Verified against mj_getState by tests/compat/test_adapter.py.
  """
  full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  out: dict[str, tuple[int, int]] = {}
  offset = 0
  members = sorted(mujoco.mjtState.__members__.items(), key=lambda kv: int(kv[1]))
  for name, bit in members:
    b = int(bit)
    if b == 0 or b & (b - 1) or not (b & full):
      continue
    size = int(mujoco.mj_stateSize(model, b))
    out[name.removeprefix("mjSTATE_").lower()] = (offset, size)
    offset += size
  if offset != int(mujoco.mj_stateSize(model, full)):
    raise MJBatchIncompatible("mjSTATE_INTEGRATION is not the concatenation of its bits")
  return out


class BatchHandle:
  """The single owner of an ``mjbatch.Batch``.

  Calls are made only from the owning (mutation) thread. Every call reports a
  per-slot ``CallOutcome``. For step/forward/reset the outcome is exact without
  re-execution: before the call the world body's ``xpos`` entry of each selected row
  is set to a sentinel; MJBatch copies bound fields out only for sims that completed
  (M6/M8), and a completed sim's world body position is exactly zero. ``set_const``
  returns early before copy-out, so failures there are isolated by re-running the
  remaining selection: ``mj_setConst`` is a pure function of a sim's parameter rows and
  mutates no integration state, so re-running it replays no transition.
  """

  _SENTINEL = -1.0e300

  def __init__(self, model: mujoco.MjModel, num_sims: int, num_threads: int):
    self._batch = _Batch(model, num_sims, num_threads=num_threads, forward=False)
    self.model = model
    self._xpos = self._batch.bind("xpos")
    self._state = self._batch.bind("state")

  # -- identity -------------------------------------------------------------------
  @property
  def num_sims(self) -> int:
    return int(self._batch.num_sims)

  @property
  def num_threads(self) -> int:
    return int(self._batch.num_threads)

  @property
  def nstate(self) -> int:
    return int(self._batch.nstate)

  @property
  def state(self) -> np.ndarray:
    return self._state

  def bind(self, name: str) -> np.ndarray:
    if name == "state":
      return self._state
    if name == "xpos":
      return self._xpos
    return self._batch.bind(name)

  def expand(self, name: str) -> np.ndarray:
    return self._batch.expand(name)

  # -- calls ------------------------------------------------------------------------
  def step1(self, ids: np.ndarray) -> CallOutcome:
    return self._sentinel_call(ids, lambda sel: self._batch.step(sel, nstep=1))

  def stepn(self, ids: np.ndarray, nstep: int, history: np.ndarray) -> CallOutcome:
    """``nstep`` substeps in one call (bitwise equal to ``nstep`` single steps, M13).
    ``history[j, k]`` receives selected sim j's state row after substep k (M17); the
    rows of a failed sim are undefined."""
    return self._sentinel_call(ids, lambda sel: self._batch.step(sel, nstep=nstep, history=history))

  def forward(self, ids: np.ndarray) -> CallOutcome:
    return self._sentinel_call(ids, lambda sel: self._batch.forward(sel))

  def reset(self, ids: np.ndarray, keyframe: int = -1) -> CallOutcome:
    return self._sentinel_call(ids, lambda sel: self._batch.reset(sel, keyframe=keyframe))

  def set_const(self, ids: np.ndarray) -> CallOutcome:
    ids = _as_ids(ids)
    failed = np.zeros(ids.shape[0], bool)
    remaining = ids
    message: str | None = None
    first: int | None = None
    while remaining.size:
      try:
        self._batch.set_const(remaining)
        break
      except RuntimeError as exc:
        sim, msg = _parse_first(str(exc))
        if sim is None or sim not in remaining:
          raise MJBatchIncompatible(f"unattributable set_const error: {exc}") from exc
        if message is None:
          message, first = msg, sim
        failed[np.searchsorted(ids, sim)] = True
        remaining = remaining[remaining != sim]
    return CallOutcome(ids, failed, message, first)

  def _sentinel_call(self, ids: np.ndarray, call) -> CallOutcome:
    ids = _as_ids(ids)
    if ids.size == 0:
      return CallOutcome(ids, np.zeros(0, bool))
    # A contiguous run of ids (the common case) is addressed as a slice (MJBatch itself
    # rejects ids that are not sorted and unique).
    sel = slice(int(ids[0]), int(ids[-1]) + 1) if int(ids[-1]) - int(ids[0]) + 1 == ids.size else ids
    self._xpos[sel, 0, 0] = self._SENTINEL
    message: str | None = None
    first: int | None = None
    try:
      call(ids)
    except RuntimeError as exc:
      first, message = _parse_first(str(exc))
      if first is None:
        self._xpos[ids, 0, 0] = 0.0
        raise MJBatchIncompatible(f"unattributable MJBatch error: {exc}") from exc
    failed = self._xpos[sel, 0, 0] == self._SENTINEL
    if failed.any():
      self._xpos[ids[failed], 0, 0] = 0.0
      if message is None:
        raise MJBatchIncompatible("sims failed without an error: sentinel contract broken")
    elif message is not None:
      raise MJBatchIncompatible(f"MJBatch raised but every sim completed: {message}")
    return CallOutcome(ids, failed, message, first)


def _as_ids(ids: np.ndarray) -> np.ndarray:
  # MJBatch's nanobind signature requires a *writable* C-contiguous array (M7);
  # arrays deserialized from the Ray object store are read-only.
  ids = np.ascontiguousarray(ids, dtype=np.int32)
  if not ids.flags.writeable:
    ids = ids.copy()
  if ids.ndim != 1:
    raise ValueError("ids must be one-dimensional")
  return ids


def _parse_first(text: str) -> tuple[int | None, str]:
  m = _FIRST_ERROR.match(text)
  if not m:
    return None, text
  return int(m.group(1)), m.group(2)


# ------------------------------------------------------------------------ probes

_PROBE_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
    </body>
    <body name="ball" pos="0 1 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass=".2"/>
    </body>
  </worldbody>
  <equality><connect body1="ball" body2="world" anchor="0 1 .8"/></equality>
  <actuator><motor joint="slide" gear="5"/></actuator>
</mujoco>
"""


def _probe(model: mujoco.MjModel) -> dict[str, bool]:
  """Re-establish the upstream semantics RaySimBatch depends on (a few milliseconds)."""
  n = 4
  results: dict[str, bool] = {}
  h = BatchHandle(model, n, num_threads=2)
  ctrl, time_v = h.bind("ctrl"), h.bind("time")
  h.step1(np.arange(n, dtype=np.int32))
  eq = h.expand("eq_type")
  eq[1] = eq[3] = 99
  saved = ctrl.copy()
  ctrl[:, 0] = 0.5
  t0 = time_v.copy()
  out = h.step1(np.arange(n, dtype=np.int32))
  # With several pool threads the reported sim is the first to fail *in time* (M8).
  results["first_error_attributed"] = out.first_failed in (1, 3)
  results["failed_slots_exact"] = list(out.failed_ids) == [1, 3]
  results["failed_keep_state"] = bool(np.all(time_v[[1, 3]] == t0[[1, 3]]))
  results["others_advanced"] = bool(np.all(time_v[[0, 2]] > t0[[0, 2]]))
  eq[:] = model.eq_type
  ctrl[[1, 3]] = saved[[1, 3]]  # neutralize the pending writes
  before = h.state[[1, 3]].copy()
  ref = BatchHandle(model, 1, num_threads=1)
  ref.state[0] = before[0]
  ref.step1(np.array([0], np.int32))
  h.step1(np.array([1, 3], np.int32))
  results["neutralize_cancels_pending"] = bool(np.array_equal(h.state[1], ref.state[0]))
  st = h.state.copy()
  h.forward(np.arange(n, dtype=np.int32))
  results["forward_pure"] = bool(np.array_equal(st, h.state))
  ctrl[2, 0] = 0.25
  h.reset(np.array([2], np.int32))
  results["reset_applies_pending_field_writes"] = bool(ctrl[2, 0] == 0.25)
  ctrl[2, 0] = 0.0
  h.forward(np.array([2], np.int32))
  layout = state_layout(model)
  d = mujoco.MjData(model)
  mujoco.mj_setState(model, d, h.state[0], int(mujoco.mjtState.mjSTATE_INTEGRATION))
  off, size = layout["qpos"]
  results["state_layout"] = bool(np.array_equal(h.state[0, off : off + size], d.qpos))
  # Fused intervals (AM-2): one step(nstep=k) equals k single steps, and history row k
  # is the state after substep k (M13, M17).
  a, b = BatchHandle(model, 2, num_threads=1), BatchHandle(model, 2, num_threads=1)
  ids = np.arange(2, dtype=np.int32)
  hist = np.zeros((2, 3, a.nstate))
  a.stepn(ids, 3, hist)
  rows = []
  for _ in range(3):
    b.step1(ids)
    rows.append(b.state.copy())
  results["fused_substeps_equal"] = bool(np.array_equal(a.state, b.state))
  results["history_rows_are_states"] = all(np.array_equal(hist[:, k], rows[k]) for k in range(3))
  return results


_REPORT: CompatReport | None = None


def check_environment(force: bool = False) -> CompatReport:
  """Verify the installed mjbatch/MuJoCo pair; cached per process."""
  global _REPORT
  if _REPORT is not None and not force:
    return _REPORT
  version = importlib.metadata.version("mjbatch")
  mj_ver = mujoco.__version__
  candidate = os.environ.get(CANDIDATE_ENV) == "1"
  entry = QUALIFIED.get(version)
  qualified = entry is not None and entry["mujoco"] == mj_ver
  if not qualified and not candidate:
    raise MJBatchIncompatible(
      f"mjbatch {version} with MuJoCo {mj_ver} is not a qualified pair "
      f"(qualified: {QUALIFIED}). Rebuild from the pinned submodule (`git submodule update`, "
      f"`uv sync`) or qualify the candidate with tools/mjbatch_upgrade.py; see "
      f"docs/operations/mjbatch-upgrade.md."
    )
  for attr in ("bind", "expand", "step", "forward", "reset", "set_const", "nstate", "num_sims", "num_threads"):
    if not hasattr(_Batch, attr):
      raise MJBatchIncompatible(f"mjbatch.Batch lacks required attribute {attr!r}")
  probes = _probe(mujoco.MjModel.from_xml_string(_PROBE_XML))
  bad = [k for k, v in probes.items() if not v]
  if bad:
    raise MJBatchIncompatible(
      f"mjbatch {version} violates contracts RaySimBatch depends on: {bad}. "
      "Run tests/upstream against this build and update the adapter before use."
    )
  _REPORT = CompatReport(
    mjbatch_version=version,
    mujoco_version=mj_ver,
    mj_version=int(mujoco.mj_version()),
    qualified=qualified,
    candidate_mode=candidate and not qualified,
    binary_sha256=binary_sha256(),
    probes=probes,
  )
  return _REPORT
