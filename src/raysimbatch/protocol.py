# SPDX-License-Identifier: Apache-2.0
"""Transport records (wire version RSB/1).

A payload that crosses the object store is a ``dict`` of primitive metadata (ints,
bytes, str, tuples) plus C-contiguous little-endian NumPy arrays. Object dtypes,
ragged row lists and pickled user classes are rejected by ``decode_*``: consumers
validate endianness, dtype, contiguity and shape before use. Frames are split by
schema group and never padded to the largest robot (design §7.1).

The dataclasses here are the in-process view; ``to_wire``/``from_wire`` convert.
"""

from __future__ import annotations

import enum
import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

WIRE_VERSION = 1
_LE = "<"


class Flag(enum.IntFlag):
  TERMINATED = 1
  TRUNCATED = 2
  ABORTED = 4
  RESET = 8
  VALID = 16


class FaultCode(enum.IntEnum):
  """uint16 fault codes. 1-9 mirror native/src/kernels.h."""

  NONE = 0
  NONFINITE_STATE = 1
  VELOCITY_BOUND = 2
  ACCELERATION_BOUND = 3
  QUATERNION_NORM = 4
  FATAL_WARNING = 5
  TIME_DISCONTINUITY = 6
  MUJOCO_ERROR = 10
  RESET_FAILED = 11
  RANDOMIZATION_FAILED = 12
  SET_CONST_FAILED = 13
  QUARANTINED = 14
  STALE_GENERATION = 100
  STALE_ROUTING_EPOCH = 101
  SEQUENCE_MISMATCH = 102
  EPISODE_MISMATCH = 103
  DUPLICATE_ROW = 104
  SCHEMA_MISMATCH = 105
  NONFINITE_ACTION = 106
  SLOT_NOT_READY = 107
  POLICY_MISMATCH = 108
  REPLAY_REQUIRED = 109
  FENCED = 110
  DEADLINE_EXCEEDED = 111
  CANCELED = 112
  UNKNOWN_ENV = 113
  PUBLICATION_FAILED = 200
  INFERENCE_DEADLINE = 201
  INFERENCE_ERROR = 202
  CREDIT_EXHAUSTED = 203
  OBJECT_LOST = 204
  SINK_REJECTED = 205
  WORKER_DIED = 300
  COORDINATOR_LOST = 301
  STORAGE_UNAVAILABLE = 302
  ARTIFACT_CORRUPT = 303
  LEASE_EXPIRED = 304
  RESTARTS_EXHAUSTED = 305


HEALTH_FAULTS = frozenset(
  {
    FaultCode.NONFINITE_STATE,
    FaultCode.VELOCITY_BOUND,
    FaultCode.ACCELERATION_BOUND,
    FaultCode.QUATERNION_NORM,
    FaultCode.FATAL_WARNING,
    FaultCode.TIME_DISCONTINUITY,
  }
)
NATIVE_FAULTS = frozenset({FaultCode.MUJOCO_ERROR, FaultCode.SET_CONST_FAILED, FaultCode.RESET_FAILED})


class SlotStatus(enum.IntEnum):
  """Per-slot outcome of one worker command (uint8 on the wire)."""

  OK = 0
  TERMINATED = 1
  TRUNCATED = 2
  NATIVE_FAILURE = 3
  HEALTH_FAILURE = 4
  REJECTED = 5
  ABORTED = 6


class WorkerState(enum.IntEnum):
  CREATED = 0
  LOADING = 1
  WARMING = 2
  READY = 3
  BUSY = 4
  DRAINING = 5
  STOPPED = 6
  SUSPECT = 7
  FAILED = 8
  FENCED = 9
  RESTARTING = 10
  QUARANTINED = 11


class SlotState(enum.IntEnum):
  UNINITIALIZED = 0
  RESETTING = 1
  READY = 2
  WAITING_ACTION = 3
  STEPPING = 4
  TERMINAL = 5
  FAILED = 6
  QUARANTINED = 7
  DRAINED = 8
  DISABLED = 9


def new_id() -> bytes:
  return uuid.uuid4().bytes


def hexid(b: bytes) -> str:
  return b.hex()


def sha256_hex(*parts: bytes | str) -> str:
  h = hashlib.sha256()
  for p in parts:
    b = p.encode() if isinstance(p, str) else p
    h.update(len(b).to_bytes(8, "little"))
    h.update(b)
  return h.hexdigest()


# ------------------------------------------------------------------ array checks


class WireError(ValueError):
  """A payload violates the RSB/1 wire contract."""


def check_array(name: str, a: Any, dtype: str, shape: tuple[int | None, ...]) -> np.ndarray:
  """Validate one wire array: exact dtype (little-endian), C-contiguous, shape."""
  if not isinstance(a, np.ndarray):
    raise WireError(f"{name}: expected ndarray, got {type(a).__name__}")
  want = np.dtype(dtype).newbyteorder(_LE) if np.dtype(dtype).itemsize > 1 else np.dtype(dtype)
  if a.dtype == object or a.dtype != want:
    raise WireError(f"{name}: dtype {a.dtype} != {want}")
  if a.dtype.byteorder == ">":
    raise WireError(f"{name}: big-endian")
  if not a.flags.c_contiguous:
    raise WireError(f"{name}: not C-contiguous")
  if a.ndim != len(shape) or any(s is not None and a.shape[i] != s for i, s in enumerate(shape)):
    raise WireError(f"{name}: shape {a.shape} does not match {shape}")
  return a


def _bytes(name: str, v: Any, n: int) -> bytes:
  if not isinstance(v, bytes) or len(v) != n:
    raise WireError(f"{name}: expected {n} bytes")
  return v


def _int(name: str, v: Any) -> int:
  if not isinstance(v, int) or isinstance(v, bool) or v < 0:
    raise WireError(f"{name}: expected a non-negative int")
  return v


# ------------------------------------------------------------------------ frames

_ROW_FIELDS: dict[str, str] = {
  "env_id": "uint64",
  "episode_id": "uint64",
  "control_seq": "uint64",
  "sim_time_ns": "int64",
  "slot_generation": "uint64",
  "slot": "int32",
  "reward": "float32",
  "discount": "float32",
  "behavior_logp": "float32",
  "flags": "uint8",
  "policy_version": "uint64",
  "curriculum_version": "uint64",
  "fault_code": "uint16",
}


@dataclass(frozen=True)
class GroupFrame:
  """Rows of one schema group (same model/task/schema/policy)."""

  group_hash: bytes
  schema_hash: bytes
  policy_hash: bytes
  env_id: np.ndarray
  episode_id: np.ndarray
  control_seq: np.ndarray
  sim_time_ns: np.ndarray
  slot_generation: np.ndarray
  slot: np.ndarray
  observation: np.ndarray
  reward: np.ndarray
  discount: np.ndarray
  behavior_logp: np.ndarray
  flags: np.ndarray
  policy_version: np.ndarray
  curriculum_version: np.ndarray
  fault_code: np.ndarray
  action: np.ndarray | None = None
  value: np.ndarray | None = None
  recurrent: Mapping[str, np.ndarray] = field(default_factory=dict)
  sample_keys: np.ndarray | None = None  # uint64 [N, 2] counter keys for stochastic heads

  @property
  def rows(self) -> int:
    return int(self.env_id.shape[0])

  def to_wire(self) -> dict[str, Any]:
    d: dict[str, Any] = {
      "group_hash": self.group_hash,
      "schema_hash": self.schema_hash,
      "policy_hash": self.policy_hash,
      "rows": self.rows,
      "observation": self.observation,
      "recurrent": tuple(sorted((k, v) for k, v in self.recurrent.items())),
    }
    for k in _ROW_FIELDS:
      d[k] = getattr(self, k)
    if self.action is not None:
      d["action"] = self.action
    if self.value is not None:
      d["value"] = self.value
    if self.sample_keys is not None:
      d["sample_keys"] = self.sample_keys
    return d

  @classmethod
  def from_wire(cls, d: Mapping[str, Any]) -> GroupFrame:
    n = _int("rows", d["rows"])
    kw: dict[str, Any] = {}
    for k, dt in _ROW_FIELDS.items():
      kw[k] = check_array(k, d[k], dt, (n,))
    obs = check_array("observation", d["observation"], "float32", (n, None))
    action = d.get("action")
    if action is not None:
      action = check_array("action", action, "float32", (n, None))
    value = d.get("value")
    if value is not None:
      value = check_array("value", value, "float32", (n,))
    rec = {}
    for name, arr in d.get("recurrent", ()):
      if not isinstance(name, str) or not isinstance(arr, np.ndarray) or arr.shape[:1] != (n,):
        raise WireError("recurrent: malformed")
      rec[name] = check_array(f"recurrent.{name}", arr, "float32", (n,) + (None,) * (arr.ndim - 1))
    sk = d.get("sample_keys")
    return cls(
      group_hash=_bytes("group_hash", d["group_hash"], 32),
      schema_hash=_bytes("schema_hash", d["schema_hash"], 32),
      policy_hash=_bytes("policy_hash", d["policy_hash"], 32),
      observation=obs,
      action=action,
      value=value,
      recurrent=rec,
      sample_keys=None if sk is None else check_array("sample_keys", sk, "uint64", (n, 2)),
      **kw,
    )


@dataclass(frozen=True)
class HeteroFrame:
  run_id: bytes
  request_id: bytes
  producer_generation: int
  routing_epoch: int
  created_unix_ns: int
  trace_id: bytes
  groups: tuple[GroupFrame, ...]

  @property
  def rows(self) -> int:
    return sum(g.rows for g in self.groups)

  def to_wire(self) -> dict[str, Any]:
    return {
      "wire_version": WIRE_VERSION,
      "kind": "hetero_frame",
      "run_id": self.run_id,
      "request_id": self.request_id,
      "producer_generation": self.producer_generation,
      "routing_epoch": self.routing_epoch,
      "created_unix_ns": self.created_unix_ns,
      "trace_id": self.trace_id,
      "groups": tuple(g.to_wire() for g in self.groups),
    }

  @classmethod
  def from_wire(cls, d: Mapping[str, Any]) -> HeteroFrame:
    if d.get("wire_version") != WIRE_VERSION or d.get("kind") != "hetero_frame":
      raise WireError("not an RSB/1 hetero frame")
    return cls(
      run_id=_bytes("run_id", d["run_id"], 16),
      request_id=_bytes("request_id", d["request_id"], 16),
      producer_generation=_int("producer_generation", d["producer_generation"]),
      routing_epoch=_int("routing_epoch", d["routing_epoch"]),
      created_unix_ns=_int("created_unix_ns", d["created_unix_ns"]),
      trace_id=_bytes("trace_id", d["trace_id"], 16),
      groups=tuple(GroupFrame.from_wire(g) for g in d["groups"]),
    )


_ACTION_ROW_FIELDS: dict[str, str] = {
  "env_id": "uint64",
  "episode_id": "uint64",
  "expected_control_seq": "uint64",
  "slot_generation": "uint64",
  "slot": "int32",
  "behavior_logp": "float32",
  "policy_version": "uint64",
}


@dataclass(frozen=True)
class ActionGroup:
  group_hash: bytes
  schema_hash: bytes
  policy_hash: bytes
  env_id: np.ndarray
  episode_id: np.ndarray
  expected_control_seq: np.ndarray
  slot_generation: np.ndarray
  slot: np.ndarray
  action: np.ndarray
  behavior_logp: np.ndarray
  policy_version: np.ndarray
  value: np.ndarray | None = None
  recurrent: Mapping[str, np.ndarray] = field(default_factory=dict)

  @property
  def rows(self) -> int:
    return int(self.env_id.shape[0])

  def to_wire(self) -> dict[str, Any]:
    d = {
      "group_hash": self.group_hash,
      "schema_hash": self.schema_hash,
      "policy_hash": self.policy_hash,
      "rows": self.rows,
      "action": self.action,
      "recurrent": tuple(sorted(self.recurrent.items())),
    }
    for k in _ACTION_ROW_FIELDS:
      d[k] = getattr(self, k)
    if self.value is not None:
      d["value"] = self.value
    return d

  @classmethod
  def from_wire(cls, d: Mapping[str, Any]) -> ActionGroup:
    n = _int("rows", d["rows"])
    kw = {k: check_array(k, d[k], dt, (n,)) for k, dt in _ACTION_ROW_FIELDS.items()}
    value = d.get("value")
    rec = {}
    for name, arr in d.get("recurrent", ()):
      rec[name] = check_array(f"recurrent.{name}", arr, "float32", (n,) + (None,) * (arr.ndim - 1))
    return cls(
      group_hash=_bytes("group_hash", d["group_hash"], 32),
      schema_hash=_bytes("schema_hash", d["schema_hash"], 32),
      policy_hash=_bytes("policy_hash", d["policy_hash"], 32),
      action=check_array("action", d["action"], "float32", (n, None)),
      value=None if value is None else check_array("value", value, "float32", (n,)),
      recurrent=rec,
      **kw,
    )


@dataclass(frozen=True)
class ActionFrame:
  run_id: bytes
  request_id: bytes
  routing_epoch: int
  created_unix_ns: int
  trace_id: bytes
  groups: tuple[ActionGroup, ...]

  def to_wire(self) -> dict[str, Any]:
    return {
      "wire_version": WIRE_VERSION,
      "kind": "action_frame",
      "run_id": self.run_id,
      "request_id": self.request_id,
      "routing_epoch": self.routing_epoch,
      "created_unix_ns": self.created_unix_ns,
      "trace_id": self.trace_id,
      "groups": tuple(g.to_wire() for g in self.groups),
    }

  @classmethod
  def from_wire(cls, d: Mapping[str, Any]) -> ActionFrame:
    if d.get("wire_version") != WIRE_VERSION or d.get("kind") != "action_frame":
      raise WireError("not an RSB/1 action frame")
    return cls(
      run_id=_bytes("run_id", d["run_id"], 16),
      request_id=_bytes("request_id", d["request_id"], 16),
      routing_epoch=_int("routing_epoch", d["routing_epoch"]),
      created_unix_ns=_int("created_unix_ns", d["created_unix_ns"]),
      trace_id=_bytes("trace_id", d["trace_id"], 16),
      groups=tuple(ActionGroup.from_wire(g) for g in d["groups"]),
    )


@dataclass(frozen=True)
class ParameterFrame:
  """Explicit reset-time parameter rows: field -> float64 [rows, *native_shape]."""

  values: Mapping[str, np.ndarray]
  randomization_hash: np.ndarray  # uint8 [rows, 32]

  def to_wire(self) -> dict[str, Any]:
    return {
      "wire_version": WIRE_VERSION,
      "kind": "parameter_frame",
      "values": tuple(sorted(self.values.items())),
      "randomization_hash": self.randomization_hash,
    }

  @classmethod
  def from_wire(cls, d: Mapping[str, Any]) -> ParameterFrame:
    if d.get("kind") != "parameter_frame":
      raise WireError("not a parameter frame")
    rh = check_array("randomization_hash", d["randomization_hash"], "uint8", (None, 32))
    n = rh.shape[0]
    vals = {}
    for name, arr in d["values"]:
      vals[name] = check_array(f"values.{name}", arr, "float64", (n,) + (None,) * (arr.ndim - 1))
    return cls(vals, rh)


EMPTY_PARAMETERS = ParameterFrame({}, np.zeros((0, 32), np.uint8))


# ---------------------------------------------------------------------- fragments

_FRAG_ROW: dict[str, str] = {
  "env_id": "uint64",
  "episode_id": "uint64",
  "control_seq": "uint64",
  "reward": "float32",
  "behavior_logp": "float32",
  "flags": "uint8",
  "policy_version": "uint64",
  "curriculum_version": "uint64",
  "sim_dt_ns": "int64",
}
_FRAG_SEG: dict[str, str] = {
  "seg_env_id": "uint64",
  "seg_episode_id": "uint64",
  "seg_first_seq": "uint64",
  "seg_flags": "uint8",
}


@dataclass(frozen=True)
class Fragment:
  """Columnar transitions of up to ``fragment_steps`` per env for one slice.

  Rows are grouped into segments (one per env/episode span) by ``episode_offsets``.
  Row t holds the observation the action was taken in, the policy action, the
  adapter-applied action, and the reward/flags that followed. A segment's successor
  observation is the next row, or ``bootstrap_observation`` (non-terminal end) /
  ``terminal_observation`` (terminated or truncated end), captured before any reset.
  """

  run_id: bytes
  schema_hash: bytes
  policy_hash: bytes
  env_id: np.ndarray
  episode_id: np.ndarray
  control_seq: np.ndarray
  observation: np.ndarray
  action: np.ndarray
  applied_action: np.ndarray
  reward: np.ndarray
  behavior_logp: np.ndarray
  flags: np.ndarray
  policy_version: np.ndarray
  curriculum_version: np.ndarray
  sim_dt_ns: np.ndarray
  episode_offsets: np.ndarray
  seg_env_id: np.ndarray
  seg_episode_id: np.ndarray
  seg_first_seq: np.ndarray
  seg_flags: np.ndarray
  bootstrap_observation: np.ndarray
  terminal_observation: np.ndarray
  reward_components: np.ndarray | None = None
  initial_recurrent: Mapping[str, np.ndarray] = field(default_factory=dict)

  @property
  def transitions(self) -> int:
    return int(self.env_id.shape[0])

  @property
  def segments(self) -> int:
    return int(self.seg_env_id.shape[0])

  def fragment_id(self) -> str:
    parts: list[bytes | str] = [self.run_id, self.schema_hash]
    offs = self.episode_offsets
    for e in range(self.segments):
      length = int(offs[e + 1] - offs[e])
      parts.append(
        f"{int(self.seg_env_id[e])}:{int(self.seg_episode_id[e])}:{int(self.seg_first_seq[e])}:{length}"
      )
    return sha256_hex(*parts)

  def digest(self) -> str:
    h = hashlib.sha256()
    for k, v in sorted(self.to_wire().items()):
      h.update(k.encode())
      if isinstance(v, np.ndarray):
        h.update(str(v.dtype).encode() + str(v.shape).encode())
        h.update(v.tobytes())
      elif isinstance(v, tuple):
        for name, arr in v:
          h.update(name.encode())
          h.update(arr.tobytes())
      else:
        h.update(repr(v).encode())
    return h.hexdigest()

  def to_wire(self) -> dict[str, Any]:
    d: dict[str, Any] = {
      "wire_version": WIRE_VERSION,
      "kind": "fragment",
      "run_id": self.run_id,
      "schema_hash": self.schema_hash,
      "policy_hash": self.policy_hash,
      "rows": self.transitions,
      "segments": self.segments,
      "observation": self.observation,
      "action": self.action,
      "applied_action": self.applied_action,
      "episode_offsets": self.episode_offsets,
      "bootstrap_observation": self.bootstrap_observation,
      "terminal_observation": self.terminal_observation,
      "initial_recurrent": tuple(sorted(self.initial_recurrent.items())),
    }
    for k in _FRAG_ROW:
      d[k] = getattr(self, k)
    for k in _FRAG_SEG:
      d[k] = getattr(self, k)
    if self.reward_components is not None:
      d["reward_components"] = self.reward_components
    return d

  @classmethod
  def from_wire(cls, d: Mapping[str, Any]) -> Fragment:
    if d.get("wire_version") != WIRE_VERSION or d.get("kind") != "fragment":
      raise WireError("not an RSB/1 fragment")
    n, e = _int("rows", d["rows"]), _int("segments", d["segments"])
    kw = {k: check_array(k, d[k], dt, (n,)) for k, dt in _FRAG_ROW.items()}
    kw.update({k: check_array(k, d[k], dt, (e,)) for k, dt in _FRAG_SEG.items()})
    offs = check_array("episode_offsets", d["episode_offsets"], "uint32", (e + 1,))
    if e and (offs[0] != 0 or offs[-1] != n or np.any(np.diff(offs.astype(np.int64)) <= 0)):
      raise WireError("episode_offsets must be strictly increasing from 0 to rows")
    obs = check_array("observation", d["observation"], "float32", (n, None))
    o = obs.shape[1]
    rc = d.get("reward_components")
    return cls(
      run_id=_bytes("run_id", d["run_id"], 16),
      schema_hash=_bytes("schema_hash", d["schema_hash"], 32),
      policy_hash=_bytes("policy_hash", d["policy_hash"], 32),
      observation=obs,
      action=check_array("action", d["action"], "float32", (n, None)),
      applied_action=check_array("applied_action", d["applied_action"], "float32", (n, None)),
      episode_offsets=offs,
      bootstrap_observation=check_array("bootstrap_observation", d["bootstrap_observation"], "float32", (e, o)),
      terminal_observation=check_array("terminal_observation", d["terminal_observation"], "float32", (e, o)),
      reward_components=None if rc is None else check_array("reward_components", rc, "float32", (n, None)),
      initial_recurrent={
        name: check_array(f"initial_recurrent.{name}", a, "float32", (e,) + (None,) * (a.ndim - 1))
        for name, a in d.get("initial_recurrent", ())
      },
      **kw,
    )


# -------------------------------------------------------------- command metadata


@dataclass(frozen=True)
class Route:
  actor_name: str
  shard_id: int
  slot: int
  generation: int
  routing_epoch: int


@dataclass(frozen=True)
class Artifact:
  uri: str
  sha256: str
  byte_length: int
  media_type: str


@dataclass(frozen=True)
class Lease:
  owner: str
  epoch: int
  expires_unix_ns: int


@dataclass(frozen=True)
class AdvanceCommand:
  command_id: bytes
  expected_generation: int
  routing_epoch: int
  slots: np.ndarray  # int32 sorted unique
  episode_id: np.ndarray  # uint64 per slot
  control_seq: np.ndarray  # uint64 per slot (expected current seq)
  policy_hash: bytes
  deadline_unix_ns: int
  credit_token: int


@dataclass(frozen=True)
class LocalRunCommand:
  command_id: bytes
  expected_generation: int
  routing_epoch: int
  slots: np.ndarray
  policy_hash: bytes
  policy_version: int
  deadline_unix_ns: int
  credit_token: int
  max_control_steps: int = 8
  autoreset: bool = True


def status_table(n: int) -> dict[str, np.ndarray]:
  """Allocate the per-slot status columns carried in a StepEnvelope."""
  return {
    "slot": np.zeros(n, np.int32),
    "env_id": np.zeros(n, np.uint64),
    "episode_id": np.zeros(n, np.uint64),
    "status": np.zeros(n, np.uint8),
    "fault_code": np.zeros(n, np.uint16),
    "completed_substeps": np.zeros(n, np.uint32),
    "failed_substep": np.full(n, -1, np.int32),
    "warning_delta": np.zeros(n, np.int32),
    "control_steps": np.zeros(n, np.uint32),
    "control_seq": np.zeros(n, np.uint64),
  }


def validate_status(d: Mapping[str, np.ndarray]) -> None:
  n = d["slot"].shape[0]
  for k, dt in (
    ("slot", "int32"),
    ("env_id", "uint64"),
    ("episode_id", "uint64"),
    ("status", "uint8"),
    ("fault_code", "uint16"),
    ("completed_substeps", "uint32"),
    ("failed_substep", "int32"),
    ("warning_delta", "int32"),
    ("control_steps", "uint32"),
    ("control_seq", "uint64"),
  ):
    check_array(k, d[k], dt, (n,))
