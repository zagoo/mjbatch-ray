# SPDX-License-Identifier: Apache-2.0
"""Per-slot transition buffers assembled into columnar Fragments (design §6.5, §7.1).

Each slot buffers up to ``capacity`` transitions split into segments (one per
env/episode span). A segment closes with a terminal observation (terminated or
truncated: captured before any reset), with no successor (aborted: the invalid
suffix is unusable), or stays open and receives a bootstrap observation when the
fragment is cut mid-episode. Buffers are preallocated; ``flush`` copies the selected
rows into one Fragment whose arrays are then published with a single ``ray.put``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raysimbatch.protocol import Flag, Fragment

END_OPEN, END_TERMINAL, END_ABORTED = 0, 1, 2


@dataclass
class _Segment:
  env_id: int
  episode_id: int
  first_seq: int
  start: int  # row offset in the slot buffer
  end_kind: int = END_OPEN
  flags: int = 0
  terminal_obs: np.ndarray | None = None
  initial_recurrent: dict[str, np.ndarray] = field(default_factory=dict)


class FragmentBuilder:
  def __init__(self, run_id: bytes, schema_hash: bytes, num_slots: int, obs_dim: int, action_dim: int,
               num_components: int, fragment_steps: int, recurrent_shapes: dict[str, tuple[int, ...]] | None = None):
    self.run_id, self.schema_hash = run_id, schema_hash
    self.N, self.O, self.A, self.K = num_slots, obs_dim, action_dim, num_components
    self.steps = fragment_steps
    self.cap = 2 * fragment_steps  # hard bound: beyond this the slot must not advance
    cap, N = self.cap, num_slots
    self.obs = np.zeros((N, cap, obs_dim), np.float32)
    self.action = np.zeros((N, cap, action_dim), np.float32)
    self.applied = np.zeros((N, cap, action_dim), np.float32)
    self.reward = np.zeros((N, cap), np.float32)
    self.comps = np.zeros((N, cap, max(num_components, 1)), np.float32)
    self.logp = np.zeros((N, cap), np.float32)
    self.flags = np.zeros((N, cap), np.uint8)
    self.pver = np.zeros((N, cap), np.uint64)
    self.cver = np.zeros((N, cap), np.uint64)
    self.seq = np.zeros((N, cap), np.uint64)
    self.dt = np.zeros((N, cap), np.int64)
    self.count = np.zeros(N, np.int64)
    self.segments: list[list[_Segment]] = [[] for _ in range(N)]
    self.rec_shapes = dict(recurrent_shapes or {})
    self.policy_hash = b"\0" * 32

  def rows(self, slot: int) -> int:
    return int(self.count[slot])

  def full(self, slots: np.ndarray) -> np.ndarray:
    return self.count[slots] >= self.cap

  def begin(self, slot: int, env_id: int, episode_id: int, first_seq: int,
            recurrent: dict[str, np.ndarray] | None = None) -> None:
    self.segments[slot].append(
      _Segment(env_id, episode_id, first_seq, int(self.count[slot]),
               initial_recurrent={k: v.copy() for k, v in (recurrent or {}).items()})
    )

  def append(self, slots: np.ndarray, obs_t: np.ndarray, action: np.ndarray, applied: np.ndarray,
             reward: np.ndarray, comps: np.ndarray, logp: np.ndarray, flags: np.ndarray, seq_t: np.ndarray,
             policy_version: np.ndarray, curriculum_version: np.ndarray, dt_ns: int, next_obs: np.ndarray) -> None:
    """Record one transition per slot; closes segments that ended in this step."""
    c = self.count[slots]
    if np.any(c >= self.cap):
      raise RuntimeError("fragment buffer full: the caller must stop advancing these slots")
    self.obs[slots, c] = obs_t
    self.action[slots, c] = action
    self.applied[slots, c] = applied
    self.reward[slots, c] = reward
    if self.K:
      self.comps[slots, c, : self.K] = comps
    self.logp[slots, c] = logp
    self.flags[slots, c] = flags
    self.seq[slots, c] = seq_t
    self.pver[slots, c] = policy_version
    self.cver[slots, c] = curriculum_version
    self.dt[slots, c] = dt_ns
    self.count[slots] = c + 1
    ended = (flags & (Flag.TERMINATED | Flag.TRUNCATED | Flag.ABORTED)) != 0
    for j in np.flatnonzero(ended):
      seg = self.segments[slots[j]][-1]
      if flags[j] & Flag.ABORTED:
        seg.end_kind = END_ABORTED
      else:
        seg.end_kind = END_TERMINAL
        seg.terminal_obs = next_obs[j].copy()
      seg.flags = int(flags[j])

  def ready(self, slots: np.ndarray, force: bool = False) -> np.ndarray:
    """Slots whose buffers should be flushed now."""
    has = self.count[slots] > 0
    if force:
      return slots[has]
    return slots[has & (self.count[slots] >= self.steps)]

  def flush(self, slots: np.ndarray, next_obs: dict[int, np.ndarray],
            recurrent_now: dict[int, dict[str, np.ndarray]] | None = None) -> Fragment | None:
    """Emit one Fragment for ``slots``; open segments get ``next_obs[slot]`` as their
    bootstrap observation and continue in a fresh segment."""
    if slots.size == 0:
      return None
    rows_idx: list[tuple[int, int, int]] = []  # (slot, start, end)
    segs: list[tuple[int, _Segment, int]] = []
    for s in slots.tolist():
      n = int(self.count[s])
      for i, seg in enumerate(self.segments[s]):
        end = self.segments[s][i + 1].start if i + 1 < len(self.segments[s]) else n
        if end > seg.start:
          rows_idx.append((s, seg.start, end))
          segs.append((s, seg, end - seg.start))
    if not segs:
      return None
    obs = np.concatenate([self.obs[s, a:b] for s, a, b in rows_idx])
    E = len(segs)
    offsets = np.zeros(E + 1, np.uint32)
    offsets[1:] = np.cumsum([n for _, _, n in segs])
    boot = np.zeros((E, self.O), np.float32)
    term = np.zeros((E, self.O), np.float32)
    seg_flags = np.zeros(E, np.uint8)
    rec = {k: np.zeros((E, *shape), np.float32) for k, shape in self.rec_shapes.items()}
    for e, (s, seg, _) in enumerate(segs):
      if seg.end_kind == END_TERMINAL:
        term[e] = seg.terminal_obs
      elif seg.end_kind == END_OPEN:
        boot[e] = next_obs[s]
      seg_flags[e] = seg.flags
      for k in rec:
        if k in seg.initial_recurrent:
          rec[k][e] = seg.initial_recurrent[k]

    def cat(a: np.ndarray) -> np.ndarray:
      return np.ascontiguousarray(np.concatenate([a[s, x:y] for s, x, y in rows_idx]))

    frag = Fragment(
      run_id=self.run_id,
      schema_hash=self.schema_hash,
      policy_hash=self.policy_hash,
      env_id=np.concatenate([np.full(n, seg.env_id, np.uint64) for _, seg, n in segs]),
      episode_id=np.concatenate([np.full(n, seg.episode_id, np.uint64) for _, seg, n in segs]),
      control_seq=cat(self.seq),
      observation=np.ascontiguousarray(obs),
      action=cat(self.action),
      applied_action=cat(self.applied),
      reward=cat(self.reward),
      behavior_logp=cat(self.logp),
      flags=cat(self.flags),
      policy_version=cat(self.pver),
      curriculum_version=cat(self.cver),
      sim_dt_ns=cat(self.dt),
      episode_offsets=offsets,
      seg_env_id=np.array([seg.env_id for _, seg, _ in segs], np.uint64),
      seg_episode_id=np.array([seg.episode_id for _, seg, _ in segs], np.uint64),
      seg_first_seq=np.array([seg.first_seq for _, seg, _ in segs], np.uint64),
      seg_flags=seg_flags,
      bootstrap_observation=boot,
      terminal_observation=term,
      reward_components=np.ascontiguousarray(cat(self.comps)[:, : self.K]) if self.K else None,
      initial_recurrent=rec,
    )
    # Reset buffers; open segments continue in a new segment at the next sequence.
    for s in slots.tolist():
      last = self.segments[s][-1] if self.segments[s] else None
      self.count[s] = 0
      self.segments[s] = []
      if last is not None and last.end_kind == END_OPEN:
        n_rows = sum(n for ss, seg, n in segs if ss == s and seg is last)
        self.begin(s, last.env_id, last.episode_id, last.first_seq + n_rows,
                   (recurrent_now or {}).get(s))
    return frag

  def close_open(self, slot: int) -> bool:
    """Close an open segment as aborted (its last row gets ABORTED, valid cleared)."""
    segs = self.segments[slot]
    if not segs or segs[-1].end_kind != END_OPEN:
      return False
    seg = segs[-1]
    n = int(self.count[slot])
    if n > seg.start:
      self.flags[slot, n - 1] = (self.flags[slot, n - 1] | Flag.ABORTED) & ~np.uint8(Flag.VALID)
      seg.end_kind, seg.flags = END_ABORTED, int(self.flags[slot, n - 1])
    else:
      segs.pop()  # empty segment
    return True

  def discard(self, slots: np.ndarray) -> int:
    """Drop buffered rows (actor-local abort); returns the discarded row count."""
    lost = int(self.count[slots].sum())
    for s in slots.tolist():
      self.count[s] = 0
      self.segments[s] = []
    return lost
