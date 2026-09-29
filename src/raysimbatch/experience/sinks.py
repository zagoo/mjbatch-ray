# SPDX-License-Identifier: Apache-2.0
"""Replay sinks (design §6.5).

``DurableReplaySink`` is idempotent with durable acknowledgement: each fragment is a
content-addressed object plus an index row keyed by fragment id, so re-delivery is a
no-op and an ACK means the data is durable. It declares V-trace capability (behavior
log-probabilities and policy versions are in every row).

``LearnerQueueSink`` delivers to an in-memory learner actor: fast but neither durable
nor idempotent (at-least-once). A run with ``require_idempotent_sink=true`` rejects it
at admission; with a durable outbox the loss window is covered by replay from the
outbox, and duplicates are the learner's to drop by fragment id.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import numpy as np

from raysimbatch.protocol import Fragment


@dataclass(frozen=True)
class SinkCapabilities:
  idempotent_append: bool
  durable_ack: bool
  off_policy_correction: Literal["none", "vtrace", "importance_sampling"]
  max_lag_versions: int
  recurrent: bool
  schema_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SinkAck:
  fragment_id: str
  digest: str
  durable_offset: str | None
  accepted_transitions: int
  rejected_reason: str | None = None


@dataclass(frozen=True)
class SinkOffset:
  offset: str
  manifest_uri: str | None


class ReplaySink(Protocol):
  async def capabilities(self) -> SinkCapabilities: ...
  async def append(self, fragment: Fragment, idempotency_key: str) -> SinkAck: ...
  async def flush(self, run_id: str) -> SinkOffset: ...


def fragment_bytes(frag: Fragment) -> bytes:
  buf = io.BytesIO()
  wire = frag.to_wire()
  arrays: dict[str, Any] = {k: v for k, v in wire.items() if isinstance(v, np.ndarray)}  # savez(**arrays)
  for name, arr in wire.get("initial_recurrent", ()):
    arrays[f"initial_recurrent.{name}"] = arr
  meta = {k: (v.hex() if isinstance(v, bytes) else v) for k, v in wire.items()
          if not isinstance(v, np.ndarray | tuple)}
  np.savez(buf, __meta__=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
  return buf.getvalue()


def fragment_from_bytes(blob: bytes) -> Fragment:
  with np.load(io.BytesIO(blob)) as z:
    meta = json.loads(z["__meta__"].tobytes())
    wire: dict[str, Any] = {k: z[k] for k in z.files if k != "__meta__" and not k.startswith("initial_recurrent.")}
    wire["initial_recurrent"] = tuple((k.split(".", 1)[1], z[k]) for k in z.files if k.startswith("initial_recurrent."))
  for k in ("run_id", "schema_hash", "policy_hash"):
    wire[k] = bytes.fromhex(meta[k])
  for k in ("wire_version", "kind", "rows", "segments"):
    wire[k] = meta[k]
  return Fragment.from_wire(wire)


class DurableReplaySink:
  def __init__(self, store, run_id: str):
    self.store, self.run_id = store, run_id
    self.lock = threading.Lock()
    self._init()

  def _init(self) -> None:
    self.store.db.tx(lambda cur: cur.execute(
      "CREATE TABLE IF NOT EXISTS replay_index (run_id TEXT NOT NULL, fragment_id TEXT NOT NULL, seq BIGSERIAL, "
      "digest TEXT NOT NULL, object_uri TEXT NOT NULL, sha256 TEXT NOT NULL, bytes BIGINT NOT NULL, "
      "transitions INTEGER NOT NULL, policy_version BIGINT NOT NULL, PRIMARY KEY (run_id, fragment_id))"))

  async def capabilities(self) -> SinkCapabilities:
    return SinkCapabilities(True, True, "vtrace", 1 << 30, True)

  def append_sync(self, frag: Fragment, key: str, digest: str) -> SinkAck:
    blob = fragment_bytes(frag)
    art = self.store.blobs.put(blob, "application/x-rsb-fragment")

    def run(cur):
      cur.execute(
        "INSERT INTO replay_index (run_id, fragment_id, digest, object_uri, sha256, bytes, transitions, policy_version) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id, fragment_id) DO NOTHING RETURNING seq",
        (self.run_id, key, digest, art.uri, art.sha256, art.byte_length, frag.transitions,
         int(frag.policy_version.max()) if frag.transitions else 0))
      row = cur.fetchone()
      if row is None:
        cur.execute("SELECT seq, digest FROM replay_index WHERE run_id=%s AND fragment_id=%s", (self.run_id, key))
        row = cur.fetchone()
        if row["digest"] != digest:
          from raysimbatch.storage.db import DigestConflict

          raise DigestConflict(f"replay fragment {key}: different digest")
      return int(row["seq"])

    seq = self.store.db.tx(run)
    return SinkAck(key, digest, str(seq), frag.transitions)

  async def append(self, fragment: Fragment, idempotency_key: str) -> SinkAck:
    import asyncio

    return await asyncio.to_thread(self.append_sync, fragment, idempotency_key, fragment.digest())

  async def flush(self, run_id: str) -> SinkOffset:
    import asyncio

    row = await asyncio.to_thread(self.store.db.tx, lambda cur: cur.execute(
      "SELECT coalesce(max(seq),0) AS s, count(*) AS n, coalesce(sum(transitions),0) AS t FROM replay_index "
      "WHERE run_id=%s", (run_id,)).fetchone())
    return SinkOffset(f"{row['s']}:{row['n']}:{row['t']}", None)

  # Learner-side reader.
  def read(self, after_seq: int = 0, limit: int = 1000) -> list[tuple[int, Fragment]]:
    rows = self.store.db.tx(lambda cur: cur.execute(
      "SELECT seq, object_uri, sha256, bytes FROM replay_index WHERE run_id=%s AND seq > %s ORDER BY seq LIMIT %s",
      (self.run_id, after_seq, limit)).fetchall())
    from raysimbatch.protocol import Artifact

    out = []
    for r in rows:
      blob = self.store.blobs.get(Artifact(r["object_uri"], r["sha256"], int(r["bytes"]), "application/x-rsb-fragment"))
      out.append((int(r["seq"]), fragment_from_bytes(blob)))
    return out


class LearnerQueueSink:
  """At-least-once delivery into a learner actor's bounded queue."""

  def __init__(self, learner):
    self.learner = learner

  async def capabilities(self) -> SinkCapabilities:
    return SinkCapabilities(False, False, "vtrace", 8, True)

  async def append(self, fragment: Fragment, idempotency_key: str) -> SinkAck:
    n = await self.learner.put.remote(idempotency_key, fragment.to_wire())
    return SinkAck(idempotency_key, fragment.digest(), None, int(n))

  async def flush(self, run_id: str) -> SinkOffset:
    return SinkOffset(str(await self.learner.size.remote()), None)


def to_rllib_batch(frag: Fragment) -> dict[str, np.ndarray]:
  """RLlib boundary adapter: column dict in SampleBatch/episode naming (converted at
  the sink boundary, never per physics substep)."""
  nxt = np.empty_like(frag.observation)
  term = (frag.flags & 1) != 0
  trunc = (frag.flags & 2) != 0
  for e in range(frag.segments):
    a, b = int(frag.episode_offsets[e]), int(frag.episode_offsets[e + 1])
    nxt[a : b - 1] = frag.observation[a + 1 : b]
    nxt[b - 1] = frag.terminal_observation[e] if (term[b - 1] or trunc[b - 1]) else frag.bootstrap_observation[e]
  return {
    "obs": frag.observation, "new_obs": nxt, "actions": frag.action, "rewards": frag.reward,
    "terminateds": term, "truncateds": trunc, "action_logp": frag.behavior_logp,
    "eps_id": frag.episode_id.astype(np.int64), "t": frag.control_seq.astype(np.int64),
    "agent_index": np.zeros(frag.transitions, np.int64), "valid": (frag.flags & 16) != 0,
  }


def caps_dict(c: SinkCapabilities) -> dict[str, Any]:
  return asdict(c)
