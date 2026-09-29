# SPDX-License-Identifier: Apache-2.0
"""Seed derivation ("rsb-seed-v1") and counter-based draws ("rsb-philox-v1").

A stream key is HMAC-SHA256 keyed by the unsigned 128-bit run seed (16 bytes,
little-endian) over a length-prefixed message
``(version, task_hash, id_a, id_b, stream_name, draw_index)``. For rollouts
``(id_a, id_b) = (env_id, episode_id)``; for evaluation they are
``(unit_seed, episode_ordinal)`` so keys never depend on placement. The first 128
digest bits are the Philox4x64-10 key (two little-endian uint64 words).

Draws are addressed by the Philox counter ``(seq, index, stream_id, 0)``; see
native/src/philox.h. numpy.random.Philox reproduces any block exactly (tests pin it).
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np

from raysimbatch import _native

SEED_VERSION = "rsb-seed-v1"
GENERATOR = "philox4x64-10/rsb-philox-v1"
STREAMS: dict[str, int] = {"reset": 1, "domain": 2, "obs_noise": 3, "action": 4, "task": 5}
_U64 = (1 << 64) - 1


def _field(b: bytes) -> bytes:
  return len(b).to_bytes(4, "little") + b


def derive_key(
  run_seed: int,
  task_hash: str,
  id_a: int,
  id_b: int,
  stream: str,
  draw_index: int = 0,
) -> tuple[int, int]:
  if not 0 <= run_seed < 1 << 128:
    raise ValueError("run_seed must be an unsigned 128-bit integer")
  if stream not in STREAMS:
    raise ValueError(f"unknown stream {stream!r}")
  th = bytes.fromhex(task_hash) if task_hash else b""
  msg = b"".join(
    _field(x)
    for x in (
      SEED_VERSION.encode(),
      th,
      int(id_a).to_bytes(8, "little"),
      int(id_b).to_bytes(8, "little"),
      stream.encode(),
      int(draw_index).to_bytes(8, "little"),
    )
  )
  digest = hmac.new(run_seed.to_bytes(16, "little"), msg, hashlib.sha256).digest()
  k = int.from_bytes(digest[:16], "little")
  return k & _U64, k >> 64


def derive_keys(
  run_seed: int, task_hash: str, id_a: np.ndarray, id_b: np.ndarray, stream: str
) -> np.ndarray:
  """uint64 [n, 2] keys for paired identity columns."""
  out = np.empty((len(id_a), 2), np.uint64)
  for i, (a, b) in enumerate(zip(id_a.tolist(), id_b.tolist(), strict=True)):
    out[i] = derive_key(run_seed, task_hash, a, b, stream)
  return out


def uniform(keys: np.ndarray, seq: int, count: int, stream: str, index0: int = 0) -> np.ndarray:
  out = np.empty((keys.shape[0], count), np.float64)
  _native.philox_uniform(np.ascontiguousarray(keys, np.uint64), seq, index0, STREAMS[stream], out)
  return out


def normal(keys: np.ndarray, seq: int, count: int, stream: str, index0: int = 0) -> np.ndarray:
  out = np.empty((keys.shape[0], count), np.float64)
  _native.philox_normal(np.ascontiguousarray(keys, np.uint64), seq, index0, STREAMS[stream], out)
  return out


def numpy_block(key: tuple[int, int], seq: int, index: int, stream: str) -> tuple[int, int, int, int]:
  """Reference: the same block through numpy.random.Philox (counter pre-increment)."""
  ctr = seq + (index << 64) + (STREAMS[stream] << 128)
  bg = np.random.Philox(key=key[0] + (key[1] << 64), counter=(ctr - 1) % (1 << 256))
  a, b, c, d = (int(x) for x in bg.random_raw(4))
  return a, b, c, d
