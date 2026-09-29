# SPDX-License-Identifier: Apache-2.0
"""Per-node AssetCache actor (design §5.1, §7.3).

Reads verified durable artifacts, ``ray.put``s the immutable bytes as uint8 arrays and
keeps strong references until released, so consumers on that node share one store
copy (zero-copy numpy reads). It is not the durable owner: a restarted cache simply
re-materializes from storage. Keys include the content hash.

Lifecycle (found in kind qualification: a live helper actor and its pinned objects keep
a node from ever being idle, so the autoscaler never scaled down). Entries are *held*
by the runs that use them; the cache owns the ObjectRefs, so a held entry is never
evicted (a worker restarted late in a run re-reads it). Unheld entries expire after
``idle_exit_s``, and an empty, idle cache exits; the next user re-creates it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import ray
import ray.exceptions  # explicit: pyright does not see the submodule through `import ray`

from raysimbatch.compat.ray_typing import actor_class
from raysimbatch.protocol import Artifact


class AssetCache:
  def __init__(self, artifact_root: str | None, max_bytes: int = 17_179_869_184, idle_exit_s: float = 300.0,
               background: bool = True):
    self.root = artifact_root
    self.max_bytes = max_bytes
    self.idle_exit_s = idle_exit_s
    self.refs: dict[str, tuple[ray.ObjectRef, int, float]] = {}
    self.inline: dict[str, bytes] = {}
    self.holders: dict[str, set[str]] = {}
    self.bytes = 0
    self.hits = 0
    self.misses = 0
    self.last_activity = time.monotonic()
    self._lock = threading.Lock()
    if background:
      threading.Thread(target=self._reaper, daemon=True, name="rsb-asset-reaper").start()

  def get(self, uri: str, sha256: str, byte_length: int, media_type: str, holder: str | None = None) -> ray.ObjectRef:
    """ObjectRef of the verified bytes (a uint8 array)."""
    with self._lock:
      self.last_activity = time.monotonic()
      self._hold(sha256, holder)
      if sha256 in self.refs:
        ref, n, _ = self.refs[sha256]
        self.refs[sha256] = (ref, n, time.monotonic())
        self.hits += 1
        return ref
      self.misses += 1
      data = self.inline.get(sha256)
    if data is None:
      from raysimbatch.storage.store import read_ref

      data = read_ref(Artifact(uri, sha256, byte_length, media_type))
    return self._put(sha256, data)

  def put_bytes(self, sha256: str, data: bytes, holder: str | None = None) -> ray.ObjectRef:
    """Register bytes produced at admission (e.g. a compiled MJB) under their digest."""
    import hashlib

    if hashlib.sha256(data).hexdigest() != sha256:
      raise ValueError("digest mismatch")
    with self._lock:
      self.last_activity = time.monotonic()
      self._hold(sha256, holder)
      self.inline[sha256] = data
    return self._put(sha256, data)

  def _hold(self, sha256: str, holder: str | None) -> None:
    if holder:
      self.holders.setdefault(sha256, set()).add(holder)

  def _put(self, sha256: str, data: bytes) -> ray.ObjectRef:
    with self._lock:
      if sha256 in self.refs:
        return self.refs[sha256][0]
      while self.bytes + len(data) > self.max_bytes:
        free = [k for k in self.refs if not self.holders.get(k)]
        if not free:
          break  # every entry is held by a live run: exceed the budget rather than break it
        self._evict(min(free, key=lambda k: self.refs[k][2]))
      ref = ray.put(np.frombuffer(data, np.uint8))
      self.refs[sha256] = (ref, len(data), time.monotonic())
      self.bytes += len(data)
      return ref

  def _evict(self, sha256: str) -> None:
    self.bytes -= self.refs.pop(sha256)[1]
    self.inline.pop(sha256, None)

  def release(self, sha256: str) -> None:
    with self._lock:
      if sha256 in self.refs and not self.holders.get(sha256):
        self._evict(sha256)

  def release_holder(self, holder: str) -> int:
    """A run finished: its entries become evictable once idle."""
    with self._lock:
      self.last_activity = time.monotonic()
      n = 0
      for sha in list(self.holders):
        if holder in self.holders[sha]:
          self.holders[sha].discard(holder)
          n += 1
        if not self.holders[sha]:
          del self.holders[sha]
      return n

  def retain_only(self, live_holders: list[str]) -> int:
    """Janitor: drop holders that are no longer live runs (a controller that died
    without cleanup). Returns the number of released holds."""
    live = set(live_holders)
    with self._lock:
      stale = {h for hs in self.holders.values() for h in hs if h not in live}
    return sum(self.release_holder(h) for h in stale)

  def expire(self, now: float | None = None) -> bool:
    """Evict idle unheld entries; True when the cache is empty and idle (may exit)."""
    now = time.monotonic() if now is None else now
    with self._lock:
      for sha in [k for k, (_, _, used) in self.refs.items()
                  if not self.holders.get(k) and now - used >= self.idle_exit_s]:
        self._evict(sha)
      return not self.refs and not self.holders and now - self.last_activity >= self.idle_exit_s

  def _reaper(self) -> None:
    while True:
      time.sleep(min(30.0, max(0.5, self.idle_exit_s / 4)))
      if self.expire():
        _exit_self()
        return

  def stats(self) -> dict[str, Any]:
    with self._lock:
      return {"entries": len(self.refs), "bytes": self.bytes, "hits": self.hits, "misses": self.misses,
              "held": {k: sorted(v) for k, v in self.holders.items()}}


def _exit_self() -> None:
  """Terminate this detached actor (no restart): the next user re-creates it."""
  ray.kill(ray.get_runtime_context().current_actor, no_restart=True)


RemoteAssetCache = actor_class(AssetCache, num_cpus=0, max_restarts=-1, max_task_retries=0)


def asset_cache_for_node(node_id: str, artifact_root: str | None, namespace: str | None = None,
                         idle_exit_s: float = 300.0):
  """Get-or-create the named cache pinned to ``node_id``."""
  from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

  name = f"rsb-asset-cache-{node_id[:16]}"
  try:
    return ray.get_actor(name, namespace=namespace)
  except ValueError:
    opts: dict[str, Any] = {"name": name, "get_if_exists": True, "lifetime": "detached",
                            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id, soft=False)}
    if namespace:
      opts["namespace"] = namespace
    return RemoteAssetCache.options(**opts).remote(artifact_root, idle_exit_s=idle_exit_s)


def call_node_helper(resolve, call, attempts: int = 3):
  """``call(resolve())`` retrying on a helper that exited between lookup and call
  (idle per-node helpers exit so nodes can scale down)."""
  last: Exception | None = None
  for _ in range(attempts):
    try:
      return ray.get(call(resolve()))
    except (ray.exceptions.RayActorError, ray.exceptions.ActorDiedError) as exc:
      last = exc
      time.sleep(0.2)
  raise RuntimeError(f"node helper unavailable after {attempts} attempts: {last}") from last
