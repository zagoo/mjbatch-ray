# SPDX-License-Identifier: Apache-2.0
"""Executable record of the Ray 2.58.0 semantics RaySimBatch depends on (design §2.4,
§7.2, §8.4). Run against every Ray upgrade candidate before the release lock moves."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.ray


def test_pinned_version(ray_session):
  assert ray_session.__version__ == "2.58.0"


def test_same_node_numpy_reads_are_zero_copy_and_read_only(ray_session):
  """ray.put makes one store snapshot; same-node readers map it without copying."""
  ray = ray_session
  src = np.arange(1 << 20, dtype=np.float32)
  ref = ray.put({"obs": src})
  a = ray.get(ref)["obs"]
  b = ray.get(ref)["obs"]
  assert not a.flags.writeable
  assert np.shares_memory(a, b)  # both reads view the same store buffer
  assert not np.shares_memory(a, src)  # the snapshot is a copy of the producer's buffer
  src[:] = -1  # mutating the producer buffer after put cannot change the snapshot
  assert a[5] == 5.0


def test_concurrency_groups_only_on_the_decorator(ray_session):
  ray = ray_session

  class A:
    def f(self):
      return 1

  with pytest.raises(ValueError, match="concurrency_groups"):
    ray.remote(A).options(concurrency_groups={"x": 1})
  h = ray.remote(concurrency_groups={"x": 1})(A).remote()
  assert ray.get(h.f.remote()) == 1


def test_cancelled_actor_task_ref_raises_even_if_the_method_returns(ray_session):
  ray = ray_session

  @ray.remote(concurrency_groups={"m": 1, "c": 1})
  class Slow:
    def __init__(self):
      self.done = False
      self.started = False

    @ray.method(concurrency_group="m")
    def work(self):
      self.started = True
      ctx = ray.get_runtime_context()
      for _ in range(500):
        if ctx.is_canceled():
          self.done = True
          return "graceful"
        time.sleep(0.01)
      return "finished"

    @ray.method(concurrency_group="c")
    def graceful(self):
      return self.done

    @ray.method(concurrency_group="c")
    def running(self):
      return self.started

  s = Slow.remote()
  ref = s.work.remote()
  # Cancel only once the body runs: a task cancelled while still queued never executes
  # (also TaskCancelledError), which is the other case the coordinator's resend covers.
  deadline = time.monotonic() + 30
  while not ray.get(s.running.remote()):
    assert time.monotonic() < deadline
    time.sleep(0.01)
  with pytest.raises(ValueError):
    ray.cancel(ref, force=True)  # force is rejected for actor tasks
  ray.cancel(ref, force=False)
  with pytest.raises(ray.exceptions.TaskCancelledError):
    ray.get(ref)
  # The ref resolves as cancelled immediately; the body observes is_canceled() at its
  # next poll and returns normally (so a cancelled mutation may still have completed).
  deadline = time.monotonic() + 5
  while not ray.get(s.graceful.remote()):
    assert time.monotonic() < deadline, "the method body never observed the cancellation"
    time.sleep(0.01)


def test_actor_restart_reruns_only_the_constructor_and_no_task_retry(ray_session):
  ray = ray_session

  @ray.remote(max_restarts=1, max_task_retries=0)
  class Counter:
    def __init__(self):
      self.n = 0

    def inc(self):
      self.n += 1
      return self.n

    def pid(self):
      import os

      return os.getpid()

    def die(self):
      import os

      os._exit(1)

  c = Counter.remote()
  assert ray.get(c.inc.remote()) == 1
  pid = ray.get(c.pid.remote())
  with pytest.raises(ray.exceptions.RayActorError):
    ray.get(c.die.remote())  # in-flight mutation is not retried
  deadline = time.time() + 30
  while True:
    try:
      if ray.get(c.pid.remote(), timeout=5) != pid:
        break
    except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
      pass
    assert time.time() < deadline
    time.sleep(0.2)
  assert ray.get(c.inc.remote()) == 1  # state was not restored


def test_put_objects_die_with_their_owner(ray_session):
  """ray.put output is owned by the putting actor: it is not lineage-reconstructable."""
  ray = ray_session

  @ray.remote(max_restarts=0)
  class Producer:
    def make(self):
      return [ray.put(np.ones(1000))]

  p = Producer.remote()
  [ref] = ray.get(p.make.remote())
  assert ray.get(ref).sum() == 1000
  ray.kill(p, no_restart=True)
  time.sleep(1.0)
  with pytest.raises((ray.exceptions.OwnerDiedError, ray.exceptions.ObjectLostError)):
    ray.get(ref, timeout=10)


def test_top_level_refs_resolve_nested_refs_do_not(ray_session):
  ray = ray_session

  @ray.remote
  def kinds(a, b):
    return type(a).__name__, type(b[0]).__name__

  r = ray.put(np.zeros(3))
  ta, tb = ray.get(kinds.remote(r, [r]))
  assert ta == "ndarray" and tb == "ObjectRef"


def test_wait_without_fetch(ray_session):
  ray = ray_session

  @ray.remote
  def big():
    return np.zeros(1 << 20)

  refs = [big.remote() for _ in range(3)]
  ready, pending = [], refs
  while pending:
    r, pending = ray.wait(pending, num_returns=1, timeout=5.0, fetch_local=False)
    ready += r
  assert len(ready) == 3
