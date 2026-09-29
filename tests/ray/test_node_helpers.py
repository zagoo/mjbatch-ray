# SPDX-License-Identifier: Apache-2.0
"""Per-node helper lifecycle. Found in kind qualification: live detached helpers (and
the objects a cache pins) keep a Ray node busy, so the autoscaler never scaled down.
Helpers must release what no live run holds and exit when idle, without ever dropping
bytes a live run may still need."""

from __future__ import annotations

import hashlib
import time

import pytest
import ray

from raysimbatch.control.assets import AssetCache, asset_cache_for_node, call_node_helper
from raysimbatch.worker.affinity_agent import agent_for_node


def _sha(b: bytes) -> str:
  return hashlib.sha256(b).hexdigest()


def test_held_entries_survive_idle_expiry_and_budget_pressure(ray_session):
  c = AssetCache(None, max_bytes=10, idle_exit_s=5.0, background=False)
  a, b = b"a" * 8, b"b" * 8
  c.put_bytes(_sha(a), a, holder="run-1")
  c.put_bytes(_sha(b), b)  # unheld; over budget, but the held entry is never evicted for it
  now = time.monotonic() + 60
  assert c.expire(now) is False and set(c.refs) == {_sha(a)}  # idle unheld entry expired, held one kept
  assert c.release_holder("run-1") == 1
  assert c.expire(now) is True and not c.refs  # nothing held, idle: may exit


def test_janitor_retain_only_drops_holds_of_dead_runs(ray_session):
  c = AssetCache(None, idle_exit_s=5.0, background=False)
  x = b"x" * 4
  c.put_bytes(_sha(x), x, holder="dead-run")
  c.put_bytes(_sha(x), x, holder="live-run")
  assert c.retain_only(["live-run"]) == 1
  assert c.stats()["held"] == {_sha(x): ["live-run"]}


def _gone(name: str, namespace: str, within: float = 20.0) -> bool:
  t0 = time.monotonic()
  while time.monotonic() - t0 < within:
    try:
      ray.get_actor(name, namespace=namespace)
    except ValueError:
      return True
    time.sleep(0.25)
  return False


def test_idle_cache_exits_and_is_recreated_on_demand(ray_session):
  ns = "rsb-helpers"
  node = ray.get_runtime_context().get_node_id()
  data = b"model-bytes"
  mk = lambda: asset_cache_for_node(node, None, ns, idle_exit_s=1.0)  # noqa: E731
  ref = call_node_helper(mk, lambda c: c.put_bytes.remote(_sha(data), data, "run-x"))
  assert bytes(ray.get(ref)) == data
  assert not _gone(f"rsb-asset-cache-{node[:16]}", ns, within=4)  # held: stays
  ray.get(mk().release_holder.remote("run-x"))
  assert _gone(f"rsb-asset-cache-{node[:16]}", ns)  # empty and idle: exits
  ref = call_node_helper(mk, lambda c: c.put_bytes.remote(_sha(data), data, "run-y"))  # re-created
  assert bytes(ray.get(ref)) == data
  ray.get(mk().release_holder.remote("run-y"))


@pytest.mark.skipif(not hasattr(__import__("os"), "sched_getaffinity"), reason="Linux CPU affinity")
def test_idle_affinity_agent_exits(ray_session, tmp_path):
  ns = "rsb-helpers"
  node = ray.get_runtime_context().get_node_id()
  agent = agent_for_node(node, namespace=ns, state_dir=str(tmp_path), idle_exit_s=1.0)
  ray.get(agent.status.remote())
  assert _gone(f"rsb-affinity-{node[:16]}", ns)
