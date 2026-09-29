# SPDX-License-Identifier: Apache-2.0
"""Run controllers belong off the head: the head runs GCS and the dashboard in little
memory, and Ray's memory monitor killed controllers placed there (kind qualification).
The affinity must be hard: a freshly registered worker raylet answered a soft lease with
spillback and the GCS placed the controller on the head. The janitor starts with the
head, so it waits for a worker node before resuming anything."""

from __future__ import annotations

import pytest

from raysimbatch.control import driver

HEAD = {"NodeID": "h" * 56, "Alive": True, "Resources": {"node:__internal_head__": 1.0, "memory": 8e9}}
SMALL = {"NodeID": "a" * 56, "Alive": True, "Resources": {"memory": 1e9}}
BIG = {"NodeID": "b" * 56, "Alive": True, "Resources": {"memory": 4e9}}
DEAD = {"NodeID": "d" * 56, "Alive": False, "Resources": {"memory": 64e9}}


def _nodes(monkeypatch, *snapshots):
  seq = list(snapshots)
  monkeypatch.setattr(driver.ray, "nodes", lambda: seq.pop(0) if len(seq) > 1 else seq[0])


def test_off_head_prefers_the_alive_worker_with_most_memory_even_if_the_head_has_more(monkeypatch):
  _nodes(monkeypatch, [HEAD, SMALL, BIG, DEAD])
  strategy = driver._off_head()["scheduling_strategy"]
  assert strategy.node_id == BIG["NodeID"] and not strategy.soft  # soft spilled back to the head


def test_off_head_without_workers_falls_back_to_default_placement(monkeypatch):
  _nodes(monkeypatch, [HEAD, DEAD])
  assert driver._off_head() == {}


def test_wait_returns_once_a_worker_joins(monkeypatch):
  _nodes(monkeypatch, [HEAD], [HEAD], [HEAD, SMALL])
  assert driver.wait_for_non_head_node(timeout_s=5, poll_s=0)
  assert driver._off_head()["scheduling_strategy"].node_id == SMALL["NodeID"]


@pytest.mark.parametrize("timeout_s", [0.0, 0.05])
def test_wait_is_bounded_on_a_head_only_cluster(monkeypatch, timeout_s):
  _nodes(monkeypatch, [HEAD, DEAD])
  assert not driver.wait_for_non_head_node(timeout_s=timeout_s, poll_s=0.01)


def test_janitor_loop_waits_for_a_worker_before_its_first_pass(monkeypatch):
  import raysimbatch.cli as cli
  import raysimbatch.recovery.watchdog as watchdog
  import raysimbatch.storage.store as store

  calls = []

  class Stop(Exception):
    pass

  async def fake_pass(*_a, **_k):
    calls.append("pass")
    return {}

  def fake_sleep(_s):  # the loop never ends by itself; stop it after two passes
    if calls.count("pass") >= 2:
      raise Stop

  monkeypatch.setattr(cli, "_load", lambda _a: object())
  monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
  monkeypatch.setattr(driver, "wait_for_non_head_node", lambda t: calls.append(("wait", t)) or True)
  monkeypatch.setattr(watchdog, "janitor_pass", fake_pass)
  monkeypatch.setattr(store.DurableStore, "from_config", staticmethod(lambda *_a, **_k: object()))
  monkeypatch.setattr("time.sleep", fake_sleep)
  with pytest.raises(Stop):
    cli.main(["janitor", "--config", "unused.yaml", "--loop", "1", "--startup-wait-s", "7"])
  assert calls == [("wait", 7.0), "pass", "pass"]  # once, before the first pass only
