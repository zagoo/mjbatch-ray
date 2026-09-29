# SPDX-License-Identifier: Apache-2.0
"""Policy and curriculum updates are handed from a control thread to the loop thread
through a one-value slot. Taking the value and clearing the slot must be one step: an
update that lands in between used to be erased, so the learner's newest policy (or
curriculum) silently never reached the workers.

Each test forces exactly that interleaving: reading the slot starts a thread that
publishes a newer update and gives it time to land before the take clears the slot."""

from __future__ import annotations

import threading

from raysimbatch.control.controller import RunController
from raysimbatch.control.coordinator import RolloutProgram, ShardCoordinator


def racing_slot(name: str, publish):
  """A property whose first read publishes ``newer`` from another thread mid-take."""
  store = f"_{name}_value"
  racers: list[threading.Thread] = []

  def get(self):
    value = getattr(self, store, None)
    if not racers:
      racers.append(threading.Thread(target=publish, args=(self,)))
      racers[0].start()
      racers[0].join(0.3)  # lands here unless the take holds the lock
    return value

  def set_(self, value):
    setattr(self, store, value)

  return property(get, set_), racers


def test_controller_keeps_an_update_that_lands_mid_take():
  slot, racers = racing_slot("policy_update", lambda c: RunController.update_policy(c, {"version": 2}))

  class Racy(RunController):
    policy_update = slot

  ctl = Racy.__new__(Racy)
  ctl.lock, ctl.curriculum = threading.Lock(), None
  ctl._policy_update_value = {"version": 1}
  _, taken = ctl._take_updates()
  racers[0].join(5)
  assert taken == {"version": 1}
  assert ctl._policy_update_value == {"version": 2}, "the newer update was erased"


def test_coordinator_keeps_a_policy_that_lands_mid_take():
  slot, racers = racing_slot("pending_policy", lambda p: ShardCoordinator.update_policy(p.co, {"version": 2}))

  class Racy(RolloutProgram):
    pending_policy = slot

  co = ShardCoordinator.__new__(ShardCoordinator)
  prog = Racy.__new__(Racy)
  co.lock, co.program, prog.co = threading.Lock(), prog, co
  prog._pending_policy_value = {"version": 1}
  taken = prog._take_policy()
  racers[0].join(5)
  assert taken == {"version": 1}
  assert prog._pending_policy_value == {"version": 2}, "the newer policy was erased"
