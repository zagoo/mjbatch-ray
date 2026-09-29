# SPDX-License-Identifier: Apache-2.0
"""The native slot bookkeeping (``_native.Book``) replaces advance_control's NumPy tail.
It must do exactly what that tail did, bit for bit, for every mix of outcomes: the
reference below *is* the former implementation, kept as the oracle."""

from __future__ import annotations

import numpy as np
import pytest

from raysimbatch import _native
from raysimbatch.protocol import Flag, SlotState, SlotStatus

N, K, MAX_STEPS = 32, 3, 50


def fresh_book(rng):
  book = {
    "slot_state": np.full(N, SlotState.READY, np.uint8),
    "physics_steps": rng.integers(0, 1000, N).astype(np.uint64),
    "warning_total": rng.integers(0, 5, N).astype(np.int64),
    "control_seq": rng.integers(MAX_STEPS - 3, MAX_STEPS, (N, 1)).astype(np.uint64),
    "sim_time_ns": np.zeros((N, 1), np.int64),
    "return_sum": rng.choice([-0.0, 0.0, 1.5, -2.25], N).astype(np.float64),  # signed zeros matter
    "component_sums": rng.choice([-0.0, 0.0, 0.5], (N, K)).astype(np.float64),
  }
  return book


def native(book):
  cfg = {"max_control_steps": MAX_STEPS, "num_components": K,
         "status": {k: int(SlotStatus[k]) for k in ("OK", "TERMINATED", "TRUNCATED", "NATIVE_FAILURE",
                                                     "HEALTH_FAILURE")},
         "state": {k: int(SlotState[k]) for k in ("READY", "WAITING_ACTION", "STEPPING", "TERMINAL", "FAILED")},
         "flag": {k: int(Flag[k]) for k in ("VALID", "TERMINATED", "TRUNCATED", "ABORTED")}}
  return _native.Book(cfg, book)


def reference(b, s, status, completed, warn, ended, reward, comps, forward_failed):
  """advance_control's former tail (forward failures applied where it applied them)."""
  b["physics_steps"][s] += completed
  b["warning_total"][s] += warn
  good = (status == SlotStatus.OK) | (status == SlotStatus.TERMINATED)
  b["control_seq"][s, 0] += 1
  seq = b["control_seq"][s, 0].copy()
  trunc = (status == SlotStatus.OK) & (seq >= MAX_STEPS)
  status[trunc] = SlotStatus.TRUNCATED
  ended |= trunc
  bad = good & forward_failed
  status[bad] = SlotStatus.NATIVE_FAILURE
  good[bad] = False
  b["return_sum"][s] += np.where(good, reward, 0.0)
  b["component_sums"][s, :K] += np.where(good[:, None], comps, 0.0)
  flags = np.zeros(s.size, np.uint8)
  flags[good] |= np.uint8(Flag.VALID)
  flags[status == SlotStatus.TERMINATED] |= np.uint8(Flag.TERMINATED)
  flags[status == SlotStatus.TRUNCATED] |= np.uint8(Flag.TRUNCATED)
  aborted = (status == SlotStatus.NATIVE_FAILURE) | (status == SlotStatus.HEALTH_FAILURE)
  flags[aborted] |= np.uint8(Flag.ABORTED)
  ended |= aborted
  b["slot_state"][s] = np.where(aborted, SlotState.FAILED,
                                np.where(ended, SlotState.TERMINAL, SlotState.READY)).astype(np.uint8)
  return seq, good, flags


@pytest.mark.parametrize("seed", range(40))
def test_native_book_equals_the_former_numpy_tail(seed):
  rng = np.random.default_rng(seed)
  s = np.sort(rng.choice(N, rng.integers(1, N + 1), replace=False)).astype(np.int32)
  n = s.size
  outcomes = [SlotStatus.OK, SlotStatus.OK, SlotStatus.TERMINATED, SlotStatus.NATIVE_FAILURE,
              SlotStatus.HEALTH_FAILURE]
  status = rng.choice(outcomes, n).astype(np.uint8)
  completed = rng.integers(0, 11, n).astype(np.uint32)
  warn = rng.integers(0, 3, n).astype(np.int32)
  ended = status == SlotStatus.TERMINATED
  reward = rng.choice([-0.0, 0.0, 1.0, -0.5], n).astype(np.float32)
  comps = rng.choice([-0.0, 0.0, 0.25], (n, K)).astype(np.float32)
  forward_failed = rng.random(n) < 0.15

  ref_book, nat_book = fresh_book(np.random.default_rng(1000 + seed)), fresh_book(np.random.default_rng(1000 + seed))
  ref_status, ref_ended = status.copy(), ended.copy()
  seq_r, good_r, flags_r = reference(ref_book, s, ref_status, completed, warn, ref_ended, reward, comps, forward_failed)

  book = native(nat_book)
  nat_status, nat_ended = status.copy(), ended.copy()
  seq, good = np.empty(n, np.uint64), np.empty(n, bool)
  book.account(s, nat_status, completed, warn, nat_ended, seq, good)
  bad = good & forward_failed  # what advance_control does after a failed boundary forward
  nat_status[bad] = SlotStatus.NATIVE_FAILURE
  good[bad] = False
  flags = np.empty(n, np.uint8)
  book.settle(s, nat_status, good, reward, comps, nat_ended, flags)

  for name, x, y in (("seq", seq, seq_r), ("good", good, good_r), ("flags", flags, flags_r),
                     ("status", nat_status, ref_status), ("ended", nat_ended, ref_ended)):
    np.testing.assert_array_equal(x, y, err_msg=name)
  for k in nat_book:
    a, r = nat_book[k], ref_book[k]
    assert a.tobytes() == r.tobytes(), k  # bitwise, including the sign of zero sums


def test_ready_and_begin():
  rng = np.random.default_rng(0)
  b = fresh_book(rng)
  book = native(b)
  s = np.array([1, 4, 9], np.int32)
  b["slot_state"][4] = SlotState.WAITING_ACTION
  assert book.ready(s) == 0
  book.begin(s)
  assert (b["slot_state"][s] == SlotState.STEPPING).all() and b["slot_state"][2] == SlotState.READY
  assert book.ready(s) == 2  # stepping slots cannot be advanced again
  b["slot_state"][[1, 4, 9]] = SlotState.READY
  b["slot_state"][9] = SlotState.TERMINAL
  assert book.ready(s) == 2
  for bad in ([4, 1], [1, 1], [-1, 2]):  # unsorted, duplicate, negative
    assert book.ready(np.array(bad, np.int32)) == 1
  with pytest.raises(IndexError):
    book.ready(np.array([1, N], np.int32))
