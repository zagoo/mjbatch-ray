# SPDX-License-Identifier: Apache-2.0
"""Seed-cluster statistics (design §6.6). The seed is the independent unit: episodes
within a seed are correlated, so they are averaged before resampling."""

from __future__ import annotations

import numpy as np

from raysimbatch.evaluation import stats


def test_bootstrap_is_over_seeds_not_episodes():
  """Adding episodes inside the same seeds must not manufacture confidence."""
  rng = np.random.default_rng(0)
  per_seed = rng.normal(0.5, 0.2, 10)
  few = stats.seed_means([(s, v) for s, v in enumerate(per_seed)])
  many = stats.seed_means([(s, v) for s, v in enumerate(per_seed) for _ in range(50)])
  a = stats.bootstrap_ci(np.asarray(list(few.values())), 5000, 0.95, 1)
  b = stats.bootstrap_ci(np.asarray(list(many.values())), 5000, 0.95, 1)
  assert (a.low, a.high) == (b.low, b.high)


def test_interval_is_reproducible_and_brackets_the_mean():
  xs = np.linspace(0, 1, 12)
  a = stats.bootstrap_ci(xs, 4000, 0.95, stats.bootstrap_seed("run", "x"))
  b = stats.bootstrap_ci(xs, 4000, 0.95, stats.bootstrap_seed("run", "x"))
  assert a == b and a.low < xs.mean() < a.high
  assert stats.bootstrap_ci(np.asarray([0.3]), 100, 0.95, 1) is None  # one seed: no interval


def test_holm_step_down_matches_a_hand_computed_example():
  # sorted p = .01, .02, .03, .04 with m = 4 -> .04, .06, .06 (monotone), .06; None passes through
  assert stats.holm([0.03, 0.01, None, 0.04, 0.02]) == [0.06, 0.04, None, 0.06, 0.06]


def test_bootstrap_p_value_is_never_zero_and_is_symmetric():
  b = 9999
  all_negative = np.full(b, -1.0)
  assert stats._bootstrap_p(all_negative) == 2 / (b + 1)  # not 0.0 from finitely many resamples
  centred = np.concatenate([np.full(5000, -1.0), np.full(4999, 1.0)])
  assert stats._bootstrap_p(centred) == 1.0
  assert stats._bootstrap_p(-centred) == stats._bootstrap_p(centred)
