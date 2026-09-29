# SPDX-License-Identifier: Apache-2.0
"""Circuit breaker and restart accounting (design §8.4).

CLOSED -> OPEN after ``circuit_restart_count`` worker restarts within
``circuit_window_s`` (or an unhealthy-slot fraction >= 5%); OPEN for
``circuit_open_s``; HALF_OPEN admits one canary actor for ``canary_episodes``; a fault
reopens it, success closes it; ``max_canary_failures`` failed canaries mark the model
family unavailable. No unbounded reconstruction loop is possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from raysimbatch.config.models import FailureConfig

CLOSED, OPEN, HALF_OPEN, UNAVAILABLE = 0, 1, 2, 3


@dataclass
class Circuit:
  cfg: FailureConfig
  state: int = CLOSED
  restarts: list[float] = field(default_factory=list)
  opened_at: float = 0.0
  canary_failures: int = 0
  canary_episodes: int = 0

  def record_restart(self, now: float | None = None) -> int:
    now = time.monotonic() if now is None else now
    self.restarts = [t for t in self.restarts if now - t <= self.cfg.circuit_window_s] + [now]
    if self.state == HALF_OPEN:
      self._canary_failed(now)
    elif self.state == CLOSED and len(self.restarts) >= self.cfg.circuit_restart_count:
      self.state, self.opened_at = OPEN, now
    return self.state

  def record_unhealthy_fraction(self, fraction: float, now: float | None = None) -> int:
    now = time.monotonic() if now is None else now
    if self.state == CLOSED and fraction >= 0.05:
      self.state, self.opened_at = OPEN, now
    return self.state

  def tick(self, now: float | None = None) -> int:
    now = time.monotonic() if now is None else now
    if self.state == OPEN and now - self.opened_at >= self.cfg.circuit_open_s:
      self.state, self.canary_episodes = HALF_OPEN, 0
    return self.state

  def admits_restart(self, now: float | None = None) -> bool:
    return self.tick(now) in (CLOSED, HALF_OPEN)

  def record_canary_episode(self, ok: bool, now: float | None = None) -> int:
    now = time.monotonic() if now is None else now
    if self.state != HALF_OPEN:
      return self.state
    if not ok:
      self._canary_failed(now)
      return self.state
    self.canary_episodes += 1
    if self.canary_episodes >= self.cfg.canary_episodes:
      self.state, self.restarts = CLOSED, []
    return self.state

  def _canary_failed(self, now: float) -> None:
    self.canary_failures += 1
    if self.canary_failures >= self.cfg.max_canary_failures:
      self.state = UNAVAILABLE
    else:
      self.state, self.opened_at = OPEN, now
