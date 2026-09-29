# SPDX-License-Identifier: Apache-2.0
"""Structured JSON logs (design §11.2) with rate-limited repeats and a byte-bounded
queue: when the collector falls behind, records are dropped and counted, never allowed
to stall physics. Fences, commits, quarantine and configuration changes are durable
audit events in PostgreSQL, not just log lines."""

from __future__ import annotations

import collections
import json
import logging
import sys
import threading
import time
from typing import Any

FIELDS = ("run_id", "job_epoch", "coordinator_id", "actor_name", "generation", "node_id", "model_family",
          "task_family", "command_id", "unit_id", "env_id", "episode_id", "trace_id", "span_id", "error_code",
          "diagnostic_uri")


class JsonFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    d: dict[str, Any] = {
      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
      "severity": record.levelname,
      "event": getattr(record, "event", record.name),
      "message": record.getMessage(),
    }
    for f in FIELDS:
      v = getattr(record, f, None)
      if v is not None:
        d[f] = v
    if record.exc_info:
      d["exception"] = self.formatException(record.exc_info)
    return json.dumps(d, default=str, separators=(",", ":"))


class BoundedHandler(logging.Handler):
  """Writes on a background thread; drops (and counts) beyond ``max_bytes`` queued."""

  def __init__(self, stream=None, max_bytes: int = 16 * 2**20, repeat_window_s: float = 10.0):
    super().__init__()
    self.stream = stream or sys.stderr
    self.max_bytes = max_bytes
    self.queue: collections.deque[str] = collections.deque()
    self.queued = 0
    self.dropped = 0
    self.suppressed = 0
    self.window = repeat_window_s
    self.last: dict[tuple[str, str], float] = {}
    self.cv = threading.Condition()
    threading.Thread(target=self._drain, daemon=True, name="rsb-log").start()

  def emit(self, record: logging.LogRecord) -> None:
    key = (getattr(record, "event", record.name), record.getMessage()[:120])
    now = time.monotonic()
    if record.levelno < logging.ERROR and now - self.last.get(key, -1e9) < self.window:
      self.suppressed += 1
      return
    self.last[key] = now
    line = self.format(record) + "\n"
    with self.cv:
      if self.queued + len(line) > self.max_bytes:
        self.dropped += 1
        return
      self.queue.append(line)
      self.queued += len(line)
      self.cv.notify()

  def _drain(self) -> None:
    while True:
      with self.cv:
        while not self.queue:
          self.cv.wait()
        line = self.queue.popleft()
        self.queued -= len(line)
      try:
        self.stream.write(line)
        self.stream.flush()
      except Exception:  # noqa: BLE001
        self.dropped += 1


def configure(level: str = "INFO", max_bytes: int = 16 * 2**20) -> BoundedHandler:
  h = BoundedHandler(max_bytes=max_bytes)
  h.setFormatter(JsonFormatter())
  root = logging.getLogger("raysimbatch")
  root.handlers[:] = [h]
  root.setLevel({"WARN": "WARNING"}.get(level, level))
  root.propagate = False
  return h


def log(event: str, level: int = logging.INFO, **fields: Any) -> None:
  logging.getLogger("raysimbatch").log(level, fields.pop("message", event), extra={"event": event, **fields})
