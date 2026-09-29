# SPDX-License-Identifier: Apache-2.0
"""Per-process telemetry setup (design §11): structured logs and OpenTelemetry tracing
are configured once per process from the run's ``TelemetryConfig``; metrics exporters
are cached per label set so every component in a process shares one registry."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

_LOCK = threading.Lock()
_CONFIGURED: dict[str, Any] = {}
_EXPORTERS: dict[tuple[tuple[str, str], ...], Any] = {}


def setup(service: str, telemetry: Mapping[str, Any] | Any | None = None) -> None:
  """Configure JSON logging and tracing for this process (first call wins)."""
  from raysimbatch.config.models import TelemetryConfig

  with _LOCK:
    if _CONFIGURED:
      return
    cfg = telemetry if isinstance(telemetry, TelemetryConfig) else TelemetryConfig.from_obj(dict(telemetry or {}))
    from raysimbatch.telemetry import logs, tracing

    handler = logs.configure(cfg.log_level, max_bytes=cfg.log_queue_bytes)
    tracing.configure(service, fraction=cfg.trace_fraction, errors=cfg.trace_errors, endpoint=cfg.otel_endpoint)
    _CONFIGURED.update(service=service, handler=handler, prefix=cfg.metrics_prefix)


def exporter(**tags: str):
  """A process-wide exporter for this bounded label set."""
  from raysimbatch.telemetry.metrics import MetricsExporter

  key = tuple(sorted(tags.items()))
  with _LOCK:
    m = _EXPORTERS.get(key)
    if m is None:
      m = MetricsExporter(tags, prefix=_CONFIGURED.get("prefix", "raysimbatch"))
      _EXPORTERS[key] = m
    return m


def log_drops() -> tuple[int, int]:
  """(dropped, suppressed) log records in this process."""
  h = _CONFIGURED.get("handler")
  return (h.dropped, h.suppressed) if h is not None else (0, 0)
