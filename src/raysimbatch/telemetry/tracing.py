# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry spans (design §11.2): admission, asset load, lease, inference, native
slice, fragment delivery, durable commit, migration, recovery. Successful data-path
spans are sampled at ``trace_fraction``; failures are always recorded. Trace IDs travel
explicitly in RSB/1 envelopes, not through implicit Ray propagation."""

from __future__ import annotations

import contextlib
import random
import threading
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_STATE: dict[str, Any] = {"fraction": 0.001, "errors": True, "memory": None, "dropped": 0}
_DROP_LOCK = threading.Lock()


class _CountingBatchSpanProcessor(BatchSpanProcessor):
  """A BatchSpanProcessor whose dropped spans (full export queue, or after shutdown) are
  counted for ``traces_dropped_total``: the SDK only logs them. It hooks the SDK's own
  drop accounting, an internal interface that tests/telemetry/test_tracing_drops.py
  checks on every upgrade."""

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    metrics = getattr(getattr(self, "_batch_processor", None), "_metrics", None)
    inner = getattr(metrics, "drop_items", None)
    if inner is not None:

      def drop_items(count: int, error_type: str = "queue_full") -> None:
        with _DROP_LOCK:
          _STATE["dropped"] += int(count)
        inner(count, error_type)

      metrics.drop_items = drop_items  # type: ignore[union-attr]


def drops() -> int:
  """Spans this process dropped because the export queue was full."""
  return int(_STATE["dropped"])


def configure(service: str, fraction: float = 0.001, errors: bool = True, endpoint: str | None = None) -> None:
  provider = TracerProvider(resource=Resource.create({"service.name": service}))
  if endpoint:
    try:
      from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
      raise RuntimeError("telemetry.otel_endpoint is set but the OTLP exporter is not installed in this image") from exc
    provider.add_span_processor(_CountingBatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint), max_queue_size=2048))
  else:
    mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    _STATE["memory"] = mem
  trace.set_tracer_provider(provider)
  _STATE.update(fraction=fraction, errors=errors)


@contextlib.contextmanager
def span(name: str, *, trace_id: bytes | None = None, force: bool = False, **attrs: Any):
  """Sampled span; an exception inside always records (and re-raises)."""
  sampled = force or random.random() < _STATE["fraction"]
  tracer = trace.get_tracer("raysimbatch")
  if not sampled:
    try:
      yield None
    except Exception as exc:
      if _STATE["errors"]:
        with tracer.start_as_current_span(name) as sp:
          _attrs(sp, trace_id, attrs)
          sp.record_exception(exc)
          sp.set_status(trace.Status(trace.StatusCode.ERROR))
      raise
    return
  with tracer.start_as_current_span(name) as sp:
    _attrs(sp, trace_id, attrs)
    try:
      yield sp
    except Exception as exc:
      sp.record_exception(exc)
      sp.set_status(trace.Status(trace.StatusCode.ERROR))
      raise


def _attrs(sp, trace_id: bytes | None, attrs: dict[str, Any]) -> None:
  if trace_id is not None:
    sp.set_attribute("rsb.trace_id", trace_id.hex())
  for k, v in attrs.items():
    sp.set_attribute(f"rsb.{k}", v if isinstance(v, bool | int | float | str) else str(v))


def finished_spans() -> list[Any]:
  mem = _STATE.get("memory")
  return [] if mem is None else list(mem.get_finished_spans())
