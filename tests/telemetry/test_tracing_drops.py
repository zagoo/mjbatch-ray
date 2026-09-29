# SPDX-License-Identifier: Apache-2.0
"""traces_dropped_total must equal the spans the OpenTelemetry SDK actually dropped on a
full export queue (the SDK only logs them). The count relies on the SDK's queue, so this
test pins it: offered = exported + still queued + counted as dropped."""

from __future__ import annotations

import threading
import time

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from raysimbatch.telemetry import tracing


class Stuck(SpanExporter):
  """Takes the first batch and never returns: the queue then fills and overflows."""

  def __init__(self):
    self.received = 0
    self.release = threading.Event()

  def export(self, spans):
    self.received += len(spans)
    self.release.wait(30)
    return SpanExportResult.SUCCESS

  def shutdown(self):
    self.release.set()


def test_dropped_spans_are_counted_exactly():
  exporter = Stuck()
  proc = tracing._CountingBatchSpanProcessor(exporter, max_queue_size=4, max_export_batch_size=4,
                                             schedule_delay_millis=60_000)
  provider = TracerProvider()
  provider.add_span_processor(proc)
  tracer = provider.get_tracer("t")
  before = tracing.drops()
  for i in range(4):  # one batch: the worker takes it and blocks in export
    with tracer.start_as_current_span(f"s{i}"):
      pass
  deadline = time.monotonic() + 10
  while exporter.received < 4:
    assert time.monotonic() < deadline, "the exporter never took the first batch"
    time.sleep(0.01)
  for i in range(4, 40):  # 4 fill the queue again, 32 overflow it
    with tracer.start_as_current_span(f"s{i}"):
      pass
  received, queued = exporter.received, len(proc._batch_processor._queue)
  dropped = tracing.drops() - before
  exporter.release.set()
  provider.shutdown()  # (its final flush exports the queued spans)
  assert dropped > 0, "the queue never overflowed: the test did not exercise dropping"
  assert received + queued + dropped == 40, (received, queued, dropped)
