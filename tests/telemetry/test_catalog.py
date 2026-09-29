# SPDX-License-Identifier: Apache-2.0
"""Metric recording never raises into the data path, so conformance to the §11.1
catalog (names, kinds, bounded labels) is enforced here, statically, over the source."""

from __future__ import annotations

import ast
from pathlib import Path

from raysimbatch.telemetry.metrics import ALLOWED_LABELS, CATALOG

SRC = Path(__file__).resolve().parents[2] / "src" / "raysimbatch"
RECEIVERS = {"self.metrics", "self.co.metrics", "prog.co.metrics", "m"}
KIND = {"count": "counter", "gauge": "gauge", "observe": "histogram"}


def _calls():
  for path in SRC.rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in KIND
          and ast.unparse(node.func.value) in RECEIVERS and node.args
          and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        yield path.relative_to(SRC), node.lineno, node.func.attr, node.args[0].value, [k.arg for k in node.keywords]


def test_every_recorded_metric_is_in_the_catalog_with_its_kind_and_bounded_labels():
  calls = list(_calls())
  assert len(calls) > 30  # the scan actually found the instrumentation
  bad = []
  for where, line, method, name, labels in calls:
    if CATALOG.get(name) != KIND[method]:
      bad.append(f"{where}:{line} {method}({name!r}) catalog={CATALOG.get(name)}")
    if set(labels) - ALLOWED_LABELS:
      bad.append(f"{where}:{line} {name} labels {sorted(set(labels) - ALLOWED_LABELS)}")
  assert not bad, "\n".join(bad)


def test_event_counters_are_catalog_counters():
  from raysimbatch.control.coordinator import _EVENT_COUNTERS

  assert all(CATALOG[v] == "counter" for v in _EVENT_COUNTERS.values())


def test_recording_succeeds_for_every_kind_and_failures_are_absorbed():
  from raysimbatch.telemetry import metrics as M

  m = M.MetricsExporter({"actor_role": "test"})
  m._ray = False  # the local registry, whether or not this test process has started Ray
  before = M.ERRORS
  m.count("control_steps_total", 3)
  m.gauge("slots_active", 7.0)
  m.observe("batch_step_seconds", 0.003)
  assert M.ERRORS == before  # found: a dispatch bug silently dropped every counter
  text = M.exposition().decode()
  assert 'raysimbatch_control_steps_total{actor_role="test"} 3.0' in text
  assert 'raysimbatch_slots_active{actor_role="test"} 7.0' in text
  assert 'raysimbatch_batch_step_seconds_count{actor_role="test"} 1.0' in text
  m.count("control_steps_total", 0)  # Ray rejects inc(0); must be a no-op, not an exception
  m.gauge("not_a_metric", 1.0)  # absorbed and counted, never raised into the caller
  assert M.ERRORS == before + 1
