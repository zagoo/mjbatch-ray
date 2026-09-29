# SPDX-License-Identifier: Apache-2.0
"""Mode C end to end: gates (cached per bundle), paired diagnostic axes as Mode B units,
source-side outcomes, and the transfer scorecard."""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

from raysimbatch.config.models import EvalConfig, Sim2SimConfig
from raysimbatch.sim2sim import source as S
from raysimbatch.sim2sim.examples import policy_layers, quadruped_bundle
from raysimbatch.sim2sim.orchestrator import Sim2SimOrchestrator
from raysimbatch.testing import runs as R

pytestmark = [pytest.mark.ray, pytest.mark.postgres, pytest.mark.slow]


def test_mode_c_end_to_end(ray_session, tmp_path):
  if not os.environ.get("RSB_TEST_DSN"):
    pytest.skip("RSB_TEST_DSN not set")
  root = f"file://{tmp_path}/artifacts"
  b = quadruped_bundle(max_control_steps=150)
  base = R.run_config(root, "env:RSB_TEST_DSN", [], [], mode="sim2sim",
                      evaluation=EvalConfig(bootstrap_replicates=1000, wave_actors=4, lease_s=30, lease_renew_s=5),
                      worker={"envs": 8, "slice_size": 8, "slice_min": 8})
  orch = Sim2SimOrchestrator(None, Sim2SimConfig(), base, namespace=f"rsb-c-{uuid.uuid4().hex[:6]}")
  pol = R.policy_spec(root, b.policy, b.executable, "src_policy")
  seeds = [0, 1, 2, 3]
  conv = S.quadruped_conventions()
  dim = sum(t["dim"] * t.get("history", 1) for t in conv.terms)
  src = S.simulate_source(b.canonical_xml, conv, policy_layers(dim), seeds=seeds, episodes=2, steps=150, dt=0.004,
                          min_height=0.12, min_up_z=0.3, solver={"iterations": b.group.iterations})

  async def go():
    t0 = time.monotonic()
    gates = await orch.validate(b)
    first = time.monotonic() - t0
    t1 = time.monotonic()
    again = await orch.validate(b)  # cached by bundle digest: no re-execution
    second = time.monotonic() - t1
    assert gates.passed and again.to_dict() == gates.to_dict() and second < first
    return await orch.evaluate(b, pol, seeds, axes=("inertia", "contact", "filter"), episodes_per_seed=2,
                               source_success=src, options={"actors": 4}, timeout_s=900)

  out = asyncio.run(go())
  card = out["scorecard"]
  assert all(g["passed"] for g in card["gates"].values())
  assert card["target"]["coverage"] == 1.0
  axes = {d["axis"]: d for d in card["diagnostics"]}
  assert set(axes) == {"inertia", "contact", "filter"}
  assert all(d["paired_coverage"] == 1.0 for d in axes.values())
  assert card["source"]["episodes"] == 8
  if card["source"]["rate"] > 0:
    assert card["conservative_transfer_ratio"] is not None
  assert out["results"]["complete"]
