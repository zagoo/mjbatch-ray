# SPDX-License-Identifier: Apache-2.0
"""Mode C: gate a source-engine policy transfer, then evaluate it along diagnostic axes.

  eval "$(tools/local_services.sh env)"
  python examples/mode_c_sim2sim.py

Uses the reference quadruped transfer bundle (canonical MJCF, source conventions,
policy) and outcomes from its reference source simulation. Real Isaac exports use the
same format (rsb-source-export/1); none are available in this repository.
"""

from __future__ import annotations

import asyncio

from _common import args, start_ray

from raysimbatch.config.models import EvalConfig, Sim2SimConfig
from raysimbatch.sim2sim import source as S
from raysimbatch.sim2sim.examples import policy_layers, quadruped_bundle
from raysimbatch.sim2sim.orchestrator import Sim2SimOrchestrator
from raysimbatch.testing import runs as R


def main() -> None:
  a = args(__doc__.splitlines()[0])
  start_ray(a.address)
  bundle = quadruped_bundle(max_control_steps=150)
  base = R.run_config(a.root, a.database_secret, [], [], mode="sim2sim",
                      evaluation=EvalConfig(bootstrap_replicates=1000, wave_actors=4),
                      worker={"envs": 8, "slice_size": 8, "slice_min": 8})
  orch = Sim2SimOrchestrator(None, Sim2SimConfig(), base, namespace="rsb-examples")
  policy = R.policy_spec(a.root, bundle.policy, bundle.executable, "src_policy")
  seeds = [0, 1, 2, 3]
  conv = S.quadruped_conventions()
  dim = sum(t["dim"] * t.get("history", 1) for t in conv.terms)
  source = S.simulate_source(bundle.canonical_xml, conv, policy_layers(dim), seeds=seeds, episodes=2, steps=150,
                             dt=0.004, min_height=0.12, min_up_z=0.3, solver={"iterations": bundle.group.iterations})

  async def go():
    gates = await orch.validate(bundle)  # 12 automated gates, cached per bundle digest
    print("gates:", {g.code: ("pass" if g.passed else "FAIL") for g in gates.results})
    return await orch.evaluate(bundle, policy, seeds, axes=("inertia", "contact", "filter"), episodes_per_seed=2,
                               source_success=source, options={"actors": 4}, timeout_s=900)

  out = asyncio.run(go())
  card = out["scorecard"]
  print(f"target success {card['target']['success']}, source success {card['source']['rate']}, "
        f"conservative transfer ratio {card['conservative_transfer_ratio']}")
  for d in card["diagnostics"]:
    print(f"  axis {d['axis']:8s} paired coverage {d['paired_coverage']:.2f}")


if __name__ == "__main__":
  main()
