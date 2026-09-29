# SPDX-License-Identifier: Apache-2.0
"""Mode B: evaluate two policies on the cart-pole task and compare them.

  eval "$(tools/local_services.sh env)"
  python examples/mode_b_evaluation.py

Registers a cart-pole model, an LQR policy and a do-nothing policy as content-addressed
artifacts, runs the seed x episode matrix as durable units, and prints each policy's
success rate with its seed-cluster bootstrap interval and the paired difference.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
from _common import args, start_ray

from raysimbatch.config.models import EvalConfig
from raysimbatch.control.driver import RaySimBatch
from raysimbatch.policy.bundle import make_bundle
from raysimbatch.policy.export import cartpole_bundle, linear_policy
from raysimbatch.protocol import Artifact
from raysimbatch.storage.blob import open_blob_store
from raysimbatch.testing import fixtures as F
from raysimbatch.testing import runs as R


def main() -> None:
  a = args(__doc__.splitlines()[0])
  start_ray(a.address)
  model = R.group_from_fixture(a.root, F.cartpole(max_control_steps=300))  # registers MJCF, MJB, adapter, kernel
  lqr, lqr_exe = cartpole_bundle()
  zero_exe = linear_policy(np.zeros((1, 4)))
  zero = make_bundle("zero", 1, "onnx", zero_exe, 4, 1, certified_batch_sizes=(1, 4, 8, 16, 32, 64, 128))
  policies = [R.policy_spec(a.root, lqr, lqr_exe, "lqr"), R.policy_spec(a.root, zero, zero_exe, "zero")]
  task = R.task_spec(a.root, "balance", [model])
  cfg = R.run_config(a.root, a.database_secret, [model], policies, [task], mode="eval",
                     evaluation=EvalConfig(seeds=tuple(range(8)), episodes_per_seed=4, wave_actors=2,
                                           bootstrap_replicates=2000),
                     worker={"envs": 8, "slice_size": 8, "slice_min": 8})

  async def go():
    rsb = RaySimBatch(cfg, namespace="rsb-examples")
    run_id = await rsb.evaluate()
    status = await rsb.wait(run_id, timeout_s=600)
    return run_id, status

  run_id, status = asyncio.run(go())
  print(f"run {run_id}: {status.state}, complete={status.artifacts['complete']}, "
        f"coverage={status.artifacts['coverage']}")
  doc = json.loads(open_blob_store(a.root).get(Artifact(**status.artifacts["aggregate_json"])))
  names = {lqr.policy_hash: "lqr", zero.policy_hash: "zero"}
  for agg in doc["aggregates"]:
    if agg["metric"] == "success":
      print(f"  {names[agg['policy_hash']]:5s} success {agg['mean']:.3f}  "
            f"95% CI [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}]  "
            f"episodes {agg['valid_episodes']}/{agg['expected_episodes']}")
  for p in doc["paired"]:
    if p["metric"] == "success":
      print(f"  paired {names[p['policy_a']]} - {names[p['policy_b']]}: {p['mean_difference']:+.3f} "
            f"(95% CI [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], Holm p={p['holm_adjusted_p']})")
  print(f"exports: {sorted(k for k, v in status.artifacts.items() if isinstance(v, dict) and 'uri' in v)}")


if __name__ == "__main__":
  main()
