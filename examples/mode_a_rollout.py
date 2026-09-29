# SPDX-License-Identifier: Apache-2.0
"""Mode A: heterogeneous rollouts (cart-pole + quadruped) into the durable replay sink.

  eval "$(tools/local_services.sh env)"
  python examples/mode_a_rollout.py

Two model groups with different observation/action widths run concurrently under a
curriculum; fragments are committed exactly once by digest and read back as RLlib-style
sample batches. Each group keeps its own schema (nothing is padded to the larger one).
"""

from __future__ import annotations

import asyncio
import collections

from _common import args, start_ray

from raysimbatch.config.models import CurriculumConfig, InferenceConfig, RolloutConfig
from raysimbatch.control.driver import RaySimBatch
from raysimbatch.experience.sinks import DurableReplaySink, to_rllib_batch
from raysimbatch.policy.export import gaussian_bundle
from raysimbatch.testing import fixtures as F
from raysimbatch.testing import runs as R


def main() -> None:
  a = args(__doc__.splitlines()[0])
  start_ray(a.address)
  cartpole = R.group_from_fixture(a.root, F.cartpole(max_control_steps=200))
  quadruped = R.group_from_fixture(a.root, F.quadruped(max_control_steps=150))
  pc, ec = gaussian_bundle(4, 1, seed=1, name="cp_pol")  # stand-ins for learner checkpoints
  pq, eq = gaussian_bundle(36, 8, seed=2, name="qd_pol")
  cfg = R.run_config(
    a.root, a.database_secret, [cartpole, quadruped], [R.policy_spec(a.root, pc, ec), R.policy_spec(a.root, pq, eq)],
    mode="rollout", target_envs=32,
    rollout=RolloutConfig(target_transitions=20_000, fragment_steps=16),
    curriculum=CurriculumConfig(version=1, weights={"cartpole": 0.5, "quadruped": 0.5}),
    inference=InferenceConfig(placement="local"),  # "central" batches through PolicyInferenceService
    worker={"envs": 16, "slice_size": 16, "slice_min": 16})

  async def go():
    rsb = RaySimBatch(cfg, namespace="rsb-examples", options={"actors": 2})
    run_id = await rsb.rollout()
    return run_id, await rsb.wait(run_id, timeout_s=600)

  run_id, status = asyncio.run(go())
  print(f"run {run_id}: {status.state}; gaps={status.artifacts.get('gaps')}")
  sink = DurableReplaySink(rsb_store(cfg), run_id)
  by_width, transitions = collections.Counter(), 0
  for _, frag in sink.read(0, 100_000):
    batch = to_rllib_batch(frag)
    by_width[batch["obs"].shape[1]] += len(batch["obs"])
    transitions += frag.transitions
  print(f"fragments hold {transitions} transitions; rows by observation width: {dict(by_width)}")


def rsb_store(cfg):
  from raysimbatch.storage.store import DurableStore

  return DurableStore.from_config(cfg, migrate=False)


if __name__ == "__main__":
  main()
