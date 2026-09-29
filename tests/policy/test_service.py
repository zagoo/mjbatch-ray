# SPDX-License-Identifier: Apache-2.0
"""PolicyInferenceService: retiring a policy version (runbook "Inference overload":
retire drained versions) must never pull the runtime out from under a batch that is
already executing. Those requests were accepted while the version was resident; they
must complete, and the version retires once it is idle."""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from raysimbatch import protocol as P
from raysimbatch.policy.bundle import make_bundle
from raysimbatch.policy.export import linear_policy
from raysimbatch.policy.service import PolicyInferenceService


def frame(policy_hash: bytes, obs: np.ndarray) -> P.HeteroFrame:
  n = obs.shape[0]
  z = np.zeros(n, np.uint64)
  g = P.GroupFrame(group_hash=b"\0" * 32, schema_hash=b"\1" * 32, policy_hash=policy_hash, env_id=np.arange(n, dtype=np.uint64),
                   episode_id=z, control_seq=z, sim_time_ns=np.zeros(n, np.int64), slot_generation=z,
                   slot=np.arange(n, dtype=np.int32), observation=obs, reward=np.zeros(n, np.float32),
                   discount=np.ones(n, np.float32), behavior_logp=np.zeros(n, np.float32),
                   flags=np.zeros(n, np.uint8), policy_version=z, curriculum_version=z,
                   fault_code=np.zeros(n, np.uint16))
  return P.HeteroFrame(b"\2" * 16, P.new_id(), 0, 0, 0, P.new_id(), (g,))


async def test_retiring_a_version_waits_for_its_executing_batch():
  svc = PolicyInferenceService({"device": "cpu"})
  exe = linear_policy(np.array([[0.1, 0.2, 0.3, 0.4]]))  # action = clip(-K obs)
  b = make_bundle("p", 1, "onnx", exe, 4, 1, certified_batch_sizes=(1, 2))
  await svc.load(b.to_json().decode(), np.frombuffer(exe, np.uint8))
  rt = svc.policies[b.policy_hash]
  executing, release = threading.Event(), threading.Event()
  infer = rt.infer

  def slow_infer(*a, **kw):
    executing.set()
    release.wait(10)
    return infer(*a, **kw)

  rt.infer = slow_infer
  ph = bytes.fromhex(b.policy_hash)
  obs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], np.float32)
  req = {"request_id": P.new_id(), "policy_hash": ph, "deadline_unix_ns": 2**62, "deterministic": True}
  pending = asyncio.ensure_future(svc.infer(req, frame(ph, obs).to_wire()))
  assert await asyncio.to_thread(executing.wait, 10)  # taken off the queue, now executing
  await svc.retire(b.policy_hash)  # nothing is queued for it
  release.set()
  out = P.ActionFrame.from_wire(await pending)  # completes; before: "policy execution failed"
  np.testing.assert_allclose(out.groups[0].action[:, 0], [-0.1, -0.2], rtol=1e-6)
  await svc.health()  # idle now: the retirement completes
  assert b.policy_hash not in svc.policies and rt.closed
