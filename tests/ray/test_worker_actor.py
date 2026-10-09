# SPDX-License-Identifier: Apache-2.0
"""SimulationWorker as a real Ray actor: publication, fencing, credits, recovery."""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from raysimbatch import protocol as P
from raysimbatch.config.models import FailureConfig, WorkerConfig
from raysimbatch.policy.export import cartpole_bundle
from raysimbatch.protocol import FaultCode, Flag, SlotStatus
from raysimbatch.testing import fixtures as F
from raysimbatch.worker.actor import RemoteSimulationWorker, worker_options

pytestmark = pytest.mark.ray
N = 8


def make_worker(ray, name=None, fx=None, **kw):
  fx = fx or F.cartpole()
  run_id = uuid.uuid4().hex
  name = name or f"w-{uuid.uuid4().hex[:8]}"
  spec, mjb = F.worker_spec(fx, run_id, name, num_slots=N, threads=2, **kw)
  wc = WorkerConfig.model_validate(spec.worker)
  opts = worker_options(wc, FailureConfig(), name=name, memory_bytes=256 * 2**20)
  actor = RemoteSimulationWorker.options(**opts).remote(spec.to_dict(), f"mem://{run_id}/{name}")
  return actor, spec, ray.put(mjb)


def lease(epoch, ttl=60):
  return {"owner": "test", "epoch": epoch, "expires_unix_ns": time.time_ns() + int(ttl * 1e9)}


def reset_all(ray, actor, gen, episode=0, slots=None):
  slots = np.arange(N, dtype=np.int32) if slots is None else slots
  env = ray.get(actor.reset.remote({
    "command_id": P.new_id(), "generation": gen, "slots": slots,
    "env_id": slots.astype(np.uint64), "episode_id": np.full(slots.size, episode, np.uint64),
  }))
  assert env["rejected"] == 0, env
  return env


def action_frame(run_id, frame: P.HeteroFrame, actions, policy_hash=b"\0" * 32, seq_offset=0):
  g = frame.groups[0]
  n = g.rows
  return P.ActionFrame(
    run_id=run_id, request_id=P.new_id(), routing_epoch=0, created_unix_ns=time.time_ns(), trace_id=P.new_id(),
    groups=(P.ActionGroup(
      group_hash=g.group_hash, schema_hash=g.schema_hash, policy_hash=policy_hash,
      env_id=g.env_id.copy(), episode_id=g.episode_id.copy(),
      expected_control_seq=g.control_seq.copy() + np.uint64(seq_offset),
      slot_generation=g.slot_generation.copy(), slot=g.slot.copy(),
      action=np.ascontiguousarray(actions, np.float32), behavior_logp=np.zeros(n, np.float32),
      policy_version=np.zeros(n, np.uint64),
    ),),
  ).to_wire()


def command(gen, frame: P.HeteroFrame, policy_hash=b"\0" * 32, deadline_s=30.0):
  g = frame.groups[0]
  return {
    "command_id": P.new_id(), "expected_generation": gen, "routing_epoch": 0, "slots": g.slot.copy(),
    "episode_id": g.episode_id.copy(), "control_seq": g.control_seq.copy(), "policy_hash": policy_hash,
    "deadline_unix_ns": time.time_ns() + int(deadline_s * 1e9),
  }


def test_initialize_advance_and_immutable_publication(ray_session):
  ray = ray_session
  actor, spec, mjb = make_worker(ray)
  ready = ray.get(actor.initialize.remote(mjb, None, lease(1)))
  assert ready["generation"] == 1 and ready["obs_dim"] == 4 and ready["mujoco"] == "3.14.0"
  env = reset_all(ray, actor, 1)
  frame0 = P.HeteroFrame.from_wire(ray.get(env["frame_ref"]))
  obs0 = frame0.groups[0].observation.copy()
  assert not frame0.groups[0].observation.flags.writeable  # object-store snapshot is read-only
  run_id = bytes.fromhex(spec.run_id)
  acts = F.cartpole_policy(obs0)
  env1 = ray.get(actor.advance.remote(command(1, frame0), action_frame(run_id, frame0, acts)))
  assert env1["rejected"] == 0
  st = env1["status"]
  assert (st["status"] == SlotStatus.OK).all() and (st["completed_substeps"] == 5).all()
  frame1 = P.HeteroFrame.from_wire(ray.get(env1["frame_ref"]))
  assert (frame1.groups[0].control_seq == 1).all()
  # The earlier publication is unchanged although the worker kept mutating.
  np.testing.assert_array_equal(P.HeteroFrame.from_wire(ray.get(env["frame_ref"])).groups[0].observation, obs0)
  # Stale sequence (replaying the old frame) is rejected before mutation.
  health0 = ray.get(actor.health.remote())
  stale = ray.get(actor.advance.remote(command(1, frame0), action_frame(run_id, frame0, acts)))
  assert (stale["status"]["fault_code"] == FaultCode.SEQUENCE_MISMATCH).all()
  assert (stale["status"]["status"] == SlotStatus.REJECTED).all()
  # Wrong generation: whole command rejected.
  wrong = ray.get(actor.advance.remote(command(2, frame1), action_frame(run_id, frame1, acts)))
  assert wrong["rejected"] == FaultCode.STALE_GENERATION
  del health0


def test_duplicate_commands_return_the_cached_output_then_replay_required(ray_session):
  ray = ray_session
  actor, spec, mjb = make_worker(ray)
  ray.get(actor.initialize.remote(mjb, None, lease(1)))
  env = reset_all(ray, actor, 1)
  frame = P.HeteroFrame.from_wire(ray.get(env["frame_ref"]))
  run_id = bytes.fromhex(spec.run_id)
  cmd = command(1, frame)
  af = action_frame(run_id, frame, np.zeros((N, 1)))
  a = ray.get(actor.advance.remote(cmd, af))
  progress = ray.get(actor.health.remote())["progress"]
  b = ray.get(actor.advance.remote(cmd, af))  # retried delivery of the same command
  assert a["frame_ref"] == b["frame_ref"]  # cached output, never re-executed
  assert ray.get(actor.health.remote())["progress"] == progress
  ray.get(actor.ack.remote(a["credit_ids"]))
  ray.get(actor.ack.remote(env["credit_ids"]))
  c = ray.get(actor.advance.remote(cmd, af))
  assert c["rejected"] == FaultCode.REPLAY_REQUIRED


def test_output_credits_bound_unacknowledged_publications(ray_session):
  ray = ray_session
  actor, spec, mjb = make_worker(ray, flow={"inflight_frames_per_worker": 2})
  ray.get(actor.initialize.remote(mjb, None, lease(1)))
  env = reset_all(ray, actor, 1)  # credit 1 of 2
  frame = P.HeteroFrame.from_wire(ray.get(env["frame_ref"]))
  run_id = bytes.fromhex(spec.run_id)
  e1 = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, np.zeros((N, 1)))))
  f1 = P.HeteroFrame.from_wire(ray.get(e1["frame_ref"]))
  progress = ray.get(actor.health.remote())["progress"]
  e2 = ray.get(actor.advance.remote(command(1, f1), action_frame(run_id, f1, np.zeros((N, 1)))))
  assert e2["rejected"] == FaultCode.CREDIT_EXHAUSTED  # queue saturation stops production
  assert ray.get(actor.health.remote())["progress"] == progress  # nothing stepped
  assert ray.get(actor.ack.remote(env["credit_ids"])) == 1
  assert ray.get(actor.ack.remote(env["credit_ids"])) == 0  # duplicate ACKs free nothing
  e3 = ray.get(actor.advance.remote(command(1, f1), action_frame(run_id, f1, np.zeros((N, 1)))))
  assert e3["rejected"] == 0


def test_local_policy_run_control_plane_and_cancellation(ray_session):
  ray = ray_session
  bundle, exe = cartpole_bundle()
  actor, spec, mjb = make_worker(ray, fx=F.cartpole(max_control_steps=10**6))
  ray.get(actor.initialize.remote(mjb, ray.put(np.frombuffer(exe, np.uint8)), lease(1), bundle.to_json().decode()))
  reset_all(ray, actor, 1)
  slots = np.arange(N, dtype=np.int32)
  cmd = {
    "command_id": P.new_id(), "expected_generation": 1, "slots": slots, "policy_hash": bundle.policy_hash_bytes,
    "deadline_unix_ns": time.time_ns() + int(60e9), "max_control_steps": 10**6, "credit_token": 0,
  }
  ref = actor.run_local.remote(cmd)
  time.sleep(1.0)
  t0 = time.perf_counter()
  h = ray.get(actor.health.remote())  # control group answers while mutate is busy
  assert time.perf_counter() - t0 < 0.5
  assert h["active_command"] is not None and h["state"] == P.WorkerState.BUSY
  ray.cancel(ref, force=False)  # cooperative: checked between native calls
  # Ray 2.58: a cancelled actor task's ref raises even when the method returned
  # normally, so its outputs are unreachable through it (tests/ray_contract). The
  # worker cached the envelope under the command id: re-sending recovers it without
  # re-executing anything.
  with pytest.raises(ray.exceptions.TaskCancelledError):
    ray.get(ref)
  out = ray.get(actor.run_local.remote(cmd))
  assert out["canceled"] and out["rejected"] == 0
  steps = out["status"]["control_steps"]
  assert (steps > 10).all() and (out["status"]["status"] == SlotStatus.OK).all()  # LQR keeps them up
  h2 = ray.get(actor.health.remote())
  assert h2["active_command"] is None and h2["progress"] > h["progress"]


def test_fence_blocks_mutation_and_restart_requires_reinitialization(ray_session):
  ray = ray_session
  actor, spec, mjb = make_worker(ray)
  ray.get(actor.initialize.remote(mjb, None, lease(1)))
  env = reset_all(ray, actor, 1)
  ray.get(actor.fence.remote(1))
  frame = P.HeteroFrame.from_wire(ray.get(env["frame_ref"]))
  run_id = bytes.fromhex(spec.run_id)
  r = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, np.zeros((N, 1)))))
  assert r["rejected"] == FaultCode.STALE_GENERATION and "fenced" in r["message"]
  # Kill with restart allowed: Ray reruns only the constructor (no state restoration).
  pid = ray.get(actor.health.remote())["pid"]
  ray.kill(actor, no_restart=False)
  deadline = time.time() + 30
  while True:
    try:
      h = ray.get(actor.health.remote(), timeout=5)
      if h["pid"] != pid:
        break
    except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
      pass  # restarting
    if time.time() > deadline:
      raise AssertionError("actor did not restart")
    time.sleep(0.2)
  assert h["generation"] == 0 and h["state"] == P.WorkerState.CREATED
  r = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, np.zeros((N, 1)))))
  assert r["rejected"] == FaultCode.STALE_GENERATION and "not initialized" in r["message"]
  ready = ray.get(actor.initialize.remote(mjb, None, lease(2)))
  assert ready["generation"] == 2
  env2 = reset_all(ray, actor, 2, episode=5)
  old = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, np.zeros((N, 1)))))
  assert old["rejected"] == FaultCode.STALE_GENERATION  # old-generation work never lands
  f2 = P.HeteroFrame.from_wire(ray.get(env2["frame_ref"]))
  assert (f2.groups[0].slot_generation == 2).all() and (f2.groups[0].episode_id == 5).all()


def test_snapshot_restore_through_the_actor(ray_session):
  ray = ray_session
  fx = F.quadruped(max_control_steps=10**6)
  actor, spec, mjb = make_worker(ray, fx=fx)
  ray.get(actor.initialize.remote(mjb, None, lease(1)))
  env = reset_all(ray, actor, 1)
  run_id = bytes.fromhex(spec.run_id)
  frame = P.HeteroFrame.from_wire(ray.get(env["frame_ref"]))
  rng = np.random.default_rng(0)
  for _ in range(5):
    e = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, rng.uniform(-1, 1, (N, 8)))))
    ray.get(actor.ack.remote(e["credit_ids"]))
    frame = P.HeteroFrame.from_wire(ray.get(e["frame_ref"]))
  snap = ray.get(actor.snapshot.remote(np.arange(N, dtype=np.int32), 1))
  acts = [rng.uniform(-1, 1, (N, 8)) for _ in range(4)]
  first = []
  f = frame
  for a in acts:
    e = ray.get(actor.advance.remote(command(1, f), action_frame(run_id, f, a)))
    ray.get(actor.ack.remote(e["credit_ids"]))
    f = P.HeteroFrame.from_wire(ray.get(e["frame_ref"]))
    first.append(f.groups[0].observation.copy())
  assert ray.get(actor.restore.remote(snap, np.arange(N, dtype=np.int32), 1))["ok"]
  f = frame
  for a, want in zip(acts, first, strict=True):
    e = ray.get(actor.advance.remote(command(1, f), action_frame(run_id, f, a)))
    ray.get(actor.ack.remote(e["credit_ids"]))
    f = P.HeteroFrame.from_wire(ray.get(e["frame_ref"]))
    np.testing.assert_array_equal(f.groups[0].observation, want)


def test_fragments_are_consistent_transitions(ray_session):
  ray = ray_session
  bundle, exe = cartpole_bundle()
  fx = F.cartpole(max_control_steps=12)
  actor, spec, mjb = make_worker(ray, fx=fx, collect_fragments=True, fragment_steps=8,
                                 flow={"fragment_slabs_per_worker": 64, "inflight_frames_per_worker": 64})
  ray.get(actor.initialize.remote(mjb, ray.put(np.frombuffer(exe, np.uint8)), lease(1), bundle.to_json().decode()))
  reset_all(ray, actor, 1)
  slots = np.arange(N, dtype=np.int32)
  frags = []
  for k in range(3):
    out = ray.get(actor.run_local.remote({
      "command_id": P.new_id(), "expected_generation": 1, "slots": slots, "policy_hash": bundle.policy_hash_bytes,
      "deadline_unix_ns": time.time_ns() + int(60e9), "max_control_steps": 10, "autoreset": True,
      "episode_ids": np.arange(1 + k * N, 1 + (k + 1) * N, dtype=np.uint64), "flush": k == 2,
    }))
    frags += [P.Fragment.from_wire(ray.get(r)) for r in out["fragment_refs"]]
  assert frags
  total = 0
  for fr in frags:
    for e in range(fr.segments):
      a, b = int(fr.episode_offsets[e]), int(fr.episode_offsets[e + 1])
      total += b - a
      assert (fr.env_id[a:b] == fr.seg_env_id[e]).all() and (fr.episode_id[a:b] == fr.seg_episode_id[e]).all()
      np.testing.assert_array_equal(fr.control_seq[a:b], fr.seg_first_seq[e] + np.arange(b - a))
      last = fr.flags[b - 1]
      if last & Flag.TRUNCATED:
        assert fr.control_seq[b - 1] == 11  # 12-step time limit, 0-based
        assert np.any(fr.terminal_observation[e] != 0)
      elif not last & (Flag.TERMINATED | Flag.ABORTED):
        assert np.any(fr.bootstrap_observation[e] != 0)
    assert len(fr.fragment_id()) == 64 and len(fr.digest()) == 64
  assert total == 3 * 10 * N  # every stepped transition is in exactly one fragment


def _fill_fragment_buffers(ray, actor, bundle, slots):
  """With one fragment credit held unacknowledged, step until every slot's builder is at
  its hard cap (2 x fragment_steps) and run_local can no longer advance."""
  held = []
  for _ in range(10):
    out = ray.get(actor.run_local.remote({
      "command_id": P.new_id(), "expected_generation": 1, "slots": slots, "policy_hash": bundle.policy_hash_bytes,
      "deadline_unix_ns": time.time_ns() + int(60e9), "max_control_steps": 4}))
    ray.get(actor.ack.remote([out["frame_credit"]]))
    held += out["fragment_credits"]
    if int(out["status"]["control_steps"].sum()) == 0:
      return held
  raise AssertionError("buffers never filled")


def test_full_fragment_buffers_flush_once_credits_return(ray_session):
  """Found by looping Mode A: slots that filled their fragment buffers while every
  fragment credit was in flight could never flush afterwards (run_local checked
  fullness before stepping and flushed only after a step), a silent livelock."""
  ray = ray_session
  bundle, exe = cartpole_bundle()
  actor, spec, mjb = make_worker(ray, fx=F.cartpole(max_control_steps=10**6), collect_fragments=True,
                                 fragment_steps=4, flow={"fragment_slabs_per_worker": 1})
  ray.get(actor.initialize.remote(mjb, ray.put(np.frombuffer(exe, np.uint8)), lease(1), bundle.to_json().decode()))
  reset_all(ray, actor, 1)
  slots = np.arange(N, dtype=np.int32)
  held = _fill_fragment_buffers(ray, actor, bundle, slots)
  assert len(held) == 1
  ray.get(actor.ack.remote(held))  # the sink acknowledged: the credit is free again
  out = ray.get(actor.run_local.remote({
    "command_id": P.new_id(), "expected_generation": 1, "slots": slots, "policy_hash": bundle.policy_hash_bytes,
    "deadline_unix_ns": time.time_ns() + int(60e9), "max_control_steps": 4}))
  assert out["fragments"], "full buffers were not flushed"
  assert int(out["status"]["control_steps"].sum()) > 0, "no progress after the credit returned"


def test_full_fragment_buffers_flush_on_advance(ray_session):
  """The central-inference path (advance) has the same shape: full slots are rejected
  as CREDIT_EXHAUSTED and must flush as soon as a fragment credit is free."""
  ray = ray_session
  bundle, exe = cartpole_bundle()
  actor, spec, mjb = make_worker(ray, fx=F.cartpole(max_control_steps=10**6), collect_fragments=True,
                                 fragment_steps=4, flow={"fragment_slabs_per_worker": 1})
  ray.get(actor.initialize.remote(mjb, None, lease(1), None))
  frame = P.HeteroFrame.from_wire(ray.get(reset_all(ray, actor, 1)["frame_ref"]))
  run_id = bytes.fromhex(spec.run_id)
  held, rejected = [], False
  for _ in range(20):  # 4 steps -> flush (credit held), 8 more buffered to the cap, then rejected
    out = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, np.zeros((N, 1)))))
    ray.get(actor.ack.remote([out["frame_credit"]]))
    held += out["fragment_credits"]
    if (out["status"]["fault_code"] == P.FaultCode.CREDIT_EXHAUSTED).all():
      rejected = True
      break
    frame = P.HeteroFrame.from_wire(ray.get(out["frame_ref"]))
  assert rejected and len(held) == 1
  ray.get(actor.ack.remote(held))
  out = ray.get(actor.advance.remote(command(1, frame), action_frame(run_id, frame, np.zeros((N, 1)))))
  assert out["fragments"] and (out["status"]["status"] == P.SlotStatus.OK).all(), out["status"]
