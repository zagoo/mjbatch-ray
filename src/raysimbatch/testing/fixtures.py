# SPDX-License-Identifier: Apache-2.0
"""Reference model groups used by tests, examples and qualification harnesses.

Each fixture bundles MJCF bytes, a ModelGroup, an attested AdapterContract, a task
kernel document and an initial-state distribution. They are small but exercise every
kernel path: torque control (cart-pole, reacher), PD control over a floating base
(quadruped), sensors, commands, site observations and contact.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import mujoco
import numpy as np

from raysimbatch.adapter import CompiledAdapter, compile_adapter
from raysimbatch.config.models import (
  ActuatorContract,
  AdapterContract,
  HealthConfig,
  ModelGroup,
  ObservationTerm,
  RandomizationRule,
)
from raysimbatch.tasks import TaskBinding, bind_task
from raysimbatch.worker.core import CoreSpec, MJBatchCore
from raysimbatch.worker.model import ModelMetadata, compile_model


def asset(name: str) -> bytes:
  return resources.files("raysimbatch.testing").joinpath("assets", name).read_bytes()


@dataclass(frozen=True)
class Fixture:
  name: str
  xml: bytes
  group: ModelGroup
  contract: AdapterContract
  task_doc: dict[str, Any]
  reset_distribution: dict[str, Any] | None = None
  randomization: tuple[RandomizationRule, ...] = ()
  keyframe: int = -1
  extra: dict[str, Any] = field(default_factory=dict)

  def model(self) -> mujoco.MjModel:
    return compile_model(self.xml, None, self.group)

  def meta(self) -> ModelMetadata:
    return ModelMetadata.from_model(self.model())

  def binding(self) -> TaskBinding:
    return bind_task(self.task_doc, self.meta())

  def compiled(self) -> CompiledAdapter:
    return compile_adapter(self.contract, self.meta(), num_commands=self.binding().num_commands)

  def core(self, num_slots: int = 16, threads: int = 2, **spec: Any) -> MJBatchCore:
    model = self.model()
    meta = ModelMetadata.from_model(model)
    binding = bind_task(self.task_doc, meta)
    compiled = compile_adapter(self.contract, meta, num_commands=binding.num_commands)
    cs = CoreSpec(
      num_slots=num_slots,
      threads=threads,
      decimation=self.group.decimation,
      timestep_ns=self.group.timestep_ns,
      keyframe=spec.pop("keyframe", self.keyframe),
      max_control_steps=spec.pop("max_control_steps", self.group.max_control_steps),
      health=spec.pop("health", self.group.health),
      randomization=spec.pop("randomization", self.randomization),
      reset_distribution=spec.pop("reset_distribution", self.reset_distribution),
      **spec,
    )
    return MJBatchCore(model, meta, compiled, binding, cs)

  def digest(self) -> str:
    return hashlib.sha256(self.xml + self.contract.model_dump_json().encode()).hexdigest()


def cartpole(**group_kw: Any) -> Fixture:
  group = ModelGroup(
    name="cartpole",
    timestep_ns=2_000_000,
    decimation=5,
    solver="Newton",
    integrator="implicitfast",
    max_control_steps=group_kw.pop("max_control_steps", 500),
    health=HealthConfig(max_abs_qvel=500.0),
    **group_kw,
  )
  contract = AdapterContract(
    joint_names=("slider",),
    joint_signs=(1,),
    observation_terms=(
      ObservationTerm(name="pos", source="joint_pos", dimension=2, selectors=("slider", "hinge"),
                      offsets=(0.0, 0.0), signs=(1, 1)),
      ObservationTerm(name="vel", source="joint_vel", dimension=2, selectors=("slider", "hinge"), signs=(1, 1)),
    ),
    action_scale=(20.0,),
    action_offset=(0.0,),
    action_clip=1.0,
    actuator=ActuatorContract(kind="torque", effort_limit=(20.0,)),
  )
  task = {"name": "balance", "version": 1,
          "spec": {"angle_joint": "hinge", "slider_joint": "slider", "angle_limit": 0.6, "slider_limit": 2.4}}
  reset = {"qpos": [{"joints": ["slider", "hinge"], "low": -0.05, "high": 0.05}],
           "qvel": [{"joints": ["slider", "hinge"], "low": -0.05, "high": 0.05}]}
  return Fixture("cartpole", asset("cartpole.xml"), group, contract, task, reset)


QUAD_JOINTS = ("fl_hip", "fl_knee", "fr_hip", "fr_knee", "hl_hip", "hl_knee", "hr_hip", "hr_knee")
QUAD_STAND = (0.5, -1.0) * 4


def quadruped(history: int = 1, **group_kw: Any) -> Fixture:
  group = ModelGroup(
    name="quadruped",
    timestep_ns=4_000_000,
    decimation=4,
    solver="Newton",
    iterations=20,
    tolerance=1e-8,
    integrator="implicitfast",
    max_control_steps=group_kw.pop("max_control_steps", 400),
    initial_keyframe=0,
    **group_kw,
  )
  terms = (
    ObservationTerm(name="gravity", source="gravity", dimension=3, history=history),
    ObservationTerm(name="ang_vel", source="base_ang_vel", dimension=3, scale=0.25, history=history),
    ObservationTerm(name="lin_vel", source="base_lin_vel", dimension=3, scale=2.0, history=history),
    ObservationTerm(name="command", source="command", dimension=3, history=history),
    ObservationTerm(name="joint_pos", source="joint_pos", dimension=8, selectors=QUAD_JOINTS,
                    offsets=QUAD_STAND, history=history),
    ObservationTerm(name="joint_vel", source="joint_vel", dimension=8, selectors=QUAD_JOINTS,
                    scale=0.05, history=history),
    ObservationTerm(name="last_action", source="previous_action", dimension=8, history=history),
  )
  contract = AdapterContract(
    joint_names=QUAD_JOINTS,
    joint_signs=(1,) * 8,
    observation_terms=terms,
    base_joint="root",
    history_reset="repeat_initial",
    action_scale=(0.5,),
    action_offset=QUAD_STAND,
    action_clip=4.0,
    actuator=ActuatorContract(kind="pd", kp=(20.0,) * 8, kd=(0.5,) * 8, effort_limit=(12.0,) * 8),
  )
  task = {
    "name": "velocity_tracking",
    "version": 1,
    "spec": {
      "base_joint": "root", "sigma_lin": 0.25, "sigma_ang": 0.25, "min_height": 0.12, "min_up_z": 0.3,
      "command_low": [-0.5, -0.2, -0.5], "command_high": [0.8, 0.2, 0.5], "resample_steps": 125,
      "weights": {"lin": 1.0, "ang": 0.5, "alive": 0.1},
    },
  }
  reset = {"qpos": [{"joints": list(QUAD_JOINTS), "low": -0.05, "high": 0.05}]}
  rand = (
    RandomizationRule(field="body_mass", distribution="uniform", operation="multiply", low=0.8, high=1.2),
    RandomizationRule(field="geom_friction", indices=(0,), distribution="uniform", operation="replace",
                      low=0.5, high=1.1),
  )
  return Fixture("quadruped", asset("quadruped.xml"), group, contract, task, reset, rand, keyframe=0)


def reacher(**group_kw: Any) -> Fixture:
  group = ModelGroup(
    name="reacher",
    timestep_ns=2_000_000,
    decimation=10,
    max_control_steps=group_kw.pop("max_control_steps", 200),
    **group_kw,
  )
  joints = ("j1", "j2", "j3")
  contract = AdapterContract(
    joint_names=joints,
    joint_signs=(1, 1, 1),
    observation_terms=(
      ObservationTerm(name="q", source="joint_pos", dimension=3, selectors=joints, offsets=(0.0, 0.0, 0.0)),
      ObservationTerm(name="dq", source="joint_vel", dimension=3, selectors=joints, scale=0.1),
      ObservationTerm(name="target", source="command", dimension=3),
    ),
    action_scale=(1.0,),
    action_offset=(0.0,),
    actuator=ActuatorContract(kind="torque", effort_limit=(1.0, 1.0, 1.0)),
  )
  task = {"name": "reach", "version": 1,
          "spec": {"site": "fingertip", "target_low": [-0.3, -0.3, 0.1], "target_high": [0.3, 0.3, 0.1],
                   "success_tol": 0.05}}
  reset = {"qpos": [{"joints": list(joints), "low": -0.3, "high": 0.3}]}
  return Fixture("reacher", asset("reacher.xml"), group, contract, task, reset)


FIXTURES = {"cartpole": cartpole, "quadruped": quadruped, "reacher": reacher}


# ------------------------------------------------------------------ reference policy


def no_termination(fx: Fixture) -> Fixture:
  """Long physics runs (parity, throughput): disable task termination only."""
  spec = dict(fx.task_doc["spec"])
  if fx.task_doc["name"] == "balance":
    spec.update(angle_limit=1e9, slider_limit=1e9)
  elif fx.task_doc["name"] == "velocity_tracking":
    spec.update(min_height=-1e9, min_up_z=-2.0)
  return Fixture(fx.name, fx.xml, fx.group, fx.contract, {**fx.task_doc, "spec": spec},
                 fx.reset_distribution, fx.randomization, fx.keyframe)


@functools.lru_cache(maxsize=4)
def cartpole_lqr_gain(decimation: int = 5) -> np.ndarray:
  """Discrete LQR gain K (action = -K x, x = [x, theta, xdot, thetadot]) for the
  cart-pole fixture at its control interval, from MuJoCo's finite-difference
  linearization (mjd_transitionFD). Action units are the policy's [-1, 1] command."""
  fx = cartpole()
  m = fx.model()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  nv = m.nv
  A = np.zeros((2 * nv, 2 * nv))
  B = np.zeros((2 * nv, m.nu))
  mujoco.mjd_transitionFD(m, d, 1e-6, True, A, B, None, None)
  # One control interval holds the action for `decimation` substeps.
  Ad = np.linalg.matrix_power(A, decimation)
  Bd = sum((np.linalg.matrix_power(A, k) @ B for k in range(decimation)), np.zeros_like(B))
  Bd = Bd * 20.0  # action_scale: policy command -> force
  Q = np.diag([1.0, 20.0, 0.1, 1.0])
  R = np.array([[0.5]])
  P = Q.copy()
  for _ in range(5000):
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    P_next = Q + Ad.T @ P @ (Ad - Bd @ K)
    if np.max(np.abs(P_next - P)) < 1e-10:
      P = P_next
      break
    P = P_next
  return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


def cartpole_policy(obs: np.ndarray) -> np.ndarray:
  """numpy reference of the exported cart-pole policy (obs = [x, theta, xdot, thetadot])."""
  K = cartpole_lqr_gain()
  return np.clip(-(obs.astype(np.float64) @ K.T), -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------- worker specs


def worker_spec(
  fx: Fixture,
  run_id: str,
  name: str,
  *,
  mode: str = "rollout",
  num_slots: int = 16,
  threads: int = 2,
  collect_fragments: bool = False,
  run_seed: int = 7,
  fragment_steps: int = 8,
  flow: dict[str, Any] | None = None,
  artifact_root: str | None = None,
):
  """A WorkerSpec plus the MJB bytes for ``fx`` (small shards for tests/examples)."""
  from raysimbatch.config.models import FailureConfig, FlowConfig, RolloutConfig, WorkerConfig
  from raysimbatch.routing.groups import group_key
  from raysimbatch.worker.actor import WorkerSpec
  from raysimbatch.worker.model import model_hash, to_mjb

  model = fx.model()
  meta = ModelMetadata.from_model(model)
  binding = bind_task(fx.task_doc, meta)
  compiled = compile_adapter(fx.contract, meta, num_commands=binding.num_commands)
  wc = WorkerConfig(envs=num_slots, threads=threads, num_cpus=threads, slice_size=min(num_slots, 64),
                    slice_min=min(num_slots, 16), slice_max=max(num_slots, 128), memory_gib=1, affinity="none")
  group = fx.group.model_copy(update={"randomization": fx.randomization, "initial_keyframe": fx.keyframe})
  spec = WorkerSpec(
    run_id=run_id,
    actor_name=name,
    shard_id=0,
    coordinator_id=0,
    mode=mode,
    group=group.model_dump(mode="json"),
    worker=wc.model_dump(mode="json"),
    failure=FailureConfig().model_dump(mode="json"),
    flow=FlowConfig(**(flow or {})).model_dump(mode="json"),
    rollout=RolloutConfig(fragment_steps=fragment_steps).model_dump(mode="json"),
    contract=fx.contract.model_dump(mode="json"),
    task_doc=fx.task_doc,
    group_hash=group_key(model_hash(fx.xml, None, group), binding, compiled, group),
    config_hash="0" * 64,
    run_seed=run_seed,
    reset_distribution=fx.reset_distribution,
    num_slots=num_slots,
    collect_fragments=collect_fragments,
    artifact_root=artifact_root,
  )
  return spec, np.frombuffer(to_mjb(model), np.uint8)
