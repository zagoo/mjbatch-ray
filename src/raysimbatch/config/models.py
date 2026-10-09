# SPDX-License-Identifier: Apache-2.0
"""Closed, typed configuration records (design §10, Appendix B.2/B.3).

Every leaf has a default. Models are strict, frozen and reject unknown keys. Input
is parsed from JSON text (YAML is converted first) so enumerations are plain string
literals. Fields marked ``# AMENDMENT`` are additions recorded in
docs/design-amendments.md.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

NonNeg = Annotated[int, Field(ge=0)]
Pos = Annotated[int, Field(gt=0)]
NonNegF = Annotated[float, Field(ge=0)]
PosF = Annotated[float, Field(gt=0)]
Frac = Annotated[float, Field(ge=0, le=1)]


class Model(BaseModel):
  model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)

  @classmethod
  def from_obj(cls, obj: object):
    """Validate plain Python data (e.g. a model_dump) through the strict JSON path, so
    lists become tuples and string literals become enumerations."""
    import json

    return cls.model_validate_json(json.dumps(obj))


class ArtifactRef(Model):
  uri: str | None = None
  sha256: str | None = None

  @model_validator(mode="after")
  def _both_or_neither(self) -> ArtifactRef:
    if (self.uri is None) != (self.sha256 is None):
      raise ValueError("uri and sha256 must be populated together")
    if self.sha256 is not None and not _HEX64.match(self.sha256):
      raise ValueError("sha256 must be 64 lowercase hex characters")
    return self

  @property
  def empty(self) -> bool:
    return self.uri is None


EMPTY = ArtifactRef()


# ----------------------------------------------------------------------- cluster


class ClusterConfig(Model):
  ray_version: str = "2.58.0"
  kuberay_version: str = "1.7.0"
  address: str = "auto"
  namespace: str = "raysimbatch"
  image_digest: str | None = None
  cpu_node_cores: Pos = 128
  cpu_node_memory_gib: Pos = 512
  ray_cpus: Pos = 120
  simulation_cpus: Pos = 112
  object_store_gib: Pos = 64
  shm_gib: Pos = 80
  spill_path: str = "/var/lib/ray-spill"
  spill_gib: NonNeg = 512
  min_cpu_nodes: NonNeg = 1
  max_cpu_nodes: NonNeg = 64
  min_gpu_nodes: NonNeg = 0
  max_gpu_nodes: NonNeg = 32
  gpu_type: str = "L40S"
  head_cpu: Pos = 16
  head_memory_gib: Pos = 64
  coordinator_fanout: Pos = 8
  placement: Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"] = "PACK"
  gang_timeout_s: PosF = 600.0
  gcs_ft: Literal["off", "redis_qualified"] = "off"
  gcs_qualification: ArtifactRef = EMPTY
  redis_secret: str | None = None
  autoscale_up_backlog_s: PosF = 60.0
  autoscale_up_hold_s: PosF = 30.0
  autoscale_up_fraction_per_min: Frac = 0.2
  autoscale_down_utilization: Frac = 0.4
  autoscale_down_hold_s: PosF = 600.0
  tenant_cpu_limit: Pos = 8192
  tenant_gpu_limit: NonNeg = 32
  runtime_env_vars: dict[str, str] = Field(default_factory=dict)
  # Appendix B.3
  gpu_pod_cpus: Pos = 16
  gpu_ray_cpus: Pos = 14
  gpu_pod_memory_gib: Pos = 128
  gpu_devices_per_pod: Pos = 1
  termination_grace_s: Pos = 120
  service_account: str = "raysimbatch-worker"
  platform_inventory: ArtifactRef = EMPTY


class WorkerConfig(Model):
  envs: Pos = 256
  threads: Pos = 4
  num_cpus: Pos = 4
  slice_size: Pos = 64
  slice_min: Pos = 16
  slice_max: Pos = 128
  model_shards: Pos = 1
  memory_gib: Pos = 8
  native_backend: Literal["mjbatch", "scalar"] = "mjbatch"
  affinity: Literal["numa_exclusive", "none"] = "numa_exclusive"
  smt: bool = False
  engine_threads: NonNeg = 0
  forward_at_control_boundary: bool = True
  # One MJBatch call per policy interval where results are provably identical to
  # substep-by-substep stepping (AM-2): health.check_each_substep=false and a torque or
  # position controller without physics-clock delay. False forces per-substep stepping.
  fuse_substeps: bool = True
  local_control_steps: Pos = 8
  checkpoint_interval_s: PosF = 60.0
  recovery: Literal["reset", "snapshot"] = "reset"
  omp_threads: Pos = 1
  blas_threads: Pos = 1
  torch_threads: Pos = 1
  max_pending_calls: Pos = 8
  native_extension: ArtifactRef = EMPTY


class RandomizationRule(Model):
  field: str = "body_mass"
  indices: tuple[NonNeg, ...] = ()
  distribution: Literal["fixed", "uniform", "log_uniform", "normal_clipped", "categorical"] = "fixed"
  operation: Literal["replace", "multiply"] = "multiply"
  low: float = 1.0
  high: float = 1.0
  mean: float = 1.0
  std: NonNegF = 0.0
  values: tuple[float, ...] = ()
  probabilities: tuple[float, ...] = ()
  refresh: Literal["auto", "always"] = "auto"
  stream: str = "domain"

  @model_validator(mode="after")
  def _distribution(self) -> RandomizationRule:
    if self.distribution in ("uniform", "log_uniform", "normal_clipped") and not self.low <= self.high:
      raise ValueError("low must be <= high")
    if self.distribution == "log_uniform" and self.low <= 0:
      raise ValueError("log_uniform needs low > 0")
    if self.distribution == "categorical":
      if not self.values or len(self.values) != len(self.probabilities):
        raise ValueError("categorical needs equal nonempty values/probabilities")
      if any(p < 0 for p in self.probabilities) or abs(sum(self.probabilities) - 1.0) > 1e-9:
        raise ValueError("categorical probabilities must be nonnegative and sum to one")
    if self.stream != "domain":
      raise ValueError("randomization draws use the 'domain' stream")
    return self


class HealthConfig(Model):
  check_each_substep: bool = True
  max_abs_qvel: PosF = 1000.0
  max_abs_qacc: PosF = 1_000_000.0
  max_energy_j: PosF = 100_000_000.0
  max_contacts: Pos = 100_000
  quaternion_norm_tolerance: PosF = 0.001
  fatal_warning_codes: tuple[str, ...] = ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL", "CONTACTFULL", "CNSTRFULL")


class ModelGroup(Model):
  name: str = "default"
  mjcf: ArtifactRef = EMPTY
  assets: ArtifactRef = EMPTY
  mjb: ArtifactRef = EMPTY
  task_kernel: ArtifactRef = EMPTY
  timestep_ns: Pos = 2_000_000
  decimation: Pos = 4
  solver: Literal["Newton", "CG", "PGS"] = "Newton"
  integrator: Literal["Euler", "RK4", "implicit", "implicitfast"] = "implicitfast"
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"
  iterations: Pos = 100
  tolerance: NonNegF = 1e-8
  resource_pool: Literal["sim_general", "sim_contact"] = "sim_general"
  initial_keyframe: Annotated[int, Field(ge=-1)] = -1
  max_control_steps: Pos = 1000
  observation_schema: ArtifactRef = EMPTY
  action_schema: ArtifactRef = EMPTY
  adapter: ArtifactRef = EMPTY
  randomization: tuple[RandomizationRule, ...] = ()
  health: HealthConfig = HealthConfig()
  expected_model_mib: PosF = 8.0
  expected_data_mib: PosF = 16.0
  plugins: tuple[ArtifactRef, ...] = ()
  sleep: bool = False


class InferenceConfig(Model):
  placement: Literal["local", "central", "auto"] = "local"
  backend: Literal["onnx", "torchscript"] = "onnx"
  gpu_replicas: Pos = 1
  cpus_per_replica: Pos = 2
  gpus_per_replica: NonNeg = 1
  max_batch_rows: Pos = 1024
  flush_ms: PosF = 2.0
  queue_requests: Pos = 64
  queue_rows: Pos = 8192
  deadline_s: PosF = 0.05
  max_resident_versions: Pos = 2
  utilization_ceiling: Annotated[float, Field(gt=0, le=1)] = 0.7
  auto_min_speedup: PosF = 1.15
  auto_max_p99_s: PosF = 0.05
  # AMENDMENT (AM-11): device of the central service; "cuda" in production. "cpu"
  # and "mps" exist for functional qualification on hosts without CUDA.
  device: Literal["cuda", "cpu", "mps"] = "cuda"


class PolicySpec(Model):
  name: str = "default"
  bundle: ArtifactRef = EMPTY
  executable: ArtifactRef = EMPTY
  weights: ArtifactRef = EMPTY
  version: NonNeg = 0
  deterministic: bool = True
  normalization: Literal["embedded", "external", "none"] = "embedded"
  normalizer: ArtifactRef = EMPTY
  distribution: Literal["deterministic", "diagonal_gaussian", "categorical"] = "deterministic"
  recurrent_schema: ArtifactRef = EMPTY
  certified_batch_sizes: tuple[Pos, ...] = (1, 16, 32, 64, 128, 256, 512, 1024)


class RolloutConfig(Model):
  target_transitions: Pos = 1_000_000
  collection: Literal["async", "on_policy_epoch"] = "async"
  fragment_steps: Pos = 32
  fragment_flush_s: PosF = 1.0
  sink_uri: str | None = None
  require_idempotent_sink: bool = True
  durable_outbox: bool = True
  max_policy_lag_versions: NonNeg = 8
  max_policy_lag_s: PosF = 2.0
  allow_quarantine_exclusion: bool = False
  rllib_adapter: bool = False


class EvalConfig(Model):
  seeds: tuple[NonNeg, ...] = tuple(range(20))
  episodes_per_seed: Pos = 10
  max_attempts: Pos = 3
  lease_s: PosF = 120.0
  lease_renew_s: PosF = 20.0
  unit_block: Pos = 128
  wave_actors: Pos = 28
  gang: bool = True
  resume: bool = True
  allow_partial: bool = True
  bootstrap_replicates: Pos = 10_000
  confidence: Annotated[float, Field(gt=0, lt=1)] = 0.95
  task_weights: dict[str, NonNegF] = Field(default_factory=dict)
  exports: tuple[Literal["parquet", "csv", "json", "markdown"], ...] = ("parquet", "csv", "json", "markdown")


class CurriculumConfig(Model):
  version: NonNeg = 0
  weights: dict[str, NonNegF] = Field(default_factory=dict)
  weight_basis: Literal["episode_starts"] = "episode_starts"
  rebalance_s: PosF = 60.0
  boundary: Literal["episode"] = "episode"


class FailureConfig(Model):
  max_restarts: NonNeg = 3
  max_task_retries: NonNeg = 0
  heartbeat_s: PosF = 1.0
  suspect_s: PosF = 10.0
  native_min_deadline_s: PosF = 2.0
  native_p99_multiplier: PosF = 10.0
  native_max_deadline_s: PosF = 30.0
  cancel_grace_s: PosF = 2.0
  lease_s: PosF = 20.0
  lease_renew_s: PosF = 5.0
  circuit_restart_count: Pos = 3
  circuit_window_s: PosF = 600.0
  circuit_open_s: PosF = 60.0
  canary_episodes: Pos = 100
  max_canary_failures: Pos = 3
  native_quarantine_count: Pos = 3
  health_quarantine_count: Pos = 10
  health_quarantine_window: Pos = 100
  minimum_healthy_fraction: Frac = 0.95
  on_insufficient_capacity: Literal["pause", "fail"] = "pause"


class FlowConfig(Model):
  inflight_frames_per_worker: Pos = 4
  fragment_slabs_per_worker: Pos = 2
  assembler_bytes: Pos = 268_435_456
  asset_cache_bytes: Pos = 17_179_869_184
  policy_load_bytes: Pos = 4_294_967_296
  object_target_bytes: Pos = 1_572_864
  object_max_bytes: Pos = 4_194_304
  admission_store_fraction: Frac = 0.60
  throttle_fraction: Frac = 0.65
  stop_episode_fraction: Frac = 0.75
  pause_fraction: Frac = 0.85
  resume_fraction: Frac = 0.60
  resume_hold_s: PosF = 30.0
  spill_hot_objects_allowed: bool = False


# ----------------------------------------------------------- transfer contracts


class ParityConfig(Model):
  observation_atol: NonNegF = 1e-5
  observation_rtol: NonNegF = 1e-5
  action_atol: NonNegF = 1e-4
  mass_rtol: NonNegF = 0.01
  com_atol_m: NonNegF = 0.001
  inertia_rtol: NonNegF = 0.02
  pose_atol_m: NonNegF = 0.001
  angle_atol_deg: NonNegF = 0.1
  effort_atol_nm: NonNegF = 0.01
  effort_rtol: NonNegF = 0.01
  filter_atol: NonNegF = 1e-6
  slide_rtol: NonNegF = 0.05
  contact_curve_rtol: NonNegF = 0.10
  initial_state_atol: NonNegF = 1e-6
  contact_fit_grid: ArtifactRef = EMPTY
  held_out_fixtures: ArtifactRef = EMPTY


class Sim2SimConfig(Model):
  bundle: ArtifactRef = EMPTY
  source_trace: ArtifactRef = EMPTY
  static_gates: bool = True
  dynamics_gates: bool = True
  fixture_count: Pos = 1000
  diagnostic_seed_count: Pos = 20
  diagnostic_fraction: Frac = 0.01
  axes: tuple[str, ...] = ("joint_map", "normalization", "actuator", "filter", "timestep", "inertia", "contact")
  parity: ParityConfig = ParityConfig()


ObservationSource = Literal[
  "joint_pos",
  "joint_vel",
  "base_lin_vel",
  "base_ang_vel",
  "gravity",
  "command",
  "previous_action",
  "phase",
  "rot6d",
  "sensor",
  "custom",
  # AMENDMENT (AM-8): the policy-rate adapter-applied action and the raw base
  # orientation in the contract's quaternion order; the injected quaternion-
  # convention mismatch (§13.4) needs a term that depends on that order.
  "applied_action",
  "base_quat",
]


class ObservationTerm(Model):
  name: str = ""
  source: ObservationSource = "joint_pos"
  dimension: NonNeg = 0
  frame: str = "body"
  units: str = "SI"
  scale: float = 1.0
  clip: PosF | None = None
  history: Pos = 1
  noise_std: NonNegF = 0.0
  rot6d_layout: Literal["columns", "rows"] = "columns"
  reference_frame: str = "world"
  kernel: ArtifactRef = EMPTY
  # AMENDMENT (AM-8): named selectors resolved against ModelMetadata (joint names for
  # joint terms, sensor names, command/action indices), and a phase period.
  selectors: tuple[str, ...] = ()
  offsets: tuple[float, ...] = ()
  signs: tuple[Literal[-1, 1], ...] = ()
  period_s: NonNegF = 0.0


class ActuatorContract(Model):
  kind: Literal["torque", "position", "pd", "dc_motor"] = "pd"
  kp: tuple[float, ...] = ()
  kd: tuple[float, ...] = ()
  effort_limit: tuple[float, ...] = ()
  velocity_limit: tuple[float, ...] = ()
  gear: tuple[float, ...] = ()
  armature: tuple[float, ...] = ()
  envelope: ArtifactRef = EMPTY
  feedforward: float = 0.0
  # AMENDMENT (AM-8): actuator names in policy-action order; resolved to ctrl indices.
  actuator_names: tuple[str, ...] = ()


class AdapterContract(Model):
  schema_version: Literal[1] = 1
  joint_names: tuple[str, ...] = ()
  joint_signs: tuple[Literal[-1, 1], ...] = ()
  observation_terms: tuple[ObservationTerm, ...] = ()
  quaternion: Literal["wxyz", "xyzw"] = "wxyz"
  velocity_frame: Literal["body", "world"] = "body"
  history_layout: Literal["term_major", "time_major"] = "term_major"
  history_reset: Literal["zeros", "repeat_initial"] = "zeros"
  normalization: Literal["embedded", "external", "none"] = "embedded"
  normalizer: ArtifactRef = EMPTY
  normalization_epsilon: PosF = 1e-8
  action_order: Literal["clip_scale_offset", "scale_clip_offset"] = "clip_scale_offset"
  action_clip: PosF = 1.0
  action_scale: tuple[float, ...] = ()
  action_offset: tuple[float, ...] = ()
  ema_alpha: Annotated[float, Field(gt=0, le=1)] = 1.0
  ema_initial: float = 0.0
  delay_ticks: NonNeg = 0
  delay_clock: Literal["policy", "physics"] = "policy"
  actuator: ActuatorContract = ActuatorContract()
  recurrent_schema: ArtifactRef = EMPTY
  golden_fixtures: ArtifactRef = EMPTY
  # AMENDMENT (AM-8): ordering facts the exporter must attest and gates must check.
  history_order: Literal["oldest_first", "newest_first"] = "oldest_first"
  term_op_order: Literal["noise_clip_scale", "scale_clip_noise"] = "noise_clip_scale"
  normalization_clip: NonNegF = 0.0
  base_joint: str | None = None


class TensorField(Model):
  name: str = ""
  dtype: Literal["float32", "float64", "int32", "int64", "uint8", "uint16", "uint64"] = "float32"
  shape: tuple[NonNeg, ...] = ()
  units: str = "SI"
  frame: str = "none"
  optional: bool = False


class TensorSchema(Model):
  version: Literal[1] = 1
  fields: tuple[TensorField, ...] = ()
  semantic_hash: str | None = None


class FixtureSpec(Model):
  name: str = ""
  kind: Literal["observation", "actuator", "drop", "slide", "rest", "kinematics"] = "observation"
  inputs: ArtifactRef = EMPTY
  expected: ArtifactRef = EMPTY
  seed: NonNeg = 0
  duration_s: PosF = 1.0
  held_out: bool = False


class AssetContract(Model):
  version: Literal[1] = 1
  source_engine: str = "IsaacLab"
  source_version: str | None = None
  target_version: str = "3.14.0"
  length_unit: str = "m"
  mass_unit: str = "kg"
  up_axis: str = "Z"
  forward_axis: str = "X"
  root_convention: str = "floating_wxyz"
  converter: ArtifactRef = EMPTY
  resolved_properties: ArtifactRef = EMPTY
  collision_pairs: ArtifactRef = EMPTY
  mesh_manifest: ArtifactRef = EMPTY
  edits: ArtifactRef = EMPTY
  fixtures: tuple[FixtureSpec, ...] = ()


class TelemetryConfig(Model):
  metrics_prefix: str = "raysimbatch"
  metrics_port: Annotated[int, Field(ge=1, le=65535)] = 8080
  scrape_s: PosF = 15.0
  log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
  trace_fraction: Frac = 0.001
  trace_errors: bool = True
  otel_endpoint: str | None = None
  log_queue_bytes: Pos = 16_777_216
  metrics_flush_s: PosF = 5.0
  per_env_metrics: bool = False
  diagnostic_trace_steps: NonNeg = 100
  diagnostic_max_bytes: Pos = 67_108_864
  task_events_hot: bool = False
  task_events_control: bool = True
  run_label_retention_hours: Pos = 24


class StorageConfig(Model):
  blob_request_timeout_s: PosF = 30.0
  database_timeout_s: PosF = 5.0
  retry_attempts: NonNeg = 3
  retry_backoff_s: tuple[NonNegF, ...] = (0.1, 0.5, 2.0)
  result_commit_rows: Pos = 1000
  result_commit_interval_s: PosF = 1.0
  result_retention_days: Pos = 365
  fragment_retention_days: Pos = 7
  diagnostic_retention_days: Pos = 30
  orphan_retention_days: Pos = 7


class TaskScoring(Model):
  success_metric: str = "success"
  success_threshold: Frac = 0.8
  minimum_valid_coverage: Frac = 0.95
  tracking_metric: str | None = None
  maximum_tracking_error: NonNegF | None = None
  reward_aggregation: Literal["sum"] = "sum"
  suite_weight: NonNegF = 1.0
  comparison: Literal["absolute", "paired_source"] = "absolute"

  @model_validator(mode="after")
  def _tracking(self) -> TaskScoring:
    if self.tracking_metric is not None and self.maximum_tracking_error is None:
      raise ValueError("tracking_metric requires maximum_tracking_error")
    return self


class TaskSpec(Model):
  version: Literal[1] = 1
  name: str = ""
  kernel: ArtifactRef = EMPTY
  observation_schema: ArtifactRef = EMPTY
  action_schema: ArtifactRef = EMPTY
  reset_distribution: ArtifactRef = EMPTY
  termination_definition: ArtifactRef = EMPTY
  reward_definition: ArtifactRef = EMPTY
  scoring: TaskScoring = TaskScoring()
  model_groups: tuple[str, ...] = ()


class RunConfig(Model):
  schema_version: Literal[1] = 1
  run_id: str | None = None
  mode: Literal["rollout", "eval", "sim2sim"] = "eval"
  target_envs: Pos = 10_000
  run_seed: Annotated[int, Field(ge=0, lt=1 << 128)] = 0
  deterministic: bool = True
  exploratory: bool = False
  artifact_root: str | None = None
  database_secret: str | None = None
  release_lock: ArtifactRef = EMPTY
  cluster: ClusterConfig = ClusterConfig()
  worker: WorkerConfig = WorkerConfig()
  inference: InferenceConfig = InferenceConfig()
  rollout: RolloutConfig = RolloutConfig()
  evaluation: EvalConfig = EvalConfig()
  sim2sim: Sim2SimConfig = Sim2SimConfig()
  failure: FailureConfig = FailureConfig()
  flow: FlowConfig = FlowConfig()
  telemetry: TelemetryConfig = TelemetryConfig()
  storage: StorageConfig = StorageConfig()
  models: tuple[ModelGroup, ...] = ()
  policies: tuple[PolicySpec, ...] = ()
  curriculum: CurriculumConfig = CurriculumConfig()
  # AMENDMENT (AM-10): the evaluation task suite for CLI submission; the Python API
  # also accepts tasks directly (RaySimBatch.evaluate).
  tasks: tuple[TaskSpec, ...] = ()
