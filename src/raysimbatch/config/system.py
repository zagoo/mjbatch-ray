# SPDX-License-Identifier: Apache-2.0
"""ConfigurationSystem: load, validate, canonical hash, and restart diff (design §10)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from raysimbatch.config.models import RunConfig

Severity = Literal["error", "warning"]
Classification = Literal["hot", "boundary", "new_run", "forbidden"]

THREAD_CAP_VARS = frozenset(
  {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"}
)


@dataclass(frozen=True)
class Issue:
  path: str
  code: str
  message: str
  severity: Severity = "error"


@dataclass(frozen=True)
class ValidationReport:
  issues: tuple[Issue, ...] = ()

  @property
  def ok(self) -> bool:
    return not any(i.severity == "error" for i in self.issues)

  def errors(self) -> list[Issue]:
    return [i for i in self.issues if i.severity == "error"]

  def raise_for_errors(self) -> None:
    if not self.ok:
      lines = "\n".join(f"  {i.path}: [{i.code}] {i.message}" for i in self.errors())
      raise ConfigError(f"configuration rejected:\n{lines}", self)


class ConfigError(ValueError):
  def __init__(self, message: str, report: ValidationReport | None = None):
    super().__init__(message)
    self.report = report


@dataclass(frozen=True)
class NodeProfile:
  name: str
  physical_cores: int
  numa_domains: tuple[tuple[int, ...], ...]
  memory_gib: int
  gpus: int = 0
  gpu_type: str | None = None


@dataclass(frozen=True)
class ClusterInventory:
  """Node pools, physical cores/NUMA maps, memory, GPU types, image/ABI hashes."""

  nodes: tuple[NodeProfile, ...] = ()
  image_abi_hashes: Mapping[str, str] = field(default_factory=dict)
  production: bool = False


@dataclass(frozen=True)
class ConfigDiff:
  changes: tuple[tuple[str, Classification, Any, Any], ...]

  @property
  def classification(self) -> Classification | None:
    order = {"hot": 0, "boundary": 1, "new_run": 2, "forbidden": 3}
    if not self.changes:
      return None
    return max((c[1] for c in self.changes), key=order.__getitem__)

  def paths(self, kind: Classification) -> list[str]:
    return [p for p, c, _, _ in self.changes if c == kind]


# ------------------------------------------------------------------------ loading


def _set_path(d: dict[str, Any], dotted: str, value: Any) -> None:
  parts = dotted.split(".")
  cur: Any = d
  for i, p in enumerate(parts[:-1]):
    if isinstance(cur, list):
      cur = cur[int(p)]
      continue
    if p not in cur or cur[p] is None:
      cur[p] = {}
    cur = cur[p]
    if not isinstance(cur, dict | list):
      raise ConfigError(f"override {dotted}: {'.'.join(parts[: i + 1])} is not an object")
  last = parts[-1]
  if isinstance(cur, list):
    cur[int(last)] = value
  else:
    cur[last] = value


def _jsonable(x: Any) -> Any:
  if isinstance(x, Mapping):
    return {str(k): _jsonable(v) for k, v in x.items()}
  if isinstance(x, list | tuple):
    return [_jsonable(v) for v in x]
  if isinstance(x, float) and not math.isfinite(x):
    raise ConfigError("NaN/Inf are not valid configuration values")
  return x


class _Loader(yaml.SafeLoader):
  """SafeLoader with YAML 1.2 floats: PyYAML (YAML 1.1) reads ``1e-6`` as a *string*,
  which strict validation then rejects (found loading a rendered run-config)."""


_Loader.add_implicit_resolver(
  "tag:yaml.org,2002:float",
  re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                 |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                 |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                 |[-+]?\.(?:inf|Inf|INF)
                 |\.(?:nan|NaN|NAN))$""", re.X),
  list("-+0123456789."))


class ConfigurationSystem:
  """Typed configuration with complete defaults, validation, hashing and diffs.

  YAML is an input format only: no interpolation, anchors are resolved by the YAML
  parser, and overrides are explicit dotted paths. Environment variables never change
  semantic configuration.
  """

  def load(self, path: str | Path | None, overrides: Mapping[str, object] | None = None) -> RunConfig:
    raw: dict[str, Any] = {}
    if path is not None:
      text = Path(path).read_text()
      loaded = json.loads(text) if str(path).endswith(".json") else yaml.load(text, Loader=_Loader)  # noqa: S506
      if loaded is None:
        loaded = {}
      if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
      raw = loaded
    raw = copy.deepcopy(raw)
    for k, v in (overrides or {}).items():
      _set_path(raw, k, v)
    return self.parse(raw)

  def parse(self, raw: Mapping[str, Any]) -> RunConfig:
    try:
      return RunConfig.model_validate_json(json.dumps(_jsonable(raw), allow_nan=False))
    except ValidationError as exc:
      issues = tuple(
        Issue(".".join(str(p) for p in e["loc"]) or "<root>", "SCHEMA", e["msg"]) for e in exc.errors()
      )
      raise ConfigError(str(exc), ValidationReport(issues)) from exc

  # ---------------------------------------------------------------- validation

  def validate(
    self,
    config: RunConfig,
    inventory: ClusterInventory | None = None,
    *,
    submission: bool = True,
  ) -> ValidationReport:
    """All independent cross-field errors in one report (design §10.3)."""
    inv = inventory or ClusterInventory()
    out: list[Issue] = []

    def err(path: str, code: str, msg: str, severity: Severity = "error") -> None:
      out.append(Issue(path, code, msg, severity))

    w, c, f, fl = config.worker, config.cluster, config.failure, config.flow
    production = inv.production
    if production:
      # An unqualified MJBatch admitted by RSB_MJBATCH_CANDIDATE exists only for the
      # upgrade workflow; it never runs production work.
      from raysimbatch.compat.mjbatch_adapter import check_environment

      if check_environment().candidate_mode:
        err("<environment>", "CANDIDATE_BINARY", "unqualified MJBatch candidate binary in a production run")
    # CPU accounting and scalar lane.
    if w.threads != w.num_cpus:
      err("worker.threads", "CPU_MISMATCH", "physics threads must equal reserved Ray CPUs")
    if w.envs < w.threads:
      err("worker.envs", "SLOTS_LT_THREADS", "slots must be >= threads")
    if w.native_backend == "scalar" and not (w.envs == w.threads == w.num_cpus == 1):
      err("worker.native_backend", "SCALAR_SHAPE", "scalar backend requires envs=threads=num_cpus=1")
    if w.model_shards != 1:
      err("worker.model_shards", "V1_SINGLE_SHARD", "model_shards must equal 1 in version 1")
    if not w.slice_min <= w.slice_size <= w.slice_max:
      err("worker.slice_size", "SLICE_BOUNDS", "slice_min <= slice_size <= slice_max required")
    if w.slice_size > w.envs:
      err("worker.slice_size", "SLICE_GT_ENVS", "slice_size must be <= envs")
    if w.engine_threads != 0:
      err("worker.engine_threads", "NESTED_POOL", "engine-internal threads must stay disabled (0)")
    if w.smt:
      err("worker.smt", "SMT_UNQUALIFIED", "SMT requires a qualification artifact", "warning")
    for name, v in (("omp_threads", w.omp_threads), ("blas_threads", w.blas_threads), ("torch_threads", w.torch_threads)):
      if v != 1:
        err(f"worker.{name}", "THREAD_CAP", "BLAS/OMP/Torch thread caps must be 1")
    if w.max_pending_calls > 8:
      err("worker.max_pending_calls", "PENDING_BOUND", "coordinators hold at most eight pending RPCs per actor")
    if production and w.affinity == "none":
      err("worker.affinity", "AFFINITY_REQUIRED", "production rejects affinity=none")
    # Per-node budgets.
    actors_per_node = c.simulation_cpus // w.num_cpus
    if c.simulation_cpus > c.ray_cpus:
      err("cluster.simulation_cpus", "NODE_BUDGET", "simulation_cpus must be <= ray_cpus")
    if c.ray_cpus > c.cpu_node_cores:
      err("cluster.ray_cpus", "NODE_BUDGET", "ray_cpus must be <= cpu_node_cores")
    if actors_per_node < 1:
      err("cluster.simulation_cpus", "NODE_BUDGET", "node cannot host one worker")
    if actors_per_node * w.memory_gib > c.cpu_node_memory_gib - c.object_store_gib:
      err("worker.memory_gib", "MEMORY_BUDGET", "worker memory requests exceed node memory minus object store")
    if c.shm_gib < c.object_store_gib:
      err("cluster.shm_gib", "SHM_LT_STORE", "shared-memory mount must hold the object store")
    for node in inv.nodes:
      domains = node.numa_domains or (tuple(range(node.physical_cores)),)
      fit = sum(len(d) // w.num_cpus for d in domains)
      if fit < actors_per_node:
        err("cluster.simulation_cpus", "NUMA_FIT", f"node {node.name}: {fit} NUMA-local actors < {actors_per_node}")
    if c.coordinator_fanout > 8:
      err("cluster.coordinator_fanout", "FANOUT", "coordinator fanout is resolved at eight actors")
    if c.gcs_ft == "redis_qualified" and (c.gcs_qualification.empty or not c.redis_secret):
      err("cluster.gcs_ft", "GCS_FT_UNQUALIFIED", "redis_qualified requires gcs_qualification and redis_secret")
    for k in c.runtime_env_vars:
      if k in THREAD_CAP_VARS:
        err(f"cluster.runtime_env_vars.{k}", "THREAD_CAP_OVERRIDE", "runtime_env_vars cannot override thread caps")
    if c.min_cpu_nodes > c.max_cpu_nodes or c.min_gpu_nodes > c.max_gpu_nodes:
      err("cluster", "AUTOSCALE_BOUNDS", "min nodes must be <= max nodes")
    # Found in kind qualification: an unbounded plan starts actors that can never be
    # scheduled and only fails at the worker initialize deadline.
    cap = c.max_cpu_nodes * max(actors_per_node, 0)
    # Evaluation runs one runtime per model x policy, each with at least one actor.
    floor = len(config.models) * max(1, len(config.policies)) if config.mode != "rollout" else len(config.models)
    need = max(math.ceil(config.target_envs / w.envs) if config.mode == "rollout" else config.evaluation.wave_actors,
               floor)
    if need > cap:
      what = "target_envs" if config.mode == "rollout" else "evaluation.wave_actors"
      err(what, "CAPACITY", f"plan needs {need} simulation actors but the cluster maximum is {cap} "
          f"({c.max_cpu_nodes} nodes x {actors_per_node} actors)")
    # Failure and lease timing.
    if f.max_task_retries != 0:
      err("failure.max_task_retries", "MUTATION_RETRY", "mutation calls must never be retried (0)")
    if f.lease_renew_s >= f.lease_s:
      err("failure.lease_renew_s", "LEASE_RENEW", "lease renewal must be shorter than the lease")
    if config.evaluation.lease_renew_s >= config.evaluation.lease_s:
      err("evaluation.lease_renew_s", "LEASE_RENEW", "lease renewal must be shorter than the lease")
    if not f.native_min_deadline_s <= f.native_max_deadline_s:
      err("failure.native_min_deadline_s", "DEADLINE_BOUNDS", "native min deadline must be <= max deadline")
    if f.heartbeat_s >= f.suspect_s:
      err("failure.heartbeat_s", "HEARTBEAT", "heartbeat must be shorter than the suspect interval")
    # Flow thresholds.
    if not (fl.resume_fraction <= fl.admission_store_fraction <= fl.throttle_fraction
            < fl.stop_episode_fraction < fl.pause_fraction <= 1.0):
      err("flow", "FLOW_ORDER", "resume <= admission <= throttle < stop_episode < pause <= 1 required")
    if fl.object_target_bytes > fl.object_max_bytes:
      err("flow.object_target_bytes", "OBJECT_SIZE", "target object size must be <= max")
    if fl.spill_hot_objects_allowed:
      err("flow.spill_hot_objects_allowed", "HOT_SPILL", "spilling active closed-loop frames is an SLO violation", "warning")
    # Inference.
    inf = config.inference
    if inf.queue_rows < inf.max_batch_rows:
      err("inference.queue_rows", "QUEUE_BOUNDS", "queue_rows must be >= max_batch_rows")
    if production and inf.placement in ("central", "auto") and inf.device != "cuda":
      err("inference.device", "DEVICE", "production central inference runs on CUDA")
    # Rollout.
    ro = config.rollout
    if config.mode == "rollout" and ro.fragment_steps > 32:
      err("rollout.fragment_steps", "FRAGMENT", "fragments hold at most 32 transitions per env", "warning")
    # Models.
    names = [m.name for m in config.models]
    if len(set(names)) != len(names):
      err("models", "DUPLICATE_GROUP", "model group names must be unique")
    for i, m in enumerate(config.models):
      p = f"models.{i}"
      if m.sleep and w.native_backend != "scalar":
        err(f"{p}.sleep", "SLEEP_UNSUPPORTED", "MJBatch rejects sleep; use the certified scalar lane")
      if m.plugins and w.native_backend != "scalar":
        err(f"{p}.plugins", "PLUGIN_LANE", "plugins require the scalar lane plus replay certification")
      if m.max_control_steps < 1:
        err(f"{p}.max_control_steps", "TIME_LIMIT", "must be positive")
      for j, r in enumerate(m.randomization):
        if r.field in ("body_mass", "body_inertia") and r.operation == "multiply" and r.distribution != "categorical":
          if r.low <= 0 and r.distribution != "fixed":
            err(f"{p}.randomization.{j}.low", "PHYSICAL_BOUNDS", "mass/inertia multipliers must be > 0")
          if r.distribution == "fixed" and r.mean <= 0:
            err(f"{p}.randomization.{j}.mean", "PHYSICAL_BOUNDS", "mass/inertia multipliers must be > 0")
      for code in m.health.fatal_warning_codes:
        if not _warning_known(code):
          err(f"{p}.health.fatal_warning_codes", "UNKNOWN_WARNING", f"engine has no warning {code}")
      if submission:
        for fld in ("mjcf", "task_kernel", "adapter"):
          if getattr(m, fld).empty and not (fld == "mjcf" and not m.mjb.empty):
            err(f"{p}.{fld}", "MISSING_ARTIFACT", f"model group requires {fld}")
    # Curriculum.
    cw = config.curriculum.weights
    if cw:
      unknown = set(cw) - set(names)
      if unknown:
        err("curriculum.weights", "UNKNOWN_GROUP", f"weights name unknown groups {sorted(unknown)}")
      if sum(cw.values()) <= 0:
        err("curriculum.weights", "WEIGHT_SUM", "weights must have a positive sum")
    # Policies.
    for i, pol in enumerate(config.policies):
      if w.slice_size not in pol.certified_batch_sizes:
        err(f"policies.{i}.certified_batch_sizes", "BATCH_SIZE", "slice_size must be a certified batch size")
      if pol.normalization == "external" and pol.normalizer.empty:
        err(f"policies.{i}.normalizer", "MISSING_ARTIFACT", "external normalization requires a normalizer")
      if submission and pol.bundle.empty and pol.executable.empty:
        err(f"policies.{i}.bundle", "MISSING_ARTIFACT", "policy requires a bundle or executable")
    # Evaluation.
    ev = config.evaluation
    if len(set(ev.seeds)) != len(ev.seeds):
      err("evaluation.seeds", "DUPLICATE_SEED", "seeds must be unique")
    for t in ev.task_weights:
      if config.tasks and t not in {x.name for x in config.tasks}:
        err("evaluation.task_weights", "UNKNOWN_TASK", f"weight for unknown task {t}")
    # Sim2Sim.
    if config.mode == "sim2sim" and submission:
      s2 = config.sim2sim
      if s2.bundle.empty and not config.exploratory:
        err("sim2sim.bundle", "MISSING_ARTIFACT", "sim2sim requires a validated transfer bundle")
      if not s2.static_gates or not s2.dynamics_gates:
        if not config.exploratory:
          err("sim2sim", "GATES_DISABLED", "disabling gates requires an exploratory run")
    # Submission requirements.
    if submission:
      if not config.models:
        err("models", "REQUIRED", "at least one model group on submission")
      if not config.policies:
        err("policies", "REQUIRED", "at least one policy on submission")
      if not config.artifact_root:
        err("artifact_root", "REQUIRED", "durable artifact root required for submit")
      if not config.database_secret:
        err("database_secret", "REQUIRED", "database secret reference required")
      if config.mode == "eval" and not config.tasks:
        err("tasks", "REQUIRED", "evaluation requires a task suite")
    if production:
      if config.release_lock.empty:
        err("release_lock", "REQUIRED", "production requires a signed release lock")
      if not c.image_digest:
        err("cluster.image_digest", "REQUIRED", "production requires a signed image digest")
      if w.native_extension.empty:
        err("worker.native_extension", "REQUIRED", "production requires the signed native extension ABI")
      if c.platform_inventory.empty:
        err("cluster.platform_inventory", "REQUIRED", "production deployment requires a platform inventory")
      if config.artifact_root and not config.artifact_root.startswith("s3://"):
        err("artifact_root", "DURABILITY", "production artifacts live in S3-compatible storage")
      if config.exploratory:
        err("exploratory", "PRODUCTION", "exploratory runs are never production releases")
    return ValidationReport(tuple(out))

  # ------------------------------------------------------------------- hashing

  def resolved(self, config: RunConfig) -> dict[str, Any]:
    """The canonical resolved form: defaults applied, weights normalized."""
    d = config.model_dump(mode="json")
    cw = d["curriculum"]["weights"]
    groups = [m["name"] for m in d["models"]]
    if not cw and groups:
      cw = {g: 1.0 for g in groups}
    total = sum(cw.values())
    d["curriculum"]["weights"] = {k: v / total for k, v in sorted(cw.items())} if total > 0 else {}
    tw = d["evaluation"]["task_weights"]
    tasks = [t["name"] for t in d["tasks"]]
    if not tw and tasks:
      tw = {t: 1.0 for t in tasks}
    total = sum(tw.values())
    d["evaluation"]["task_weights"] = {k: v / total for k, v in sorted(tw.items())} if total > 0 else {}
    return d

  def canonical_json(self, config: RunConfig) -> str:
    return json.dumps(self.resolved(config), sort_keys=True, separators=(",", ":"), allow_nan=False)

  def canonical_hash(self, config: RunConfig) -> str:
    return hashlib.sha256(self.canonical_json(config).encode()).hexdigest()

  # ---------------------------------------------------------------------- diff

  def diff(self, previous: RunConfig, proposed: RunConfig) -> ConfigDiff:
    a, b = _flatten(self.resolved(previous)), _flatten(self.resolved(proposed))
    changes = []
    for path in sorted(set(a) | set(b)):
      va, vb = a.get(path, _MISSING), b.get(path, _MISSING)
      if va != vb:
        changes.append((path, _classify(path, va, vb), _undef(va), _undef(vb)))
    return ConfigDiff(tuple(changes))

  def json_schema(self) -> dict[str, Any]:
    return RunConfig.model_json_schema()


_MISSING = object()


def _undef(v: Any) -> Any:
  return None if v is _MISSING else v


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
  out: dict[str, Any] = {}
  if isinstance(d, dict):
    if not d and prefix:
      out[prefix] = {}
    for k, v in d.items():
      out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
  elif isinstance(d, list):
    if not d and prefix:
      out[prefix] = []
    for i, v in enumerate(d):
      out.update(_flatten(v, f"{prefix}.{i}"))
  else:
    out[prefix] = d
  return out


_HOT = ("telemetry.log_level", "telemetry.trace_fraction", "telemetry.trace_errors", "telemetry.otel_endpoint")
_BOUNDARY = (
  "curriculum.",
  "target_envs",
  "rollout.target_transitions",
  "flow.",
  "inference.gpu_replicas",
  "inference.max_resident_versions",
  "cluster.min_cpu_nodes",
  "cluster.max_cpu_nodes",
  "cluster.min_gpu_nodes",
  "cluster.max_gpu_nodes",
)
# Loosening any of these inside an active epoch is forbidden (design §10.3).
_LOOSEN_UP = ("sim2sim.parity.", "models.*.health.max_", "models.*.health.quaternion_norm_tolerance")


def _match(path: str, pattern: str) -> bool:
  pp, qq = path.split("."), pattern.rstrip(".").split(".")
  if pattern.endswith("."):
    return len(pp) >= len(qq) and all(q in ("*", p) for p, q in zip(pp, qq, strict=False))
  return len(pp) == len(qq) and all(q in ("*", p) for p, q in zip(pp, qq, strict=True)) or (
    "*" not in pattern and path.startswith(pattern)
  )


def _number(x: Any) -> float | None:
  """``float(x)``, or None when x is not numeric (missing or non-numeric health values)."""
  try:
    return float(x)
  except (TypeError, ValueError):
    return None


def _classify(path: str, old: Any, new: Any) -> Classification:
  if path == "failure.max_task_retries" and new not in (0, _MISSING):
    return "forbidden"
  if path in ("run_id",) and old not in (None, _MISSING):
    return "forbidden"
  if _match(path, "models.*.health.check_each_substep") and old is True and new is False:
    return "forbidden"
  if _match(path, "models.*.health.fatal_warning_codes.") and new is _MISSING:
    return "forbidden"
  if path in ("sim2sim.static_gates", "sim2sim.dynamics_gates") and old is True and new is False:
    return "forbidden"
  for pat in _LOOSEN_UP:
    if _match(path, pat if pat.endswith(".") else pat + ".") or path.startswith(pat.replace("*.", "")):
      if isinstance(old, int | float) and isinstance(new, int | float) and new > old:
        return "forbidden"
  if path.startswith("models.") and ".health." in path and path.split(".")[-1].startswith("max_"):
    n, o = _number(new), _number(old)
    if n is not None and o is not None and n > o:
      return "forbidden"
  if any(path == h or path.startswith(h + ".") for h in _HOT):
    return "hot"
  if _match(path, "policies.*.version"):
    return "boundary"
  for b in _BOUNDARY:
    if b.endswith(".") and path.startswith(b) or path == b:
      if b == "flow." and isinstance(old, int | float) and isinstance(new, int | float) and new > old:
        return "new_run"  # only queue *reductions* are boundary changes
      return "boundary"
  return "new_run"


_WARNINGS: frozenset[str] | None = None


def _warning_known(code: str) -> bool:
  global _WARNINGS
  if _WARNINGS is None:
    import mujoco

    _WARNINGS = frozenset(
      k.removeprefix("mjWARN_") for k in mujoco.mjtWarning.__members__ if k.startswith("mjWARN_")
    )
  return code in _WARNINGS
