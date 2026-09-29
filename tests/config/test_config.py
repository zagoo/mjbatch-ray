# SPDX-License-Identifier: Apache-2.0
"""Configuration system (design §10): strict parsing, one report with every independent
error, capacity bounds, canonical hashing and change classification."""

from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest
import yaml

from raysimbatch.config import ConfigurationSystem
from raysimbatch.config.system import ConfigError

ROOT = Path(__file__).resolve().parents[2]
CS = ConfigurationSystem()


def codes(cfg, **kw):
  return {(i.path, i.code) for i in CS.validate(cfg, **kw).errors()}


def test_unknown_and_mistyped_keys_are_rejected_with_paths():
  with pytest.raises(ConfigError) as e:
    CS.parse({"worker": {"envs": "many", "thraeds": 2}})
  paths = {i.path for i in e.value.report.issues}
  assert "worker.envs" in paths and "worker.thraeds" in paths  # both reported, not just the first


def test_independent_errors_are_reported_together():
  cfg = CS.parse({"cluster": {"simulation_cpus": 200, "ray_cpus": 100, "cpu_node_cores": 128},
                  "failure": {"max_task_retries": 1}})
  got = codes(cfg)
  assert ("cluster.simulation_cpus", "NODE_BUDGET") in got
  assert ("failure.max_task_retries", "MUTATION_RETRY") in got



def test_nodes_whose_numa_domains_cannot_host_the_plan_are_rejected():
  """Design B.3 [VERIFY] (profile side): a node's actors must fit NUMA-locally. 112
  simulation CPUs at 4 per actor need 28 actors; two 64-core domains hold 32, but a node
  with one 64-core domain and one 40-core domain holds only 26 and must be refused."""
  from raysimbatch.config.system import ClusterInventory, NodeProfile

  cfg = CS.parse({})
  fits = NodeProfile("fits", 128, (tuple(range(64)), tuple(range(64, 128))), 512)
  short = NodeProfile("short", 104, (tuple(range(64)), tuple(range(64, 104))), 512)
  assert ("cluster.simulation_cpus", "NUMA_FIT") not in codes(cfg, inventory=ClusterInventory(nodes=(fits,)))
  issues = [i for i in CS.validate(cfg, ClusterInventory(nodes=(fits, short))).errors() if i.code == "NUMA_FIT"]
  assert len(issues) == 1 and "short" in issues[0].message

@pytest.mark.parametrize("mode,field,raw", [
  # 2 nodes x (4 // 2) actors = 4 actors max; 10k envs / 8 per actor needs 1250.
  ("rollout", "target_envs", {"target_envs": 10_000}),
  ("eval", "evaluation.wave_actors", {"evaluation": {"wave_actors": 5}}),
])
def test_plan_beyond_cluster_maximum_is_rejected(mode, field, raw):
  """Found in kind qualification: a rollout planned 1250 actors on a 4-actor cluster
  and only failed at the 120 s worker-initialize deadline. It must fail validation."""
  base = {"mode": mode, "cluster": {"max_cpu_nodes": 2, "simulation_cpus": 4, "ray_cpus": 4, "cpu_node_cores": 8,
                                    "cpu_node_memory_gib": 5, "object_store_gib": 1, "shm_gib": 1},
          "worker": {"envs": 8, "num_cpus": 2, "memory_gib": 1}}
  cfg = CS.parse({**base, **raw})
  assert (field, "CAPACITY") in codes(cfg)
  fits = CS.parse({**base, **({"target_envs": 32} if mode == "rollout" else {"evaluation": {"wave_actors": 4}})})
  assert not any(c == "CAPACITY" for _, c in codes(fits))


def test_plan_capacity_enforces_the_same_bound():
  from raysimbatch.control.placement import CapacityError, plan_capacity

  cfg = CS.parse({"mode": "rollout", "target_envs": 10_000,
                  "cluster": {"max_cpu_nodes": 2, "simulation_cpus": 4, "ray_cpus": 4},
                  "worker": {"envs": 8, "num_cpus": 2}})
  with pytest.raises(CapacityError, match="1250 simulation actors .* maximum is 4"):
    plan_capacity(cfg, "r" * 32, {"g": 1.0})
  assert len(plan_capacity(cfg, "r" * 32, {"g": 1.0}, actors=4).assignments) == 4


def test_canonical_hash_ignores_spelling_but_not_semantics():
  a = CS.parse({"curriculum": {"weights": {"a": 1.0, "b": 1.0}}, "target_envs": 64})
  b = CS.parse({"target_envs": 64, "curriculum": {"weights": {"b": 2.0, "a": 2.0}}})  # same normalized weights
  c = CS.parse({"curriculum": {"weights": {"a": 1.0, "b": 3.0}}, "target_envs": 64})
  assert CS.canonical_hash(a) == CS.canonical_hash(b) != CS.canonical_hash(c)


@pytest.mark.parametrize("change,kind", [
  ({"telemetry": {"log_level": "DEBUG"}}, "hot"),
  ({"target_envs": 5000}, "boundary"),
  ({"worker": {"envs": 16}}, "new_run"),
  ({"failure": {"max_task_retries": 1}}, "forbidden"),
])
def test_diff_classification(change, kind):
  prev = CS.parse({})
  assert CS.diff(prev, CS.parse(change)).classification == kind


def test_published_json_schema_accepts_shipped_configs():
  schema = CS.json_schema()
  for p in [ROOT / "deploy/kind/qual-config.yaml"]:
    jsonschema.validate(yaml.safe_load(p.read_text()), schema)
    rep = CS.validate(CS.load(p), submission=False)  # a base: the smoke test adds models/tasks
    assert rep.ok, rep.errors()


def test_production_rejects_an_unqualified_candidate_binary(monkeypatch):
  """RSB_MJBATCH_CANDIDATE admits an unqualified MJBatch only for the upgrade workflow."""
  from raysimbatch.compat import mjbatch_adapter as M
  from raysimbatch.config.system import ClusterInventory

  cfg = CS.parse({})
  real = M.check_environment()
  monkeypatch.setattr(M, "check_environment", lambda force=False: M.CompatReport(**{**real.__dict__,
                                                                                    "candidate_mode": True}))
  assert ("<environment>", "CANDIDATE_BINARY") in codes(cfg, inventory=ClusterInventory(production=True))
  assert ("<environment>", "CANDIDATE_BINARY") not in codes(cfg)  # development validation allows it


def test_exponent_floats_load_from_yaml_and_rendered_json(tmp_path):
  """Found in kind: `raysimbatch cancel --config run-config.json` failed because PyYAML
  (YAML 1.1) reads 1e-06 as a string; the system could not load its own rendered config."""
  y = tmp_path / "c.yaml"
  y.write_text("models: []\nevaluation: {seeds: [0]}\ntelemetry: {trace_fraction: 1e-3}\n")
  assert CS.load(y).telemetry.trace_fraction == 1e-3
  j = tmp_path / "run-config.json"
  cfg = CS.parse({"telemetry": {"trace_fraction": 1e-6}})
  j.write_text(cfg.model_dump_json(indent=2))  # what the renderer puts in the ConfigMap
  assert "1e-6" in j.read_text() and CS.load(j) == cfg
