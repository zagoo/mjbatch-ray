# SPDX-License-Identifier: Apache-2.0
"""KubeRay rendering (design §11.4). Each assertion encodes a failure observed while
deploying to the kind qualification cluster, or a production admission rule."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import jsonschema
import pytest
import yaml

from raysimbatch.config import ConfigurationSystem
from raysimbatch.deployment.render import PlatformInventory, release_label, render

ROOT = Path(__file__).resolve().parents[2]
CRD = ROOT / "deploy/crds/kuberay-1.7.0-ray.io_rayclusters.yaml"


def objects(tag="blue"):
  cfg = ConfigurationSystem().load(ROOT / "deploy/kind/qual-config.yaml")
  inv = yaml.safe_load((ROOT / "deploy/kind/inventory.yaml").read_text().replace("${TAG}", tag))
  return cfg, render(cfg, PlatformInventory.from_dict(inv), tenant="qual", cluster_name=f"rsb-{tag}",
                     include_gpu=False)


def raycluster():
  return next(o for o in objects()[1] if o["kind"] == "RayCluster")


def _crd_schema():
  crd = yaml.safe_load(CRD.read_text())
  return next(v for v in crd["spec"]["versions"] if v["name"] == "v1")["schema"]["openAPIV3Schema"]


def _unknown_fields(obj, schema, path="") -> list[str]:
  """Fields the API server would prune (structural schema without preserve-unknown)."""
  out = []
  if isinstance(obj, dict) and "properties" in schema and not schema.get("x-kubernetes-preserve-unknown-fields"):
    for k, v in obj.items():
      if k not in schema["properties"]:
        out.append(f"{path}.{k}")
      else:
        out += _unknown_fields(v, schema["properties"][k], f"{path}.{k}")
  elif isinstance(obj, dict) and isinstance(schema.get("additionalProperties"), dict):
    for k, v in obj.items():
      out += _unknown_fields(v, schema["additionalProperties"], f"{path}.{k}")
  elif isinstance(obj, list) and "items" in schema:
    for i, v in enumerate(obj):
      out += _unknown_fields(v, schema["items"], f"{path}[{i}]")
  return out


def test_raycluster_matches_kuberay_1_7_0_crd():
  rc, schema = raycluster(), _crd_schema()
  jsonschema.validate(rc, schema)
  assert _unknown_fields(rc, schema) == []


def test_ray_start_json_survives_kuberay_bash_splicing():
  """KubeRay 1.7.0 builds `bash -c "ray start --k=v ..."` with values unquoted; JSON
  values must arrive at ray start intact (AM-14)."""
  bash = shutil.which("bash")
  if bash is None:
    pytest.skip("bash not available")
  rc = raycluster()
  params = [("system-config", rc["spec"]["headGroupSpec"]["rayStartParams"]["system-config"])]
  params += [("resources", g["rayStartParams"]["resources"]) for g in rc["spec"]["workerGroupSpecs"]]
  for key, value in params:
    argv = subprocess.run([bash, "-c", f"printf '%s\\n' --{key}={value}"], capture_output=True, text=True,
                          check=True).stdout.splitlines()
    assert len(argv) == 1, argv  # one argument, not split on spaces
    json.loads(argv[0].split("=", 1)[1])


def test_system_config_is_head_only():
  rc = raycluster()
  assert "system-config" in rc["spec"]["headGroupSpec"]["rayStartParams"]
  assert all("system-config" not in g["rayStartParams"] for g in rc["spec"]["workerGroupSpecs"])


def test_pod_identity_and_secret_access():
  rc = raycluster()
  head = rc["spec"]["headGroupSpec"]["template"]["spec"]
  # The in-tree autoscaler sidecar needs the KubeRay-managed service account token.
  assert "serviceAccountName" not in head and "automountServiceAccountToken" not in head
  for g in rc["spec"]["workerGroupSpecs"]:
    spec = g["template"]["spec"]
    assert spec["serviceAccountName"] == "raysimbatch-worker" or spec["serviceAccountName"]
    assert spec["automountServiceAccountToken"] is False
  for spec in [head] + [g["template"]["spec"] for g in rc["spec"]["workerGroupSpecs"]]:
    sc = spec["securityContext"]
    assert sc["runAsNonRoot"] and sc["runAsUser"] == 1000 and sc["fsGroup"] == 1000
    secret = next(v for v in spec["volumes"] if "secret" in v)["secret"]
    assert secret["defaultMode"] & 0o040, "uid 1000 reads the secret through fsGroup"


def test_labels_are_valid_for_any_image_reference():
  label = re.compile(r"^([A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?)?$")
  for img in ("raysimbatch-cpu:blue", "registry.local:5000/rsb/cpu:1.2.3",
              "registry.local:5000/rsb/cpu@sha256:" + "a" * 64):
    v = release_label(img)
    assert label.match(v) and len(v) <= 63, v


def test_production_requires_digests_selectors_and_secrets():
  cfg = ConfigurationSystem().load(ROOT / "deploy/kind/qual-config.yaml")
  inv = PlatformInventory(head_image="rsb:tag", cpu_image="rsb:tag")
  with pytest.raises(ValueError) as e:
    render(cfg, inv, production=True)
  msg = str(e.value)
  assert "pinned by digest" in msg and "node selectors" in msg and "database secret" in msg


def test_head_runs_the_cluster_janitor_and_every_container_knows_its_cluster():
  rc = raycluster()
  head = rc["spec"]["headGroupSpec"]["template"]["spec"]["containers"]
  janitor = next(c for c in head if c["name"] == "janitor")
  assert janitor["command"][:2] == ["raysimbatch", "janitor"] and "--loop" in janitor["command"]
  pods = [rc["spec"]["headGroupSpec"]["template"]] + [g["template"] for g in rc["spec"]["workerGroupSpecs"]]
  for p in pods:
    for c in p["spec"]["containers"]:
      env = {e["name"]: e.get("value") for e in c["env"]}
      assert env["RSB_CLUSTER_NAME"] == rc["metadata"]["name"], c["name"]


def test_simulation_pods_declare_their_exclusive_cpu_request_for_admission():
  """The affinity agent compares the pod's real CPU set with this request and refuses the
  node on a mismatch (design B.3 [VERIFY]); the value must be the container's own request."""
  rc = raycluster()
  sims = [g for g in rc["spec"]["workerGroupSpecs"] if g["template"]["spec"]["containers"][0]["name"] == "ray-worker"]
  assert sims
  for g in sims:
    c = g["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env["RSB_POD_CPUS"] == c["resources"]["requests"]["cpu"] == c["resources"]["limits"]["cpu"]
    assert env["RSB_POD_CPUS"].isdigit()  # whole CPUs: the only way to get an exclusive set
