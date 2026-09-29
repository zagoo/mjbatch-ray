# SPDX-License-Identifier: Apache-2.0
"""KubeRay manifest rendering (design §11.4, Appendix B.3).

The renderer is a product component, not a hand-edited template: it reads the closed
RunConfig and a signed platform inventory and emits Namespace, ServiceAccount,
NetworkPolicy, ResourceQuota, LimitRange, the resolved-config ConfigMap and the
RayCluster (head with Ray CPU capacity zero; sim-general, sim-contact and GPU worker
groups). Pods use requests == limits with integer CPUs (Guaranteed QoS, which the
kubelet static CPU Manager policy turns into exclusive cores), thread caps, read-only
secret mounts, spill storage and a preStop drain. Secrets are referenced, never
embedded. An empty inventory blocks production rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from raysimbatch.config.models import RunConfig

THREAD_ENV = [{"name": k, "value": "1"} for k in
              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")]


@dataclass
class Pool:
  node_selector: dict[str, str] = field(default_factory=dict)
  tolerations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PlatformInventory:
  """Installation inputs: concrete selectors, tolerations, registry and image digests,
  service DNS names, spill volume class and secret references."""

  name: str = "local"
  head_image: str = ""
  cpu_image: str = ""
  gpu_image: str = ""
  image_pull_policy: str = "IfNotPresent"
  pools: dict[str, Pool] = field(default_factory=dict)  # head | sim_general | sim_contact | gpu
  spill_volume: dict[str, Any] = field(default_factory=lambda: {"hostPath": {"path": "/var/lib/raysimbatch-spill",
                                                                             "type": "DirectoryOrCreate"}})
  database_secret: str | None = None  # k8s:<name>/<key>
  s3_secret: str | None = None  # k8s:<name> with AWS_* keys
  s3_endpoint: str | None = None
  extra_env: dict[str, str] = field(default_factory=dict)
  scale: dict[str, Any] = field(default_factory=dict)  # qualification-only resource overrides

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> PlatformInventory:
    d = dict(d)
    d["pools"] = {k: Pool(**v) for k, v in d.get("pools", {}).items()}
    return cls(**d)

  def check(self, production: bool) -> list[str]:
    problems = []
    if not self.cpu_image or not self.head_image:
      problems.append("head_image and cpu_image are required")
    if production:
      for img in (self.head_image, self.cpu_image, self.gpu_image):
        if img and "@sha256:" not in img:
          problems.append(f"production images must be pinned by digest: {img}")
      for p in ("head", "sim_general"):
        if p not in self.pools or not self.pools[p].node_selector:
          problems.append(f"production inventory needs node selectors for pool {p}")
      if not self.database_secret:
        problems.append("database secret reference is required")
    return problems


def _secret_volume(ref: str | None, name: str) -> tuple[list[dict], list[dict]]:
  if not ref or not ref.startswith("k8s:"):
    return [], []
  secret = ref[4:].split("/", 1)[0]
  vol = {"name": name, "secret": {"secretName": secret, "defaultMode": 0o440}}  # group-readable via fsGroup
  mount = {"name": name, "mountPath": f"/var/run/secrets/raysimbatch/{secret}", "readOnly": True}
  return [vol], [mount]


def _container(name: str, image: str, cpu: str, mem: str, inv: PlatformInventory, env_extra=(), gpus: int = 0,
               mounts=(), pre_stop: bool = False, cluster: str = "") -> dict[str, Any]:
  env = list(THREAD_ENV) + [{"name": k, "value": v} for k, v in sorted(inv.extra_env.items())] + list(env_extra)
  env.append({"name": "RSB_CLUSTER_NAME", "value": cluster})  # the janitor resumes only its own runs
  if inv.s3_endpoint:
    env.append({"name": "RSB_S3_ENDPOINT", "value": inv.s3_endpoint})
  env_from = []
  if inv.s3_secret and inv.s3_secret.startswith("k8s:"):
    env_from.append({"secretRef": {"name": inv.s3_secret[4:]}})
  res: dict[str, Any] = {"requests": {"cpu": cpu, "memory": mem}, "limits": {"cpu": cpu, "memory": mem}}
  if gpus:
    res["requests"]["nvidia.com/gpu"] = str(gpus)
    res["limits"]["nvidia.com/gpu"] = str(gpus)
  c: dict[str, Any] = {"name": name, "image": image, "imagePullPolicy": inv.image_pull_policy, "resources": res,
                       "env": env, "volumeMounts": list(mounts),
                       "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": False,
                                           "capabilities": {"drop": ["ALL"], "add": ["SYS_NICE"]}}}
  if env_from:
    c["envFrom"] = env_from
  if pre_stop:
    c["lifecycle"] = {"preStop": {"exec": {"command": ["python", "-m", "raysimbatch.cli", "drain-node",
                                                       "--grace-s", "110"]}}}
  return c


def _pod(pool: Pool | None, containers: list[dict], volumes: list[dict], sa: str | None, grace: int) -> dict[str, Any]:
  """Workers run under the restricted service account without an API token. The head
  (sa=None) uses the service account KubeRay creates for the in-tree autoscaler, whose
  RBAC is scoped to this RayCluster, and needs its token."""
  spec: dict[str, Any] = {"containers": containers, "volumes": volumes, "terminationGracePeriodSeconds": grace,
                          # The image's non-root user; fsGroup makes mounted secrets readable by it.
                          "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000,
                                              "fsGroup": 1000, "seccompProfile": {"type": "RuntimeDefault"}}}
  if sa is not None:
    spec["serviceAccountName"] = sa
    spec["automountServiceAccountToken"] = False
  if pool is not None:
    if pool.node_selector:
      spec["nodeSelector"] = dict(pool.node_selector)
    if pool.tolerations:
      spec["tolerations"] = list(pool.tolerations)
  return {"spec": spec}


def render(cfg: RunConfig, inv: PlatformInventory, *, tenant: str = "default", production: bool = False,
           cluster_name: str | None = None, include_gpu: bool = True, include_contact: bool = True) -> list[dict]:
  problems = inv.check(production)
  if problems:
    raise ValueError("platform inventory rejected: " + "; ".join(problems))
  c = cfg.cluster
  ns = f"{c.namespace}-{tenant}"
  name = cluster_name or f"raysimbatch-{tenant}"
  sc = inv.scale
  sa = c.service_account
  grace = c.termination_grace_s
  actors_per_node = c.simulation_cpus // cfg.worker.num_cpus
  store_bytes = int(sc.get("object_store_bytes", c.object_store_gib * 2**30))
  shm = sc.get("shm", f"{c.shm_gib}Gi")
  worker_cpu = str(sc.get("worker_cpu", c.ray_cpus))
  worker_mem = sc.get("worker_memory", f"{c.cpu_node_memory_gib - 32}Gi")
  ray_cpus = str(sc.get("ray_cpus", c.ray_cpus))
  # KubeRay 1.7.0 splices rayStartParams values unquoted into `bash -c "ray start ..."`:
  # JSON values must be compact and wrapped in single quotes (amendment AM-14).
  spill_cfg = _shell_json({"object_spilling_config": json.dumps(
    {"type": "filesystem", "params": {"directory_path": c.spill_path}}, separators=(",", ":"))})
  sec_vol, sec_mount = _secret_volume(inv.database_secret, "db-secret")
  common_vol = [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": shm}},
                {"name": "spill", **inv.spill_volume}, *sec_vol,
                {"name": "rsb-config", "configMap": {"name": f"{name}-config"}}]
  common_mounts = [{"name": "shm", "mountPath": "/dev/shm"}, {"name": "spill", "mountPath": c.spill_path},
                   {"name": "rsb-config", "mountPath": "/etc/raysimbatch", "readOnly": True}, *sec_mount]
  head_c = _container("ray-head", inv.head_image, str(sc.get("head_cpu", c.head_cpu)),
                      sc.get("head_memory", f"{c.head_memory_gib}Gi"), inv, mounts=common_mounts, cluster=name)
  # Cluster janitor (design §7): reconciles detached controllers, placement groups, cache
  # holds and orphaned runs against durable leases every minute, next to the GCS.
  janitor_c = _container("janitor", inv.head_image, "100m", sc.get("janitor_memory", "384Mi"), inv,
                         mounts=common_mounts, cluster=name)
  janitor_c["command"] = ["raysimbatch", "janitor", "--config", "/etc/raysimbatch/run-config.json",
                          # Ray Client: a sidecar has no access to the head's /tmp/ray session
                          # (raylet/plasma sockets), so it cannot attach as a local driver.
                          "--address", "ray://127.0.0.1:10001", "--namespace", c.namespace, "--loop", "60"]
  head_c["ports"] = [{"containerPort": 6379, "name": "gcs"}, {"containerPort": 8265, "name": "dashboard"},
                     {"containerPort": 10001, "name": "client"},
                     {"containerPort": cfg.telemetry.metrics_port, "name": "metrics"}]
  head = {
    "rayStartParams": {"num-cpus": "0", "metrics-export-port": str(cfg.telemetry.metrics_port),
                       "dashboard-host": "0.0.0.0", "system-config": spill_cfg},
    "template": _pod(inv.pools.get("head"), [head_c, janitor_c], common_vol, None, grace),
  }

  def sim_group(group: str, resource: str) -> dict[str, Any]:
    # RSB_POD_CPUS: the affinity agent refuses a node whose actual CPU set is not this
    # exclusive allocation (static CPU Manager not in effect).
    wc = _container("ray-worker", inv.cpu_image, worker_cpu, worker_mem, inv, mounts=common_mounts, pre_stop=True,
                    cluster=name, env_extra=({"name": "RSB_POD_CPUS", "value": worker_cpu},))
    return {
      "groupName": group,
      "replicas": max(c.min_cpu_nodes, int(sc.get("replicas", c.min_cpu_nodes))),
      "minReplicas": c.min_cpu_nodes, "maxReplicas": c.max_cpu_nodes,
      "rayStartParams": {"num-cpus": ray_cpus, "object-store-memory": str(store_bytes),
                         "metrics-export-port": str(cfg.telemetry.metrics_port),
                         # system-config is head-only in Ray 2.58 (propagates to workers)
                         "resources": _shell_json({resource: actors_per_node})},
      "template": _pod(inv.pools.get(resource), [wc], common_vol, sa, grace),
    }

  groups = [sim_group("sim-general", "sim_general")]
  if include_contact:
    g = sim_group("sim-contact", "sim_contact")
    g["replicas"] = g["minReplicas"] = 0
    groups.append(g)
  if include_gpu and inv.gpu_image:
    gc = _container("ray-gpu", inv.gpu_image, str(c.gpu_pod_cpus), f"{c.gpu_pod_memory_gib}Gi", inv,
                    gpus=c.gpu_devices_per_pod, mounts=common_mounts, pre_stop=True, cluster=name)
    groups.append({
      "groupName": "gpu", "replicas": c.min_gpu_nodes, "minReplicas": c.min_gpu_nodes, "maxReplicas": c.max_gpu_nodes,
      "rayStartParams": {"num-cpus": str(c.gpu_ray_cpus), "num-gpus": str(c.gpu_devices_per_pod),
                         "resources": _shell_json({"policy_gpu": c.gpu_devices_per_pod}),
                         "metrics-export-port": str(cfg.telemetry.metrics_port)},
      "template": _pod(inv.pools.get("gpu"), [gc], common_vol, sa, grace),
    })
  rc: dict[str, Any] = {
    "apiVersion": "ray.io/v1", "kind": "RayCluster",
    "metadata": {"name": name, "namespace": ns,
                 "labels": {"app.kubernetes.io/part-of": "raysimbatch", "raysimbatch.io/release": release_label(inv.cpu_image)},
                 "annotations": {"raysimbatch.io/cpu-image": inv.cpu_image, "raysimbatch.io/head-image": inv.head_image,
                                 "raysimbatch.io/config-hash": _config_hash(cfg)}},
    "spec": {"rayVersion": c.ray_version, "enableInTreeAutoscaling": True,
             "autoscalerOptions": {"upscalingMode": "Default", "idleTimeoutSeconds": int(c.autoscale_down_hold_s)},
             "headGroupSpec": head, "workerGroupSpecs": groups},
  }
  if c.gcs_ft == "redis_qualified":
    rc["spec"]["gcsFaultToleranceOptions"] = {
      "redisAddress": inv.extra_env.get("RSB_REDIS_ADDRESS", ""),
      "redisPassword": {"valueFrom": {"secretKeyRef": {"name": (c.redis_secret or "").removeprefix("k8s:").split("/")[0],
                                                       "key": "password"}}},
    }
  cpu_quota = c.tenant_cpu_limit
  out = [
    {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns, "labels": {"raysimbatch.io/tenant": tenant}}},
    {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": sa, "namespace": ns},
     "automountServiceAccountToken": False},
    {"apiVersion": "v1", "kind": "ResourceQuota", "metadata": {"name": "raysimbatch-quota", "namespace": ns},
     "spec": {"hard": {"limits.cpu": str(cpu_quota), "requests.nvidia.com/gpu": str(c.tenant_gpu_limit)}}},
    {"apiVersion": "v1", "kind": "LimitRange", "metadata": {"name": "raysimbatch-limits", "namespace": ns},
     "spec": {"limits": [{"type": "Container", "defaultRequest": {"cpu": "1", "memory": "1Gi"},
                          "default": {"cpu": "1", "memory": "1Gi"}}]}},
    {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
     "metadata": {"name": "raysimbatch-intra", "namespace": ns},
     "spec": {"podSelector": {}, "policyTypes": ["Ingress"],
              "ingress": [{"from": [{"podSelector": {}}]},
                          {"from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "monitoring"}}}],
                           "ports": [{"port": cfg.telemetry.metrics_port}]}]}},
    {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": f"{name}-config", "namespace": ns},
     "data": {"run-config.json": cfg.model_dump_json(indent=2)}},
    rc,
  ]
  return out


def _shell_json(obj: Any) -> str:
  text = json.dumps(obj, separators=(",", ":"))
  if "'" in text:
    raise ValueError("rayStartParams JSON must not contain single quotes")
  return f"'{text}'"


def release_label(image: str) -> str:
  """A valid Kubernetes label value identifying the release image (<= 63 chars)."""
  import re

  v = image.split("@sha256:")[1][:16] if "@sha256:" in image else image.rsplit("/", 1)[-1]
  v = re.sub(r"[^A-Za-z0-9_.-]", "-", v)[:63].strip("-_.")
  return v or "unknown"


def _config_hash(cfg: RunConfig) -> str:
  from raysimbatch.config.system import ConfigurationSystem

  return ConfigurationSystem().canonical_hash(cfg)


def to_yaml(objs: list[dict]) -> str:
  return yaml.safe_dump_all(objs, sort_keys=False)
