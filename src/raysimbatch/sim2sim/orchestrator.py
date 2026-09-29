# SPDX-License-Identifier: Apache-2.0
"""Sim2SimOrchestrator: gate a transfer bundle once per digest, then evaluate it as
Mode B units over policy x task x seed x experiment axis (design §5.4, §6.6, §6.7)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from typing import Any

from raysimbatch.config.models import EvalConfig, RunConfig, Sim2SimConfig
from raysimbatch.control.driver import ClusterDriver, RunRequest
from raysimbatch.protocol import Artifact
from raysimbatch.sim2sim.bundle import GateReport, GateResult, TransferBundle
from raysimbatch.sim2sim.gates import run_gates
from raysimbatch.sim2sim.harness import axis_variant, build_scorecard, scorecard_markdown
from raysimbatch.storage.store import DurableStore


class Sim2SimOrchestrator:
  def __init__(self, run_id: str | None, config: Sim2SimConfig, base: RunConfig, *, namespace: str = "raysimbatch",
               store: DurableStore | None = None):
    self.run_id = run_id
    self.config = config
    self.base = base
    self.namespace = namespace
    self.store = store or DurableStore.from_config(base)

  async def validate(self, bundle: TransferBundle) -> GateReport:
    """Gates run once per unique bundle digest; the report is committed durably."""
    profile = json.dumps(self.config.parity.model_dump(mode="json"), sort_keys=True)
    key = f"gates/{self.store.blobs.root}/{bundle.digest}/{profile}"
    rows = await asyncio.to_thread(self.store.db.tx, lambda cur: cur.execute(
      "SELECT artifact_uri, artifact_hash FROM commits WHERE key=%s", (key,)).fetchone())
    if rows:
      try:
        doc = json.loads(await asyncio.to_thread(self.store.blobs.get, _art(rows)))
        return GateReport(doc["bundle_digest"], [GateResult(**r) for r in doc["results"]], doc["parity_profile"])
      except Exception:  # noqa: BLE001 - an unreadable cached report is recomputed, never trusted
        pass
    report = await asyncio.to_thread(run_gates, bundle, self.config.parity, 128, self.config.static_gates,
                                     self.config.dynamics_gates)
    doc = json.dumps(report.to_dict(), sort_keys=True, default=float).encode()
    art = await asyncio.to_thread(self.store.put_blob_sync, doc, "application/json")
    try:
      await asyncio.to_thread(self.store.db.commit, key, 1, report.digest, art)
    except Exception:  # noqa: BLE001 - digest differs from an unreadable stale entry
      await asyncio.to_thread(self.store.db.tx, lambda cur: cur.execute(
        "UPDATE commits SET digest=%s, artifact_uri=%s, artifact_hash=%s WHERE key=%s",
        (report.digest, art.uri, art.sha256, key)))
    await asyncio.to_thread(self.store.db.audit, None, "sim2sim_gates",
                            {"bundle": bundle.digest, "passed": report.passed, "failed": report.failed()})
    from raysimbatch import telemetry

    m = telemetry.exporter(mode="sim2sim", actor_role="sim2sim")
    for code in report.failed():
      m.count("sim2sim_gate_failures_total", fault_code=str(code))
    if report.passed:
      m.count("sim2sim_validated_bundles_total")
    return report

  def _register(self, payload: bytes, media: str):
    from raysimbatch.config.models import ArtifactRef

    a = self.store.put_blob_sync(payload, media)
    self.store.db.register_artifact(a)
    return ArtifactRef(uri=a.uri, sha256=a.sha256)

  def build_config(self, bundle: TransferBundle, seeds: Sequence[int], axes: Sequence[str], episodes_per_seed: int,
                   policy_spec) -> tuple[RunConfig, dict[str, str], tuple]:
    from raysimbatch.config.models import TaskSpec

    groups, tasks, axis_of_group = [], [], {}
    kernel_doc = json.dumps({**bundle.task_doc, "reset_distribution": {
      "qpos": [{"joints": list(bundle.contract.joint_names), "low": -0.05, "high": 0.05}]}}, sort_keys=True).encode()
    kernel = self._register(kernel_doc, "application/json")
    base_name = bundle.group.name
    for axis in ("", *axes):
      xml, group, contract = (bundle.canonical_xml, bundle.group, bundle.contract) if axis == "" else axis_variant(
        axis, bundle.canonical_xml, bundle.group, bundle.contract)
      name = base_name if axis == "" else f"{base_name}__{axis}"
      g = group.model_copy(update={
        "name": name, "mjcf": self._register(xml, "application/xml"),
        "adapter": self._register(contract.model_dump_json().encode(), "application/json"), "task_kernel": kernel,
      })
      groups.append(g)
      axis_of_group[name] = axis
    task = TaskSpec(name="transfer", kernel=kernel, model_groups=tuple(g.name for g in groups))
    tasks.append(task)
    cfg = self.base.model_copy(update={
      "mode": "sim2sim", "models": tuple(groups), "policies": (policy_spec,), "tasks": tuple(tasks),
      "evaluation": self.base.evaluation.model_copy(update={"seeds": tuple(seeds),
                                                            "episodes_per_seed": episodes_per_seed}),
      "sim2sim": self.config.model_copy(update={"axes": tuple(axes)}),
    })
    return cfg, axis_of_group, tuple(tasks)

  async def evaluate(self, bundle: TransferBundle, policy_spec, seeds: Sequence[int], *, axes: Sequence[str] = (),
                     episodes_per_seed: int | None = None, exploratory: bool = False,
                     source_success: dict[str, Any] | None = None, options: dict[str, Any] | None = None,
                     timeout_s: float = 3600) -> dict[str, Any]:
    gates = await self.validate(bundle)
    if not gates.passed and not exploratory:
      raise ValueError(f"transfer admission refused: gates failed {gates.failed()} (exploratory runs only)")
    for a in axes:
      if a not in self.config.axes:
        raise ValueError(f"axis {a!r} is not configured / has no perturbation manifest")
    cfg, axis_of_group, tasks = self.build_config(bundle, seeds, axes,
                                                  episodes_per_seed or self.base.evaluation.episodes_per_seed,
                                                  policy_spec)
    # The run references the validated bundle: gate reports bind asset, adapter,
    # engine and policy hashes to the exact digest being scored.
    import hashlib

    import mujoco

    src_ref = self._register(bundle.source.to_bytes(), "application/x-rsb-source-export")
    manifest = {
      "format": "rsb-transfer-bundle/1", "bundle_digest": bundle.digest, "gate_report_digest": gates.digest,
      "gates_passed": gates.passed, "canonical_sha256": hashlib.sha256(bundle.canonical_xml).hexdigest(),
      "contract_sha256": hashlib.sha256(bundle.contract.model_dump_json().encode()).hexdigest(),
      "policy_hash": bundle.policy.policy_hash, "source_export": src_ref.sha256, "engine": mujoco.__version__,
      "asset_contract": bundle.asset_contract.model_dump(mode="json"),
    }
    bref = self._register(json.dumps(manifest, sort_keys=True).encode(), "application/json")
    cfg = cfg.model_copy(update={"sim2sim": cfg.sim2sim.model_copy(update={"bundle": bref, "source_trace": src_ref})})
    if exploratory:
      cfg = cfg.model_copy(update={"exploratory": True})
    driver = ClusterDriver(cfg, self.store, self.namespace)
    rid = await driver.submit(RunRequest("sim2sim", cfg, tasks, "sim2sim", {**(options or {}),
                                                                          "axis_of_group": axis_of_group}),
                              uuid.uuid4().hex)
    st = await driver.wait(rid, timeout_s=timeout_s)
    if st.state != "COMPLETED":
      raise RuntimeError(f"transfer evaluation {rid} ended {st.state}: {st.error}")
    arts = st.artifacts
    agg = json.loads(await asyncio.to_thread(self.store.blobs.get, Artifact(**arts["aggregate_json"])))
    card = build_scorecard(agg, gates, source_success=source_success,
                           minimum_paired_coverage=0.95, scoring=tasks[0].scoring)
    card["run_id"] = rid
    card["exploratory"] = exploratory
    from raysimbatch import telemetry

    m = telemetry.exporter(mode="sim2sim", actor_role="sim2sim")
    if card["conservative_transfer_ratio"] is not None:
      m.gauge("sim2sim_transfer_ratio", float(card["conservative_transfer_ratio"]))
    m.gauge("sim2sim_paired_coverage_ratio", float(card["target"]["coverage"] or 0.0))  # what the release gate checks
    card["release_qualified"] = card["release_qualified"] and not exploratory
    ja = await asyncio.to_thread(self.store.put_blob_sync, json.dumps(card, indent=2, sort_keys=True,
                                                                      default=float).encode(), "application/json")
    ma = await asyncio.to_thread(self.store.put_blob_sync, scorecard_markdown(card).encode(), "text/markdown")
    return {"run_id": rid, "scorecard": card, "scorecard_json": ja.__dict__, "scorecard_markdown": ma.__dict__,
            "results": arts, "gates": gates.to_dict()}


def _art(row) -> Artifact:
  return Artifact(row["artifact_uri"], row["artifact_hash"], _len(row), "application/json")


def _len(row) -> int:
  from pathlib import Path
  from urllib.parse import urlparse

  u = urlparse(row["artifact_uri"])
  if u.scheme == "file":
    return Path(u.path).stat().st_size
  import boto3

  return int(boto3.client("s3").head_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["ContentLength"])


def default_eval(seeds: Sequence[int], episodes: int) -> EvalConfig:
  return EvalConfig(seeds=tuple(seeds), episodes_per_seed=episodes)
