# SPDX-License-Identifier: Apache-2.0
"""Transfer bundles and gate reports (design §6.7, §10.2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from raysimbatch.config.models import (
  ActuatorContract,
  AdapterContract,
  AssetContract,
  ModelGroup,
  ObservationTerm,
)
from raysimbatch.policy.bundle import PolicyBundle
from raysimbatch.sim2sim.source import SourceConventions, SourceExport

GATES = ("XML_ABI_MISMATCH", "JOINT_MAPPING", "OBS_CONTRACT", "NORMALIZATION", "STATE_LEAK", "POLICY_EXPORT",
         "CONTROL_RATE", "ACTUATOR", "INERTIAL", "ASSET_FRAME", "CONTACT", "PHYSICS_CONFIG")


@dataclass
class TransferBundle:
  """Canonical MJCF, source export (property tables, traces, fixtures), attested
  adapter and asset contracts, and the exported policy."""

  canonical_xml: bytes
  group: ModelGroup
  contract: AdapterContract
  asset_contract: AssetContract
  policy: PolicyBundle
  executable: bytes
  source: SourceExport
  task_doc: dict[str, Any]
  normalizer: dict[str, list[float]] | None = None
  envelope: dict[str, Any] | None = None

  @property
  def digest(self) -> str:
    h = hashlib.sha256()
    for part in (self.canonical_xml, self.group.model_dump_json().encode(), self.contract.model_dump_json().encode(),
                 self.asset_contract.model_dump_json().encode(), self.policy.to_json(), self.executable,
                 self.source.digest.encode(), json.dumps(self.task_doc, sort_keys=True).encode(),
                 json.dumps(self.normalizer, sort_keys=True).encode()):
      h.update(hashlib.sha256(part).digest())
    return h.hexdigest()


@dataclass(frozen=True)
class GateResult:
  code: str
  passed: bool
  metrics: dict[str, float] = field(default_factory=dict)
  message: str = ""
  skipped: bool = False


@dataclass
class GateReport:
  bundle_digest: str
  results: list[GateResult]
  parity_profile: dict[str, Any]

  @property
  def passed(self) -> bool:
    return all(r.passed for r in self.results if not r.skipped) and not any(r.skipped for r in self.results)

  def failed(self) -> list[str]:
    return [r.code for r in self.results if not r.passed and not r.skipped]

  def to_dict(self) -> dict[str, Any]:
    return {"bundle_digest": self.bundle_digest, "passed": self.passed, "failed": self.failed(),
            "results": [asdict(r) for r in self.results], "parity_profile": self.parity_profile}

  @property
  def digest(self) -> str:
    return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=float).encode()).hexdigest()


def contract_for(conv: SourceConventions, canonical_defaults_sign: bool = True, **overrides: Any) -> AdapterContract:
  """The attested adapter contract matching a source export's conventions."""
  J = len(conv.joint_order)
  signs = tuple(int(conv.joint_sign.get(n, 1)) for n in conv.joint_order)
  terms = []
  for t in conv.terms:
    src = t["source"]
    kw: dict[str, Any] = {"name": t["name"], "source": "previous_action" if src == "previous_action" else src,
                          "dimension": int(t["dim"]), "history": int(t.get("history", 1)),
                          "scale": float(t.get("scale", 1.0)), "clip": t.get("clip")}
    if src in ("joint_pos", "joint_vel"):
      kw["selectors"] = tuple(conv.joint_order)
      if src == "joint_pos":
        kw["offsets"] = tuple(float(conv.default_pos.get(n, 0.0)) for n in conv.joint_order)
    if src == "rot6d":
      kw["rot6d_layout"] = t.get("layout", "columns")
    terms.append(ObservationTerm(**kw))
  quat = next((t.get("quaternion", "wxyz") for t in conv.terms if t["source"] == "base_quat"), "wxyz")
  c = AdapterContract.model_validate(dict(  # validates the Literal fields from strings
    joint_names=tuple(conv.joint_order),
    joint_signs=signs,
    observation_terms=tuple(terms),
    quaternion=quat,
    history_layout=conv.history_layout,
    history_reset=conv.history_reset,
    history_order=conv.history_order,
    base_joint=conv.base_joint,
    action_clip=conv.action_clip,
    action_scale=(conv.action_scale,),
    action_offset=tuple(float(conv.default_pos.get(n, 0.0)) for n in conv.joint_order),
    ema_alpha=conv.ema_alpha,
    normalization="none",
    actuator=ActuatorContract(kind="pd", kp=(conv.kp,) * J, kd=(conv.kd,) * J, effort_limit=(conv.effort_limit,) * J),
  ))
  return c.model_copy(update=overrides) if overrides else c


def as_numpy(x: Any) -> np.ndarray:
  return np.asarray(x, np.float64)
