# SPDX-License-Identifier: Apache-2.0
"""Policy bundles (design §6.4, §6.7): executable + explicit tensor contracts.

A bundle is a closed JSON manifest plus the executable bytes (ONNX or TorchScript).
The policy hash covers both. Recurrent state is explicit: every state tensor is a
named graph input and output, so inference is stateless with respect to environments
and requests can be retried or rerouted without cross-env leakage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

BUNDLE_FORMAT = "rsb-policy-bundle/1"


class BundleError(ValueError):
  pass


@dataclass(frozen=True)
class StateSpec:
  name: str
  shape: tuple[int, ...]
  input: str
  output: str


@dataclass(frozen=True)
class PolicyBundle:
  name: str
  version: int
  runtime: Literal["onnx", "torchscript"]
  executable_sha256: str
  obs_dim: int
  action_dim: int
  observation_input: str
  action_output: str
  distribution: Literal["deterministic", "diagonal_gaussian", "categorical"] = "deterministic"
  deterministic_inference: bool = True
  normalization: Literal["embedded", "external", "none"] = "embedded"
  log_std_output: str | None = None
  value_output: str | None = None
  recurrent: tuple[StateSpec, ...] = ()
  certified_batch_sizes: tuple[int, ...] = (1, 16, 32, 64, 128, 256, 512, 1024)
  schema_hash: str = ""
  training_manifest: Mapping[str, Any] = field(default_factory=dict)

  def manifest(self) -> dict[str, Any]:
    return {
      "format": BUNDLE_FORMAT,
      "name": self.name,
      "version": self.version,
      "runtime": self.runtime,
      "executable_sha256": self.executable_sha256,
      "obs_dim": self.obs_dim,
      "action_dim": self.action_dim,
      "observation_input": self.observation_input,
      "action_output": self.action_output,
      "distribution": self.distribution,
      "deterministic_inference": self.deterministic_inference,
      "normalization": self.normalization,
      "log_std_output": self.log_std_output,
      "value_output": self.value_output,
      "recurrent": [
        {"name": s.name, "shape": list(s.shape), "input": s.input, "output": s.output} for s in self.recurrent
      ],
      "certified_batch_sizes": list(self.certified_batch_sizes),
      "schema_hash": self.schema_hash,
      "training_manifest": dict(self.training_manifest),
    }

  def to_json(self) -> bytes:
    return json.dumps(self.manifest(), sort_keys=True, separators=(",", ":")).encode()

  @property
  def policy_hash(self) -> str:
    """Identity of manifest + executable (the executable digest is in the manifest)."""
    return hashlib.sha256(self.to_json()).hexdigest()

  @property
  def policy_hash_bytes(self) -> bytes:
    return bytes.fromhex(self.policy_hash)

  @classmethod
  def from_json(cls, blob: bytes | str) -> PolicyBundle:
    d = json.loads(blob)
    required = {
      "format", "name", "version", "runtime", "executable_sha256", "obs_dim", "action_dim",
      "observation_input", "action_output",
    }
    missing = required - set(d)
    if missing:
      raise BundleError(f"policy bundle missing contract fields {sorted(missing)}")
    if d["format"] != BUNDLE_FORMAT:
      raise BundleError(f"unsupported bundle format {d['format']!r}")
    allowed = required | {
      "distribution", "deterministic_inference", "normalization", "log_std_output", "value_output",
      "recurrent", "certified_batch_sizes", "schema_hash", "training_manifest",
    }
    extra = set(d) - allowed
    if extra:
      raise BundleError(f"unknown bundle keys {sorted(extra)}")
    b = cls(
      name=d["name"],
      version=int(d["version"]),
      runtime=d["runtime"],
      executable_sha256=d["executable_sha256"],
      obs_dim=int(d["obs_dim"]),
      action_dim=int(d["action_dim"]),
      observation_input=d["observation_input"],
      action_output=d["action_output"],
      distribution=d.get("distribution", "deterministic"),
      deterministic_inference=bool(d.get("deterministic_inference", True)),
      normalization=d.get("normalization", "embedded"),
      log_std_output=d.get("log_std_output"),
      value_output=d.get("value_output"),
      recurrent=tuple(
        StateSpec(r["name"], tuple(int(x) for x in r["shape"]), r["input"], r["output"]) for r in d.get("recurrent", ())
      ),
      certified_batch_sizes=tuple(int(x) for x in d.get("certified_batch_sizes", (1, 16, 32, 64, 128, 256, 512, 1024))),
      schema_hash=d.get("schema_hash", ""),
      training_manifest=d.get("training_manifest", {}),
    )
    b.check()
    return b

  def check(self) -> None:
    if self.runtime not in ("onnx", "torchscript"):
      raise BundleError("runtime must be onnx or torchscript")
    if self.distribution == "diagonal_gaussian" and not self.log_std_output:
      raise BundleError("diagonal_gaussian needs log_std_output")
    if self.distribution == "categorical":
      raise BundleError("categorical action heads are not supported by the continuous adapters in v1")
    if len({s.name for s in self.recurrent}) != len(self.recurrent):
      raise BundleError("duplicate recurrent state names")


def make_bundle(
  name: str,
  version: int,
  runtime: Literal["onnx", "torchscript"],
  executable: bytes,
  obs_dim: int,
  action_dim: int,
  **kw: Any,
) -> PolicyBundle:
  b = PolicyBundle(
    name=name,
    version=version,
    runtime=runtime,
    executable_sha256=hashlib.sha256(executable).hexdigest(),
    obs_dim=obs_dim,
    action_dim=action_dim,
    observation_input=kw.pop("observation_input", "obs"),
    action_output=kw.pop("action_output", "action"),
    **kw,
  )
  b.check()
  return b


def verify_executable(bundle: PolicyBundle, executable: bytes | Sequence[int]) -> bytes:
  data = bytes(executable)
  if hashlib.sha256(data).hexdigest() != bundle.executable_sha256:
    raise BundleError("policy executable digest mismatch: refusing to load corrupt weights")
  return data
