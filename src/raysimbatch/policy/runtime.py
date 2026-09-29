# SPDX-License-Identifier: Apache-2.0
"""LocalPolicyRuntime: single-threaded CPU inference inside the simulation actor.

Runs on the actor's calling thread while the MJBatch pool is parked (design §5.2),
with intra/inter-op threads fixed at one. Stochastic heads sample with counter-based
keys (``action`` stream), so a retried request reproduces the same action. There is
no automatic numerical substitution: a runtime that cannot load fails loudly.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from raysimbatch import seeds
from raysimbatch.policy.bundle import BundleError, PolicyBundle, verify_executable


@dataclass
class PolicyOutput:
  action: np.ndarray  # float32 [n, A]
  logp: np.ndarray  # float32 [n]
  value: np.ndarray | None  # float32 [n]
  recurrent: dict[str, np.ndarray]


class PolicyRuntimeError(RuntimeError):
  pass


def _gaussian_logp(x: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
  z = (x - mean) * np.exp(-log_std)
  return (-0.5 * z * z - log_std - 0.5 * math.log(2 * math.pi)).sum(axis=-1)


class LocalPolicyRuntime:
  def __init__(self, bundle: PolicyBundle, artifact: bytes | np.ndarray, device: str = "cpu", threads: int = 1):
    self.bundle = bundle
    data = verify_executable(bundle, artifact.tobytes() if isinstance(artifact, np.ndarray) else artifact)
    self.device = device
    if bundle.runtime == "onnx":
      import onnxruntime as ort

      # Observed: an exit-time abort in ORT's built-in telemetry (1DS) worker thread.
      # Platform telemetry is never wanted from simulation/inference processes.
      ort.disable_telemetry_events()
      opts = ort.SessionOptions()
      opts.intra_op_num_threads = threads
      opts.inter_op_num_threads = threads
      opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
      opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
      if device not in ("cpu", "cuda"):
        # Found by review: "mps" silently ran ONNX on the CPU provider.
        raise PolicyRuntimeError(f"ONNX policies run on cpu or cuda, not {device!r} (no silent CPU fallback); "
                                 "use a TorchScript bundle for mps")
      providers = ["CPUExecutionProvider"]
      if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
      self._sess = ort.InferenceSession(data, sess_options=opts, providers=providers)
      if device == "cuda" and "CUDAExecutionProvider" not in self._sess.get_providers():
        raise PolicyRuntimeError("CUDA requested but ONNX Runtime has no CUDA provider (no silent CPU fallback)")
      names = {i.name for i in self._sess.get_inputs()}
      outs = {o.name for o in self._sess.get_outputs()}
      need_in = {bundle.observation_input} | {s.input for s in bundle.recurrent}
      need_out = {bundle.action_output} | {s.output for s in bundle.recurrent}
      need_out |= {x for x in (bundle.log_std_output, bundle.value_output) if x}
      if not need_in <= names or not need_out <= outs:
        raise BundleError(f"graph does not expose the declared tensors: inputs {names}, outputs {outs}")
      self._outputs = sorted(need_out)
    else:
      import torch

      torch.set_num_threads(threads)
      self._torch = torch
      self._module = torch.jit.load(io.BytesIO(data), map_location=device)
      self._module.eval()
    self.closed = False

  @property
  def policy_hash(self) -> str:
    return self.bundle.policy_hash

  def infer(
    self,
    observations: np.ndarray,
    recurrent: Mapping[str, np.ndarray] | None = None,
    sample_keys: np.ndarray | None = None,
    sample_seq: np.ndarray | None = None,
    deterministic: bool | None = None,
  ) -> PolicyOutput:
    b = self.bundle
    obs = np.ascontiguousarray(observations, np.float32)
    n = obs.shape[0]
    if obs.ndim != 2 or obs.shape[1] != b.obs_dim:
      raise PolicyRuntimeError(f"observations must be [n, {b.obs_dim}]")
    rec = dict(recurrent or {})
    for s in b.recurrent:
      if s.name not in rec:
        raise PolicyRuntimeError(f"missing recurrent input {s.name!r} (state travels with the request)")
      if rec[s.name].shape != (n, *s.shape):
        raise PolicyRuntimeError(f"recurrent {s.name!r} must be [n, {s.shape}]")
    if n == 0:
      return PolicyOutput(np.zeros((0, b.action_dim), np.float32), np.zeros(0, np.float32), None,
                          {s.name: np.zeros((0, *s.shape), np.float32) for s in b.recurrent})
    outs = self._run(obs, rec)
    mean = np.asarray(outs[b.action_output], np.float32).reshape(n, b.action_dim)
    if not np.isfinite(mean).all():
      raise PolicyRuntimeError("policy produced nonfinite actions (no substitution is permitted)")
    det = b.deterministic_inference if deterministic is None else deterministic
    if b.distribution == "diagonal_gaussian":
      assert b.log_std_output is not None  # required for this distribution (PolicyBundle validation)
      log_std = np.broadcast_to(np.asarray(outs[b.log_std_output], np.float32), mean.shape)
      if det:
        action = mean
      else:
        if sample_keys is None or sample_seq is None:
          raise PolicyRuntimeError("stochastic sampling needs counter keys")
        z = np.empty_like(mean, dtype=np.float64)
        for i in range(n):  # per-row sequence counters
          z[i] = seeds.normal(sample_keys[i : i + 1], int(sample_seq[i]), b.action_dim, "action")[0]
        action = (mean + np.exp(log_std) * z).astype(np.float32)
      logp = _gaussian_logp(action.astype(np.float64), mean, log_std).astype(np.float32)
    else:
      action, logp = mean, np.zeros(n, np.float32)
    value = None
    if b.value_output:
      value = np.asarray(outs[b.value_output], np.float32).reshape(n)
    nxt = {s.name: np.ascontiguousarray(np.asarray(outs[s.output], np.float32).reshape(n, *s.shape)) for s in b.recurrent}
    return PolicyOutput(np.ascontiguousarray(action, np.float32), logp, value, nxt)

  def _run(self, obs: np.ndarray, rec: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    b = self.bundle
    if b.runtime == "onnx":
      feeds = {b.observation_input: obs}
      for s in b.recurrent:
        feeds[s.input] = np.ascontiguousarray(rec[s.name], np.float32)
      if self._sess is None:
        raise PolicyRuntimeError("policy runtime is closed")
      vals = self._sess.run(self._outputs, feeds)
      return {k: np.asarray(v) for k, v in zip(self._outputs, vals, strict=True)}
    torch = self._torch
    with torch.inference_mode():
      args = [torch.from_numpy(obs).to(self.device)]
      for s in b.recurrent:
        args.append(torch.from_numpy(np.ascontiguousarray(rec[s.name], np.float32)).to(self.device))
      if self._module is None:
        raise PolicyRuntimeError("policy runtime is closed")
      res = self._module(*args)
    if not isinstance(res, dict):
      raise PolicyRuntimeError("TorchScript policies must return a dict of named tensors")
    return {k: v.detach().to("cpu").numpy() for k, v in res.items()}

  def close(self) -> None:
    self.closed = True
    self._sess = None  # type: ignore[assignment]
    self._module = None  # type: ignore[assignment]
