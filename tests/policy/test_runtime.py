# SPDX-License-Identifier: Apache-2.0
"""Policy runtimes never substitute a device: a request the runtime cannot honour fails."""

from __future__ import annotations

import numpy as np
import pytest

from raysimbatch.policy.bundle import make_bundle
from raysimbatch.policy.export import linear_policy
from raysimbatch.policy.runtime import LocalPolicyRuntime, PolicyRuntimeError


def test_onnx_on_mps_is_refused_not_silently_cpu():
  exe = linear_policy(np.zeros((1, 4)))
  bundle = make_bundle("zero", 1, "onnx", exe, 4, 1, certified_batch_sizes=(1,))
  with pytest.raises(PolicyRuntimeError, match="no silent CPU fallback"):
    LocalPolicyRuntime(bundle, exe, device="mps")
  assert LocalPolicyRuntime(bundle, exe, device="cpu").infer(np.zeros((1, 4), np.float32), {}).action.shape == (1, 1)


torch = pytest.importorskip("torch")


def _pair(obs_dim=4, act_dim=2):
  from raysimbatch.policy.export import torchscript_linear

  K = np.random.default_rng(0).normal(size=(act_dim, obs_dim))
  onnx_exe = linear_policy(K)
  ts_exe = torchscript_linear(K)
  mk = lambda rt, exe: make_bundle(f"lin-{rt}", 1, rt, exe, obs_dim, act_dim,  # noqa: E731
                                    certified_batch_sizes=(1, 8), deterministic_inference=True)
  return (mk("onnx", onnx_exe), onnx_exe), (mk("torchscript", ts_exe), ts_exe)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torchscript_matches_onnx_for_the_same_policy(device):
  if device == "mps" and not torch.backends.mps.is_available():
    pytest.skip("no MPS device")
  (ob, oe), (tb, te) = _pair()
  obs = np.random.default_rng(1).normal(size=(8, 4)).astype(np.float32)
  a = LocalPolicyRuntime(ob, oe, device="cpu").infer(obs, {}, deterministic=True).action
  b = LocalPolicyRuntime(tb, te, device=device).infer(obs, {}, deterministic=True).action
  np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_embedded_normalizer_needs_mean_and_std():
  """Half a normalizer used to export silently: a mean without a std divided by NaN (every
  action NaN, found only later as "nonfinite actions"); a std without a mean was dropped."""
  from raysimbatch.policy.export import mlp_onnx

  W, b = np.eye(2, dtype=np.float32), np.zeros(2, np.float32)
  for kw in ({"obs_mean": np.zeros(2)}, {"obs_std": np.ones(2)}):
    with pytest.raises(ValueError, match="together"):
      mlp_onnx([(W, b)], **kw)
  import onnxruntime as ort

  exe = mlp_onnx([(W, b)], obs_mean=np.array([1.0, 2.0]), obs_std=np.array([2.0, 4.0]))
  sess = ort.InferenceSession(exe, providers=["CPUExecutionProvider"])
  out = sess.run(["action"], {"obs": np.array([[3.0, 6.0]], np.float32)})[0]
  np.testing.assert_allclose(out, [[1.0, 1.0]])  # (obs - mean) / std through the identity layer
