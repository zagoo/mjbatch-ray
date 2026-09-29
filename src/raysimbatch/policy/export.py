# SPDX-License-Identifier: Apache-2.0
"""Build ONNX/TorchScript policy executables with explicit, batch-dynamic tensors.

Used for examples, tests and golden fixtures. Graphs keep the batch axis symbolic
(unlike exporters that bake example shapes) and expose recurrent state as explicit
inputs/outputs, the contract §6.7 requires for concurrent batched use.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from raysimbatch.policy.bundle import PolicyBundle, StateSpec, make_bundle

OPSET = 17


def _init(name: str, a: np.ndarray) -> onnx.TensorProto:
  return numpy_helper.from_array(np.asarray(a, np.float32), name)


def mlp_onnx(
  layers: Sequence[tuple[np.ndarray, np.ndarray]],
  *,
  activation: str = "Tanh",
  out_clip: float | None = None,
  log_std: np.ndarray | None = None,
  value_head: tuple[np.ndarray, np.ndarray] | None = None,
  obs_mean: np.ndarray | None = None,
  obs_std: np.ndarray | None = None,
) -> bytes:
  """y = clip(MLP(normalize(obs))) with weights W:[in,out]. ``obs_mean/std`` embed the
  normalizer in the graph (``normalization: embedded``); they come together or not at all."""
  if (obs_mean is None) != (obs_std is None):
    raise ValueError("obs_mean and obs_std must be given together (embedded normalization needs both)")
  nodes, inits = [], []
  x = "obs"
  if obs_mean is not None:
    assert obs_std is not None  # checked above: both or neither
    inits += [_init("nm", obs_mean), _init("ns", obs_std)]
    nodes += [helper.make_node("Sub", ["obs", "nm"], ["obs_c"]), helper.make_node("Div", ["obs_c", "ns"], ["obs_n"])]
    x = "obs_n"
  h = x
  for i, (W, b) in enumerate(layers):
    inits += [_init(f"W{i}", W), _init(f"b{i}", b)]
    nodes += [helper.make_node("MatMul", [h, f"W{i}"], [f"m{i}"]), helper.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"])]
    h = f"a{i}"
    if i < len(layers) - 1:
      nodes.append(helper.make_node(activation, [h], [f"h{i}"]))
      h = f"h{i}"
  if out_clip is not None:
    inits += [_init("lo", np.array(-out_clip)), _init("hi", np.array(out_clip))]
    nodes.append(helper.make_node("Clip", [h, "lo", "hi"], ["action"]))
  else:
    nodes.append(helper.make_node("Identity", [h], ["action"]))
  outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", layers[-1][0].shape[1]])]
  if log_std is not None:
    inits.append(_init("log_std_c", log_std.reshape(1, -1)))
    nodes.append(helper.make_node("Identity", ["log_std_c"], ["log_std"]))
    outputs.append(helper.make_tensor_value_info("log_std", TensorProto.FLOAT, [1, log_std.size]))
  if value_head is not None:
    inits += [_init("Wv", value_head[0]), _init("bv", value_head[1]),
              numpy_helper.from_array(np.array([-1], np.int64), "flat")]
    src = x if len(layers) == 1 else f"h{len(layers) - 2}"
    nodes += [helper.make_node("MatMul", [src, "Wv"], ["vm"]), helper.make_node("Add", ["vm", "bv"], ["va"]),
              helper.make_node("Reshape", ["va", "flat"], ["value"])]
    outputs.append(helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"]))
  graph = helper.make_graph(
    nodes, "rsb_mlp",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", layers[0][0].shape[0]])],
    outputs, inits,
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
  model.ir_version = 8
  onnx.checker.check_model(model)
  return model.SerializeToString()


def recurrent_onnx(Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, Wo: np.ndarray, bo: np.ndarray) -> bytes:
  """h' = tanh(x Wx + h Wh + b); action = h' Wo + bo; explicit h_in/h_out."""
  H = Wh.shape[0]
  inits = [_init(n, a) for n, a in (("Wx", Wx), ("Wh", Wh), ("b", b), ("Wo", Wo), ("bo", bo))]
  nodes = [
    helper.make_node("MatMul", ["obs", "Wx"], ["xw"]),
    helper.make_node("MatMul", ["h_in", "Wh"], ["hw"]),
    helper.make_node("Add", ["xw", "hw"], ["s0"]),
    helper.make_node("Add", ["s0", "b"], ["s1"]),
    helper.make_node("Tanh", ["s1"], ["h_out"]),
    helper.make_node("MatMul", ["h_out", "Wo"], ["o0"]),
    helper.make_node("Add", ["o0", "bo"], ["action"]),
  ]
  graph = helper.make_graph(
    nodes, "rsb_rnn",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", Wx.shape[0]]),
     helper.make_tensor_value_info("h_in", TensorProto.FLOAT, ["batch", H])],
    [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", Wo.shape[1]]),
     helper.make_tensor_value_info("h_out", TensorProto.FLOAT, ["batch", H])],
    inits,
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
  model.ir_version = 8
  onnx.checker.check_model(model)
  return model.SerializeToString()


def linear_policy(K: np.ndarray, clip: float = 1.0) -> bytes:
  """action = clip(-obs K^T): a stabilizing linear feedback as an ONNX graph."""
  return mlp_onnx([(-K.T.astype(np.float32), np.zeros(K.shape[0], np.float32))], out_clip=clip)


def random_mlp(obs_dim: int, act_dim: int, hidden: int = 64, seed: int = 0, **kw) -> bytes:
  rng = np.random.default_rng(seed)
  layers = [
    (rng.normal(0, 1 / np.sqrt(obs_dim), (obs_dim, hidden)).astype(np.float32), np.zeros(hidden, np.float32)),
    (rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)).astype(np.float32), np.zeros(hidden, np.float32)),
    (rng.normal(0, 0.1 / np.sqrt(hidden), (hidden, act_dim)).astype(np.float32), np.zeros(act_dim, np.float32)),
  ]
  return mlp_onnx(layers, **kw)


def cartpole_bundle(version: int = 1) -> tuple[PolicyBundle, bytes]:
  from raysimbatch.testing.fixtures import cartpole_lqr_gain

  exe = linear_policy(cartpole_lqr_gain())
  return make_bundle("cartpole_lqr", version, "onnx", exe, 4, 1, certified_batch_sizes=(1, 4, 8, 16, 32, 64, 128, 256)), exe


def gaussian_bundle(obs_dim: int, act_dim: int, version: int = 1, seed: int = 0, value: bool = True,
                    name: str = "mlp_gaussian") -> tuple[PolicyBundle, bytes]:
  rng = np.random.default_rng(seed)
  hidden = 64
  exe = random_mlp(
    obs_dim, act_dim, hidden, seed, out_clip=4.0, log_std=np.full(act_dim, -1.0, np.float32),
    value_head=(rng.normal(0, 0.1, (hidden, 1)).astype(np.float32), np.zeros(1, np.float32)) if value else None,
  )
  return make_bundle(
    name, version, "onnx", exe, obs_dim, act_dim, distribution="diagonal_gaussian",
    deterministic_inference=False, log_std_output="log_std", value_output="value" if value else None,
    certified_batch_sizes=(1, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
  ), exe


def recurrent_bundle(obs_dim: int, act_dim: int, hidden: int = 16, seed: int = 0) -> tuple[PolicyBundle, bytes]:
  rng = np.random.default_rng(seed)
  exe = recurrent_onnx(
    rng.normal(0, 0.3, (obs_dim, hidden)).astype(np.float32), rng.normal(0, 0.3, (hidden, hidden)).astype(np.float32),
    np.zeros(hidden, np.float32), rng.normal(0, 0.3, (hidden, act_dim)).astype(np.float32),
    np.zeros(act_dim, np.float32),
  )
  return make_bundle(
    "rnn", 1, "onnx", exe, obs_dim, act_dim, recurrent=(StateSpec("h", (hidden,), "h_in", "h_out"),),
    certified_batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256),
  ), exe


def torchscript_linear(K: np.ndarray, clip: float = 1.0) -> bytes:
  import torch

  class Linear(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.register_buffer("K", torch.from_numpy(np.ascontiguousarray(K, np.float32)))
      self.clip = float(clip)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
      # torch types a registered buffer as Tensor | Module, and TorchScript rejects a
      # class-level `K: torch.Tensor` declaration here (torch is imported locally).
      return {"action": torch.clamp(-obs @ self.K.t(), -self.clip, self.clip)}  # pyright: ignore[reportCallIssue]

  import warnings

  buf = io.BytesIO()
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # TorchScript is the bundle format, not a choice here
    torch.jit.save(torch.jit.script(Linear()), buf)
  return buf.getvalue()
