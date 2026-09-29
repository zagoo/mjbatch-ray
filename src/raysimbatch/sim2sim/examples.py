# SPDX-License-Identifier: Apache-2.0
"""The reference quadruped transfer bundle used by tests, examples and qualification."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from raysimbatch.config.models import AssetContract
from raysimbatch.policy.bundle import make_bundle
from raysimbatch.policy.export import mlp_onnx
from raysimbatch.sim2sim import source as S
from raysimbatch.sim2sim.bundle import TransferBundle, contract_for
from raysimbatch.testing import fixtures as F


@functools.cache
def policy_layers(obs_dim: int, act_dim: int = 8, seed: int = 42):
  rng = np.random.default_rng(seed)
  return (
    (rng.normal(0, 1 / np.sqrt(obs_dim), (obs_dim, 64)).astype(np.float32), rng.normal(0, 0.1, 64).astype(np.float32)),
    (rng.normal(0, 0.1, (64, act_dim)).astype(np.float32), np.zeros(act_dim, np.float32)),
  )


def quadruped_bundle(conv: S.SourceConventions | None = None, *, xml: bytes | None = None,
                     contract_kw: dict[str, Any] | None = None, group_kw: dict[str, Any] | None = None,
                     policy_exe: bytes | None = None, normalizer: dict | None = None, n_fixtures: int = 1000,
                     max_control_steps: int = 400) -> TransferBundle:
  conv = conv or S.quadruped_conventions()
  fx = F.quadruped(max_control_steps=max_control_steps)
  dim = sum(t["dim"] * t.get("history", 1) for t in conv.terms)
  ls = policy_layers(dim)
  solver = {"iterations": fx.group.iterations, "tolerance": fx.group.tolerance}
  src = S.export_reference(fx.xml, conv, ls, n_fixtures=n_fixtures, solver=solver)
  exe = policy_exe or mlp_onnx(list(ls), out_clip=conv.action_clip)
  pol = make_bundle("src_policy", 1, "onnx", exe, dim, 8, normalization="none" if normalizer is None else "embedded",
                    certified_batch_sizes=(1, 8, 16, 64, 256))
  group = fx.group.model_copy(update={"initial_keyframe": 0, **(group_kw or {})})
  return TransferBundle(xml or fx.xml, group, contract_for(conv, **(contract_kw or {})), AssetContract(), pol, exe,
                        src, fx.task_doc, normalizer=normalizer)
