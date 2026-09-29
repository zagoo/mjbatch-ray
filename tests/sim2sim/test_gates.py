# SPDX-License-Identifier: Apache-2.0
"""Sim2Sim golden fixtures (design §13.4): an aligned bundle passes every gate, and each
intentionally injected checklist error makes its own gate fail."""

from __future__ import annotations

import mujoco
import numpy as np

from raysimbatch.policy.export import mlp_onnx
from raysimbatch.sim2sim import source as S
from raysimbatch.sim2sim.bundle import TransferBundle, contract_for
from raysimbatch.sim2sim.examples import policy_layers, quadruped_bundle
from raysimbatch.sim2sim.gates import run_gates
from raysimbatch.testing import fixtures as F


def layers(obs_dim: int, act_dim: int = 8):
  return policy_layers(obs_dim, act_dim)


def bundle(conv=None, **kw) -> TransferBundle:
  return quadruped_bundle(conv, **kw)


def failed(b) -> set[str]:
  return set(run_gates(b).failed())


def test_aligned_bundle_passes_every_gate():
  r = run_gates(bundle())
  assert r.passed, r.to_dict()
  metrics = {x.code: x.metrics for x in r.results}
  assert metrics["OBS_CONTRACT"]["fixtures"] >= 1000
  assert metrics["POLICY_EXPORT"]["max_action_error"] <= 1e-4


def _swap(t, i, j):
  t = list(t)
  t[i], t[j] = t[j], t[i]
  return tuple(t)


def test_permuted_joints_fail_joint_mapping():
  c = contract_for(S.quadruped_conventions())
  b = bundle(contract_kw={"joint_names": _swap(c.joint_names, 0, 4)})
  assert "JOINT_MAPPING" in failed(b)


def test_reversed_sign_fails_joint_mapping_and_observation():
  c = contract_for(S.quadruped_conventions())
  signs = list(c.joint_signs)
  signs[1] = -signs[1]  # fr_hip
  terms = []
  for t in c.observation_terms:
    if t.source in ("joint_pos", "joint_vel"):
      t = t.model_copy(update={"signs": tuple(signs)})
    terms.append(t)
  f = failed(bundle(contract_kw={"joint_signs": tuple(signs), "observation_terms": tuple(terms)}))
  assert {"JOINT_MAPPING", "OBS_CONTRACT"} <= f


def test_quaternion_convention_fails_observation():
  assert "OBS_CONTRACT" in failed(bundle(contract_kw={"quaternion": "wxyz"}))


def test_rot6d_row_column_swap_fails_observation():
  c = contract_for(S.quadruped_conventions())
  terms = tuple(t.model_copy(update={"rot6d_layout": "rows"}) if t.source == "rot6d" else t
                for t in c.observation_terms)
  assert "OBS_CONTRACT" in failed(bundle(contract_kw={"observation_terms": terms}))


def test_double_normalization_fails_normalization():
  conv = S.quadruped_conventions()
  dim = sum(t["dim"] * t.get("history", 1) for t in conv.terms)
  mean, std = np.zeros(dim, np.float32), np.ones(dim, np.float32)
  exe = mlp_onnx(list(layers(dim)), out_clip=conv.action_clip, obs_mean=mean, obs_std=std)
  norm = {"mean": mean.tolist(), "var": (std**2).tolist()}
  b = bundle(contract_kw={"normalization": "external"}, policy_exe=exe, normalizer=norm)
  assert "NORMALIZATION" in failed(b)


def test_swapped_history_order_fails_observation():
  assert "OBS_CONTRACT" in failed(bundle(contract_kw={"history_order": "newest_first"}))


def test_unsupported_xml_feature_fails_xml_abi():
  xml = F.quadruped().xml.replace(b'<option timestep="0.004"/>',
                                  b'<option timestep="0.004"><flag sleep="enable"/></option>')
  assert "XML_ABI_MISMATCH" in failed(bundle(xml=xml))


def test_torque_vs_position_mismatch_fails_actuator():
  c = contract_for(S.quadruped_conventions())
  act = c.actuator.model_copy(update={"kind": "torque"})
  assert "ACTUATOR" in failed(bundle(contract_kw={"actuator": act}))


def test_missing_ema_fails_actuator():
  conv = S.perturb_conventions(S.quadruped_conventions(), ema_alpha=0.5)
  assert "ACTUATOR" in failed(bundle(conv, contract_kw={"ema_alpha": 1.0}))


def test_decimation_error_fails_control_rate():
  assert "CONTROL_RATE" in failed(bundle(group_kw={"decimation": 5}))


def test_inertia_scale_fails_inertial():
  spec = mujoco.MjSpec.from_string(F.quadruped().xml.decode())
  for g in spec.geoms:
    if g.mass:
      g.mass = g.mass * 1.1
  assert "INERTIAL" in failed(bundle(xml=spec.to_xml().encode()))


def test_altered_contact_response_fails_contact():
  xml = F.quadruped().xml.replace(b'friction="0.8 0.02 0.01"', b'friction="0.5 0.02 0.01"')
  assert "CONTACT" in failed(bundle(xml=xml))
