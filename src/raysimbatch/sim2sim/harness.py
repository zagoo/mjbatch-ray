# SPDX-License-Identifier: Apache-2.0
"""AssetAlignmentHarness, contact fitting, diagnostic axes and transfer scorecards
(design §5.4, §6.7).

Diagnostics are paired one-axis experiments: the aligned baseline plus one named
perturbation at a time (joint map, normalization, actuator, filter, timestep, inertia,
contact) with fixed seeds. Their paired effects are sensitivity measurements, not
proof of causal attribution for uncontrolled source/target differences, and the
scorecard says so.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from raysimbatch.config.models import AssetContract, ParityConfig
from raysimbatch.sim2sim.bundle import GateReport, TransferBundle
from raysimbatch.sim2sim.gates import run_gates
from raysimbatch.sim2sim.source import SourceExport, contact_fixtures

AXES = ("joint_map", "normalization", "actuator", "filter", "timestep", "inertia", "contact")


class AssetAlignmentHarness:
  def __init__(self, config: ParityConfig | None = None, registry=None):
    self.parity = config or ParityConfig()
    self.registry = registry

  async def ingest(self, source: SourceExport, canonical_xml: bytes, manifest: AssetContract, **bundle_kw) -> TransferBundle:
    """Bind a source export to a canonical asset; the contract facts must be attested by
    the caller (``bundle_kw``: group, contract, policy, executable, task_doc, ...)."""
    if manifest.target_version != mujoco.__version__:
      raise ValueError(f"XML_ABI_MISMATCH: asset targets MuJoCo {manifest.target_version}, engine is {mujoco.__version__}")
    missing = [k for k in ("group", "contract", "policy", "executable", "task_doc") if k not in bundle_kw]
    if missing:
      raise ValueError(f"transfer bundle is missing attested contract fields: {missing}")
    return TransferBundle(canonical_xml=canonical_xml, asset_contract=manifest, source=source, **bundle_kw)

  async def validate_static(self, bundle: TransferBundle) -> GateReport:
    return run_gates(bundle, self.parity, dynamics=False)

  async def replay_observations(self, bundle: TransferBundle, trace: SourceExport | None = None) -> GateReport:
    b = bundle if trace is None else TransferBundle(**{**bundle.__dict__, "source": trace})
    r = run_gates(b, self.parity, dynamics=False)
    r.results = [x for x in r.results if x.code in ("OBS_CONTRACT", "STATE_LEAK", "POLICY_EXPORT", "NORMALIZATION")]
    return r

  async def validate_dynamics(self, bundle: TransferBundle, fixtures: Sequence[Any] = ()) -> GateReport:
    return run_gates(bundle, self.parity, static=False)

  async def scorecard(self, results: dict[str, Any], gates: GateReport) -> dict[str, Any]:
    return build_scorecard(results, gates)


# --------------------------------------------------------------- contact fitting


@dataclass(frozen=True)
class ContactFit:
  friction: float
  timeconst: float
  objective: float
  holdout_slide_rel: float
  holdout_drop_err: float
  passed_holdout: bool
  grid_size: int


def fit_contact(source: SourceExport, friction0: float, solref0: Sequence[float], solimp: Sequence[float], dt: float,
                *, friction_span: float = 0.3, n_friction: int = 25, timeconsts: Sequence[float] = (0.01, 0.02, 0.04),
                parity: ParityConfig | None = None, holdout: dict[str, np.ndarray] | None = None) -> ContactFit:
  """Bounded deterministic grid search of (friction, solref timeconst) against the
  source fit fixtures, then frozen and validated on held-out fixtures. A failure is
  reported, never "fixed" by optimizing task reward."""
  p = parity or ParityConfig()
  arr = source.arrays
  grid = list(itertools.product(np.linspace(friction0 * (1 - friction_span), friction0 * (1 + friction_span), n_friction),
                                timeconsts))
  best = None
  for mu, tc in grid:
    f = contact_fixtures(np.array([mu, 0.02, 0.01]), np.array([tc, solref0[1]]), np.asarray(solimp), dt)
    so = float(arr["contact/slide_onset_force"][0])
    obj = abs(float(f["contact/slide_onset_force"][0]) - so) / max(so, 1e-9) + float(
      np.max(np.abs(f["contact/drop_z"] - arr["contact/drop_z"])) / 0.27)
    if best is None or obj < best[0] - 1e-15:
      best = (obj, mu, tc)
  assert best is not None, "empty calibration grid"
  obj, mu, tc = best
  ho = holdout or {}
  hs = hd = 0.0
  if ho:
    f = contact_fixtures(np.array([mu, 0.02, 0.01]), np.array([tc, solref0[1]]), np.asarray(solimp), dt)
    hs = abs(float(f["contact/slide_onset_force"][0]) - float(ho["contact/slide_onset_force"][0])) / max(
      float(ho["contact/slide_onset_force"][0]), 1e-9)
    hd = float(np.max(np.abs(f["contact/drop_z"] - ho["contact/drop_z"])) / 0.27)
  return ContactFit(float(mu), float(tc), float(obj), hs, hd,
                    hs <= p.slide_rtol and hd <= p.contact_curve_rtol, len(grid))


# ---------------------------------------------------------- diagnostic variants


def axis_variant(axis: str, xml: bytes, group, contract):
  """The registered fixed perturbation manifest of one diagnostic axis."""
  if axis == "inertia":
    spec = mujoco.MjSpec.from_string(xml.decode())
    for g in spec.geoms:
      if g.mass:
        g.mass = g.mass * 1.10
    return spec.to_xml().encode(), group, contract
  if axis == "contact":
    return xml.replace(b'friction="0.8 0.02 0.01"', b'friction="0.56 0.02 0.01"'), group, contract
  if axis == "timestep":
    return xml, group.model_copy(update={"timestep_ns": group.timestep_ns // 2, "decimation": group.decimation * 2}), contract
  if axis == "actuator":
    act = contract.actuator
    return xml, group, contract.model_copy(update={"actuator": act.model_copy(update={"kp": tuple(k * 0.8 for k in act.kp)})})
  if axis == "filter":
    return xml, group, contract.model_copy(update={"ema_alpha": 0.5})
  if axis == "joint_map":
    names = list(contract.joint_names)
    signs = list(contract.joint_signs)
    signs[0] = -signs[0]
    terms = []
    for t in contract.observation_terms:
      if t.source in ("joint_pos", "joint_vel"):
        ts = [signs[names.index(n)] for n in t.selectors]
        t = t.model_copy(update={"signs": tuple(ts)})
      terms.append(t)
    return xml, group, contract.model_copy(update={"joint_signs": tuple(signs), "observation_terms": tuple(terms)})
  if axis == "normalization":
    terms = tuple(t.model_copy(update={"scale": t.scale * 1.25}) if t.source == "joint_vel" else t
                  for t in contract.observation_terms)
    return xml, group, contract.model_copy(update={"observation_terms": terms})
  raise ValueError(f"axis {axis!r} has no registered perturbation manifest")


# ----------------------------------------------------------------- scorecards


def build_scorecard(results: dict[str, Any], gates: GateReport, *, source_success: dict[str, Any] | None = None,
                    minimum_paired_coverage: float = 0.95, scoring=None) -> dict[str, Any]:
  """results: the aggregate JSON document of the evaluation (aggregates + paired)."""
  aggs = results["aggregates"]
  base = {a["metric"]: a for a in aggs if a["experiment_axis"] == ""}
  axes = {}
  for a in aggs:
    if a["experiment_axis"] and a["metric"] in ("success", "return"):
      axes.setdefault(a["experiment_axis"], {})[a["metric"]] = a
  diag = []
  for axis, ms in sorted(axes.items()):
    row = {"axis": axis}
    for m, a in ms.items():
      b = base.get(m)
      row[f"{m}_effect"] = None if (a["mean"] is None or b is None or b["mean"] is None) else a["mean"] - b["mean"]
    pair = next((p for p in results.get("paired", []) if p.get("axis") == axis and p["metric"] == "success"), None)
    if pair:
      row["success_effect_ci"] = [pair["ci_low"], pair["ci_high"]]
      row["paired_coverage"] = pair["paired_coverage"]
    diag.append(row)
  tgt = base.get("success", {})
  src = (source_success or {}).get("rate")
  ratio = None
  if src and src > 0 and tgt.get("success_rate_conservative") is not None:
    ratio = tgt["success_rate_conservative"] / src
  coverage = tgt.get("coverage", 0.0)
  static_ok = all(r.passed for r in gates.results if r.code in ("XML_ABI_MISMATCH", "JOINT_MAPPING", "OBS_CONTRACT",
                                                                  "NORMALIZATION", "STATE_LEAK", "POLICY_EXPORT",
                                                                  "CONTROL_RATE", "ACTUATOR"))
  physical_ok = all(r.passed for r in gates.results if r.code in ("INERTIAL", "ASSET_FRAME", "CONTACT",
                                                                    "PHYSICS_CONFIG"))
  threshold_ok = None
  if scoring is not None and tgt.get("success_rate_conservative") is not None:
    threshold_ok = tgt["success_rate_conservative"] >= scoring.success_threshold and coverage >= scoring.minimum_valid_coverage
  qualified = bool(static_ok and physical_ok and coverage >= minimum_paired_coverage and threshold_ok is not False
                   and gates.passed)
  return {
    "schema": "rsb-transfer-scorecard/1",
    "bundle_digest": gates.bundle_digest,
    "gates": {r.code: {"passed": r.passed, "message": r.message} for r in gates.results},
    "target": {"success": tgt.get("mean"), "success_ci": [tgt.get("ci", {}).get("low") if isinstance(tgt.get("ci"), dict)
                                                          else tgt.get("ci_low"), tgt.get("ci_high")],
               "success_conservative": tgt.get("success_rate_conservative"), "coverage": coverage,
               "return": base.get("return", {}).get("mean"),
               "return_ci": [base.get("return", {}).get("ci_low"), base.get("return", {}).get("ci_high")],
               "tracking_rmse": base.get("tracking_rmse", {}).get("mean")},
    "source": source_success,
    "conservative_transfer_ratio": ratio,
    "diagnostics": diag,
    "release_qualified": qualified,
    "notes": [
      "Diagnostic axis effects are paired sensitivity measurements with fixed seeds; observed correlation "
      "or ablation sensitivity is not proof of causal attribution for uncontrolled source/target differences.",
      "CPU MuJoCo is a second physics implementation for evaluation, not physical ground truth.",
    ],
  }


def scorecard_markdown(card: dict[str, Any]) -> str:
  lines = [f"# Transfer scorecard `{card['bundle_digest'][:16]}`", "",
           f"**Release qualified:** {'yes' if card['release_qualified'] else 'NO'}", "", "## Gates", "",
           "| gate | result | detail |", "|---|---|---|"]
  for code, g in card["gates"].items():
    lines.append(f"| {code} | {'pass' if g['passed'] else 'FAIL'} | {g['message'] or ''} |")
  t = card["target"]
  lines += ["", "## Outcomes", "", f"- Target success (seed-cluster mean): {t['success']} CI {t['success_ci']}",
            f"- Conservative target success: {t['success_conservative']}; coverage {t['coverage']}",
            f"- Source success: {json.dumps(card['source'])}",
            f"- Conservative transfer ratio: {card['conservative_transfer_ratio']}", "", "## Diagnostic axes", "",
            "| axis | success effect | return effect | paired coverage |", "|---|---:|---:|---:|"]
  for d in card["diagnostics"]:
    lines.append(f"| {d['axis']} | {_f(d.get('success_effect'))} | {_f(d.get('return_effect'))} | "
                 f"{_f(d.get('paired_coverage'))} |")
  lines += [""] + [f"> {n}" for n in card["notes"]]
  return "\n".join(lines) + "\n"


def _f(v) -> str:
  return "null" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4g}"
