# SPDX-License-Identifier: Apache-2.0
"""Reset-time domain randomization over MJBatch ``expand`` rows (design §6.2).

* Only whitelisted fields are randomized; topology, assets, actuator types and
  solver workspace growth need a new compiled variant.
* Every draw starts from the template baseline: multiplicative randomization never
  compounds across episodes.
* Draws are counter-based (``domain`` stream keyed per episode), recorded, and hashed.
* The ``set_const`` closure is prewarmed across distribution extrema at construction
  (upstream finding M14); at reset a ``ClosureGuard`` checks, on a private scratch
  model, that the drawn parameters change no derived field outside the prewarmed
  closure. A draw that would grow the closure is refused (RANDOMIZATION_FAILED)
  instead of letting MJBatch recompute every in-flight simulation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mujoco
import numpy as np

from raysimbatch import seeds
from raysimbatch.config.models import RandomizationRule

_ASSET_PREFIX = ("mesh_", "hfield_", "tex_", "skin_", "bvh_", "oct_")


@dataclass(frozen=True)
class FieldPolicy:
  set_const: bool
  positive: bool
  option: bool = False


# Whitelist: field -> (needs set_const refresh, must stay strictly positive).
WHITELIST: dict[str, FieldPolicy] = {
  "body_mass": FieldPolicy(True, True),
  "body_inertia": FieldPolicy(True, True),
  "body_ipos": FieldPolicy(True, False),
  "geom_friction": FieldPolicy(False, False),
  "geom_solref": FieldPolicy(False, False),
  "geom_solimp": FieldPolicy(False, False),
  "geom_margin": FieldPolicy(False, False),
  "dof_damping": FieldPolicy(False, False),
  "dof_armature": FieldPolicy(True, False),
  "dof_frictionloss": FieldPolicy(False, False),
  "jnt_stiffness": FieldPolicy(False, False),
  "actuator_gainprm": FieldPolicy(False, False),
  "actuator_biasprm": FieldPolicy(False, False),
  "actuator_gear": FieldPolicy(True, False),
  "gravity": FieldPolicy(False, False, option=True),
  "wind": FieldPolicy(False, False, option=True),
  "density": FieldPolicy(False, False, option=True),
  "viscosity": FieldPolicy(False, False, option=True),
}


class RandomizationError(ValueError):
  pass


def _model_array(m: mujoco.MjModel, name: str) -> np.ndarray:
  pol = WHITELIST[name]
  return np.asarray(getattr(m.opt, name) if pol.option else getattr(m, name))


def _set_model_array(m: mujoco.MjModel, name: str, value: np.ndarray) -> None:
  target = getattr(m.opt, name) if WHITELIST[name].option else getattr(m, name)
  if isinstance(target, np.ndarray):
    target[...] = value.reshape(target.shape)
  else:  # scalar option
    setattr(m.opt, name, float(value.reshape(-1)[0]))


@dataclass(frozen=True)
class CompiledRule:
  rule: RandomizationRule
  flat_indices: np.ndarray  # into the per-sim flattened field row


def compile_rules(rules: tuple[RandomizationRule, ...], model: mujoco.MjModel) -> list[CompiledRule]:
  out = []
  for i, r in enumerate(rules):
    if r.field not in WHITELIST:
      raise RandomizationError(f"randomization.{i}: {r.field!r} is not a whitelisted reset-time field")
    base = _model_array(model, r.field)
    size = int(np.prod(base.shape)) if base.ndim else 1
    if r.indices:
      idx = np.asarray(r.indices, np.int64)
      if np.any(idx >= size):
        raise RandomizationError(f"randomization.{i}: index out of range for {r.field} (size {size})")
    elif r.field == "body_mass":
      idx = np.arange(1, size)  # every valid entry excluding the world body
    elif r.field == "body_inertia":
      idx = np.arange(3, size)
    else:
      idx = np.arange(size)
    if WHITELIST[r.field].positive:
      lows = [r.low, r.high, r.mean] if r.distribution != "categorical" else list(r.values)
      if r.operation == "replace" and min(lows) <= 0:
        raise RandomizationError(f"randomization.{i}: {r.field} must stay positive")
      if r.operation == "multiply" and r.distribution != "fixed" and r.distribution != "categorical" and r.low <= 0:
        raise RandomizationError(f"randomization.{i}: {r.field} multipliers must be > 0")
    out.append(CompiledRule(r, idx))
  return out


def draw_values(rule: RandomizationRule, u: np.ndarray, z: np.ndarray) -> np.ndarray:
  """Map uniform/normal draws [n, k] to parameter values."""
  d = rule.distribution
  if d == "fixed":
    return np.full(u.shape, rule.mean)
  if d == "uniform":
    return rule.low + (rule.high - rule.low) * u
  if d == "log_uniform":
    return np.exp(np.log(rule.low) + (np.log(rule.high) - np.log(rule.low)) * u)
  if d == "normal_clipped":
    return np.clip(rule.mean + rule.std * z, rule.low, rule.high)
  if d == "categorical":
    cdf = np.cumsum(rule.probabilities)
    pick = np.minimum(np.searchsorted(cdf, u, side="right"), len(rule.values) - 1)
    return np.asarray(rule.values)[pick]
  raise RandomizationError(d)


def extremes(rule: RandomizationRule) -> tuple[float, float]:
  if rule.distribution == "fixed":
    return rule.mean, rule.mean
  if rule.distribution == "categorical":
    return min(rule.values), max(rule.values)
  return rule.low, rule.high


def derived_fields(template: mujoco.MjModel) -> list[str]:
  """Every non-asset mjModel array (the candidates mj_setConst may rewrite)."""
  names = []
  for name in dir(template):
    if name.startswith("_") or name.startswith(_ASSET_PREFIX) or name in ("opt", "stat", "vis"):
      continue
    try:
      v = getattr(template, name)
    except Exception:  # noqa: BLE001 - some properties raise on empty models
      continue
    if isinstance(v, np.ndarray) and v.dtype != object and v.size:
      names.append(name)
  return names


class ClosureGuard:
  """Computes, on a private scratch model, which derived fields a parameter set changes."""

  def __init__(self, template: mujoco.MjModel, rules: list[CompiledRule]):
    self.template = template
    self.scratch = _copy(template)
    self.data = mujoco.MjData(self.scratch)
    self.rules = rules
    self.fields = [f for f in derived_fields(template) if f not in {r.rule.field for r in rules}]
    self.baseline = {f: np.array(getattr(template, f), copy=True) for f in self.fields}
    self.closure: frozenset[str] = frozenset()

  def changed(self, values: dict[str, np.ndarray]) -> set[str]:
    m = self.scratch
    for f, base in self.baseline.items():
      getattr(m, f)[...] = base
    for r in self.rules:
      _set_model_array(m, r.rule.field, _model_array(self.template, r.rule.field))
    for name, v in values.items():
      _set_model_array(m, name, v)
    mujoco.mj_setConst(m, self.data)
    # Bytewise, like MJBatch's memcmp (NaN-safe, sign-of-zero exact).
    return {f for f, base in self.baseline.items() if getattr(m, f).tobytes() != base.tobytes()}

  def prewarm(self, samples: list[dict[str, np.ndarray]]) -> frozenset[str]:
    out: set[str] = set()
    for s in samples:
      out |= self.changed(s)
    self.closure = frozenset(out)
    return self.closure


def _copy(m: mujoco.MjModel) -> mujoco.MjModel:
  import copy

  return copy.copy(m)


class Randomizer:
  """Applies randomization rules to selected slots of a BatchHandle."""

  def __init__(self, handle, template: mujoco.MjModel, rules: tuple[RandomizationRule, ...], task_hash: str,
               run_seed: int):
    self.handle = handle
    self.template = template
    self.rules = compile_rules(rules, template)
    self.task_hash = task_hash
    self.run_seed = run_seed
    self.fields = sorted({r.rule.field for r in self.rules})
    self.rows = {f: handle.expand(f) for f in self.fields}
    self.baseline = {f: np.array(_model_array(template, f), dtype=np.float64, copy=True) for f in self.fields}
    self.needs_refresh = any(WHITELIST[f].set_const for f in self.fields) or any(
      r.rule.refresh == "always" for r in self.rules
    )
    self.guard: ClosureGuard | None = None
    self.closure: frozenset[str] = frozenset()
    if self.needs_refresh:
      self.guard = ClosureGuard(template, self.rules)
      self.closure = self.guard.prewarm(self._extreme_samples())
      for f in sorted(self.closure):
        handle.expand(f)  # seeded from the template; per-sim once set_const runs

  def _extreme_samples(self) -> list[dict[str, np.ndarray]]:
    samples = []
    for pick in (0, 1):
      vals = {f: self.baseline[f].copy() for f in self.fields}
      for r in self.rules:
        v = extremes(r.rule)[pick]
        flat = vals[r.rule.field].reshape(-1)
        flat[r.flat_indices] = flat[r.flat_indices] * v if r.rule.operation == "multiply" else v
      samples.append(vals)
    # A midpoint sample catches fields that change only strictly inside the range.
    mid = {f: self.baseline[f].copy() for f in self.fields}
    for r in self.rules:
      lo, hi = extremes(r.rule)
      flat = mid[r.rule.field].reshape(-1)
      v = 0.5 * (lo + hi)
      flat[r.flat_indices] = flat[r.flat_indices] * v if r.rule.operation == "multiply" else v
    samples.append(mid)
    return samples

  def draw(self, keys: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Per-slot values for every randomized field ([n, *shape]), randomization hashes
    ([n, 32] uint8), and the raw draws per rule (for the reproducibility manifest)."""
    n = keys.shape[0]
    values = {f: np.broadcast_to(self.baseline[f], (n, *self.baseline[f].shape)).copy() for f in self.fields}
    raw: dict[str, np.ndarray] = {}
    hashes = np.zeros((n, 32), np.uint8)
    hs = [hashlib.sha256() for _ in range(n)]
    for i, r in enumerate(self.rules):
      k = r.flat_indices.size
      u = seeds.uniform(keys, i, k, "domain")
      z = seeds.normal(keys, i, k, "domain", index0=1 << 32)
      v = draw_values(r.rule, u, z)
      raw[f"{i}:{r.rule.field}"] = v
      flat = values[r.rule.field].reshape(n, -1)
      if r.rule.operation == "multiply":
        flat[:, r.flat_indices] = flat[:, r.flat_indices] * v
      else:
        flat[:, r.flat_indices] = v
      for j in range(n):
        hs[j].update(i.to_bytes(4, "little"))
        hs[j].update(v[j].tobytes())
    for j in range(n):
      hashes[j] = np.frombuffer(hs[j].digest(), np.uint8)
    return values, hashes, raw

  def validate(self, values: dict[str, np.ndarray]) -> np.ndarray:
    """Per-slot admissibility: finite, positive where required, closure-preserving."""
    n = next(iter(values.values())).shape[0] if values else 0
    ok = np.ones(n, bool)
    for f, v in values.items():
      flat = v.reshape(n, -1)
      ok &= np.isfinite(flat).all(axis=1)
      if WHITELIST[f].positive:
        base = self.baseline[f].reshape(-1)
        ok &= ~np.any((flat <= 0) & (base > 0), axis=1)
    if self.guard is not None:
      for j in np.flatnonzero(ok):
        grown = self.guard.changed({f: values[f][j] for f in self.fields}) - self.closure
        if grown:
          ok[j] = False
    return ok

  def write(self, slots: np.ndarray, values: dict[str, np.ndarray]) -> None:
    for f, v in values.items():
      row = self.rows[f]
      row[slots] = v.reshape(row[slots].shape)

  def restore_baseline(self, slots: np.ndarray) -> None:
    for f in self.fields:
      self.rows[f][slots] = self.baseline[f]
