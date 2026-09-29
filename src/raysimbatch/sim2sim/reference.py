# SPDX-License-Identifier: Apache-2.0
"""Independent numpy reference of the policy-side pipeline (design §6.7 steps 1-8).

This is deliberately *not* the native kernel code: gates compare the compiled
adapter against it and against source-engine exports, so an error in one
implementation is not silently shared by the other. Conventions follow Isaac Lab's
observation manager (noise -> clip -> scale -> history, oldest first, term-major
flatten) unless the contract says otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
  q = np.asarray(q, np.float64)
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  w, x, y, z = np.moveaxis(q, -1, 0)
  return np.stack([
    np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
    np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
    np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
  ], -2)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
  return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
  return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def rot_inv_apply(R: np.ndarray, v: np.ndarray) -> np.ndarray:
  return np.einsum("...ji,...j->...i", R, v)


def rot_apply(R: np.ndarray, v: np.ndarray) -> np.ndarray:
  return np.einsum("...ij,...j->...i", R, v)


def term_value(term: Mapping, state: Mapping[str, np.ndarray]) -> np.ndarray:
  """Raw value of one observation term from named state (source conventions already
  converted to the contract's: quat wxyz, velocities in declared frames)."""
  src = term["source"]

  def rot() -> np.ndarray:  # only the frame-dependent sources need the base orientation
    if "base_quat_wxyz" not in state:
      raise ValueError(f"source {src!r} needs base_quat_wxyz in the state")
    return quat_wxyz_to_mat(state["base_quat_wxyz"])

  if src == "joint_pos":
    return np.asarray(state["joint_pos"], np.float64)[..., term["index"]] - np.asarray(term["offsets"])
  if src == "joint_vel":
    return np.asarray(state["joint_vel"], np.float64)[..., term["index"]] - np.asarray(term.get("offsets", 0.0))
  if src == "gravity":
    R = rot()
    return rot_inv_apply(R, np.broadcast_to([0.0, 0.0, -1.0], R.shape[:-1]))
  if src == "base_lin_vel":
    v = state["base_lin_vel_w"]
    return rot_inv_apply(rot(), v) if term.get("frame", "body") == "body" else np.asarray(v)
  if src == "base_ang_vel":
    w = state["base_ang_vel_b"]
    return np.asarray(w) if term.get("frame", "body") == "body" else rot_apply(rot(), w)
  if src == "command":
    return np.asarray(state["command"])[..., term["index"]]
  if src == "previous_action":
    return np.asarray(state["prev_action"], np.float64)[..., term["index"]]
  if src == "rot6d":
    R = rot()
    if term.get("layout", "columns") == "columns":
      return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)
  if src == "base_quat":
    q = np.asarray(state["base_quat_wxyz"], np.float64)
    return wxyz_to_xyzw(q) if term.get("quaternion", "wxyz") == "xyzw" else q
  if src == "phase":
    t = np.asarray(state["time_s"], np.float64)
    ph = np.fmod(t, term["period_s"]) / term["period_s"]
    return np.stack([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], -1)
  raise ValueError(f"reference does not implement source {src!r}")


def process_term(x: np.ndarray, term: Mapping, noise: np.ndarray | None = None,
                 order: str = "noise_clip_scale") -> np.ndarray:
  clip = term.get("clip") or np.inf
  n = 0.0 if noise is None else noise
  if order == "noise_clip_scale":
    return np.clip(x + n, -clip, clip) * term.get("scale", 1.0)
  return np.clip(x * term.get("scale", 1.0), -clip, clip) + n


class History:
  """Per-term circular history, oldest first, Isaac Lab style."""

  def __init__(self, terms: Sequence[Mapping], reset: str = "repeat_initial"):
    self.terms = list(terms)
    self.reset_mode = reset
    self.buf: list[np.ndarray | None] = [None] * len(self.terms)

  def reset(self) -> None:
    self.buf = [None] * len(self.terms)

  def push(self, values: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = []
    for i, (t, v) in enumerate(zip(self.terms, values, strict=True)):
      H = int(t.get("history", 1))
      b = self.buf[i]
      if b is None:
        b = np.repeat(v[None], H, 0) if self.reset_mode == "repeat_initial" else np.zeros((H, *v.shape))
        if self.reset_mode == "zeros":
          b[-1] = v
      else:
        b = np.concatenate([b[1:], v[None]], 0)
      self.buf[i] = b
      out.append(b.copy())
    return out


def flatten(hist: Sequence[np.ndarray], layout: str = "term_major", order: str = "oldest_first") -> np.ndarray:
  hs = [h if order == "oldest_first" else h[::-1] for h in hist]
  if layout == "term_major":
    return np.concatenate([h.reshape(-1) for h in hs])
  H = hs[0].shape[0]
  return np.concatenate([np.concatenate([h[k] for h in hs]) for k in range(H)])


def process_action(a: np.ndarray, *, clip: float, scale: np.ndarray, offset: np.ndarray, sign: np.ndarray,
                   order: str = "clip_scale_offset") -> np.ndarray:
  a = np.asarray(a, np.float64)
  if order == "clip_scale_offset":
    x = np.clip(a, -clip, clip) * scale + offset
  else:
    x = np.clip(a * scale, -clip, clip) + offset
  return x * sign


def ema(x: np.ndarray, prev: np.ndarray, alpha: float) -> np.ndarray:
  return alpha * x + (1.0 - alpha) * prev


def pd_effort(target, q, dq, kp, kd, limit, tau_ff=0.0) -> np.ndarray:
  tau = kp * (target - q) - kd * dq + tau_ff
  lim = np.where(np.asarray(limit) > 0, limit, np.inf)
  return np.clip(tau, -lim, lim)


def dc_motor_effort(target, q, dq, kp, kd, limit, speed, motor, brake, tau_ff=0.0) -> np.ndarray:
  tau = kp * (target - q) - kd * dq + tau_ff
  out = np.empty_like(np.asarray(tau, np.float64))
  for j in range(out.shape[-1]):
    motoring = tau[..., j] * dq[..., j] >= 0
    env_m = np.interp(np.abs(dq[..., j]), speed[j], motor[j])
    env_b = np.interp(np.abs(dq[..., j]), speed[j], brake[j])
    lim = np.minimum(np.where(motoring, env_m, env_b), limit[j] if limit[j] > 0 else np.inf)
    out[..., j] = np.clip(tau[..., j], -lim, lim)
  return out


def mlp_forward(layers: Sequence[tuple[np.ndarray, np.ndarray]], x: np.ndarray, clip: float | None = None) -> np.ndarray:
  """float64 forward of an MLP with tanh hidden activations (the 'source' policy)."""
  h = np.asarray(x, np.float64)
  for i, (W, b) in enumerate(layers):
    h = h @ np.asarray(W, np.float64) + np.asarray(b, np.float64)
    if i < len(layers) - 1:
      h = np.tanh(h)
  return np.clip(h, -clip, clip) if clip is not None else h
