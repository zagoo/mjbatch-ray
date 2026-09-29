// SPDX-License-Identifier: Apache-2.0
//
// Compiled per-slot kernels that run between MJBatch calls, on the calling thread,
// while the MJBatch pool is parked. They read and write only public MJBatch bind()
// buffers and RaySimBatch-owned state arrays. Nothing here links MuJoCo: the ABI
// coupling of RaySimBatch to the engine stays inside MJBatch.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "arrays.h"
#include "philox.h"

namespace rsb {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Fault codes; mirrored by raysimbatch.protocol.FaultCode.
enum Fault : uint16_t {
  kFaultNone = 0,
  kFaultNonfinite = 1,
  kFaultVelocity = 2,
  kFaultAcceleration = 3,
  kFaultQuaternion = 4,
  kFaultFatalWarning = 5,
  kFaultTimeDiscontinuity = 6,
};

// ---------------------------------------------------------------- rotation helpers

// Row-major rotation matrix of a unit (w, x, y, z) quaternion, normalized here.
inline void QuatToMat(const double* q, double R[9]) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n > 0) {
    w /= n;
    x /= n;
    y /= n;
    z /= n;
  }
  R[0] = 1 - 2 * (y * y + z * z);
  R[1] = 2 * (x * y - w * z);
  R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z);
  R[4] = 1 - 2 * (x * x + z * z);
  R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y);
  R[7] = 2 * (y * z + w * x);
  R[8] = 1 - 2 * (x * x + y * y);
}

inline void MulT(const double R[9], const double* v, double out[3]) {  // R^T v
  out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
  out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
  out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
}

inline void Mul(const double R[9], const double* v, double out[3]) {  // R v
  out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

inline double Clamp(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

// Piecewise-linear interpolation over ascending breakpoints, clamped at both ends.
inline double Interp(double x, const double* xs, const double* ys, int64_t k) {
  if (k <= 0) return kInf;
  if (x <= xs[0]) return ys[0];
  for (int64_t i = 1; i < k; ++i) {
    if (x <= xs[i]) {
      const double t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[k - 1];
}

// ------------------------------------------------------------------ action processing

// Policy action -> joint targets, in the contract's declared order:
// clip/scale/offset (or scale/clip/offset), sign map, EMA at policy rate, then a
// policy-clock delay. Physics-clock delay is applied by the Controller.
class ActionProcessor {
 public:
  ActionProcessor(const nb::dict& cfg, const nb::dict& arrays) {
    A_ = Scalar<int64_t>(cfg, "num_actions");
    order_ = Scalar<int>(cfg, "order");  // 0 clip_scale_offset, 1 scale_clip_offset
    clip_ = Scalar<double>(cfg, "clip");
    alpha_ = Scalar<double>(cfg, "ema_alpha");
    ema_initial_ = Scalar<double>(cfg, "ema_initial");
    delay_ = Scalar<int64_t>(cfg, "policy_delay_ticks");
    scale_ = Get<const double>(arrays, "scale", A_, 1);
    offset_ = Get<const double>(arrays, "offset", A_, 1);
    sign_ = Get<const double>(arrays, "sign", A_, 1);
    prev_ = Get<float>(arrays, "prev_action", -1, A_);
    N_ = prev_.rows;
    ema_ = Get<double>(arrays, "ema", N_, A_);
    buf_ = Get<double>(arrays, "delay_buffer", N_, (delay_ + 1) * A_);
    count_ = Get<int64_t>(arrays, "delay_count", N_, 1);
    targets_ = Get<double>(arrays, "targets", N_, A_);
  }

  void Reset(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      std::fill(prev_.Row(s), prev_.Row(s) + A_, 0.0f);
      std::fill(ema_.Row(s), ema_.Row(s) + A_, ema_initial_);
      std::fill(buf_.Row(s), buf_.Row(s) + buf_.cols, 0.0);
      count_.Row(s)[0] = 0;
    }
  }

  void Process(const Slots& slots,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> actions) {
    CheckSlots(slots, N_);
    if (actions.shape(0) != slots.shape(0) || static_cast<int64_t>(actions.shape(1)) != A_) {
      throw nb::value_error("actions must have shape (len(slots), num_actions)");
    }
    nb::gil_scoped_release release;
    const int64_t size = delay_ + 1;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      const float* a = actions.data() + j * A_;
      float* prev = prev_.Row(s);
      double* ema = ema_.Row(s);
      double* buf = buf_.Row(s);
      int64_t& count = count_.Row(s)[0];
      double* out = targets_.Row(s);
      const int64_t pos = count % size;
      for (int64_t i = 0; i < A_; ++i) {
        prev[i] = a[i];
        double x = a[i];
        if (order_ == 0) {
          x = Clamp(x, -clip_, clip_) * scale_.p[i] + offset_.p[i];
        } else {
          x = Clamp(x * scale_.p[i], -clip_, clip_) + offset_.p[i];
        }
        x *= sign_.p[i];
        ema[i] = alpha_ * x + (1.0 - alpha_) * ema[i];
        buf[pos * A_ + i] = ema[i];
      }
      ++count;
      const int64_t src = count <= delay_ ? 0 : (count - 1 - delay_) % size;
      for (int64_t i = 0; i < A_; ++i) out[i] = buf[src * A_ + i];
    }
  }

 private:
  int64_t A_ = 0, N_ = 0, delay_ = 0;
  int order_ = 0;
  double clip_ = 1, alpha_ = 1, ema_initial_ = 0;
  Mat<const double> scale_, offset_, sign_;
  Mat<float> prev_;
  Mat<double> ema_, buf_, targets_;
  Mat<int64_t> count_;
};

// ------------------------------------------------------------------ substep controller

// Actuator effort at every physics substep. kind: 0 torque (target is effort),
// 1 position (target is written to a native position actuator; no stiffness here),
// 2 pd (tau = kp (q* - q) + kd (0 - dq) + tau_ff), 3 dc_motor (pd, then a signed
// piecewise torque-speed envelope, motoring and braking evaluated separately).
// Efforts are clipped to effort_limit and divided by gear into ctrl.
class Controller {
 public:
  Controller(const nb::dict& cfg, const nb::dict& arrays) {
    A_ = Scalar<int64_t>(cfg, "num_actions");
    kind_ = Scalar<int>(cfg, "kind");
    tau_ff_ = Scalar<double>(cfg, "feedforward");
    delay_ = Scalar<int64_t>(cfg, "physics_delay_ticks");
    qadr_ = Get<const int32_t>(arrays, "qpos_adr", A_, 1);
    dadr_ = Get<const int32_t>(arrays, "dof_adr", A_, 1);
    cidx_ = Get<const int32_t>(arrays, "ctrl_idx", A_, 1);
    gear_ = Get<const double>(arrays, "gear", A_, 1);
    kp_ = Get<const double>(arrays, "kp", A_, 1);
    kd_ = Get<const double>(arrays, "kd", A_, 1);
    elim_ = Get<const double>(arrays, "effort_limit", A_, 1);
    env_speed_ = Get<const double>(arrays, "envelope_speed", A_, -1, false);
    env_motor_ = Get<const double>(arrays, "envelope_motor", A_, -1, false);
    env_brake_ = Get<const double>(arrays, "envelope_brake", A_, -1, false);
    if (kind_ == 3 && (!env_speed_.Valid() || !env_motor_.Valid() || !env_brake_.Valid())) {
      throw nb::value_error("dc_motor requires envelope_speed/motor/brake arrays");
    }
    qpos_ = Get<const double>(arrays, "qpos");
    N_ = qpos_.rows;
    qvel_ = Get<const double>(arrays, "qvel", N_);
    ctrl_ = Get<double>(arrays, "ctrl", N_);
    targets_ = Get<const double>(arrays, "targets", N_, A_);
    buf_ = Get<double>(arrays, "physics_delay_buffer", N_, (delay_ + 1) * A_);
    count_ = Get<int64_t>(arrays, "physics_delay_count", N_, 1);
    effort_ = Get<double>(arrays, "effort", N_, A_);
    saved_ = Get<double>(arrays, "ctrl_saved", N_, ctrl_.cols);
    sat_ = Get<int64_t>(arrays, "saturated", N_, 1);
    evals_ = Get<int64_t>(arrays, "effort_evals", N_, 1);
    energy_ = Get<double>(arrays, "energy", N_, 1);
    for (int64_t i = 0; i < A_; ++i) {
      if (qadr_.p[i] < 0 || qadr_.p[i] >= qpos_.cols || dadr_.p[i] < 0 || dadr_.p[i] >= qvel_.cols ||
          cidx_.p[i] < 0 || cidx_.p[i] >= ctrl_.cols) {
        throw nb::value_error("controller address out of range");
      }
    }
    if (Fusable()) {  // transient scratch, not slot state
      dq0_.assign(static_cast<size_t>(N_ * qvel_.cols), 0.0);
      clip_.assign(static_cast<size_t>(A_), 0.0);
    }
  }

  // ctrl is constant over a policy interval: torque or position targets, no physics-clock
  // delay (pd and dc_motor read the state at every substep).
  bool Fusable() const { return (kind_ == 0 || kind_ == 1) && delay_ == 0; }

  void Reset(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      std::fill(buf_.Row(s), buf_.Row(s) + buf_.cols, 0.0);
      std::fill(effort_.Row(s), effort_.Row(s) + A_, 0.0);
      count_.Row(s)[0] = 0;
      sat_.Row(s)[0] = 0;
      evals_.Row(s)[0] = 0;
      energy_.Row(s)[0] = 0.0;
    }
  }

  // Saves each selected ctrl row, then writes the new efforts/targets into it.
  void BeforeSubstep(const Slots& slots, double dt) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      Substep(s, qpos_.Row(s), qvel_.Row(s), dt, true);
    }
  }

  // Fused interval, part 1 (before one MJBatch step(nstep=d) call): writes the interval's
  // constant ctrl exactly as the first BeforeSubstep would, saves the previous ctrl row
  // (for Neutralize) and the velocities the first substep's accounting reads. Counters,
  // energy, the delay queue and efforts are left for FinishFused, so a slot that is
  // replayed substep by substep starts from untouched controller state.
  void PrepareFused(const Slots& slots) {
    CheckSlots(slots, N_);
    if (!Fusable()) throw nb::value_error("controller kind or physics delay is not fusable");
    nb::gil_scoped_release release;
    const int64_t nv = qvel_.cols;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      double* ctrl = ctrl_.Row(s);
      std::copy(ctrl, ctrl + ctrl_.cols, saved_.Row(s));
      std::copy(qvel_.Row(s), qvel_.Row(s) + nv, dq0_.data() + s * nv);
      const double* target = targets_.Row(s);  // no delay: the commanded targets
      for (int64_t i = 0; i < A_; ++i) {
        if (kind_ == 1) {
          ctrl[cidx_.p[i]] = target[i];
        } else {
          const double limit = elim_.p[i] > 0 ? elim_.p[i] : kInf;
          ctrl[cidx_.p[i]] = Clamp(target[i], -limit, limit) / gear_.p[i];
        }
      }
    }
  }

  // Fused interval, part 2, for slots that completed nstep substeps (the interval, or a
  // terminated slot's replay up to its terminal substep): replays each substep's
  // controller bookkeeping in the per-substep order (bitwise the same sums), with the
  // state each substep started from: the saved velocities for substep 0, then history
  // row k-1 (bind("state") layout) for substep k. The ctrl row saved for substep 0 is
  // PrepareFused's (the pre-interval one), as BeforeSubstep would have saved it.
  void FinishFused(const Slots& slots, const Slots& rows,
                   nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu> history,
                   int64_t qpos_off, int64_t qvel_off, double dt) {
    CheckSlots(slots, N_);
    if (!Fusable()) throw nb::value_error("controller kind or physics delay is not fusable");
    const int64_t nstep = static_cast<int64_t>(history.shape(1));
    const int64_t nstate = static_cast<int64_t>(history.shape(2));
    if (rows.shape(0) != slots.shape(0)) throw nb::value_error("rows length must equal len(slots)");
    if (nstep < 1) throw nb::value_error("nstep must be >= 1");
    if (qpos_off < 0 || qpos_off + qpos_.cols > nstate || qvel_off < 0 || qvel_off + qvel_.cols > nstate) {
      throw nb::value_error("state offsets out of range");
    }
    for (size_t j = 0; j < rows.shape(0); ++j) {
      if (rows.data()[j] < 0 || rows.data()[j] >= static_cast<int32_t>(history.shape(0))) {
        throw nb::index_error("history row out of range");
      }
    }
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      FinishSlot(slots.data()[j], history.data() + static_cast<int64_t>(rows.data()[j]) * nstep * nstate, nstep,
                 nstate, qpos_off, qvel_off, dt);
    }
  }

  // One slot's FinishFused from its history rows h (nstep rows of nstate): what nstep
  // Substep calls leave behind. For fusable kinds every substep writes the same ctrl,
  // effort and delay row and makes the same clipping decisions, so those are done once
  // and the counters advance by nstep; only the energy sum reads each substep's starting
  // velocities (the saved ones for substep 0, row k-1 for substep k), and it is added in
  // Substep's order (substep, then actuator), so the sum is bitwise the same.
  void FinishSlot(int64_t s, const double* h, int64_t nstep, int64_t nstate, int64_t /*qpos_off*/,
                  int64_t qvel_off, double dt) {
    double* ctrl = ctrl_.Row(s);
    const double* target = targets_.Row(s);  // no delay: the commanded targets
    double* effort = effort_.Row(s);
    if (nstep > 1) std::copy(ctrl, ctrl + ctrl_.cols, saved_.Row(s));  // saved by the last substep
    std::copy(target, target + A_, buf_.Row(s));
    count_.Row(s)[0] += nstep;
    if (kind_ == 1) {
      for (int64_t i = 0; i < A_; ++i) {
        ctrl[cidx_.p[i]] = target[i];
        effort[i] = std::numeric_limits<double>::quiet_NaN();
      }
      return;
    }
    int64_t clipped_count = 0;
    for (int64_t i = 0; i < A_; ++i) {
      const double limit = elim_.p[i] > 0 ? elim_.p[i] : kInf;
      const double clipped = Clamp(target[i], -limit, limit);
      if (clipped != target[i]) ++clipped_count;
      clip_[i] = clipped;
      effort[i] = clipped;
      ctrl[cidx_.p[i]] = clipped / gear_.p[i];
    }
    sat_.Row(s)[0] += clipped_count * nstep;
    evals_.Row(s)[0] += A_ * nstep;
    double& energy = energy_.Row(s)[0];
    for (int64_t k = 0; k < nstep; ++k) {
      const double* dq = k == 0 ? dq0_.data() + s * qvel_.cols : h + (k - 1) * nstate + qvel_off;
      for (int64_t i = 0; i < A_; ++i) energy += std::abs(clip_[i] * dq[dadr_.p[i]]) * dt;
    }
  }
  int64_t nq() const { return qpos_.cols; }
  int64_t nv() const { return qvel_.cols; }

  // Restores the rows saved by BeforeSubstep so a slot whose MJBatch call failed
  // carries no pending ctrl write (MJBatch copies in only elements that differ from
  // what it last copied out).
  void Neutralize(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      std::copy(saved_.Row(s), saved_.Row(s) + ctrl_.cols, ctrl_.Row(s));
    }
  }

 private:
  // One substep of slot s from positions q and velocities dq (full qpos/qvel rows);
  // `save` keeps the ctrl row it replaces for Neutralize.
  void Substep(int64_t s, const double* q, const double* dq, double dt, bool save) {
    const int64_t size = delay_ + 1;
    double* ctrl = ctrl_.Row(s);
    if (save) std::copy(ctrl, ctrl + ctrl_.cols, saved_.Row(s));
    const double* commanded = targets_.Row(s);
    double* buf = buf_.Row(s);
    int64_t& count = count_.Row(s)[0];
    const int64_t pos = count % size;
    std::copy(commanded, commanded + A_, buf + pos * A_);
    ++count;
    const double* target = buf + (count <= delay_ ? 0 : (count - 1 - delay_) % size) * A_;
    double* effort = effort_.Row(s);
    for (int64_t i = 0; i < A_; ++i) {
      const double qi = q[qadr_.p[i]];
      const double dqi = dq[dadr_.p[i]];
      if (kind_ == 1) {
        ctrl[cidx_.p[i]] = target[i];
        effort[i] = std::numeric_limits<double>::quiet_NaN();
        continue;
      }
      double tau = kind_ == 0 ? target[i] : kp_.p[i] * (target[i] - qi) - kd_.p[i] * dqi + tau_ff_;
      double limit = elim_.p[i] > 0 ? elim_.p[i] : kInf;
      if (kind_ == 3) {
        const bool motoring = tau * dqi >= 0.0;
        const int64_t k = env_speed_.cols;
        const double env = Interp(std::abs(dqi), env_speed_.Row(i),
                                  motoring ? env_motor_.Row(i) : env_brake_.Row(i), k);
        limit = std::min(limit, env);
      }
      const double clipped = Clamp(tau, -limit, limit);
      if (clipped != tau) ++sat_.Row(s)[0];
      ++evals_.Row(s)[0];
      effort[i] = clipped;
      energy_.Row(s)[0] += std::abs(clipped * dqi) * dt;
      ctrl[cidx_.p[i]] = clipped / gear_.p[i];
    }
  }

  int64_t A_ = 0, N_ = 0, delay_ = 0;
  int kind_ = 2;
  double tau_ff_ = 0;
  Mat<const int32_t> qadr_, dadr_, cidx_;
  Mat<const double> gear_, kp_, kd_, elim_, env_speed_, env_motor_, env_brake_;
  Mat<const double> qpos_, qvel_, targets_;
  Mat<double> ctrl_, buf_, effort_, saved_, energy_;
  Mat<int64_t> count_, sat_, evals_;
  std::vector<double> dq0_, clip_;
};

// ------------------------------------------------------------------------ health

class Health {
 public:
  Health(const nb::dict& cfg, const nb::dict& arrays) {
    max_qvel_ = Scalar<double>(cfg, "max_abs_qvel");
    max_qacc_ = Scalar<double>(cfg, "max_abs_qacc");
    quat_tol_ = Scalar<double>(cfg, "quaternion_norm_tolerance");
    time_rtol_ = Scalar<double>(cfg, "time_rtol");
    nwarn_ = Scalar<int64_t>(cfg, "num_warnings");
    quat_adr_ = Get<const int32_t>(arrays, "quat_adr", -1, 1);
    fatal_ = Get<const int32_t>(arrays, "fatal_warnings", -1, 1);
    qpos_ = Get<const double>(arrays, "qpos");
    N_ = qpos_.rows;
    qvel_ = Get<const double>(arrays, "qvel", N_);
    act_ = Get<const double>(arrays, "act", N_, -1, false);
    qacc_ = Get<const double>(arrays, "qacc", N_, -1, false);
    warning_ = Get<const int32_t>(arrays, "warning", N_, nwarn_ * 2);
    time_ = Get<const double>(arrays, "time", N_, 1);
    base_ = Get<int32_t>(arrays, "warning_base", N_, nwarn_);
    expected_ = Get<double>(arrays, "expected_time", N_, 1);
    for (int64_t i = 0; i < quat_adr_.rows; ++i) {
      if (quat_adr_.p[i] < 0 || quat_adr_.p[i] + 4 > qpos_.cols) throw nb::value_error("quat_adr");
    }
    for (int64_t i = 0; i < fatal_.rows; ++i) {
      if (fatal_.p[i] < 0 || fatal_.p[i] >= nwarn_) throw nb::value_error("fatal_warnings");
    }
  }

  // Record the counters and time a freshly reset/restored slot starts from.
  void Rebase(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      for (int64_t w = 0; w < nwarn_; ++w) base_.Row(s)[w] = warning_.Row(s)[2 * w + 1];
      expected_.Row(s)[0] = time_.Row(s)[0];
    }
  }

  // After a successful substep of length dt: fault code per slot (0 = healthy) and
  // the total warning-count increment. Healthy slots advance their baselines.
  void Check(const Slots& slots, double dt,
             nb::ndarray<uint16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> fault,
             nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> warn_delta) {
    CheckSlots(slots, N_);
    if (fault.shape(0) != slots.shape(0) || warn_delta.shape(0) != slots.shape(0)) {
      throw nb::value_error("output length must equal len(slots)");
    }
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      fault.data()[j] = CheckSlot(slots.data()[j], dt, &warn_delta.data()[j]);
    }
  }

  // One slot's check (Check's body): returns the fault code, stores the warning delta,
  // and advances the slot's baselines.
  uint16_t CheckSlot(int64_t s, double dt, int32_t* warn_delta) {
    {
      uint16_t code = kFaultNone;
      int32_t delta = 0;
      bool fatal = false;
      for (int64_t w = 0; w < nwarn_; ++w) {
        const int32_t d = warning_.Row(s)[2 * w + 1] - base_.Row(s)[w];
        delta += d;
        if (d > 0) {
          for (int64_t f = 0; f < fatal_.rows; ++f) fatal |= fatal_.p[f] == w;
        }
      }
      if (fatal) code = kFaultFatalWarning;
      if (!code && !AllFinite(s)) code = kFaultNonfinite;
      if (!code) {
        const double want = expected_.Row(s)[0] + dt;
        const double t = time_.Row(s)[0];
        if (std::abs(t - want) > time_rtol_ * std::max(1.0, std::abs(want))) code = kFaultTimeDiscontinuity;
      }
      if (!code) {
        const double* q = qpos_.Row(s);
        for (int64_t i = 0; i < quat_adr_.rows; ++i) {
          const double* x = q + quat_adr_.p[i];
          const double n = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
          if (std::abs(n - 1.0) > quat_tol_) code = kFaultQuaternion;
        }
      }
      if (!code) {
        const double* v = qvel_.Row(s);
        for (int64_t i = 0; i < qvel_.cols; ++i) {
          if (std::abs(v[i]) > max_qvel_) code = kFaultVelocity;
        }
      }
      if (!code && qacc_.Valid()) {
        const double* a = qacc_.Row(s);
        for (int64_t i = 0; i < qacc_.cols; ++i) {
          if (std::abs(a[i]) > max_qacc_) code = kFaultAcceleration;
        }
      }
      *warn_delta = delta;
      for (int64_t w = 0; w < nwarn_; ++w) base_.Row(s)[w] = warning_.Row(s)[2 * w + 1];
      expected_.Row(s)[0] = time_.Row(s)[0];
      return code;
    }
  }

 private:
  bool AllFinite(int64_t s) const {
    for (int64_t i = 0; i < qpos_.cols; ++i) {
      if (!std::isfinite(qpos_.Row(s)[i])) return false;
    }
    for (int64_t i = 0; i < qvel_.cols; ++i) {
      if (!std::isfinite(qvel_.Row(s)[i])) return false;
    }
    if (act_.Valid()) {
      for (int64_t i = 0; i < act_.cols; ++i) {
        if (!std::isfinite(act_.Row(s)[i])) return false;
      }
    }
    if (qacc_.Valid()) {
      for (int64_t i = 0; i < qacc_.cols; ++i) {
        if (!std::isfinite(qacc_.Row(s)[i])) return false;
      }
    }
    return true;
  }

  int64_t N_ = 0, nwarn_ = 0;
  double max_qvel_ = kInf, max_qacc_ = kInf, quat_tol_ = 1e-3, time_rtol_ = 1e-9;
  Mat<const int32_t> quat_adr_, fatal_, warning_;
  Mat<const double> qpos_, qvel_, act_, qacc_, time_;
  Mat<int32_t> base_;
  Mat<double> expected_;
};

// --------------------------------------------------------------- slot bookkeeping

class TaskKernel;

// The per-slot episode bookkeeping of MJBatchCore.advance_control around the physics,
// as a few passes over the slots instead of dozens of array operations. Status, state
// and flag codes come from raysimbatch.protocol through `cfg` (one source of truth).
class Book {
 public:
  using U8 = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using Bool = nb::ndarray<bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  Book(const nb::dict& cfg, const nb::dict& arrays) {
    max_steps_ = Scalar<uint64_t>(cfg, "max_control_steps");
    K_ = Scalar<int64_t>(cfg, "num_components");
    const nb::dict st = nb::cast<nb::dict>(cfg["status"]), sl = nb::cast<nb::dict>(cfg["state"]);
    const nb::dict fl = nb::cast<nb::dict>(cfg["flag"]);
    ok_ = Scalar<uint8_t>(st, "OK");
    term_ = Scalar<uint8_t>(st, "TERMINATED");
    trunc_ = Scalar<uint8_t>(st, "TRUNCATED");
    native_ = Scalar<uint8_t>(st, "NATIVE_FAILURE");
    health_ = Scalar<uint8_t>(st, "HEALTH_FAILURE");
    ready_ = Scalar<uint8_t>(sl, "READY");
    waiting_ = Scalar<uint8_t>(sl, "WAITING_ACTION");
    stepping_ = Scalar<uint8_t>(sl, "STEPPING");
    terminal_ = Scalar<uint8_t>(sl, "TERMINAL");
    failed_ = Scalar<uint8_t>(sl, "FAILED");
    f_valid_ = Scalar<uint8_t>(fl, "VALID");
    f_term_ = Scalar<uint8_t>(fl, "TERMINATED");
    f_trunc_ = Scalar<uint8_t>(fl, "TRUNCATED");
    f_abort_ = Scalar<uint8_t>(fl, "ABORTED");
    slot_state_ = Get<uint8_t>(arrays, "slot_state", -1, 1);
    N_ = slot_state_.rows;
    steps_ = Get<uint64_t>(arrays, "physics_steps", N_, 1);
    warnings_ = Get<int64_t>(arrays, "warning_total", N_, 1);
    seq_ = Get<uint64_t>(arrays, "control_seq", N_, 1);
    returns_ = Get<double>(arrays, "return_sum", N_, 1);
    sim_time_ = Get<int64_t>(arrays, "sim_time_ns", N_, 1);
    comps_ = Get<double>(arrays, "component_sums", N_);
    if (comps_.cols < K_) throw nb::value_error("component_sums narrower than num_components");
  }

  // 0 when the slots are sorted, unique and non-negative and every one is READY or
  // WAITING_ACTION; 1 for a bad slot list, 2 for a slot in another state (nothing is
  // changed; an index beyond the slots raises IndexError).
  int Ready(const Slots& slots) const {
    int32_t prev = -1;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      if (slots.data()[j] <= prev) return 1;
      prev = slots.data()[j];
    }
    CheckSlots(slots, N_);
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const uint8_t st = slot_state_.Row(slots.data()[j])[0];
      if (st != ready_ && st != waiting_) return 2;
    }
    return 0;
  }

  void Begin(const Slots& slots) {
    CheckSlots(slots, N_);
    for (size_t j = 0; j < slots.shape(0); ++j) slot_state_.Row(slots.data()[j])[0] = stepping_;
  }

  // After stepping: counters, the control sequence (copied to seq_out), `good` (the slot
  // stepped or terminated cleanly), then the time limit: OK slots at max_control_steps
  // become TRUNCATED and ended.
  void Account(const Slots& slots, U8 status,
               nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> completed,
               nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> warn, Bool ended,
               nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> seq_out, Bool good) {
    CheckSlots(slots, N_);
    const size_t n = slots.shape(0);
    if (status.shape(0) != n || completed.shape(0) != n || warn.shape(0) != n || ended.shape(0) != n ||
        seq_out.shape(0) != n || good.shape(0) != n) {
      throw nb::value_error("per-slot arrays must have len(slots) entries");
    }
    for (size_t j = 0; j < n; ++j) {
      const int64_t s = slots.data()[j];
      steps_.Row(s)[0] += completed.data()[j];
      warnings_.Row(s)[0] += warn.data()[j];
      const uint64_t seq = ++seq_.Row(s)[0];
      seq_out.data()[j] = seq;
      uint8_t& st = status.data()[j];
      good.data()[j] = st == ok_ || st == term_;
      if (st == ok_ && seq >= max_steps_) {
        st = trunc_;
        ended.data()[j] = true;
      }
    }
  }

  // After observation: return and component sums (every slot adds, a slot that is not
  // good adds zero, exactly as `sum += where(good, x, 0)`), flags, and the next state.
  void Settle(const Slots& slots, U8 status, Bool good,
              nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> reward,
              nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> comps, Bool ended, U8 flags) {
    CheckSlots(slots, N_);
    const size_t n = slots.shape(0);
    if (status.shape(0) != n || good.shape(0) != n || reward.shape(0) != n || comps.shape(0) != n ||
        static_cast<int64_t>(comps.shape(1)) != K_ || ended.shape(0) != n || flags.shape(0) != n) {
      throw nb::value_error("per-slot arrays must have len(slots) entries");
    }
    for (size_t j = 0; j < n; ++j) {
      const int64_t s = slots.data()[j];
      const bool g = good.data()[j];
      returns_.Row(s)[0] += g ? static_cast<double>(reward.data()[j]) : 0.0;
      double* cs = comps_.Row(s);
      for (int64_t c = 0; c < K_; ++c) cs[c] += g ? static_cast<double>(comps.data()[j * K_ + c]) : 0.0;
      const uint8_t st = status.data()[j];
      const bool aborted = st == native_ || st == health_;
      uint8_t f = 0;
      if (g) f |= f_valid_;
      if (st == term_) f |= f_term_;
      if (st == trunc_) f |= f_trunc_;
      if (aborted) f |= f_abort_;
      flags.data()[j] = f;
      bool& e = ended.data()[j];
      e = e || aborted;
      slot_state_.Row(s)[0] = aborted ? failed_ : (e ? terminal_ : ready_);
    }
  }

 private:
  int64_t N_ = 0, K_ = 0;
  uint64_t max_steps_ = 0;
  uint8_t ok_ = 0, term_ = 0, trunc_ = 0, native_ = 0, health_ = 0;
  uint8_t ready_ = 0, waiting_ = 0, stepping_ = 0, terminal_ = 0, failed_ = 0;
  uint8_t f_valid_ = 0, f_term_ = 0, f_trunc_ = 0, f_abort_ = 0;
  Mat<uint8_t> slot_state_;
  Mat<uint64_t> steps_, seq_;
  Mat<int64_t> warnings_, sim_time_;
  Mat<double> returns_, comps_;

 public:
  using U16 = nb::ndarray<uint16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using U32 = nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using I32 = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using History = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
  void FusedSettle(Controller& ctl, Health& health, TaskKernel& task, const Slots& slots,
                   nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu> failed, History history,
                   int64_t qpos_off, int64_t qvel_off, double dt, int64_t timestep_ns, U8 status, U16 fault,
                   U32 completed, I32 failed_sub, I32 warn, Bool ended, I32 first, Bool replay);
  int64_t SubstepSettle(Health& health, TaskKernel& task, nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig,
                        nb::device::cpu> active, nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> pos,
                        int64_t k, int64_t last, bool each_substep, double dt, int64_t timestep_ns, U8 status,
                        U16 fault, U32 completed, I32 failed_sub, I32 warn, Bool ended);
};

// ------------------------------------------------------------------- observation

// Term sources; mirrored by raysimbatch.sim2sim.contract.SOURCE_CODES.
enum Source : int32_t {
  kJointPos = 0,
  kJointVel = 1,
  kBaseLinVel = 2,
  kBaseAngVel = 3,
  kGravity = 4,
  kCommand = 5,
  kPrevAction = 6,
  kPhase = 7,
  kRot6d = 8,
  kSensor = 9,
  kAppliedAction = 10,
  kBaseQuat = 11,
};

// term_table rows: (source, dim, history, frame_or_layout, index_offset, index_count,
// history_offset); term_params rows: (scale, clip, noise_std, aux). aux is the phase
// period in seconds for kPhase. index_pool holds qpos/dof/sensor/command/action
// indices; sign_pool the per-element source/target axis signs and offset_pool the
// per-element subtrahends (default joint positions in the source convention):
// indexed terms emit sign * value - offset.
class Observation {
 public:
  Observation(const nb::dict& cfg, const nb::dict& arrays) {
    layout_ = Scalar<int>(cfg, "history_layout");      // 0 term_major, 1 time_major
    reset_fill_ = Scalar<int>(cfg, "history_reset");   // 0 zeros, 1 repeat_initial
    order_ = Scalar<int>(cfg, "history_order");        // 0 oldest_first, 1 newest_first
    op_order_ = Scalar<int>(cfg, "term_op_order");     // 0 noise_clip_scale, 1 scale_clip_noise
    quat_xyzw_ = Scalar<int>(cfg, "quaternion_xyzw");  // output order for kBaseQuat
    base_qadr_ = Scalar<int64_t>(cfg, "base_qpos_adr");
    base_dadr_ = Scalar<int64_t>(cfg, "base_dof_adr");
    obs_dim_ = Scalar<int64_t>(cfg, "obs_dim");
    hist_width_ = Scalar<int64_t>(cfg, "history_width");
    norm_clip_ = Scalar<double>(cfg, "normalization_clip");
    table_ = Get<const int32_t>(arrays, "term_table", -1, 7);
    params_ = Get<const double>(arrays, "term_params", table_.rows, 4);
    index_ = Get<const int32_t>(arrays, "index_pool", -1, 1);
    offset_ = Get<const double>(arrays, "offset_pool", index_.rows, 1);
    sign_ = Get<const double>(arrays, "sign_pool", index_.rows, 1);
    mean_ = Get<const double>(arrays, "norm_mean", obs_dim_, 1, false);
    inv_std_ = Get<const double>(arrays, "norm_inv_std", obs_dim_, 1, false);
    qpos_ = Get<const double>(arrays, "qpos");
    N_ = qpos_.rows;
    qvel_ = Get<const double>(arrays, "qvel", N_);
    sensor_ = Get<const double>(arrays, "sensordata", N_, -1, false);
    command_ = Get<const double>(arrays, "command", N_, -1, false);
    prev_ = Get<const float>(arrays, "prev_action", N_, -1, false);
    applied_ = Get<const double>(arrays, "targets", N_, -1, false);
    sim_ns_ = Get<const int64_t>(arrays, "sim_time_ns", N_, 1);
    keys_ = Get<const uint64_t>(arrays, "noise_keys", N_, 2);
    seq_ = Get<const uint64_t>(arrays, "control_seq", N_, 1);
    hist_ = Get<float>(arrays, "history", N_, hist_width_);
    filled_ = Get<uint8_t>(arrays, "history_filled", N_, 1);
    Validate();
    raw_.resize(static_cast<size_t>(MaxDim()));
  }

  void Reset(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      std::fill(hist_.Row(s), hist_.Row(s) + hist_width_, 0.0f);
      filled_.Row(s)[0] = 0;
    }
  }

  // Computes one policy-rate observation per slot: raw terms, per-term ops, history
  // push, flatten, optional external normalization.
  void Observe(const Slots& slots,
               nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> out) {
    CheckSlots(slots, N_);
    if (out.shape(0) != slots.shape(0) || static_cast<int64_t>(out.shape(1)) != obs_dim_) {
      throw nb::value_error("out must have shape (len(slots), obs_dim)");
    }
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) {
      const int64_t s = slots.data()[j];
      float* hist = hist_.Row(s);
      const bool first = filled_.Row(s)[0] == 0;
      int64_t noise_index = 0;
      for (int64_t t = 0; t < table_.rows; ++t) {
        const int32_t* row = table_.Row(t);
        const int64_t dim = row[1], H = row[2], hoff = row[6];
        Raw(s, t, raw_.data());
        const double* prm = params_.Row(t);
        for (int64_t e = 0; e < dim; ++e) {
          const double noise = prm[2] > 0 ? prm[2] * Normal(keys_.Row(s), seq_.Row(s)[0],
                                                          static_cast<uint64_t>(noise_index + e),
                                                          kStreamObsNoise)
                                           : 0.0;
          double x = raw_[e];
          const double clip = prm[1] > 0 ? prm[1] : kInf;
          if (op_order_ == 0) {
            x = Clamp(x + noise, -clip, clip) * prm[0];
          } else {
            x = Clamp(x * prm[0], -clip, clip) + noise;
          }
          raw_[e] = x;
        }
        noise_index += dim;
        float* h = hist + hoff;  // H frames of dim, oldest first in storage
        const bool fill = first && reset_fill_ == 1;
        if (fill) {
          for (int64_t k = 0; k < H; ++k) {
            for (int64_t e = 0; e < dim; ++e) h[k * dim + e] = static_cast<float>(raw_[e]);
          }
        } else {
          std::move(h + dim, h + H * dim, h);
          for (int64_t e = 0; e < dim; ++e) h[(H - 1) * dim + e] = static_cast<float>(raw_[e]);
        }
      }
      filled_.Row(s)[0] = 1;
      float* o = out.data() + j * obs_dim_;
      Flatten(hist, o);
      if (mean_.Valid()) {
        for (int64_t i = 0; i < obs_dim_; ++i) {
          const double v = (o[i] - mean_.p[i]) * inv_std_.p[i];
          o[i] = static_cast<float>(norm_clip_ > 0 ? Clamp(v, -norm_clip_, norm_clip_) : v);
        }
      }
    }
  }

  int64_t obs_dim() const { return obs_dim_; }

 private:
  int64_t MaxDim() const {
    int64_t m = 1;
    for (int64_t t = 0; t < table_.rows; ++t) m = std::max<int64_t>(m, table_.Row(t)[1]);
    return m;
  }

  void Validate() {
    int64_t width = 0, total = 0, frames = -1;
    for (int64_t t = 0; t < table_.rows; ++t) {
      const int32_t* r = table_.Row(t);
      const int32_t src = r[0], dim = r[1], H = r[2], ioff = r[4], icnt = r[5], hoff = r[6];
      if (dim <= 0 || H <= 0) throw nb::value_error("term dim/history must be positive");
      if (hoff != width) throw nb::value_error("history offsets must be packed in term order");
      if (ioff < 0 || icnt < 0 || ioff + icnt > index_.rows) throw nb::value_error("index slice");
      width += static_cast<int64_t>(dim) * H;
      total += static_cast<int64_t>(dim) * H;
      if (layout_ == 1) {
        if (frames >= 0 && frames != H) throw nb::value_error("time_major needs equal histories");
        frames = H;
      }
      const bool floating = src == kBaseLinVel || src == kBaseAngVel || src == kGravity ||
                            src == kRot6d || src == kBaseQuat;
      if (floating && (base_qadr_ < 0 || base_dadr_ < 0)) {
        throw nb::value_error("base terms need a floating base");
      }
      int32_t want = dim;
      switch (src) {
        case kBaseLinVel:
        case kBaseAngVel:
        case kGravity:
          want = 3;
          break;
        case kRot6d:
          want = 6;
          break;
        case kBaseQuat:
          want = 4;
          break;
        case kPhase:
          want = 2;
          break;
        default:
          if (icnt != dim) throw nb::value_error("indexed term needs dim indices");
      }
      if (want != dim) throw nb::value_error("term dimension does not match its source");
      const int64_t limit = src == kJointPos    ? qpos_.cols
                            : src == kJointVel  ? qvel_.cols
                            : src == kSensor    ? (sensor_.Valid() ? sensor_.cols : 0)
                            : src == kCommand   ? (command_.Valid() ? command_.cols : 0)
                            : src == kPrevAction ? (prev_.Valid() ? prev_.cols : 0)
                            : src == kAppliedAction ? (applied_.Valid() ? applied_.cols : 0)
                                                     : -1;
      if (limit >= 0) {
        for (int32_t k = 0; k < icnt; ++k) {
          if (index_.p[ioff + k] < 0 || index_.p[ioff + k] >= limit) {
            throw nb::value_error("term index out of range for its source");
          }
        }
      }
    }
    if (width != hist_width_) throw nb::value_error("history_width mismatch");
    if (total != obs_dim_) throw nb::value_error("obs_dim mismatch");
    if (base_qadr_ >= 0 && base_qadr_ + 7 > qpos_.cols) throw nb::value_error("base_qpos_adr");
    if (base_dadr_ >= 0 && base_dadr_ + 6 > qvel_.cols) throw nb::value_error("base_dof_adr");
    if (mean_.Valid() != inv_std_.Valid()) throw nb::value_error("normalizer needs mean and inv_std");
  }

  void Raw(int64_t s, int64_t t, double* out) const {
    const int32_t* r = table_.Row(t);
    const int32_t src = r[0], dim = r[1], frame = r[3], ioff = r[4];
    const int32_t* idx = index_.p + ioff;
    const double* off = offset_.p + ioff;
    const double* sg = sign_.p + ioff;
    const double* q = qpos_.Row(s);
    const double* v = qvel_.Row(s);
    double R[9];
    if (src == kBaseLinVel || src == kBaseAngVel || src == kGravity || src == kRot6d) {
      QuatToMat(q + base_qadr_ + 3, R);
    }
    switch (src) {
      case kJointPos:
        for (int32_t e = 0; e < dim; ++e) out[e] = sg[e] * q[idx[e]] - off[e];
        break;
      case kJointVel:
        for (int32_t e = 0; e < dim; ++e) out[e] = sg[e] * v[idx[e]] - off[e];
        break;
      case kSensor:
        for (int32_t e = 0; e < dim; ++e) out[e] = sg[e] * sensor_.Row(s)[idx[e]] - off[e];
        break;
      case kCommand:
        for (int32_t e = 0; e < dim; ++e) out[e] = sg[e] * command_.Row(s)[idx[e]] - off[e];
        break;
      case kPrevAction:
        for (int32_t e = 0; e < dim; ++e) out[e] = sg[e] * prev_.Row(s)[idx[e]] - off[e];
        break;
      case kAppliedAction:
        for (int32_t e = 0; e < dim; ++e) out[e] = sg[e] * applied_.Row(s)[idx[e]] - off[e];
        break;
      case kBaseLinVel:  // free-joint linear velocity is world-frame in qvel
        if (frame == 0) {
          MulT(R, v + base_dadr_, out);
        } else {
          std::copy(v + base_dadr_, v + base_dadr_ + 3, out);
        }
        break;
      case kBaseAngVel:  // free-joint angular velocity is body-frame in qvel
        if (frame == 0) {
          std::copy(v + base_dadr_ + 3, v + base_dadr_ + 6, out);
        } else {
          Mul(R, v + base_dadr_ + 3, out);
        }
        break;
      case kGravity: {
        const double g[3] = {0.0, 0.0, -1.0};
        MulT(R, g, out);
        break;
      }
      case kRot6d:
        if (frame == 0) {  // first two columns
          const double c[6] = {R[0], R[3], R[6], R[1], R[4], R[7]};
          std::copy(c, c + 6, out);
        } else {  // first two rows
          std::copy(R, R + 6, out);
        }
        break;
      case kBaseQuat: {
        const double* x = q + base_qadr_ + 3;  // MuJoCo wxyz
        if (quat_xyzw_) {
          out[0] = x[1];
          out[1] = x[2];
          out[2] = x[3];
          out[3] = x[0];
        } else {
          std::copy(x, x + 4, out);
        }
        break;
      }
      case kPhase: {
        const double period = params_.Row(t)[3];
        const double seconds = static_cast<double>(sim_ns_.Row(s)[0]) * 1e-9;
        const double phase = period > 0 ? std::fmod(seconds, period) / period : 0.0;
        out[0] = std::sin(6.283185307179586 * phase);
        out[1] = std::cos(6.283185307179586 * phase);
        break;
      }
      default:
        std::fill(out, out + dim, std::numeric_limits<double>::quiet_NaN());
    }
  }

  void Flatten(const float* hist, float* o) const {
    int64_t w = 0;
    if (layout_ == 0) {  // term-major: each term's frames together
      for (int64_t t = 0; t < table_.rows; ++t) {
        const int32_t* r = table_.Row(t);
        const int64_t dim = r[1], H = r[2], hoff = r[6];
        for (int64_t k = 0; k < H; ++k) {
          const int64_t frame = order_ == 0 ? k : H - 1 - k;
          for (int64_t e = 0; e < dim; ++e) o[w++] = hist[hoff + frame * dim + e];
        }
      }
    } else {  // time-major: each frame's terms together
      const int64_t H = table_.rows ? table_.Row(0)[2] : 0;
      for (int64_t k = 0; k < H; ++k) {
        const int64_t frame = order_ == 0 ? k : H - 1 - k;
        for (int64_t t = 0; t < table_.rows; ++t) {
          const int32_t* r = table_.Row(t);
          const int64_t dim = r[1], hoff = r[6];
          for (int64_t e = 0; e < dim; ++e) o[w++] = hist[hoff + frame * dim + e];
        }
      }
    }
  }

  int layout_ = 0, reset_fill_ = 0, order_ = 0, op_order_ = 0, quat_xyzw_ = 0;
  int64_t base_qadr_ = -1, base_dadr_ = -1, obs_dim_ = 0, hist_width_ = 0, N_ = 0;
  double norm_clip_ = 0;
  Mat<const int32_t> table_, index_;
  Mat<const double> params_, offset_, sign_, mean_, inv_std_;
  Mat<const double> qpos_, qvel_, sensor_, command_, applied_;
  Mat<const float> prev_;
  Mat<const int64_t> sim_ns_;
  Mat<const uint64_t> keys_, seq_;
  Mat<float> hist_;
  Mat<uint8_t> filled_;
  mutable std::vector<double> raw_;
};

// ------------------------------------------------------------------------- tasks

// Metrics row written by Task::Metrics: (fell, tracking_rmse, metric_a, metric_b).
constexpr int kTaskMetrics = 4;

// Termination is a pure function of a qpos row, so it can be evaluated on the live
// bind() rows after each substep or on recorded history rows (fused intervals).
class Task {
 public:
  virtual ~Task() = default;
  virtual void Reset(const int32_t* slots, size_t n) = 0;
  virtual bool Terminated(const double* qpos) const = 0;
  virtual void MarkTerminated(int64_t slot) = 0;
  virtual void Control(const int32_t* slots, size_t n, float* reward, float* components) = 0;
  virtual void Metrics(const int32_t* slots, size_t n, double* out) = 0;
  virtual int64_t NumComponents() const = 0;
  virtual int64_t NumCommands() const = 0;
  virtual int64_t NumState() const = 0;
};

struct TaskArrays {
  Mat<const double> qpos, qvel, site_xpos;
  Mat<double> command, state;
  Mat<const uint64_t> keys;
  int64_t N = 0;
};

// Pole/pendulum stabilization. iparams (angle_qadr, slider_qadr or -1); params
// (angle_limit, slider_limit). state (sum angle^2, count, fell).
class Balance : public Task {
 public:
  Balance(TaskArrays a, const Mat<const double>& p, const Mat<const int32_t>& ip) : a_(std::move(a)) {
    if (p.rows < 2 || ip.rows < 2) throw nb::value_error("balance/1 needs 2 params and 2 iparams");
    angle_ = ip.p[0];
    slider_ = ip.p[1];
    angle_limit_ = p.p[0];
    slider_limit_ = p.p[1];
    if (angle_ < 0 || angle_ >= a_.qpos.cols || slider_ >= a_.qpos.cols) throw nb::value_error("balance adr");
    if (a_.state.cols < 3) throw nb::value_error("balance state");
  }
  void Reset(const int32_t* slots, size_t n) override {
    for (size_t j = 0; j < n; ++j) std::fill(a_.state.Row(slots[j]), a_.state.Row(slots[j]) + 3, 0.0);
  }
  bool Terminated(const double* q) const override {
    return std::abs(q[angle_]) > angle_limit_ || (slider_ >= 0 && std::abs(q[slider_]) > slider_limit_);
  }
  void MarkTerminated(int64_t s) override { a_.state.Row(s)[2] = 1.0; }
  void Control(const int32_t* slots, size_t n, float* reward, float* comps) override {
    for (size_t j = 0; j < n; ++j) {
      double* st = a_.state.Row(slots[j]);
      const double ang = a_.qpos.Row(slots[j])[angle_];
      st[0] += ang * ang;
      st[1] += 1.0;
      reward[j] = 1.0f;
      comps[j] = 1.0f;
    }
  }
  void Metrics(const int32_t* slots, size_t n, double* out) override {
    for (size_t j = 0; j < n; ++j) {
      const double* st = a_.state.Row(slots[j]);
      out[j * kTaskMetrics + 0] = st[2];
      out[j * kTaskMetrics + 1] = st[1] > 0 ? std::sqrt(st[0] / st[1]) : std::nan("");
      out[j * kTaskMetrics + 2] = std::nan("");
      out[j * kTaskMetrics + 3] = std::nan("");
    }
  }
  int64_t NumComponents() const override { return 1; }
  int64_t NumCommands() const override { return 0; }
  int64_t NumState() const override { return 3; }

 private:
  TaskArrays a_;
  int64_t angle_ = 0, slider_ = -1;
  double angle_limit_ = 0, slider_limit_ = 0;
};

// Legged base velocity tracking. iparams (base_qadr, base_dadr, resample_steps);
// params (sigma_lin, sigma_ang, min_height, min_up_z, lo_vx, hi_vx, lo_vy, hi_vy,
// lo_wz, hi_wz, w_lin, w_ang, w_alive). commands (vx, vy, wz) body frame;
// state (steps, sum sq err, count, fell).
class VelocityTracking : public Task {
 public:
  VelocityTracking(TaskArrays a, const Mat<const double>& p, const Mat<const int32_t>& ip)
      : a_(std::move(a)) {
    if (p.rows < 13 || ip.rows < 3) throw nb::value_error("velocity_tracking/1 needs 13 params, 3 iparams");
    qadr_ = ip.p[0];
    dadr_ = ip.p[1];
    resample_ = ip.p[2];
    std::copy(p.p, p.p + 13, prm_);
    if (qadr_ < 0 || qadr_ + 7 > a_.qpos.cols || dadr_ < 0 || dadr_ + 6 > a_.qvel.cols) {
      throw nb::value_error("velocity_tracking base address");
    }
    if (!a_.command.Valid() || a_.command.cols != 3 || a_.state.cols < 4) {
      throw nb::value_error("velocity_tracking arrays");
    }
  }
  void Reset(const int32_t* slots, size_t n) override {
    for (size_t j = 0; j < n; ++j) {
      std::fill(a_.state.Row(slots[j]), a_.state.Row(slots[j]) + 4, 0.0);
      Draw(slots[j], 0);
    }
  }
  bool Terminated(const double* q) const override {
    double R[9];
    QuatToMat(q + qadr_ + 3, R);
    return q[qadr_ + 2] < prm_[2] || R[8] < prm_[3];
  }
  void MarkTerminated(int64_t s) override { a_.state.Row(s)[3] = 1.0; }
  void Control(const int32_t* slots, size_t n, float* reward, float* comps) override {
    for (size_t j = 0; j < n; ++j) {
      const int64_t s = slots[j];
      const double* q = a_.qpos.Row(s);
      const double* v = a_.qvel.Row(s);
      const double* c = a_.command.Row(s);
      double* st = a_.state.Row(s);
      double R[9], vb[3];
      QuatToMat(q + qadr_ + 3, R);
      MulT(R, v + dadr_, vb);
      const double wz = v[dadr_ + 5];
      const double el = (c[0] - vb[0]) * (c[0] - vb[0]) + (c[1] - vb[1]) * (c[1] - vb[1]);
      const double ea = (c[2] - wz) * (c[2] - wz);
      const double lin = prm_[10] * std::exp(-el / prm_[0]);
      const double ang = prm_[11] * std::exp(-ea / prm_[1]);
      comps[j * 3 + 0] = static_cast<float>(lin);
      comps[j * 3 + 1] = static_cast<float>(ang);
      comps[j * 3 + 2] = static_cast<float>(prm_[12]);
      reward[j] = static_cast<float>(lin + ang + prm_[12]);
      st[1] += el + ea;
      st[2] += 1.0;
      st[0] += 1.0;
      if (resample_ > 0 && static_cast<int64_t>(st[0]) % resample_ == 0) {
        Draw(s, static_cast<uint64_t>(st[0]) / resample_);
      }
    }
  }
  void Metrics(const int32_t* slots, size_t n, double* out) override {
    for (size_t j = 0; j < n; ++j) {
      const double* st = a_.state.Row(slots[j]);
      out[j * kTaskMetrics + 0] = st[3];
      out[j * kTaskMetrics + 1] = st[2] > 0 ? std::sqrt(st[1] / st[2]) : std::nan("");
      out[j * kTaskMetrics + 2] = st[0];
      out[j * kTaskMetrics + 3] = std::nan("");
    }
  }
  int64_t NumComponents() const override { return 3; }
  int64_t NumCommands() const override { return 3; }
  int64_t NumState() const override { return 4; }

 private:
  void Draw(int64_t s, uint64_t seq) {
    double* c = a_.command.Row(s);
    for (int k = 0; k < 3; ++k) {
      const double lo = prm_[4 + 2 * k], hi = prm_[5 + 2 * k];
      c[k] = lo + (hi - lo) * Uniform(a_.keys.Row(s), seq, static_cast<uint64_t>(k), kStreamTask);
    }
  }
  TaskArrays a_;
  int64_t qadr_ = 0, dadr_ = 0, resample_ = 0;
  double prm_[13] = {};
};

// End-effector reaching. iparams (site_index); params (lo_x, hi_x, lo_y, hi_y, lo_z,
// hi_z, success_tol). commands: target xyz; state (last_dist, sum d^2, count).
class Reach : public Task {
 public:
  Reach(TaskArrays a, const Mat<const double>& p, const Mat<const int32_t>& ip) : a_(std::move(a)) {
    if (p.rows < 7 || ip.rows < 1) throw nb::value_error("reach/1 needs 7 params, 1 iparam");
    site_ = ip.p[0];
    std::copy(p.p, p.p + 7, prm_);
    if (!a_.site_xpos.Valid() || site_ < 0 || 3 * site_ + 3 > a_.site_xpos.cols) {
      throw nb::value_error("reach needs site_xpos and a valid site");
    }
    if (!a_.command.Valid() || a_.command.cols != 3 || a_.state.cols < 3) throw nb::value_error("reach arrays");
  }
  void Reset(const int32_t* slots, size_t n) override {
    for (size_t j = 0; j < n; ++j) {
      const int64_t s = slots[j];
      std::fill(a_.state.Row(s), a_.state.Row(s) + 3, 0.0);
      a_.state.Row(s)[0] = std::nan("");
      for (int k = 0; k < 3; ++k) {
        const double lo = prm_[2 * k], hi = prm_[2 * k + 1];
        a_.command.Row(s)[k] = lo + (hi - lo) * Uniform(a_.keys.Row(s), 0, k, kStreamTask);
      }
    }
  }
  bool Terminated(const double*) const override { return false; }
  void MarkTerminated(int64_t) override {}
  void Control(const int32_t* slots, size_t n, float* reward, float* comps) override {
    for (size_t j = 0; j < n; ++j) {
      const int64_t s = slots[j];
      const double* x = a_.site_xpos.Row(s) + 3 * site_;
      const double* c = a_.command.Row(s);
      const double d = std::sqrt((x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]) +
                                 (x[2] - c[2]) * (x[2] - c[2]));
      double* st = a_.state.Row(s);
      st[0] = d;
      st[1] += d * d;
      st[2] += 1.0;
      reward[j] = static_cast<float>(-d);
      comps[j] = static_cast<float>(-d);
    }
  }
  void Metrics(const int32_t* slots, size_t n, double* out) override {
    for (size_t j = 0; j < n; ++j) {
      const double* st = a_.state.Row(slots[j]);
      out[j * kTaskMetrics + 0] = std::isfinite(st[0]) && st[0] <= prm_[6] ? 0.0 : 1.0;
      out[j * kTaskMetrics + 1] = st[2] > 0 ? std::sqrt(st[1] / st[2]) : std::nan("");
      out[j * kTaskMetrics + 2] = st[0];
      out[j * kTaskMetrics + 3] = std::nan("");
    }
  }
  int64_t NumComponents() const override { return 1; }
  int64_t NumCommands() const override { return 3; }
  int64_t NumState() const override { return 3; }

 private:
  TaskArrays a_;
  int64_t site_ = 0;
  double prm_[7] = {};
};

// The versioned function table exposed to Python.
class TaskKernel {
 public:
  TaskKernel(const std::string& name, int version, const nb::dict& arrays) : name_(name), version_(version) {
    if (version != 1) throw nb::value_error(("unsupported task kernel version for " + name).c_str());
    params_ = Get<const double>(arrays, "params", -1, 1);
    iparams_ = Get<const int32_t>(arrays, "iparams", -1, 1);
    TaskArrays a;
    a.qpos = Get<const double>(arrays, "qpos");
    a.N = a.qpos.rows;
    a.qvel = Get<const double>(arrays, "qvel", a.N);
    a.site_xpos = Get<const double>(arrays, "site_xpos", a.N, -1, false);
    a.command = Get<double>(arrays, "command", a.N, -1, false);
    a.state = Get<double>(arrays, "state", a.N);
    a.keys = Get<const uint64_t>(arrays, "keys", a.N, 2);
    N_ = a.N;
    qpos_ = a.qpos;
    if (name == "balance") {
      task_ = std::make_unique<Balance>(std::move(a), params_, iparams_);
    } else if (name == "velocity_tracking") {
      task_ = std::make_unique<VelocityTracking>(std::move(a), params_, iparams_);
    } else if (name == "reach") {
      task_ = std::make_unique<Reach>(std::move(a), params_, iparams_);
    } else {
      throw nb::value_error(("unknown task kernel " + name).c_str());
    }
  }

  void Reset(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    task_->Reset(slots.data(), slots.shape(0));
  }
  void MarkTerminated(const Slots& slots) {
    CheckSlots(slots, N_);
    nb::gil_scoped_release release;
    for (size_t j = 0; j < slots.shape(0); ++j) task_->MarkTerminated(slots.data()[j]);
  }
  // First substep whose recorded state terminates the episode (-1: none), from history
  // rows (bind("state") layout; rows[j] is slot j's history index). Marks nothing.
  void FirstTermination(const Slots& rows,
                        nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu> history,
                        int64_t qpos_off, nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out) {
    const int64_t nstep = static_cast<int64_t>(history.shape(1));
    const int64_t nstate = static_cast<int64_t>(history.shape(2));
    if (out.shape(0) != rows.shape(0)) throw nb::value_error("out length must equal len(rows)");
    if (qpos_off < 0 || qpos_off + qpos_.cols > nstate) throw nb::value_error("qpos offset out of range");
    for (size_t j = 0; j < rows.shape(0); ++j) {
      if (rows.data()[j] < 0 || rows.data()[j] >= static_cast<int32_t>(history.shape(0))) {
        throw nb::index_error("history row out of range");
      }
    }
    nb::gil_scoped_release release;
    for (size_t j = 0; j < rows.shape(0); ++j) {
      out.data()[j] = FirstTerminationRow(history.data() + static_cast<int64_t>(rows.data()[j]) * nstep * nstate,
                                          nstep, nstate, qpos_off);
    }
  }
  int32_t FirstTerminationRow(const double* h, int64_t nstep, int64_t nstate, int64_t qpos_off) const {
    for (int64_t k = 0; k < nstep; ++k) {
      if (task_->Terminated(h + k * nstate + qpos_off)) return static_cast<int32_t>(k);
    }
    return -1;
  }
  void MarkSlot(int64_t s) { task_->MarkTerminated(s); }
  bool TerminatedSlot(int64_t s) const { return task_->Terminated(qpos_.Row(s)); }
  int64_t num_slots() const { return N_; }
  int64_t nq() const { return qpos_.cols; }
  void Control(const Slots& slots, nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> reward,
               nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> comps) {
    CheckSlots(slots, N_);
    if (reward.shape(0) != slots.shape(0) || comps.shape(0) != slots.shape(0) ||
        static_cast<int64_t>(comps.shape(1)) != task_->NumComponents()) {
      throw nb::value_error("reward/components shape");
    }
    nb::gil_scoped_release release;
    task_->Control(slots.data(), slots.shape(0), reward.data(), comps.data());
  }
  void Metrics(const Slots& slots, nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> out) {
    CheckSlots(slots, N_);
    if (out.shape(0) != slots.shape(0) || out.shape(1) != kTaskMetrics) throw nb::value_error("metrics shape");
    nb::gil_scoped_release release;
    task_->Metrics(slots.data(), slots.shape(0), out.data());
  }
  const std::string& name() const { return name_; }
  int version() const { return version_; }
  int64_t num_components() const { return task_->NumComponents(); }
  int64_t num_commands() const { return task_->NumCommands(); }
  int64_t num_state() const { return task_->NumState(); }

 private:
  std::string name_;
  int version_;
  int64_t N_ = 0;
  Mat<const double> qpos_;
  Mat<const double> params_;
  Mat<const int32_t> iparams_;
  std::unique_ptr<Task> task_;
};

// After one fused step(slots, nstep=d, history) call (AM-2), per slot in one pass: a slot
// that failed natively, or whose history shows a termination before the last substep,
// is only marked for replay (`first` holds its terminal substep). Every other slot is
// finished as per-substep stepping would leave it: controller bookkeeping from its rows,
// d substeps completed, sim time advanced, health checked once over the interval, and a
// termination on the last substep counted if the slot is healthy.
inline void Book::FusedSettle(Controller& ctl, Health& health, TaskKernel& task, const Slots& slots,
                              nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu> failed,
                              History history, int64_t qpos_off, int64_t qvel_off, double dt, int64_t timestep_ns,
                              U8 status, U16 fault, U32 completed, I32 failed_sub, I32 warn, Bool ended, I32 first,
                              Bool replay) {
  CheckSlots(slots, N_);
  const size_t n = slots.shape(0);
  const int64_t d = static_cast<int64_t>(history.shape(1)), nstate = static_cast<int64_t>(history.shape(2));
  if (static_cast<size_t>(history.shape(0)) != n || failed.shape(0) != n || status.shape(0) != n ||
      fault.shape(0) != n || completed.shape(0) != n || failed_sub.shape(0) != n || warn.shape(0) != n ||
      ended.shape(0) != n || first.shape(0) != n || replay.shape(0) != n) {
    throw nb::value_error("per-slot arrays and history must have len(slots) rows");
  }
  if (!ctl.Fusable()) throw nb::value_error("controller kind or physics delay is not fusable");
  if (task.num_slots() != N_ || qpos_off < 0 || qpos_off + ctl.nq() > nstate || qvel_off < 0 ||
      qvel_off + ctl.nv() > nstate || task.nq() != ctl.nq()) {
    throw nb::value_error("state offsets or kernel sizes do not match");
  }
  nb::gil_scoped_release release;
  for (size_t j = 0; j < n; ++j) {
    const int64_t s = slots.data()[j];
    first.data()[j] = -1;
    replay.data()[j] = failed.data()[j];
    if (replay.data()[j]) continue;
    const double* h = history.data() + static_cast<int64_t>(j) * d * nstate;
    const int32_t k = task.FirstTerminationRow(h, d, nstate, qpos_off);
    first.data()[j] = k;
    if (k >= 0 && k < d - 1) {
      replay.data()[j] = true;
      continue;
    }
    ctl.FinishSlot(s, h, d, nstate, qpos_off, qvel_off, dt);
    completed.data()[j] = static_cast<uint32_t>(d);
    sim_time_.Row(s)[0] += d * timestep_ns;
    const uint16_t code = health.CheckSlot(s, dt * static_cast<double>(d), &warn.data()[j]);
    if (code) {
      status.data()[j] = health_;
      fault.data()[j] = code;
      failed_sub.data()[j] = static_cast<int32_t>(d - 1);
    } else if (k == d - 1) {
      task.MarkSlot(s);
      status.data()[j] = term_;
      ended.data()[j] = true;
    }
  }
}

// After one successful step(active, nstep=1) (substep k of a policy interval), per slot:
// the substep counts; health runs after every substep (`each_substep`), or once per
// interval on the slot's last substep (the one it terminates on, or `last`) with dt
// over the interval so far; a healthy terminated slot is marked TERMINATED, an
// unhealthy one HEALTH_FAILURE (its fall is not recorded). Slots still stepping stay
// in `active`/`pos`, compacted in place; returns how many.
inline int64_t Book::SubstepSettle(Health& health, TaskKernel& task,
                                   nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> active,
                                   nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> pos, int64_t k,
                                   int64_t last, bool each_substep, double dt, int64_t timestep_ns, U8 status,
                                   U16 fault, U32 completed, I32 failed_sub, I32 warn, Bool ended) {
  const size_t m = active.shape(0), n = status.shape(0);
  if (pos.shape(0) != m || fault.shape(0) != n || completed.shape(0) != n || failed_sub.shape(0) != n ||
      warn.shape(0) != n || ended.shape(0) != n) {
    throw nb::value_error("active/pos or result arrays have mismatched lengths");
  }
  for (size_t j = 0; j < m; ++j) {
    if (active.data()[j] < 0 || active.data()[j] >= N_ || pos.data()[j] < 0 ||
        static_cast<size_t>(pos.data()[j]) >= n) {
      throw nb::index_error("slot or position out of range");
    }
  }
  if (task.num_slots() != N_) throw nb::value_error("task kernel slot count differs");
  nb::gil_scoped_release release;
  int64_t kept = 0;
  const double span = each_substep ? dt : dt * static_cast<double>(k + 1);
  for (size_t j = 0; j < m; ++j) {
    const int32_t s = active.data()[j], p = pos.data()[j];
    completed.data()[p] += 1;
    sim_time_.Row(s)[0] += timestep_ns;
    const bool term = task.TerminatedSlot(s);
    if (each_substep || term || k == last) {
      int32_t delta = 0;
      const uint16_t code = health.CheckSlot(s, span, &delta);
      warn.data()[p] += delta;
      if (code) {
        status.data()[p] = health_;
        fault.data()[p] = code;
        failed_sub.data()[p] = static_cast<int32_t>(k);
        continue;
      }
      if (term) {
        task.MarkSlot(s);
        status.data()[p] = term_;
        ended.data()[p] = true;
        continue;
      }
      if (!each_substep) continue;  // checked on its last substep: done with this interval
    }
    active.data()[kept] = s;
    pos.data()[kept] = p;
    ++kept;
  }
  return kept;
}

}  // namespace rsb
