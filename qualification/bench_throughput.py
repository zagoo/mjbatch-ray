# SPDX-License-Identifier: Apache-2.0
"""Native throughput: RaySimBatch's MJBatchCore against upstream MJBatch at equal threads
(design §9.1 target: wrapper >= 90% of equal-thread upstream MJBatch).

For each model and thread count, three measurements of physics steps per second:

* ``upstream_nstep``: ``Batch.step(ids, nstep=d)``, one call per control interval
  (upstream's best case, no per-substep hook);
* ``upstream_substep``: ``Batch.step(ids, nstep=1)`` called d times, the upstream call
  pattern RaySimBatch uses so its kernels can run between substeps (AM-2);
* ``wrapper``: ``MJBatchCore.advance_control`` (controller, health, task kernels,
  per-slot status, observation) with termination disabled and the default per-substep
  health checks, so one ``nstep=1`` call per substep;
* ``wrapper_fused``: the same with ``health.check_each_substep=false``, which lets
  torque/position controllers step each interval in one ``nstep=d`` call with results
  identical to per-substep stepping (AM-2; ``stepping`` records which path ran).

The modes are measured round-robin: each of --repeats rounds times one window of every
mode back to back, and ratios are formed within a round, so drift in host load or in
the cores the OS schedules on (performance vs efficiency cores on Apple silicon) affects
numerator and denominator alike. Figures are medians over rounds; spread is reported.
These are measurements on *this* host only; they are not capacity claims.

  python qualification/bench_throughput.py --out evidence/perf/throughput.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _window(fn, seconds: float) -> tuple[int, float]:
  steps, t0 = 0, time.perf_counter()
  while (dt := time.perf_counter() - t0) < seconds:
    steps += fn()
  return steps, dt


class _TimedBatch:
  """Benchmark-side proxy around RaySimBatch's own handle: accumulates time spent inside
  upstream ``Batch.step`` so wrapper overhead is measured on identical physics."""

  def __init__(self, batch):
    self._inner, self.native_s = batch, 0.0

  def step(self, *a, **kw):
    t0 = time.perf_counter()
    try:
      return self._inner.step(*a, **kw)
    finally:
      self.native_s += time.perf_counter() - t0

  def __getattr__(self, name):
    return getattr(self._inner, name)


def bench(name: str, slots: int, threads: int, seconds: float, repeats: int, warmup: float) -> dict:
  from mjbatch import Batch  # upstream, measured directly (qualification code, not src/)

  from raysimbatch.testing import fixtures as F

  fx = F.no_termination(F.FIXTURES[name](max_control_steps=10**9))
  d = fx.group.decimation
  ids = np.arange(slots, dtype=np.int32)
  model = fx.model()

  core = fx.core(num_slots=slots, threads=threads)
  fcore = fx.core(num_slots=slots, threads=threads,
                  health=fx.group.health.model_copy(update={"check_each_substep": False}))
  s = np.arange(slots, dtype=np.int32)
  ep = np.zeros(slots, np.uint64)
  r = core.reset_exact(s, s.astype(np.uint64), ep, (s.astype(np.uint64), ep))
  assert r.ok.all(), r.message
  start_state = core.h.state.copy()
  rng = np.random.default_rng(0)
  act = (0.1 * rng.uniform(-1, 1, (slots, core.A))).astype(np.float32)
  core.advance_control(s, act)
  ctrl = core.h.bind("ctrl").copy()  # the wrapper's applied control, held by upstream

  # Upstream from the wrapper's exact start state and control (equal physics work).
  b = Batch(model, slots, num_threads=threads)
  b_state, b_ctrl = b.bind("state"), b.bind("ctrl")

  def restart_upstream():
    b_state[:] = start_state
    b_ctrl[:] = ctrl
    b.forward(ids)

  def up_nstep():
    b.step(ids, nstep=d)
    return slots * d

  def up_substep():
    for _ in range(d):
      b.step(ids, nstep=1)
    return slots * d

  timed = _TimedBatch(core.h._batch)
  core.h._batch = timed
  ftimed = _TimedBatch(fcore.h._batch)
  fcore.h._batch = ftimed

  def restart_wrapper():
    core.reset_exact(s, s.astype(np.uint64), ep, (s.astype(np.uint64), ep))

  def wrapper():
    out = core.advance_control(s, act)
    return int(out.completed_substeps.sum())

  def restart_fused():
    fcore.reset_exact(s, s.astype(np.uint64), ep, (s.astype(np.uint64), ep))

  def fused():
    out = fcore.advance_control(s, act)
    return int(out.completed_substeps.sum())

  modes = {"upstream_nstep": (up_nstep, restart_upstream, None),
           "upstream_substep": (up_substep, restart_upstream, None),
           "wrapper": (wrapper, restart_wrapper, timed),
           "wrapper_fused": (fused, restart_fused, ftimed)}
  rates: dict[str, list[float]] = {k: [] for k in modes}
  fracs: dict[str, list[float]] = {k: [] for k in modes}
  for fn, restart, _ in modes.values():
    restart()
    _window(fn, warmup)
  for _ in range(repeats):
    for k, (fn, restart, tb) in modes.items():
      restart()
      if tb is not None:
        tb.native_s = 0.0
      steps, dt = _window(fn, seconds)
      rates[k].append(steps / dt)
      if tb is not None:
        fracs[k].append(tb.native_s / dt)

  def summary(k):
    r = rates[k]
    out = {"median": statistics.median(r), "min": min(r), "max": max(r),
           "rel_spread": (max(r) - min(r)) / statistics.median(r), "runs": r}
    if fracs[k]:
      out["native_fraction"] = statistics.median(fracs[k])
    return out

  def ratio(a, b):
    return statistics.median(x / y for x, y in zip(rates[a], rates[b], strict=True))

  res = {"model": name, "slots": slots, "threads": threads, "decimation": d, "nv": int(model.nv),
         **{k: summary(k) for k in modes}, "fused_stepping": fcore.stepping}
  res["wrapper_vs_upstream_nstep"] = ratio("wrapper", "upstream_nstep")
  res["wrapper_vs_upstream_substep"] = ratio("wrapper", "upstream_substep")
  res["fused_vs_upstream_nstep"] = ratio("wrapper_fused", "upstream_nstep")
  res["fused_vs_wrapper"] = ratio("wrapper_fused", "wrapper")
  res["substep_call_overhead"] = 1 - ratio("upstream_substep", "upstream_nstep")
  core.h._batch = timed._inner
  fcore.h._batch = ftimed._inner
  core.close()
  fcore.close()
  return res


def host() -> dict:
  info = {"machine": platform.machine(), "system": platform.system(), "release": platform.release(),
          "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
          "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
  if sys.platform == "darwin":
    info["cpu_model"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                       text=True).stdout.strip()
  elif Path("/proc/cpuinfo").exists():
    info["cpu_model"] = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                              if ln.startswith(("model name", "Model"))), None)
  import mujoco

  from raysimbatch.compat.mjbatch_adapter import check_environment

  rep = check_environment()
  info.update(mujoco=mujoco.__version__, mjbatch=rep.mjbatch_version, mjbatch_binary_sha256=rep.binary_sha256)
  return info


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--models", default="cartpole,quadruped,reacher")
  ap.add_argument("--slots", type=int, default=64)
  ap.add_argument("--threads", default="1,2,4")
  ap.add_argument("--seconds", type=float, default=2.0)
  ap.add_argument("--repeats", type=int, default=5)
  ap.add_argument("--warmup", type=float, default=1.0)
  ap.add_argument("--out")
  args = ap.parse_args(argv)
  results = []
  for name in args.models.split(","):
    for t in (int(x) for x in args.threads.split(",")):
      r = bench(name, args.slots, t, args.seconds, args.repeats, args.warmup)
      results.append(r)
      print(f"{name:10s} T={t} upstream(nstep)={r['upstream_nstep']['median']:12,.0f}/s "
            f"upstream(substep)={r['upstream_substep']['median']:12,.0f}/s "
            f"wrapper={r['wrapper']['median']:12,.0f}/s  ratio={r['wrapper_vs_upstream_nstep']:.3f} "
            f"(vs substep {r['wrapper_vs_upstream_substep']:.3f}) native_fraction="
            f"{r['wrapper']['native_fraction']:.2f} | {r['fused_stepping']}={r['wrapper_fused']['median']:12,.0f}/s "
            f"ratio={r['fused_vs_upstream_nstep']:.3f} native_fraction={r['wrapper_fused']['native_fraction']:.2f}",
            flush=True)
  doc = {"schema": "rsb-bench-throughput/1", "host": host(), "args": vars(args), "results": results,
         "target": "wrapper_vs_upstream_nstep >= 0.90 (design §9.1)",
         "met": {f"{r['model']}/T{r['threads']}": r["wrapper_vs_upstream_nstep"] >= 0.90 for r in results},
         "met_with_control_rate_health": {f"{r['model']}/T{r['threads']}": r["fused_vs_upstream_nstep"] >= 0.90
                                          for r in results}}
  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
