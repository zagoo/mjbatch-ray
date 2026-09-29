# SPDX-License-Identifier: Apache-2.0
"""Serial per-call overhead of MJBatchCore.advance_control: the wall time of a call minus
the time spent inside MJBatch's own calls (step and forward). This is RaySimBatch's work
around the physics (kernels, bookkeeping, Python); it does not shrink with MJBatch
threads, so its share of an interval grows with them.

Scenarios, for each model, slice size and stepping path (per-substep: default health;
fused: health.check_each_substep=false, AM-2):

* ``steady``: no slot ends (termination disabled);
* ``termination``: before each call one slot is set to terminate a few substeps into the
  interval, so fused stepping replays it; ended slots are reset after the call (the
  re-arming and resets are not timed).

Medians over --calls calls after a warmup. Measurements on *this* host only.

  python qualification/bench_overhead.py --out evidence/perf/overhead.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from raysimbatch.testing import fixtures as F


class _Timed:
  """Proxy around the core's MJBatch Batch that accumulates time inside its calls."""

  def __init__(self, batch):
    self._inner, self.seconds = batch, 0.0

  def _timed(self, name, *a, **kw):
    t0 = time.perf_counter()
    try:
      return getattr(self._inner, name)(*a, **kw)
    finally:
      self.seconds += time.perf_counter() - t0

  def step(self, *a, **kw):
    return self._timed("step", *a, **kw)

  def forward(self, *a, **kw):
    return self._timed("forward", *a, **kw)

  def __getattr__(self, name):
    return getattr(self._inner, name)


def measure(name: str, slots: int, fused: bool, scenario: str, calls: int, threads: int) -> dict:
  fx = F.FIXTURES[name](max_control_steps=10**9)
  if scenario == "steady":
    fx = F.no_termination(fx)
  health = fx.group.health.model_copy(update={"check_each_substep": not fused})
  core = fx.core(num_slots=slots, threads=threads, health=health)
  s = np.arange(slots, dtype=np.int32)
  ep = np.zeros(slots, np.uint64)
  core.reset_exact(s, s.astype(np.uint64), ep, (s.astype(np.uint64), ep))
  act = (0.1 * np.random.default_rng(0).uniform(-1, 1, (slots, core.A))).astype(np.float32)
  q, v = core.layout["qpos"][0], core.layout["qvel"][0]
  one = np.array([3], np.int32)

  def arm():  # slot 3 falls a few substeps into the next interval (balance task)
    if scenario == "termination":
      core.reset_exact(one, one.astype(np.uint64), ep[:1], (one.astype(np.uint64), ep[:1]))
      core.h.state[3, q + 1], core.h.state[3, v + 1] = 0.595, 1.0
      core.h.forward(one)

  timed = _Timed(core.h._batch)
  core.h._batch = timed
  walls, natives = [], []
  for i in range(calls + 50):
    arm()
    timed.seconds = 0.0
    t0 = time.perf_counter()
    out = core.advance_control(s, act)
    wall = time.perf_counter() - t0
    if i >= 50:
      walls.append(wall)
      natives.append(timed.seconds)
    if scenario == "termination":
      assert out.status[3] != 0, "the armed slot did not end"
      ended = s[out.status != 0]  # other slots that fell too: start them again (not timed)
      core.reset_exact(ended, ended.astype(np.uint64), ep[: ended.size], (ended.astype(np.uint64), ep[: ended.size]))
  core.h._batch = timed._inner
  over = [w - n for w, n in zip(walls, natives, strict=True)]
  return {"model": name, "slots": slots, "threads": threads, "stepping": core.stepping, "scenario": scenario,
          "wall_us": 1e6 * statistics.median(walls), "mjbatch_us": 1e6 * statistics.median(natives),
          "overhead_us": 1e6 * statistics.median(over), "replayed_slots": core.replayed_slots}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--models", default="cartpole,reacher")
  ap.add_argument("--slots", default="64,256")
  ap.add_argument("--threads", type=int, default=1)
  ap.add_argument("--calls", type=int, default=600)
  ap.add_argument("--out")
  args = ap.parse_args(argv)
  rows = []
  for name in args.models.split(","):
    scenarios = ("steady", "termination") if name == "cartpole" else ("steady",)
    for n in (int(x) for x in args.slots.split(",")):
      for scenario in scenarios:
        for fused in (False, True):
          r = measure(name, n, fused, scenario, args.calls, args.threads)
          rows.append(r)
          print(f"{name:9s} slots={n:3d} {r['stepping']:11s} {scenario:11s} wall {r['wall_us']:8.1f} us  "
                f"in MJBatch {r['mjbatch_us']:8.1f} us  overhead {r['overhead_us']:6.1f} us", flush=True)
  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"schema": "rsb-bench-overhead/1", "args": vars(args), "results": rows},
                                         indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
