# SPDX-License-Identifier: Apache-2.0
"""Memory accounting per actor shape (design §9.2): the core's own census of RaySimBatch
arrays plus the measured RSS growth of constructing and warming a shard (MJBatch model
copies, per-thread mjData arenas, bound mirrors and expanded rows included).

  python qualification/memory_census.py --out evidence/perf/memory-census.json

Each configuration runs in a fresh subprocess; RSS is sampled with `ps` before building
the shard and after warm-up (current resident set, not the lifetime peak, which an
earlier import-time peak would mask). RSS is not PSS; untouched arena pages are not
resident, so heavy-contact models must be measured at their worst-case contact load.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CHILD = r"""
import json, os, subprocess, sys
import numpy as np
from raysimbatch.testing import fixtures as F

def rss():  # current resident set (not the lifetime peak), bytes
  return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) * 1024

name, slots, threads, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fx = F.no_termination(F.FIXTURES[name](max_control_steps=10**9))
model = fx.model()
base = rss()
core = fx.core(num_slots=slots, threads=threads)
built = rss()
s = np.arange(slots, dtype=np.int32)
ep = np.zeros(slots, np.uint64)
core.reset_exact(s, s.astype(np.uint64), ep, (s.astype(np.uint64), ep))
act = np.zeros((slots, core.A), np.float32)
for _ in range(steps):
  core.advance_control(s, act)
warm = rss()
print(json.dumps({"model": name, "slots": slots, "threads": threads, "nv": int(model.nv), "nbuffer": int(model.nbuffer),
                  "narena": int(model.narena), "census": core.memory_census(),
                  "rss_construct_bytes": built - base, "rss_after_warmup_bytes": warm - base}))
"""


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--models", default="cartpole,quadruped,reacher")
  ap.add_argument("--slots", default="64,256")
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--steps", type=int, default=200)
  ap.add_argument("--out")
  args = ap.parse_args(argv)
  rows = []
  for name in args.models.split(","):
    for n in (int(x) for x in args.slots.split(",")):
      p = subprocess.run([sys.executable, "-c", CHILD, name, str(n), str(args.threads), str(args.steps)],
                         capture_output=True, text=True, check=True)
      r = json.loads(p.stdout.strip().splitlines()[-1])
      r["rss_per_slot_bytes"] = r["rss_after_warmup_bytes"] / n
      rows.append(r)
      c = r["census"]
      print(f"{name:10s} slots={n:4d} T={args.threads} rss+={r['rss_after_warmup_bytes'] / 2**20:7.1f} MiB "
            f"({r['rss_per_slot_bytes'] / 1024:7.1f} KiB/slot)  state={c['state_bytes'] / 2**10:7.1f} KiB "
            f"kernels={c['kernel_state_bytes'] / 2**10:7.1f} KiB bound={c['bound_bytes'] / 2**10:7.1f} KiB "
            f"narena={r['narena'] / 2**20:.1f} MiB", flush=True)
  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"schema": "rsb-memory-census/1", "rows": rows,
                                          "note": "current RSS deltas sampled with ps (not PSS)"}, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
