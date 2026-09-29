# SPDX-License-Identifier: Apache-2.0
"""Policy inference service curves at the runtime level (design §6.4, §9.1): latency and
rows/s versus batch size for an MLP policy, per runtime and device available here.

  python qualification/bench_inference.py --out evidence/perf/inference-curves.json

Measures LocalPolicyRuntime (the engine inside both local and central inference). The
central service adds batching and RPC on top. CUDA curves are BLOCKED (no CUDA device).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from raysimbatch.policy.export import gaussian_bundle
from raysimbatch.policy.runtime import LocalPolicyRuntime


def curve(rt: LocalPolicyRuntime, obs_dim: int, sizes, seconds: float) -> list[dict]:
  rows = []
  for b in sizes:
    obs = np.random.default_rng(0).normal(size=(b, obs_dim)).astype(np.float32)
    rec = {}
    for _ in range(20):  # warmup (MPS kernel compilation, allocator)
      rt.infer(obs, rec, deterministic=True)
    lat, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
      t = time.perf_counter()
      rt.infer(obs, rec, deterministic=True)
      lat.append(time.perf_counter() - t)
    lat.sort()
    rows.append({"batch": b, "calls": len(lat), "p50_ms": 1e3 * statistics.median(lat),
                 "p99_ms": 1e3 * lat[min(len(lat) - 1, int(0.99 * len(lat)))],
                 "rows_per_s": b * len(lat) / sum(lat)})
  return rows


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--obs-dim", type=int, default=36)
  ap.add_argument("--act-dim", type=int, default=8)
  ap.add_argument("--sizes", default="1,8,64,256,1024")
  ap.add_argument("--seconds", type=float, default=1.0)
  ap.add_argument("--out")
  args = ap.parse_args(argv)
  sizes = [int(x) for x in args.sizes.split(",")]
  results = []
  onnx_bundle, onnx_exe = gaussian_bundle(args.obs_dim, args.act_dim, seed=0)
  configs = [("onnx", "cpu", onnx_bundle, onnx_exe)]
  try:
    import torch

    from raysimbatch.policy.bundle import make_bundle
    from raysimbatch.policy.export import torchscript_linear

    ts_exe = torchscript_linear(np.random.default_rng(0).normal(size=(args.act_dim, args.obs_dim)))
    ts_bundle = make_bundle("linear_ts", 1, "torchscript", ts_exe, args.obs_dim, args.act_dim,
                            certified_batch_sizes=tuple(sizes), deterministic_inference=True)
    configs.append(("torchscript", "cpu", ts_bundle, ts_exe))
    if torch.backends.mps.is_available():
      configs.append(("torchscript", "mps", ts_bundle, ts_exe))
  except ImportError:
    pass
  for runtime, device, bundle, exe in configs:
    rt = LocalPolicyRuntime(bundle, exe, device=device)
    for row in curve(rt, args.obs_dim, sizes, args.seconds):
      row.update(runtime=runtime, device=device, policy=bundle.name)
      results.append(row)
      print(f"{runtime:11s} {device:4s} batch={row['batch']:5d} p50={row['p50_ms']:8.3f} ms "
            f"p99={row['p99_ms']:8.3f} ms rows/s={row['rows_per_s']:12,.0f}", flush=True)
  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"schema": "rsb-bench-inference/1", "args": vars(args),
                                          "results": results}, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
