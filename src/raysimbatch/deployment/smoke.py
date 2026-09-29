# SPDX-License-Identifier: Apache-2.0
"""Deployment smoke test, run inside a RayCluster (e.g. ``kubectl exec`` on the head).

Registers reference artifacts in the cluster's artifact store, submits a small Mode B
evaluation (and optionally a Mode A rollout) through the normal driver/controller
path, waits, and checks the durable results. Prints one JSON document; exit status 0
only when every check passed.

Cross-cluster resume (release rollback qualification):

* ``--mode xresume-start`` on cluster X submits a longer evaluation, waits until some
  units are committed, prints the run id and leaves the run executing on X;
* X is then destroyed (a failed release) and traffic rolled back;
* ``--mode xresume-finish --run-id R`` on cluster Y resumes R from the durable store,
  runs the same configuration fresh as a reference, and requires both to be complete
  with identical per-episode semantic digests.

``--mode compare-runs --run-id R --reference-run Q`` starts nothing: both runs must be
COMPLETED with identical per-episode semantic digests (e.g. an orphan the janitor
resumed after its cluster was replaced, against an uninterrupted run of the same
configuration).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any

import numpy as np


def _eval_config(base, *, max_steps: int, seeds: tuple[int, ...] | None = None):
  from raysimbatch.policy.bundle import make_bundle
  from raysimbatch.policy.export import cartpole_bundle, linear_policy
  from raysimbatch.testing import fixtures as F
  from raysimbatch.testing import runs as R

  root = base.artifact_root
  g = R.group_from_fixture(root, F.cartpole(max_control_steps=max_steps))
  good, gexe = cartpole_bundle()
  zexe = linear_policy(np.zeros((1, 4)))
  zero = make_bundle("zero", 1, "onnx", zexe, 4, 1, certified_batch_sizes=(1, 4, 8, 16, 32, 64, 128))
  pols = (R.policy_spec(root, good, gexe, "lqr"), R.policy_spec(root, zero, zexe, "zero"))
  task = R.task_spec(root, "balance", [g])
  upd: dict = {"models": (g,), "policies": pols, "tasks": (task,), "mode": "eval"}
  if seeds is not None:
    upd["evaluation"] = base.evaluation.model_copy(update={"seeds": seeds})
  return base.model_copy(update=upd), good, zero


def _mode_b(driver, cfg, good, zero, timeout: float) -> dict:
  from raysimbatch.control.driver import RunRequest

  rid = asyncio.run(driver.submit(RunRequest("eval", cfg, cfg.tasks, "smoke", {}), uuid.uuid4().hex))
  st = asyncio.run(driver.wait(rid, timeout_s=timeout))
  return _eval_check(cfg, rid, st, good, zero)


def _eval_check(cfg, rid, st, good, zero) -> dict:
  from raysimbatch.protocol import Artifact
  from raysimbatch.storage.blob import open_blob_store

  arts = st.artifacts or {}
  c = {"run_id": rid, "state": st.state, "error": st.error, "complete": arts.get("complete"),
       "coverage": arts.get("coverage"), "committed_units": st.committed_units, "epoch": st.epoch}
  ok = st.state == "COMPLETED" and bool(arts.get("complete")) and arts.get("coverage") == 1.0
  if ok:
    agg = json.loads(open_blob_store(cfg.artifact_root).get(Artifact(**arts["aggregate_json"])))
    succ = {a["policy_hash"]: a["mean"] for a in agg["aggregates"] if a["metric"] == "success"}
    c["success"] = {"lqr": succ.get(good.policy_hash), "zero": succ.get(zero.policy_hash)}
    ok = succ.get(good.policy_hash) == 1.0 and succ.get(zero.policy_hash) == 0.0
  c["passed"] = bool(ok)
  return c


def main(argv=None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--config", default="/etc/raysimbatch/run-config.json")
  p.add_argument("--namespace", default="raysimbatch")
  p.add_argument("--mode", choices=("eval", "rollout", "both", "xresume-start", "xresume-finish", "xresume-compare",
                                     "compare-runs"),
                 default="eval")
  p.add_argument("--run-id")
  p.add_argument("--reference-run", help="compare-runs: the uninterrupted run")
  p.add_argument("--timeout", type=float, default=600)
  p.add_argument("--seeds", type=int, default=12, help="xresume-start: evaluation seeds")
  p.add_argument("--steps", type=int, default=1500, help="xresume-start: episode length (control steps)")
  p.add_argument("--handoff-units", type=int, default=3, help="xresume-start: return after this many units")
  args = p.parse_args(argv)
  if args.mode in ("xresume-finish", "xresume-compare", "compare-runs") and not args.run_id:
    p.error(f"--mode {args.mode} needs --run-id")
  if args.mode == "compare-runs" and not args.reference_run:
    p.error("--mode compare-runs needs --reference-run")

  import ray

  from raysimbatch.compat.mjbatch_adapter import check_environment
  from raysimbatch.config import ConfigurationSystem
  from raysimbatch.config.models import RunConfig
  from raysimbatch.control.driver import ClusterDriver, RunRequest
  from raysimbatch.testing import runs as R

  t0 = time.time()
  base = RunConfig.model_validate_json(open(args.config).read()) if args.config.endswith(".json") else (
    ConfigurationSystem().load(args.config))
  root = base.artifact_root
  if root is None:
    raise SystemExit("the smoke configuration needs an artifact_root")
  ray.init(address="auto", namespace=args.namespace, log_to_driver=False)
  report: dict = {"compat": check_environment().__dict__, "image": os.environ.get("RSB_GIT_SHA"),
                  "mjbatch": os.environ.get("RSB_MJBATCH_SHA"), "nodes": len(ray.nodes()), "checks": {}}
  ok = True
  if args.mode in ("eval", "both"):
    cfg, good, zero = _eval_config(base, max_steps=150)
    c = _mode_b(ClusterDriver(cfg, namespace=args.namespace), cfg, good, zero, args.timeout)
    report["checks"]["mode_b"] = c
    ok &= c["passed"]
  if args.mode in ("rollout", "both"):
    from raysimbatch.config.models import RolloutConfig
    from raysimbatch.policy.export import gaussian_bundle

    cfg, _, _ = _eval_config(base, max_steps=150)
    gb, gexe2 = gaussian_bundle(4, 1, seed=1, name="cp_pol")
    # Two simulation actors: fits the smallest qualification cluster (validated as CAPACITY).
    rcfg = cfg.model_copy(update={"mode": "rollout", "policies": (R.policy_spec(root, gb, gexe2),),
                                  "tasks": (), "target_envs": 2 * cfg.worker.envs,
                                  "rollout": RolloutConfig(target_transitions=5000, fragment_steps=16)})
    driver = ClusterDriver(rcfg, namespace=args.namespace)
    rid = asyncio.run(driver.submit(RunRequest("rollout", rcfg, (), "smoke", {"target_transitions": 5000}),
                                    uuid.uuid4().hex))
    st = asyncio.run(driver.wait(rid, timeout_s=args.timeout))
    arts = st.artifacts or {}
    committed = sum(int((c.get("program") or {}).get("committed_transitions", 0)) for c in arts.get("coordinators", []))
    passed = st.state == "COMPLETED" and committed >= 5000 and arts.get("gaps") == 0
    report["checks"]["mode_a"] = {"run_id": rid, "state": st.state, "error": st.error, "committed_transitions": committed,
                                  "gaps": arts.get("gaps"), "passed": passed}
    ok &= passed
  if args.mode.startswith("xresume"):
    cfg, good, zero = _eval_config(base, max_steps=args.steps, seeds=tuple(range(args.seeds)))
    if args.mode in ("xresume-finish", "xresume-compare"):
      # The run's own stored configuration, never a recomputation that could differ.
      from raysimbatch.storage.store import DurableStore

      row = _stored_run(DurableStore.from_config(base, migrate=False).db, args.run_id)
      cfg = RunConfig.model_validate_json(json.dumps(row["payload"]["resolved"])).model_copy(update={"run_id": None})
    driver = ClusterDriver(cfg, namespace=args.namespace)
    if args.mode == "xresume-start":
      rid = asyncio.run(driver.submit(RunRequest("eval", cfg, cfg.tasks, "xresume", {}), uuid.uuid4().hex))
      deadline = time.monotonic() + args.timeout
      while (n := asyncio.run(driver.status(rid)).committed_units) < args.handoff_units:
        if time.monotonic() > deadline:
          ok = False
          break
        time.sleep(0.5)
      report["checks"]["xresume_start"] = {"run_id": rid, "committed_units_at_handoff": n, "passed": ok}
    else:
      from raysimbatch.evaluation.schema import semantic_digests

      rid = args.run_id
      before = asyncio.run(driver.status(rid))
      if args.mode == "xresume-finish":
        asyncio.run(driver.resume(rid, cfg))
      st = asyncio.run(driver.wait(rid, timeout_s=args.timeout))  # compare: the run already finished
      resumed = _eval_check(cfg, rid, st, good, zero)
      resumed["committed_units_before_resume"] = before.committed_units
      resumed["state_before_resume"] = before.state
      ref = _mode_b(driver, cfg, good, zero, args.timeout)
      a, b = semantic_digests(driver.store.db, rid), semantic_digests(driver.store.db, ref["run_id"])
      same = bool(a) and a == b
      report["checks"]["xresume_finish"] = {"resumed": resumed, "reference": ref, "episodes": len(a),
                                            "identical_to_reference": same,
                                            "passed": resumed["passed"] and ref["passed"] and same}
      ok &= report["checks"]["xresume_finish"]["passed"]
  if args.mode == "compare-runs":
    from raysimbatch.evaluation.schema import semantic_digests
    from raysimbatch.storage.store import DurableStore

    db = DurableStore.from_config(base, migrate=False).db
    ref = semantic_digests(db, args.reference_run)
    runs = {}
    for rid in (args.run_id, args.reference_run):
      row, d = _stored_run(db, rid), semantic_digests(db, rid)
      runs[rid] = {"state": row["state"], "attempt": row["attempt"], "epoch": row["epoch"],
                   "cluster": row["payload"].get("cluster"), "episodes": len(d),
                   "identical_to_reference": bool(d) and d == ref}
    passed = all(r["state"] == "COMPLETED" and r["identical_to_reference"] for r in runs.values())
    report["checks"]["compare_runs"] = {"runs": runs, "passed": passed}
    ok &= passed
  report["affinity"] = _affinity_leases()
  report["passed"] = bool(ok)
  report["seconds"] = time.time() - t0
  print(json.dumps(report, default=str, indent=2))
  return 0 if ok else 1


def _stored_run(db, run_id: str) -> dict:
  row = db.get_run(run_id)
  if row is None:
    raise SystemExit(f"run {run_id} not found in the durable store")
  return row


def _affinity_leases() -> dict:
  """Per-node affinity agent state: the pod's CPU set and current leases."""
  import ray
  from ray.util.state import list_actors

  out = {}
  actors: list[Any] = list_actors(filters=[("class_name", "=", "AffinityAgent"), ("state", "=", "ALIVE")])
  for a in actors:  # ActorState is a pydantic dataclass: its fields are invisible to pyright
    try:
      out[a.name] = ray.get(ray.get_actor(a.name, namespace=a.ray_namespace).status.remote(), timeout=10)
    except Exception as exc:  # noqa: BLE001 - reported, not fatal to the smoke verdict
      out[a.name] = {"error": str(exc)}
  return out


if __name__ == "__main__":
  sys.exit(main())
