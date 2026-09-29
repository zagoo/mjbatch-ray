# SPDX-License-Identifier: Apache-2.0
"""raysimbatch command line: configuration, artifacts, runs, rendering, operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml


def _overrides(pairs):
  out = {}
  for p in pairs or ():
    k, _, v = p.partition("=")
    try:
      out[k] = yaml.safe_load(v)
    except yaml.YAMLError:
      out[k] = v
  return out


def _load(args):
  from raysimbatch.config import ConfigurationSystem

  return ConfigurationSystem().load(args.config, _overrides(args.set))


def cmd_validate(args) -> int:
  from raysimbatch.config import ClusterInventory, ConfigError, ConfigurationSystem

  cs = ConfigurationSystem()
  try:
    cfg = cs.load(args.config, _overrides(args.set))
  except ConfigError as exc:
    print(json.dumps([i.__dict__ for i in exc.report.issues] if exc.report else str(exc), indent=2))
    return 1
  rep = cs.validate(cfg, ClusterInventory(production=args.production), submission=not args.no_submission)
  print(json.dumps({"ok": rep.ok, "hash": cs.canonical_hash(cfg), "issues": [i.__dict__ for i in rep.issues]},
                   indent=2))
  return 0 if rep.ok else 1


def cmd_hash(args) -> int:
  from raysimbatch.config import ConfigurationSystem

  print(ConfigurationSystem().canonical_hash(_load(args)))
  return 0


def cmd_diff(args) -> int:
  from raysimbatch.config import ConfigurationSystem

  cs = ConfigurationSystem()
  d = cs.diff(cs.load(args.previous), cs.load(args.proposed))
  print(json.dumps({"classification": d.classification,
                    "changes": [{"path": p, "class": c, "old": o, "new": n} for p, c, o, n in d.changes]},
                   indent=2, default=str))
  return 0 if d.classification in (None, "hot", "boundary") else 2


def cmd_schema(args) -> int:
  from raysimbatch.config import ConfigurationSystem

  print(json.dumps(ConfigurationSystem().json_schema(), indent=2))
  return 0


def cmd_put(args) -> int:
  from raysimbatch.storage.blob import open_blob_store

  a = open_blob_store(args.root).put(Path(args.file).read_bytes(), args.media_type)
  print(json.dumps({"uri": a.uri, "sha256": a.sha256}))
  return 0


def _driver(args):
  import ray

  from raysimbatch.control.driver import ClusterDriver

  if not ray.is_initialized():
    ray.init(address=args.address, namespace=args.namespace, log_to_driver=False)
  cfg = _load(args) if getattr(args, "config", None) else None
  if cfg is None:
    raise SystemExit("--config is required to locate durable storage")
  return ClusterDriver(cfg, namespace=args.namespace), cfg


def cmd_submit(args) -> int:
  from raysimbatch.control.driver import RunRequest

  d, cfg = _driver(args)
  opts = json.loads(args.options) if args.options else {}
  rid = asyncio.run(d.submit(RunRequest(cfg.mode, cfg, cfg.tasks, args.caller, opts), args.key or uuid.uuid4().hex))
  print(rid)
  if args.wait:
    st = asyncio.run(d.wait(rid, timeout_s=args.wait))
    print(json.dumps(st.__dict__, indent=2, default=str))
    return 0 if st.state == "COMPLETED" else 3
  return 0


def cmd_status(args) -> int:
  d, _ = _driver(args)
  st = asyncio.run(d.wait(args.run_id, timeout_s=args.wait) if args.wait else d.status(args.run_id))
  print(json.dumps(st.__dict__, indent=2, default=str))
  return 0


def cmd_cancel(args) -> int:
  d, _ = _driver(args)
  asyncio.run(d.cancel(args.run_id, args.reason, drain=not args.no_drain))
  return 0


def cmd_resume(args) -> int:
  d, cfg = _driver(args)
  st = asyncio.run(d.resume(args.run_id, cfg))
  print(json.dumps(st.__dict__, indent=2, default=str))
  return 0


def cmd_render(args) -> int:
  from raysimbatch.deployment.render import PlatformInventory, render, to_yaml

  cfg = _load(args)
  inv = PlatformInventory.from_dict(yaml.safe_load(Path(args.inventory).read_text()))
  objs = render(cfg, inv, tenant=args.tenant, production=args.production, cluster_name=args.name,
                include_gpu=not args.no_gpu)
  text = to_yaml(objs)
  if args.out:
    Path(args.out).write_text(text)
  else:
    sys.stdout.write(text)
  return 0


def cmd_migrate(args) -> int:
  from raysimbatch.storage.db import Database
  from raysimbatch.storage.secrets import resolve

  print(json.dumps(Database(resolve(args.dsn_ref)).migrate()))
  return 0


def cmd_compat(args) -> int:
  from raysimbatch.compat.mjbatch_adapter import check_environment

  r = check_environment()
  print(json.dumps({**r.__dict__, "abi_fingerprint": r.abi_fingerprint}, indent=2))
  return 0


def cmd_janitor(args) -> int:
  import ray

  from raysimbatch.recovery.watchdog import janitor_pass
  from raysimbatch.storage.store import DurableStore

  cfg = _load(args)
  if not args.loop:
    ray.init(address=args.address, namespace=args.namespace, log_to_driver=False)
    print(json.dumps(asyncio.run(janitor_pass(DurableStore.from_config(cfg), args.namespace))))
    return 0
  # Cluster janitor (design §7): reconcile every --loop seconds for the life of the
  # process (a head-pod sidecar); every failure is logged and retried next pass.
  import time

  from raysimbatch.control.driver import wait_for_non_head_node

  store, waited = None, False
  while True:
    try:
      if not ray.is_initialized():
        ray.init(address=args.address, namespace=args.namespace, log_to_driver=False)
      if not waited:  # resumed controllers need a worker node to land on (not the head)
        print(json.dumps({"janitor_worker_node": wait_for_non_head_node(args.startup_wait_s)}), flush=True)
        waited = True
      store = store or DurableStore.from_config(cfg, migrate=False)
      print(json.dumps({"janitor": asyncio.run(janitor_pass(store, args.namespace))}), flush=True)
    except Exception as exc:  # noqa: BLE001 - the head may not be up yet, or the database may be down
      print(json.dumps({"janitor_error": f"{type(exc).__name__}: {exc}"[:500]}), flush=True)
    time.sleep(args.loop)


def cmd_drain_node(args) -> int:
  """preStop hook: drain this node's simulation workers at episode boundaries, then
  let the pod terminate within the grace period."""
  try:
    import ray

    ray.init(address="auto", log_to_driver=False)
    node = ray.get_runtime_context().get_node_id()
    from ray.util.state import list_actors

    from raysimbatch.compat.ray_typing import get_one

    deadline = time.time_ns() + int(args.grace_s * 1e9)
    handles = []
    actors: list[Any] = list_actors(filters=[("node_id", "=", node), ("state", "=", "ALIVE")], limit=10_000)
    for a in actors:  # ActorState is a pydantic dataclass: its fields are invisible to pyright
      if a.class_name == "SimulationWorker" and a.name:
        try:
          handles.append(ray.get_actor(a.name, namespace=a.ray_namespace))
        except ValueError:
          pass
    pending = {h: True for h in handles}
    while pending and time.time_ns() < deadline:
      for h in list(pending):
        try:
          if not get_one(h.drain.remote(deadline), timeout=2)["pending"]:
            pending.pop(h)
        except Exception:  # noqa: BLE001
          pending.pop(h)
      time.sleep(1)
  except Exception as exc:  # noqa: BLE001 - never block pod termination beyond the grace
    print(f"drain-node: {exc}", file=sys.stderr)
  return 0


def main(argv=None) -> int:
  p = argparse.ArgumentParser(prog="raysimbatch")
  sub = p.add_subparsers(dest="cmd", required=True)

  def cfg_args(sp, required=True):
    sp.add_argument("--config", required=required)
    sp.add_argument("--set", action="append", help="dotted override path=value (repeatable)")

  def ray_args(sp):
    sp.add_argument("--address", default="auto")
    sp.add_argument("--namespace", default="raysimbatch")

  c = sub.add_parser("config", help="validate/hash/diff/schema")
  csub = c.add_subparsers(dest="sub", required=True)
  v = csub.add_parser("validate")
  cfg_args(v)
  v.add_argument("--production", action="store_true")
  v.add_argument("--no-submission", action="store_true")
  v.set_defaults(fn=cmd_validate)
  h = csub.add_parser("hash")
  cfg_args(h)
  h.set_defaults(fn=cmd_hash)
  d = csub.add_parser("diff")
  d.add_argument("previous")
  d.add_argument("proposed")
  d.set_defaults(fn=cmd_diff)
  s = csub.add_parser("schema")
  s.set_defaults(fn=cmd_schema)
  a = sub.add_parser("artifacts", help="register content-addressed artifacts")
  asub = a.add_subparsers(dest="sub", required=True)
  put = asub.add_parser("put")
  put.add_argument("file")
  put.add_argument("--root", required=True)
  put.add_argument("--media-type", default="application/octet-stream")
  put.set_defaults(fn=cmd_put)
  sm = sub.add_parser("submit")
  cfg_args(sm)
  ray_args(sm)
  sm.add_argument("--key")
  sm.add_argument("--caller", default="cli")
  sm.add_argument("--options", help="JSON run options (e.g. actors, axis_of_group)")
  sm.add_argument("--wait", type=float, default=0.0)
  sm.set_defaults(fn=cmd_submit)
  st = sub.add_parser("status")
  cfg_args(st)
  ray_args(st)
  st.add_argument("run_id")
  st.add_argument("--wait", type=float, default=0.0)
  st.set_defaults(fn=cmd_status)
  cn = sub.add_parser("cancel")
  cfg_args(cn)
  ray_args(cn)
  cn.add_argument("run_id")
  cn.add_argument("--reason", default="operator")
  cn.add_argument("--no-drain", action="store_true")
  cn.set_defaults(fn=cmd_cancel)
  rs = sub.add_parser("resume")
  cfg_args(rs)
  ray_args(rs)
  rs.add_argument("run_id")
  rs.set_defaults(fn=cmd_resume)
  rd = sub.add_parser("render", help="KubeRay manifests from config + platform inventory")
  cfg_args(rd)
  rd.add_argument("--inventory", required=True)
  rd.add_argument("--tenant", default="default")
  rd.add_argument("--name")
  rd.add_argument("--production", action="store_true")
  rd.add_argument("--no-gpu", action="store_true")
  rd.add_argument("--out")
  rd.set_defaults(fn=cmd_render)
  db = sub.add_parser("db")
  dsub = db.add_subparsers(dest="sub", required=True)
  mg = dsub.add_parser("migrate")
  mg.add_argument("--dsn-ref", required=True, help="secret reference, e.g. env:RSB_DATABASE_DSN")
  mg.set_defaults(fn=cmd_migrate)
  cp = sub.add_parser("compat", help="verify the installed MJBatch/MuJoCo pair")
  cp.set_defaults(fn=cmd_compat)
  jn = sub.add_parser("janitor", help="reconcile controllers, placement groups, cache holds and orphaned runs")
  cfg_args(jn)
  ray_args(jn)
  jn.add_argument("--loop", type=float, default=0.0, help="run as the cluster janitor: one pass every N seconds")
  jn.add_argument("--startup-wait-s", type=float, default=120.0,
                  help="with --loop: before the first pass, wait up to N seconds for a worker node to join")
  jn.set_defaults(fn=cmd_janitor)
  dn = sub.add_parser("drain-node")
  dn.add_argument("--grace-s", type=float, default=110)
  dn.set_defaults(fn=cmd_drain_node)
  args = p.parse_args(argv)
  return int(args.fn(args) or 0)


if __name__ == "__main__":
  raise SystemExit(main())
