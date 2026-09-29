# SPDX-License-Identifier: Apache-2.0
"""deploy/release.sh against a fake kubectl (state in a JSON file, every call recorded).

What an operator relies on during a production release:
* promote and rollback route admissions: the raysimbatch-active Service selects the
  chosen color's head, and the pointer records active/previous;
* re-promoting keeps the rollback target;
* admissions are never routed to a cluster that is not ready, and the active release is
  never retired;
* every kubectl call names the context and namespace given on the command line (never
  whichever context happens to be current)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FAKE = r'''
import fcntl, json, os, sys, yaml
# release.sh pipes `kubectl create ... | kubectl apply -f -`: two fakes run at once. Each
# takes the lock for its whole read-modify-write (apply reads its stdin first, or it
# would wait on a writer that waits on the lock) and replaces the state atomically.
path = os.environ["FAKE_KUBE_STATE"]
argv = sys.argv[1:]
stdin = sys.stdin.read() if "apply" in argv else ""
lock = open(path + ".lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)
st = json.load(open(path))
st["calls"].append(argv)
ctx = ns = None
rest = []
i = 0
while i < len(argv):
  if argv[i] == "--context": ctx = argv[i + 1]; i += 2
  elif argv[i] == "-n": ns = argv[i + 1]; i += 2
  else: rest.append(argv[i]); i += 1
def done(code=0, out=""):
  with open(path + ".tmp", "w") as f:
    json.dump(st, f)
  os.replace(path + ".tmp", path)
  sys.stdout.write(out)
  sys.exit(code)
objs = st["objects"]
verb = rest[0]
if verb == "get":
  kind, name = rest[1], rest[2]
  path_ = rest[rest.index("-o") + 1].removeprefix("jsonpath=")
  if kind == "raycluster":
    if name not in st["rayclusters"]: done(1)
    done(0, st["rayclusters"][name])
  obj = objs.get(f"ConfigMap/{name}")
  if obj is None: done(1)
  if path_ == "{.data}": done(0, json.dumps(obj["data"]))
  done(0, obj["data"].get(path_[len("{.data."):-1], ""))
if verb == "create":
  lits = dict(a.removeprefix("--from-literal=").split("=", 1) for a in rest if a.startswith("--from-literal="))
  done(0, yaml.safe_dump({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": rest[2]}, "data": lits}))
if verb == "apply":
  obj = yaml.safe_load(stdin)
  old = objs.get(f"{obj['kind']}/{obj['metadata']['name']}", {})
  obj["metadata"].setdefault("annotations", {}).update(old.get("metadata", {}).get("annotations", {}) if obj["kind"] == "ConfigMap" else {})
  objs[f"{obj['kind']}/{obj['metadata']['name']}"] = obj
  done()
if verb == "annotate":
  key, value = rest[3].split("=", 1)
  objs[f"ConfigMap/{rest[2]}"]["metadata"].setdefault("annotations", {})[key] = value
  done()
if verb == "delete":
  st["rayclusters"].pop(rest[2], None)
  done()
done(9)
'''


@pytest.fixture
def kube(tmp_path):
  bin_dir = tmp_path / "bin"
  bin_dir.mkdir()
  (bin_dir / "kubectl").write_text(f"#!{sys.executable}\n{FAKE}")
  (bin_dir / "kubectl").chmod(0o755)
  state = tmp_path / "state.json"
  state.write_text(json.dumps({"rayclusters": {"rsb-blue": "ready", "rsb-green": "ready"}, "objects": {},
                               "calls": []}))

  def release(*args, context="prod-east", namespace="rsb-prod"):
    flags = [*(["--context", context] if context else []), *(["--namespace", namespace] if namespace else [])]
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_KUBE_STATE": str(state)}
    return subprocess.run(["bash", str(ROOT / "deploy/release.sh"), *flags, *args], env=env, capture_output=True,
                          text=True)

  def read():
    return json.loads(state.read_text())

  def set_cluster(name, value):
    s = read()
    s["rayclusters"][name] = value
    state.write_text(json.dumps(s))

  release.read, release.set_cluster = read, set_cluster
  return release


def routed(st):
  svc = st["objects"]["Service/raysimbatch-active"]
  sel = svc["spec"]["selector"]
  assert sel["ray.io/node-type"] == "head"
  assert {p["port"] for p in svc["spec"]["ports"]} == {10001, 8265}
  data = st["objects"]["ConfigMap/raysimbatch-active"]["data"]
  return sel["ray.io/cluster"], data["active"], data["previous"]


def test_promote_and_rollback_route_admissions(kube):
  assert kube("promote", "blue").returncode == 0
  r = kube("promote", "green")
  assert r.returncode == 0, r.stderr
  assert routed(kube.read()) == ("rsb-green", "green", "blue")
  assert kube("promote", "green").returncode == 0  # idempotent
  assert routed(kube.read()) == ("rsb-green", "green", "blue"), "re-promoting erased the rollback target"
  r = kube("rollback")
  assert r.returncode == 0, r.stderr
  assert routed(kube.read()) == ("rsb-blue", "blue", "green")
  calls = kube.read()["calls"]
  assert calls and all(c[:4] == ["--context", "prod-east", "-n", "rsb-prod"] for c in calls), calls


def test_admissions_are_never_routed_to_a_cluster_that_is_not_ready(kube):
  assert kube("promote", "blue").returncode == 0
  kube.set_cluster("rsb-green", "failed")
  r = kube("promote", "green")
  assert r.returncode == 1 and "not ready" in r.stderr
  kube.set_cluster("rsb-green", "ready")
  assert kube("promote", "green").returncode == 0
  kube.set_cluster("rsb-blue", "suspended")
  r = kube("rollback")  # the rollback target broke meanwhile
  assert r.returncode == 1 and "rsb-blue is not ready" in r.stderr
  assert routed(kube.read()) == ("rsb-green", "green", "blue")  # unchanged


def test_the_active_release_is_never_retired(kube):
  assert kube("promote", "blue").returncode == 0
  assert kube("promote", "green").returncode == 0
  r = kube("retire", "green")
  assert r.returncode == 1 and "active release" in r.stderr
  assert "rsb-green" in kube.read()["rayclusters"]
  assert kube("retire", "blue").returncode == 0
  assert "rsb-blue" not in kube.read()["rayclusters"]
  r = kube("rollback")  # its target is gone
  assert r.returncode == 1 and "absent" in r.stderr


@pytest.mark.parametrize("missing", ["context", "namespace"])
def test_context_and_namespace_are_required(kube, missing):
  r = kube("promote", "green", **{missing: ""})
  assert r.returncode == 2 and "usage" in r.stderr
  assert kube.read()["calls"] == []  # nothing touched
