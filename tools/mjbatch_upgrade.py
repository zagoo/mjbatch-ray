# SPDX-License-Identifier: Apache-2.0
"""MJBatch upgrade workflow: candidate in isolation, gates, then pin; rollback restores.

  python tools/mjbatch_upgrade.py status
  python tools/mjbatch_upgrade.py prepare <ref> [--mujoco X.Y.Z] [--no-fetch]
  python tools/mjbatch_upgrade.py qualify <sha12> [--baseline evidence/perf/<file>.json]
  python tools/mjbatch_upgrade.py promote <sha12>     # only with a passing qualification
  python tools/mjbatch_upgrade.py linux <sha12>       # required after promote, before commit
  python tools/mjbatch_upgrade.py rollback            # back to the committed pin
  python tools/mjbatch_upgrade.py cleanup <sha12>

Isolation: the candidate is a detached `git worktree` of the pinned checkout's repository
under var/upgrade/<sha12>/mjbatch, built into its own virtualenv with its build
directory outside the worktree. Neither third_party/mjbatch's working tree, the
superproject pin, the project .venv nor any user checkout is touched before `promote`.
`promote` changes the pin (submodule checkout, QUALIFIED table, pyproject) and leaves
the result uncommitted for review; `rollback` restores all three from HEAD.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "third_party/mjbatch"
UPGRADE = ROOT / "var/upgrade"
ADAPTER = ROOT / "src/raysimbatch/compat/mjbatch_adapter.py"


def sh(*cmd: str, cwd: Path | None = None, env: dict | None = None, check: bool = True,
       log: Path | None = None) -> subprocess.CompletedProcess:
  t0 = time.time()
  p = subprocess.run(list(cmd), cwd=cwd, env=env, capture_output=True, text=True)
  if log is not None:
    with log.open("a") as f:
      f.write(f"$ {' '.join(cmd)}  (cwd={cwd}, exit={p.returncode}, {time.time() - t0:.1f}s)\n{p.stdout}{p.stderr}\n")
  if check and p.returncode != 0:
    raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
  return p


def committed_pin() -> str:
  return sh("git", "ls-tree", "HEAD", "third_party/mjbatch", cwd=ROOT).stdout.split()[2]


def cmd_status(_args) -> int:
  import importlib.metadata as md

  from raysimbatch.compat.mjbatch_adapter import QUALIFIED

  print(json.dumps({
    "committed_pin": committed_pin(),
    "checkout_head": sh("git", "rev-parse", "HEAD", cwd=PIN).stdout.strip(),
    "qualified": QUALIFIED,
    "installed": {"mjbatch": md.version("mjbatch"), "mujoco": md.version("mujoco")},
    "candidates": sorted(p.name for p in UPGRADE.glob("*") if p.is_dir()) if UPGRADE.exists() else [],
  }, indent=2))
  return 0


def cmd_prepare(args) -> int:
  if not args.no_fetch:
    p = sh("git", "fetch", "origin", args.ref.removeprefix("origin/"), cwd=PIN, check=False)
    if p.returncode != 0:
      print(f"fetch failed ({p.stderr.strip()[:200]}); using refs already present", file=sys.stderr)
  ref = args.ref
  sha = sh("git", "rev-parse", "--verify", f"{ref}^{{commit}}", cwd=PIN).stdout.strip()
  d = UPGRADE / sha[:12]
  if d.exists():
    raise SystemExit(f"{d} exists; run cleanup first")
  d.mkdir(parents=True)
  log = d / "prepare.log"
  sh("git", "worktree", "add", "--detach", str(d / "mjbatch"), sha, cwd=PIN, log=log)
  version = re.search(r'^version\s*=\s*"([^"]+)"', (d / "mjbatch/pyproject.toml").read_text(), re.M).group(1)
  mujoco = args.mujoco or re.search(r'"mujoco==([^"]+)"', (ROOT / "pyproject.toml").read_text()).group(1)
  py = d / "venv/bin/python"
  sh("uv", "venv", "--python", "3.12", str(d / "venv"), cwd=ROOT, log=log)
  req = d / "requirements.txt"  # the release lock, minus the two packages built here
  req.write_text(sh("uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--no-emit-package", "mjbatch",
                    "--no-emit-package", "mujoco", "--extra", "torch", cwd=ROOT).stdout)
  sh("uv", "pip", "install", "--python", str(py), "-r", str(req), f"mujoco=={mujoco}", cwd=ROOT, log=log)
  build = d / "build"
  # --no-config: the project's own [tool.uv] build-dir for the *pinned* mjbatch must not
  # merge with the candidate's (observed: scikit-build received a list and failed).
  # No bytecode: observed, candidate 121e033's build hook (mujoco_metadata.py) otherwise
  # leaves __pycache__ inside the upstream tree.
  benv = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
  sh("uv", "pip", "install", "--no-config", "--python", str(py), "--no-deps", "-C",
     f"build-dir={build}/{{wheel_tag}}", str(d / "mjbatch"), cwd=ROOT, env=benv, log=log)
  sh("uv", "pip", "install", "--no-config", "--python", str(py), "--no-deps", "-C", f"build-dir={build}/rsb",
     str(ROOT), cwd=ROOT, log=log)
  # The version the build actually produced (observed: 121e033 declares 0.1.0 but builds
  # 0.1.0.post31300); QUALIFIED and the pyproject pin must use this one.
  built = sh(str(py), "-c", "import importlib.metadata as m; print(m.version('mjbatch'))", cwd=d).stdout.strip()
  manifest = {"ref": args.ref, "sha": sha, "version": built, "declared_version": version, "mujoco": mujoco,
              "worktree": str(d / "mjbatch"),
              "venv": str(d / "venv"), "committed_pin": committed_pin(), "prepared_at": time.time()}
  (d / "candidate.json").write_text(json.dumps(manifest, indent=2))
  print(json.dumps(manifest, indent=2))
  return 0


def _load(sha12: str) -> tuple[Path, dict]:
  d = UPGRADE / sha12
  return d, json.loads((d / "candidate.json").read_text())


def cmd_qualify(args) -> int:
  d, man = _load(args.sha12)
  py = d / "venv/bin/python"
  env = {**os.environ, "RSB_MJBATCH_CANDIDATE": "1", "PYTHONPATH": ""}
  log = d / "qualify.log"
  gates: dict[str, dict] = {}

  p = sh(str(py), "-m", "raysimbatch.cli", "compat", env=env, check=False, cwd=d, log=log)
  try:
    rep = json.loads(p.stdout)
    ok = p.returncode == 0 and all(rep.get("probes", {}).values())
  except json.JSONDecodeError:
    rep, ok = {"raw": p.stdout[-500:] + p.stderr[-500:]}, False
  gates["behaviour_probes"] = {"passed": ok, "report": rep}

  p = sh(str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests/upstream"),
         str(ROOT / "tests/core"), env=env, check=False, cwd=ROOT, log=log)
  gates["upstream_contract_and_parity_tests"] = {"passed": p.returncode == 0, "summary": p.stdout.strip().splitlines()[-1:]}

  # Performance: pinned build (project .venv) and candidate measured back to back, twice
  # each (A B A B), with the same benchmark code, so host drift and methodology cannot
  # masquerade as a regression (observed: a stored baseline confounded both).
  runs: dict[str, list[dict]] = {"pinned": [], "candidate": []}
  bench = [str(ROOT / "qualification/bench_throughput.py"), "--models", "cartpole,quadruped", "--threads", "1,2",
           "--seconds", "1.5", "--repeats", "5"]
  ok_runs = True
  for i, (label, python, e) in enumerate([("pinned", str(ROOT / ".venv/bin/python"), None), ("candidate", str(py), env)] * 2):
    out = d / f"throughput-{label}-{i}.json"
    p = sh(python, *bench, "--out", str(out), env=e, check=False, cwd=ROOT, log=log)
    ok_runs &= p.returncode == 0
    if p.returncode == 0:
      runs[label].append(json.loads(out.read_text()))
  detail, passed = {}, ok_runs
  if ok_runs:
    def med(label, key):
      vals = [r["wrapper"]["median"] for doc in runs[label] for r in doc["results"]
              if (r["model"], r["threads"]) == key]
      return sum(vals) / len(vals)
    for r in runs["pinned"][0]["results"]:
      key = (r["model"], r["threads"])
      change = med("candidate", key) / med("pinned", key) - 1
      detail[f"{key[0]}/T{key[1]}"] = round(change, 4)
      passed &= change >= -0.05  # design §11.5: <= 5% regression vs the previous release
  gates["performance_regression"] = {"passed": passed, "wrapper_change_vs_pinned": detail, "method": "ABAB"}
  if args.baseline:
    gates["performance_regression"]["reference_file"] = args.baseline  # informational only

  p = sh(sys.executable, str(ROOT / "tools/check_mjbatch_clean.py"), "--checkout", man["worktree"],
         "--only-checkout", check=False, cwd=ROOT, log=log)
  gates["candidate_tree_pristine"] = {"passed": p.returncode == 0, "output": p.stdout.strip()[-500:]}

  result = {"candidate": man, "gates": gates, "passed": all(g["passed"] for g in gates.values()),
            "qualified_at": time.time()}
  (d / "qualification.json").write_text(json.dumps(result, indent=2))
  print(json.dumps({k: v["passed"] for k, v in gates.items()} | {"passed": result["passed"]}, indent=2))
  return 0 if result["passed"] else 1


def cmd_promote(args) -> int:
  d, man = _load(args.sha12)
  q = d / "qualification.json"
  if not q.exists() or not json.loads(q.read_text())["passed"]:
    raise SystemExit("refusing to promote: no passing qualification.json for this candidate")
  sh("git", "checkout", "--detach", man["sha"], cwd=PIN)
  src = ADAPTER.read_text()
  entry = f'  "{man["version"]}": {{"mujoco": "{man["mujoco"]}", "revision": "{man["sha"]}"}},\n'
  src = re.sub(r"(QUALIFIED: dict\[str, dict\[str, str\]\] = \{\n)", r"\1" + entry.replace("\\", "\\\\"), src, count=1)
  ADAPTER.write_text(src)
  pj = (ROOT / "pyproject.toml").read_text()
  pj = re.sub(r'"mjbatch==[^"]+"', f'"mjbatch=={man["version"]}"', pj)
  pj = re.sub(r'"mujoco==[^"]+"', f'"mujoco=={man["mujoco"]}"', pj)
  (ROOT / "pyproject.toml").write_text(pj)
  sh("uv", "lock", cwd=ROOT)
  sh("uv", "sync", "--extra", "torch", cwd=ROOT)
  # Stage the pin change as one unit (gitlink + lock + table + pyproject), uncommitted.
  sh("git", "add", "third_party/mjbatch", "pyproject.toml", "uv.lock", str(ADAPTER.relative_to(ROOT)), cwd=ROOT)
  p = sh(sys.executable, str(ROOT / "tools/check_mjbatch_clean.py"), "--no-linux-gate", cwd=ROOT, check=False)
  print(p.stdout)
  print(sh("git", "status", "--short", cwd=ROOT).stdout)
  print(f"promoted in the working tree; next: `linux {args.sha12}` (required), the full suite, then commit")
  return p.returncode


def cmd_linux(args) -> int:
  """The Linux gate, after promote and before commit. The other gates run on macOS, which
  has no CPU affinity: the 0.1.4 thread-pool change passed them all and broke pinning on
  every Linux worker. Builds the `test` image from the working tree (by image ID, never a
  tag that a failed build would leave stale), checks the image has the promoted versions,
  and runs tools/linux_suite.sh with no skips allowed. tools/check_mjbatch_clean.py, which
  CI runs on every commit, refuses a pin without a passing record."""
  from raysimbatch.compat.mjbatch_adapter import QUALIFIED

  head = sh("git", "rev-parse", "HEAD", cwd=PIN).stdout.strip()
  if not head.startswith(args.sha12):
    raise SystemExit(f"third_party/mjbatch is at {head}, not {args.sha12}: promote it first")
  version = next((v for v, e in QUALIFIED.items() if e["revision"] == head), None)
  if version is None:
    raise SystemExit(f"{head} is not in QUALIFIED: promote it first")
  out = ROOT / "evidence/mjbatch-upgrade" / head[:7]
  out.mkdir(parents=True, exist_ok=True)
  log = out / "linux-gate.log"
  log.write_text("")
  iid = UPGRADE / f"linux-{head[:12]}.iid"
  iid.parent.mkdir(parents=True, exist_ok=True)
  iid.unlink(missing_ok=True)
  sh("docker", "build", "--target", "test", "--iidfile", str(iid), "-f", "deploy/docker/cpu.Dockerfile", ".",
     cwd=ROOT, log=log)
  image = iid.read_text().strip()
  probe = sh("docker", "run", "--rm", image, "python", "-c",
             "import importlib.metadata as m, mujoco, platform; "
             "print(m.version('mjbatch'), mujoco.__version__, platform.machine())", log=log).stdout.split()
  rec: dict = {"sha": head, "image": image, "mjbatch": probe[0], "mujoco": probe[1], "machine": probe[2]}
  want = (version, QUALIFIED[version]["mujoco"])
  if (probe[0], probe[1]) != want:
    rec.update(passed=False, reason=f"the image has mjbatch/mujoco {probe[0]}/{probe[1]}, QUALIFIED says {want}")
  else:
    p = sh(str(ROOT / "tools/linux_suite.sh"), image, cwd=ROOT, check=False, log=log)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    rec["summary"] = lines[-1] if lines else ""
    rec["exit"] = p.returncode
    rec["passed"] = p.returncode == 0 and " skipped" not in rec["summary"]
  rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
  (out / "linux-gate.json").write_text(json.dumps(rec, indent=2) + "\n")
  print(json.dumps(rec, indent=2))
  return 0 if rec["passed"] else 1


def cmd_rollback(_args) -> int:
  pin = committed_pin()
  sh("git", "checkout", "--detach", pin, cwd=PIN)
  sh("git", "reset", "-q", "HEAD", "--", "third_party/mjbatch", cwd=ROOT)  # unstage a promoted gitlink
  sh("git", "checkout", "HEAD", "--", "pyproject.toml", "uv.lock", str(ADAPTER.relative_to(ROOT)), cwd=ROOT)
  sh("uv", "sync", "--extra", "torch", cwd=ROOT)
  p = sh(sys.executable, str(ROOT / "tools/check_mjbatch_clean.py"), cwd=ROOT, check=False)
  print(p.stdout.strip())
  print(f"rolled back to the committed pin {pin}")
  return p.returncode


def cmd_cleanup(args) -> int:
  d = UPGRADE / args.sha12
  if (d / "mjbatch").exists():
    sh("git", "worktree", "remove", "--force", str(d / "mjbatch"), cwd=PIN)
  shutil.rmtree(d, ignore_errors=True)
  sh("git", "worktree", "prune", cwd=PIN)
  print(f"removed {d}")
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  sub.add_parser("status").set_defaults(fn=cmd_status)
  p = sub.add_parser("prepare")
  p.add_argument("ref")
  p.add_argument("--mujoco")
  p.add_argument("--no-fetch", action="store_true")
  p.set_defaults(fn=cmd_prepare)
  p = sub.add_parser("qualify")
  p.add_argument("sha12")
  p.add_argument("--baseline")
  p.set_defaults(fn=cmd_qualify)
  p = sub.add_parser("promote")
  p.add_argument("sha12")
  p.set_defaults(fn=cmd_promote)
  p = sub.add_parser("linux")
  p.add_argument("sha12")
  p.set_defaults(fn=cmd_linux)
  sub.add_parser("rollback").set_defaults(fn=cmd_rollback)
  p = sub.add_parser("cleanup")
  p.add_argument("sha12")
  p.set_defaults(fn=cmd_cleanup)
  args = ap.parse_args(argv)
  return int(args.fn(args) or 0)


if __name__ == "__main__":
  sys.exit(main())
