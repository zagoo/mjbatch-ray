# SPDX-License-Identifier: Apache-2.0
"""Pyright gate for CI: a ratchet on the committed baseline (owner decision 2026-09-29).

``tools/pyright-baseline.json`` counts the tolerated pyright errors per (file, rule,
message). It started with 409 triaged false positives (evidence/tests/
pyright-triage-e228b97.csv) and has been empty since 2026-09-29, so today any pyright
error fails. An entry may be added only for a false positive, with the reason in the
commit. The check fails when:

* a count rises or a new (file, rule, message) appears: a new error, fix it (or, if it is
  a false positive, say why in the commit that updates the baseline);
* a count falls: the baseline is stale; run ``--update`` and commit it, so fixed errors
  cannot come back unnoticed;
* the pyright version differs from the baseline's (versions report differently).

  python tools/pyright_ratchet.py                 # run pyright and check
  python tools/pyright_ratchet.py --update        # rewrite the baseline from the current errors
  python tools/pyright_ratchet.py --report r.json # check a saved `pyright --outputjson`
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "pyright-baseline.json"
Key = tuple[str, str, str]


def run_pyright() -> dict:
  out = subprocess.run([sys.executable, "-m", "pyright", "--outputjson"], cwd=ROOT, capture_output=True, text=True)
  if out.returncode not in (0, 1):  # 1 means "errors found"; anything else is a failed run
    raise SystemExit(f"pyright failed to run (exit {out.returncode}):\n{out.stderr[-2000:]}")
  return json.loads(out.stdout)


def _key(d: dict, root: Path) -> Key:
  path = Path(d["file"])
  rel = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
  return rel, d.get("rule", ""), d["message"].splitlines()[0]


def errors(report: dict, root: Path = ROOT) -> collections.Counter[Key]:
  return collections.Counter(_key(d, root) for d in report["generalDiagnostics"] if d["severity"] == "error")


def load_baseline(path: Path = BASELINE) -> tuple[str, collections.Counter[Key]]:
  data = json.loads(path.read_text())
  return data["pyright"], collections.Counter({(e["file"], e["rule"], e["message"]): e["count"] for e in data["errors"]})


def write_baseline(version: str, current: collections.Counter[Key], path: Path = BASELINE) -> None:
  rows = [{"file": f, "rule": r, "message": m, "count": n} for (f, r, m), n in sorted(current.items())]
  path.write_text(json.dumps({"pyright": version, "total": sum(current.values()), "errors": rows}, indent=1) + "\n")


def check(report: dict, baseline: Path = BASELINE, root: Path = ROOT) -> int:
  version, allowed = load_baseline(baseline)
  shown = baseline.relative_to(root) if baseline.is_relative_to(root) else baseline
  if report["version"] != version:
    print(f"pyright {report['version']} is running but the baseline is for {version}: use the pinned version "
          f"(uv sync) or regenerate the baseline with it (--update) and review the difference.")
    return 1
  current = errors(report, root)
  new = {k: n - allowed[k] for k, n in current.items() if n > allowed[k]}
  fixed = {k: allowed[k] - current[k] for k in allowed if current[k] < allowed[k]}
  if new:
    print(f"{sum(new.values())} new pyright error(s) (not in {shown}):")
    for d in report["generalDiagnostics"]:
      if d["severity"] == "error" and _key(d, root) in new:
        f, rule, msg = _key(d, root)
        print(f"  {f}:{d['range']['start']['line'] + 1}: {rule}: {msg}")
  if fixed:
    print(f"{sum(fixed.values())} baselined error(s) no longer reported, the baseline is stale. "
          f"Run `python tools/pyright_ratchet.py --update` and commit {shown}:")
    for (f, rule, msg), n in sorted(fixed.items()):
      print(f"  {f}: {rule}: {msg} (-{n})")
  if new or fixed:
    return 1
  print(f"pyright {version}: {sum(current.values())} errors, all in the baseline")
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--update", action="store_true", help="rewrite the baseline from the current errors")
  ap.add_argument("--report", type=Path, help="a saved `pyright --outputjson` instead of running pyright")
  args = ap.parse_args(argv)
  report = json.loads(args.report.read_text()) if args.report else run_pyright()
  if args.update:
    current = errors(report)
    write_baseline(report["version"], current)
    print(f"wrote {BASELINE.relative_to(ROOT)}: {sum(current.values())} errors, pyright {report['version']}")
    return 0
  return check(report)


if __name__ == "__main__":
  sys.exit(main())
