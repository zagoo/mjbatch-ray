# SPDX-License-Identifier: Apache-2.0
"""Enforce the non-invasive MJBatch integration boundary.

  python tools/check_mjbatch_clean.py [--checkout third_party/mjbatch] [--no-installed]

Checks (each failure is reported; exit 1 if any):

1. The MJBatch checkout has no modified, untracked or ignored files: no edits, no
   patches, no build artifacts inside upstream's tree (builds go to ../../build/).
2. The pin agrees everywhere: the superproject's gitlink, the checkout's HEAD, and the
   compat module's QUALIFIED table for the installed mjbatch version.
3. The installed mjbatch distribution was built from that checkout (direct_url) and
   reports the qualified version.
4. Only raysimbatch/compat/mjbatch_adapter.py imports mjbatch (tests/upstream may, to
   pin upstream behaviour); nothing assigns attributes on mjbatch objects (no
   monkey-patching) or reads a private (underscore) attribute of an mjbatch object.
5. No copy of MJBatch sources lives outside the checkout (no vendor-and-modify).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_MARKERS = ("threadpool.h", "batch.h")  # files that exist only in MJBatch's source tree


def _git(*args: str, cwd: Path) -> str:
  return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def check_checkout(checkout: Path) -> list[str]:
  problems = []
  dirty = _git("status", "--porcelain", "--ignored", "--untracked-files=all", cwd=checkout).strip()
  if dirty:
    problems.append(f"{checkout}: working tree is not pristine (modified/untracked/ignored files):\n  "
                    + "\n  ".join(dirty.splitlines()[:20]))
  return problems


def check_pin(checkout: Path, superproject: Path | None) -> tuple[list[str], str]:
  problems = []
  head = _git("rev-parse", "HEAD", cwd=checkout).strip()
  if superproject is not None:
    rel = checkout.resolve().relative_to(superproject.resolve())
    # The index gitlink: equal to HEAD normally, and the staged pin right after
    # `mjbatch_upgrade.py promote` (found: comparing HEAD made a correct promote fail).
    gitlink = _git("ls-files", "-s", str(rel), cwd=superproject).split()
    if len(gitlink) < 2 or gitlink[1] != head:
      problems.append(f"superproject pins {gitlink[1] if len(gitlink) > 1 else '?'} but {rel} is at {head}")
  return problems, head


def check_installed(head: str, checkout: Path) -> list[str]:
  import importlib.metadata as md

  from raysimbatch.compat.mjbatch_adapter import QUALIFIED

  problems = []
  version = md.version("mjbatch")
  q = QUALIFIED.get(version)
  if q is None:
    problems.append(f"installed mjbatch {version} is not in QUALIFIED {sorted(QUALIFIED)}")
  elif q["revision"] != head:
    problems.append(f"QUALIFIED[{version}] is {q['revision']} but the checkout is at {head}")
  direct = md.distribution("mjbatch").read_text("direct_url.json")
  if direct:
    url = json.loads(direct).get("url", "")
    if Path(url.removeprefix("file://")).resolve() != checkout.resolve():
      problems.append(f"installed mjbatch was built from {url}, not {checkout}")
  else:
    problems.append("installed mjbatch has no direct_url.json: cannot prove it was built from the pinned checkout")
  return problems


def _mjbatch_names(tree: ast.AST) -> set[str]:
  names = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      names |= {a.asname or a.name for a in node.names if a.name.split(".")[0] == "mjbatch"}
    elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "mjbatch":
      names |= {a.asname or a.name for a in node.names}
  return names


def check_sources(src: Path, allowed_importers: set[str]) -> list[str]:
  problems = []
  for path in sorted(src.rglob("*.py")):
    rel = str(path.relative_to(src.parent))
    tree = ast.parse(path.read_text())
    names = _mjbatch_names(tree)
    if names and rel not in allowed_importers:
      problems.append(f"{rel}: imports mjbatch ({sorted(names)}); only {sorted(allowed_importers)} may")
    # Objects that come from mjbatch in this module: imported names and anything
    # assigned from calling them (e.g. self._batch = _Batch(...)).
    tainted = set(names)
    for node in ast.walk(tree):
      if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        fn = ast.unparse(node.value.func)
        if fn in tainted:
          tainted |= {ast.unparse(t) for t in node.targets}
    for node in ast.walk(tree):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
        node, ast.AugAssign | ast.AnnAssign) else []
      for t in targets:
        if isinstance(t, ast.Attribute) and ast.unparse(t.value) in tainted:
          problems.append(f"{rel}:{node.lineno}: assigns {ast.unparse(t)} on an mjbatch object (monkey-patch)")
      if isinstance(node, ast.Call) and ast.unparse(node.func) == "setattr" and node.args \
          and ast.unparse(node.args[0]) in tainted:
        problems.append(f"{rel}:{node.lineno}: setattr on an mjbatch object (monkey-patch)")
      if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__") \
          and ast.unparse(node.value) in tainted:
        problems.append(f"{rel}:{node.lineno}: reads private attribute {ast.unparse(node)} of an mjbatch object")
  return problems


def check_no_vendored_copy(root: Path, checkout: Path) -> list[str]:
  problems = []
  skip = {".git", ".venv", ".tools", "build", "var", "node_modules"}
  for marker in UPSTREAM_MARKERS:
    for hit in root.rglob(marker):
      parts = set(hit.relative_to(root).parts)
      if parts & skip or checkout.resolve() in hit.resolve().parents:
        continue
      problems.append(f"{hit.relative_to(root)}: an MJBatch source file outside the pinned checkout")
  return problems


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkout", default=str(ROOT / "third_party/mjbatch"))
  ap.add_argument("--superproject", default=str(ROOT))
  ap.add_argument("--no-installed", action="store_true", help="skip the installed-distribution check")
  ap.add_argument("--only-checkout", action="store_true",
                  help="only check that the given checkout is pristine (upgrade candidates)")
  args = ap.parse_args(argv)
  checkout = Path(args.checkout)
  problems = check_checkout(checkout)
  if args.only_checkout:
    for p in problems:
      print(f"FAIL {p}")
    print("OK checkout pristine" if not problems else "")
    return 1 if problems else 0
  pin_problems, head = check_pin(checkout, Path(args.superproject) if args.superproject else None)
  problems += pin_problems
  if not args.no_installed:
    problems += check_installed(head, checkout)
  problems += check_sources(ROOT / "src" / "raysimbatch", {"raysimbatch/compat/mjbatch_adapter.py"})
  problems += check_no_vendored_copy(ROOT, checkout)
  for p in problems:
    print(f"FAIL {p}")
  if not problems:
    print(f"OK mjbatch checkout {checkout} pristine at {head}; boundary respected")
  return 1 if problems else 0


if __name__ == "__main__":
  sys.exit(main())
