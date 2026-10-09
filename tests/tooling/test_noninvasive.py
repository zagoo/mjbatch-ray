# SPDX-License-Identifier: Apache-2.0
"""The non-invasiveness checker must pass on this repository and must fail on each
kind of violation it exists to catch (a checker that cannot fail proves nothing)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_mjbatch_clean", ROOT / "tools/check_mjbatch_clean.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_repository_passes():
  assert C.main([]) == 0


def _pkg(tmp_path: Path, files: dict[str, str]) -> Path:
  src = tmp_path / "raysimbatch"
  for rel, text in files.items():
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
  return src


def test_detects_imports_outside_the_boundary_monkey_patches_and_private_access(tmp_path):
  src = _pkg(tmp_path, {
    "compat/mjbatch_adapter.py": "from mjbatch import Batch as _Batch\n"
                                 "class H:\n  def __init__(self, m):\n    self._batch = _Batch(m, 1)\n"
                                 "  def bad(self):\n    return self._batch._pool\n",
    "worker/core.py": "import mjbatch\nmjbatch.Batch.step = lambda *a: None\nsetattr(mjbatch, 'x', 1)\n",
  })
  problems = "\n".join(C.check_sources(src, {"raysimbatch/compat/mjbatch_adapter.py"}))
  assert "worker/core.py: imports mjbatch" in problems
  assert "assigns mjbatch.Batch.step" in problems or "monkey-patch" in problems
  assert "setattr on an mjbatch object" in problems
  assert "private attribute self._batch._pool" in problems


def test_clean_boundary_passes(tmp_path):
  src = _pkg(tmp_path, {"compat/mjbatch_adapter.py": "from mjbatch import Batch as _Batch\n"
                                                      "b = _Batch(None, 1)\nb.step([0])\n"})
  assert C.check_sources(src, {"raysimbatch/compat/mjbatch_adapter.py"}) == []


def test_detects_edits_untracked_files_and_build_artifacts(tmp_path):
  clone = tmp_path / "mjbatch"
  subprocess.run(["git", "clone", "-q", "--shared", str(ROOT / "third_party/mjbatch"), str(clone)], check=True)
  assert C.check_checkout(clone) == []
  (clone / "stray.txt").write_text("x")
  (clone / "build").mkdir()
  (clone / "build" / "_bindings.so").write_text("artifact")
  readme = next(clone.glob("README*"))
  readme.write_text(readme.read_text() + "\nlocal edit\n")
  out = "\n".join(C.check_checkout(clone))
  assert "stray.txt" in out and "build/" in out and readme.name in out


def test_detects_a_vendored_copy(tmp_path):
  checkout = tmp_path / "third_party" / "mjbatch"
  checkout.mkdir(parents=True)
  (checkout / "batch.h").write_text("")  # inside the pinned checkout: fine
  (tmp_path / "native").mkdir()
  (tmp_path / "native" / "batch.h").write_text("")  # a copied upstream source: not fine
  problems = C.check_no_vendored_copy(tmp_path, checkout)
  assert len(problems) == 1 and "native/batch.h" in problems[0]


def _gate(root: Path, sha: str, **rec) -> None:
  d = root / "evidence" / "mjbatch-upgrade" / sha[:7]
  d.mkdir(parents=True, exist_ok=True)
  (d / "linux-gate.json").write_text(json.dumps({"sha": sha, **rec}))


def test_a_pin_needs_a_passing_linux_gate_record(tmp_path):
  """The upgrade gates run on macOS, which has no CPU affinity: the 0.1.4 thread-pool change
  passed all of them and broke pinning on every Linux worker. A pin is therefore accepted
  only with a passing Linux gate record for exactly that revision."""
  pin, other = "088e2d4cd0d5878ea184010446c058710f23a671", "2f169146b643adda65e195d8ad13d997abdd2a2e"
  assert any("no Linux gate record" in p for p in C.check_linux_gate(tmp_path, pin))
  _gate(tmp_path, pin, passed=False, summary="1 failed, 152 passed")
  assert any("did not pass" in p for p in C.check_linux_gate(tmp_path, pin))
  _gate(tmp_path, pin, passed=True, summary="153 passed")
  assert C.check_linux_gate(tmp_path, pin) == []
  _gate(tmp_path, other, passed=True, summary="153 passed")
  assert C.check_linux_gate(tmp_path, other) == []
  (tmp_path / "evidence/mjbatch-upgrade" / pin[:7] / "linux-gate.json").write_text(json.dumps({"sha": other, "passed": True}))
  assert any("recorded for" in p for p in C.check_linux_gate(tmp_path, pin))
