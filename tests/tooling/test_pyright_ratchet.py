# SPDX-License-Identifier: Apache-2.0
"""The CI pyright gate (tools/pyright_ratchet.py) must fail on every new error, including
one that merely repeats a baselined message in the same file; must fail when fixed errors
leave the baseline stale (else they could return unnoticed); must refuse a different
pyright version; and must pass only on an exact match."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pyright_ratchet", ROOT / "tools" / "pyright_ratchet.py")
ratchet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ratchet)


def diag(root, file, line, rule, message):
  return {"file": str(root / file), "severity": "error", "message": message, "rule": rule,
          "range": {"start": {"line": line - 1, "character": 0}, "end": {"line": line - 1, "character": 1}}}


def report(root, diags, version="1.1.414"):
  return {"version": version, "generalDiagnostics": [*diags, {**diags[0], "severity": "warning"}]}


@pytest.fixture
def base(tmp_path):
  root = tmp_path
  today = [diag(root, "src/a.py", 3, "reportOptionalMemberAccess", '"x" is not a known attribute of "None"'),
           diag(root, "src/a.py", 9, "reportOptionalMemberAccess", '"x" is not a known attribute of "None"'),
           diag(root, "src/b.py", 1, "reportAttributeAccessIssue", '"mjtBias" is not a known attribute\n  detail')]
  path = tmp_path / "baseline.json"
  ratchet.write_baseline("1.1.414", ratchet.errors(report(root, today), root), path)
  return root, today, path


def test_an_exact_match_passes_and_warnings_are_ignored(base, capsys):
  root, today, path = base
  assert ratchet.check(report(root, today), path, root) == 0
  assert "3 errors, all in the baseline" in capsys.readouterr().out
  assert json.loads(path.read_text())["total"] == 3  # the warning is not baselined


def test_a_new_error_fails_and_is_listed(base, capsys):
  root, today, path = base
  new = diag(root, "src/c.py", 7, "reportArgumentType", 'Argument of type "None" cannot be assigned')
  assert ratchet.check(report(root, [*today, new]), path, root) == 1
  assert "src/c.py:7: reportArgumentType" in capsys.readouterr().out


def test_a_repeat_of_a_baselined_error_is_still_new(base, capsys):
  root, today, path = base
  again = diag(root, "src/a.py", 20, "reportOptionalMemberAccess", '"x" is not a known attribute of "None"')
  assert ratchet.check(report(root, [*today, again]), path, root) == 1
  assert "1 new pyright error" in capsys.readouterr().out


def test_fixed_errors_make_the_baseline_stale_until_updated(base, capsys):
  root, today, path = base
  fewer = today[1:]
  assert ratchet.check(report(root, fewer), path, root) == 1
  assert "--update" in capsys.readouterr().out
  ratchet.write_baseline("1.1.414", ratchet.errors(report(root, fewer), root), path)
  assert ratchet.check(report(root, fewer), path, root) == 0
  assert ratchet.check(report(root, today), path, root) == 1  # the fixed error cannot come back


def test_another_pyright_version_is_refused(base, capsys):
  root, today, path = base
  assert ratchet.check(report(root, today, version="1.1.999"), path, root) == 1
  assert "baseline is for 1.1.414" in capsys.readouterr().out


def test_the_committed_baseline_is_well_formed():
  version, allowed = ratchet.load_baseline()
  data = json.loads(ratchet.BASELINE.read_text())
  assert data["total"] == sum(allowed.values()) and all(n > 0 for n in allowed.values())
  assert all((ROOT / f).exists() for f, _, _ in allowed), "baseline cites a file that no longer exists"
  pinned = [line for line in (ROOT / "pyproject.toml").read_text().splitlines() if '"pyright==' in line]
  assert pinned and f"pyright=={version}" in pinned[0]  # the baseline matches the pinned version
