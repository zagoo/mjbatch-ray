# SPDX-License-Identifier: Apache-2.0
"""src/raysimbatch/_native.pyi is how pyright checks every call into the native module.
A stub that drifted from native/src/module.cpp would check against the wrong API: every
public name, and every class's public members, must be the built module's
(regenerate with tools/gen_native_stub.py)."""

from __future__ import annotations

import ast
from pathlib import Path

from raysimbatch import _native

STUB = Path(__file__).resolve().parents[2] / "src" / "raysimbatch" / "_native.pyi"


def public(names):
  return {n for n in names if not n.startswith("_")}


def test_stub_declares_exactly_the_built_module():
  tree = ast.parse(STUB.read_text())
  declared: dict[str, set[str] | None] = {}
  for node in tree.body:
    if isinstance(node, ast.ClassDef):
      declared[node.name] = public(n.name for n in node.body if isinstance(n, ast.FunctionDef))
    elif isinstance(node, ast.FunctionDef):
      declared[node.name] = None
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      declared[node.target.id] = None
  assert set(declared) == public(dir(_native)), "regenerate: python tools/gen_native_stub.py"
  for name, members in declared.items():
    if members is not None:
      assert members == public(dir(getattr(_native, name))), f"{name}: regenerate the stub"
