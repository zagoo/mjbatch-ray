# SPDX-License-Identifier: Apache-2.0
"""typings/mujoco lets pyright check every mujoco.* use against the real bindings. Stubs
for another MuJoCo version would check against the wrong API, silently: they must be
regenerated (tools/gen_mujoco_stubs.py) whenever MuJoCo changes."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]


def test_stubs_match_the_installed_mujoco():
  line = re.search(r"^generated-for: (.*)$", (ROOT / "typings/README.md").read_text(), re.M)
  assert line, "typings/README.md lacks its generated-for line"
  fields = dict(f.split("==") for f in line.group(1).split() if "==" in f)
  assert fields["mujoco"] == mujoco.__version__, (
    f"stubs are for mujoco {fields['mujoco']}, installed is {mujoco.__version__}: "
    "run `uv run --with pybind11-stubgen==3.0.0 python tools/gen_mujoco_stubs.py`")
  assert (ROOT / "typings/mujoco/__init__.pyi").is_file()
