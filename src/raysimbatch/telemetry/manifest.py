# SPDX-License-Identifier: Apache-2.0
"""Reproducibility manifest (design §11.2): everything needed to reproduce or audit a
run, including what could *not* be certified."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _git(*args: str, cwd: Path = REPO) -> str | None:
  try:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
  except Exception:  # noqa: BLE001 - not a git checkout (e.g. inside an image)
    return None


def file_sha256(path: str | Path) -> str:
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software() -> dict[str, Any]:
  from raysimbatch import _native
  from raysimbatch.compat import mjbatch_adapter as compat

  pkgs = {}
  for p in ("ray", "mujoco", "mjbatch", "numpy", "onnxruntime", "torch", "pyarrow", "pydantic", "psycopg"):
    try:
      pkgs[p] = importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
      pkgs[p] = None
  rep = compat.check_environment()
  return {
    "python": sys.version.split()[0],
    "packages": pkgs,
    "raysimbatch_revision": os.environ.get("RSB_GIT_SHA") or _git("rev-parse", "HEAD"),
    "raysimbatch_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    "mjbatch_revision": os.environ.get("RSB_MJBATCH_SHA") or _git("rev-parse", "HEAD", cwd=REPO / "third_party/mjbatch"),
    "mjbatch_binary_sha256": rep.binary_sha256,
    "native_extension_sha256": file_sha256(_native.__file__),
    "engine_abi": rep.abi_fingerprint,
    "mj_version": int(mujoco.mj_version()),
    "image_digest": os.environ.get("RSB_IMAGE_DIGEST"),
  }


def hardware() -> dict[str, Any]:
  info: dict[str, Any] = {
    "machine": platform.machine(),
    "system": platform.system(),
    "release": platform.release(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                     "NUMEXPR_NUM_THREADS")},
  }
  if sys.platform.startswith("linux"):
    try:
      cpuinfo = Path("/proc/cpuinfo").read_text()
      info["cpu_model"] = next((ln.split(":", 1)[1].strip() for ln in cpuinfo.splitlines()
                                if ln.startswith(("model name", "Model"))), None)
      info["microcode"] = next((ln.split(":", 1)[1].strip() for ln in cpuinfo.splitlines()
                                if ln.startswith("microcode")), None)
      info["numa_nodes"] = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
      smt = Path("/sys/devices/system/cpu/smt/active")
      info["smt_active"] = smt.read_text().strip() == "1" if smt.exists() else None
    except OSError:
      pass
  elif sys.platform == "darwin":
    try:
      info["cpu_model"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                         text=True, timeout=5).stdout.strip()
      info["numa_nodes"] = 1
      info["smt_active"] = False
    except Exception:  # noqa: BLE001
      pass
  try:
    info["blas"] = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
  except Exception:  # noqa: BLE001
    info["blas"] = None
  return info


def build(
  *,
  requested: dict[str, Any] | None,
  resolved: dict[str, Any],
  config_hash: str,
  run_id: str,
  artifacts: dict[str, str],
  counts: dict[str, Any],
  seeds: dict[str, Any],
  failures: dict[str, Any],
  deterministic_replay_certified: bool,
  extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  return {
    "schema": "rsb-run-manifest/1",
    "run_id": run_id,
    "config": {"requested": requested, "resolved": resolved, "hash": config_hash},
    "software": software(),
    "hardware": hardware(),
    "artifacts": artifacts,
    "seed_derivation": {"version": "rsb-seed-v1", "generator": "philox4x64-10/rsb-philox-v1", **seeds},
    "counts": counts,
    "failures": failures,
    "deterministic_replay_certified": deterministic_replay_certified,
    **(extra or {}),
  }


def dumps(m: dict[str, Any]) -> bytes:
  return json.dumps(m, indent=2, sort_keys=True, default=str).encode()
