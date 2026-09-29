# SPDX-License-Identifier: Apache-2.0
"""Secret references (configuration holds references, never values).

``env:NAME`` reads an environment variable, ``file:/abs/path`` a mounted file, and
``k8s:<secret>/<key>`` the file KubeRay mounts at
``/var/run/secrets/raysimbatch/<secret>/<key>`` (see deployment rendering).
"""

from __future__ import annotations

import os
from pathlib import Path

K8S_MOUNT = Path(os.environ.get("RSB_SECRET_MOUNT", "/var/run/secrets/raysimbatch"))


class SecretError(RuntimeError):
  pass


def resolve(ref: str) -> str:
  if ref.startswith("env:"):
    v = os.environ.get(ref[4:])
    if v is None:
      raise SecretError(f"secret reference {ref!r}: environment variable not set")
    return v
  if ref.startswith("file:"):
    p = Path(ref[5:])
  elif ref.startswith("k8s:"):
    name, _, key = ref[4:].partition("/")
    if not name or not key:
      raise SecretError("k8s secret references look like k8s:<secret>/<key>")
    p = K8S_MOUNT / name / key
  else:
    raise SecretError(f"unsupported secret reference {ref!r} (env:, file:, k8s:)")
  try:
    return p.read_text().strip()
  except OSError as exc:
    raise SecretError(f"secret reference {ref!r}: {exc}") from exc
