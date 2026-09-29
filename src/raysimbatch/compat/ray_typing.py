# SPDX-License-Identifier: Apache-2.0
"""The places where Ray's type stubs mislead pyright, kept in one module. Runtime
behavior is Ray's own; only the static types are widened to ``Any``.

* ``actor_class(cls, **options)`` is ``ray.remote(**options)(cls)``. The stub overloads
  reject valid option combinations (``concurrency_groups``, ``max_concurrency`` with
  restarts), and they type a handle's methods as the class's plain functions, so
  ``handle.method.remote(...)`` does not type-check unless every method is decorated.
* ``get_one(ref)`` is ``ray.get`` of a single reference: on an untyped reference pyright
  picks the list overload.
"""

from __future__ import annotations

from typing import Any

import ray

_remote: Any = ray.remote
_get: Any = ray.get


def actor_class(cls: type, **options: Any) -> Any:
  return _remote(**options)(cls) if options else _remote(cls)


def get_one(ref: Any, timeout: float | None = None) -> Any:
  return _get(ref, timeout=timeout)
