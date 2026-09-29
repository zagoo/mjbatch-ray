# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the examples: a local Ray, an artifact root and a database secret."""

from __future__ import annotations

import argparse
import tempfile


def args(description: str) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=description)
  p.add_argument("--root", default=f"file://{tempfile.mkdtemp(prefix='rsb-example-')}/artifacts",
                 help="artifact root: file:///path or s3://bucket/prefix?endpoint=...")
  p.add_argument("--database-secret", default="env:RSB_TEST_DSN",
                 help="secret reference resolving to a PostgreSQL DSN (env:VAR, file:/path, k8s:name/key)")
  p.add_argument("--address", default=None, help="Ray address (default: start a local Ray)")
  return p.parse_args()


def start_ray(address: str | None) -> None:
  import ray

  ray.init(address=address, namespace="rsb-examples", log_to_driver=False,
           **({} if address else {"num_cpus": 16, "include_dashboard": False}))
