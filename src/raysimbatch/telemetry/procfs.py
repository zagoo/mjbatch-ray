# SPDX-License-Identifier: Apache-2.0
"""Process memory and NUMA readings for the worker's catalog gauges (design §11.1:
``worker_private_bytes``, ``numa_remote_ratio``). The kernel is the source, so they
exist on Linux only; elsewhere the readers return None and nothing is recorded. Parsers
are pure functions over the /proc and /sys text."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

LINUX = sys.platform.startswith("linux")


def parse_cpulist(text: str) -> set[int]:
  """``0-3,8,10-11`` -> {0, 1, 2, 3, 8, 10, 11}."""
  out: set[int] = set()
  for part in text.strip().split(","):
    if not part:
      continue
    lo, _, hi = part.partition("-")
    out.update(range(int(lo), int(hi or lo) + 1))
  return out


def private_bytes(smaps_rollup: str) -> int:
  """Private (unshared) resident bytes: Private_Clean + Private_Dirty."""
  kb = 0
  for line in smaps_rollup.splitlines():
    key, _, rest = line.partition(":")
    if key in ("Private_Clean", "Private_Dirty"):
      kb += int(rest.split()[0])
  return kb * 1024


def numa_bytes(numa_maps: str) -> dict[int, int]:
  """Resident bytes per NUMA node: each mapping's ``N<node>=<pages>`` weighted by its
  ``kernelpagesize_kB`` (huge pages count at their size)."""
  out: dict[int, int] = {}
  for line in numa_maps.splitlines():
    page_kb, counts = 4, []
    for tok in line.split():
      key, eq, val = tok.partition("=")
      if not eq:
        continue
      if key == "kernelpagesize_kB":
        page_kb = int(val)
      elif key[:1] == "N" and key[1:].isdigit():
        counts.append((int(key[1:]), int(val)))
    for node, pages in counts:
      out[node] = out.get(node, 0) + pages * page_kb * 1024
  return out


def remote_ratio(by_node: dict[int, int], local_nodes: set[int]) -> float | None:
  total = sum(by_node.values())
  if not total or not local_nodes:
    return None
  return sum(b for n, b in by_node.items() if n not in local_nodes) / total


def read_private_bytes() -> int | None:
  if not LINUX:
    return None
  try:
    return private_bytes(Path("/proc/self/smaps_rollup").read_text())
  except OSError:
    return None


def read_numa_remote_ratio(cpus: Iterable[int] = (), sys_root: str = "/sys/devices/system/node") -> float | None:
  """Fraction of this process's resident memory on NUMA nodes other than those of
  ``cpus`` (its leased cores; its affinity mask when empty)."""
  if not LINUX:
    return None
  import os

  mine = set(cpus) or set(os.sched_getaffinity(0))
  try:
    local = {int(p.name[4:]) for p in Path(sys_root).glob("node[0-9]*")
             if parse_cpulist((p / "cpulist").read_text()) & mine}
    return remote_ratio(numa_bytes(Path("/proc/self/numa_maps").read_text()), local)
  except OSError:
    return None
