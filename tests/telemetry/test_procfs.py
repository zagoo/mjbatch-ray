# SPDX-License-Identifier: Apache-2.0
"""The kernel text behind worker_private_bytes and numa_remote_ratio (design §11.1). A
wrong parse would misreport placement quality silently, so each rule is pinned here on
sample text: private = Private_Clean + Private_Dirty; NUMA bytes weight each mapping's
pages by its page size (huge pages count at their size); the ratio is the share of
resident bytes outside the nodes of the leased CPUs."""

from __future__ import annotations

from raysimbatch.telemetry import procfs

SMAPS_ROLLUP = """\
55a0c0000000-7ffd3a1f2000 ---p 00000000 00:00 0                          [rollup]
Rss:              524288 kB
Pss:              400000 kB
Shared_Clean:     100000 kB
Shared_Dirty:          0 kB
Private_Clean:     24288 kB
Private_Dirty:    400000 kB
Swap:                  0 kB
"""

NUMA_MAPS = """\
55a0c0000000 default file=/usr/bin/python3.12 mapped=100 N0=100 kernelpagesize_kB=4
7f0000000000 default anon=1000 dirty=1000 N0=600 N1=400 kernelpagesize_kB=4
7f1000000000 bind:1 anon=2 dirty=2 N1=2 kernelpagesize_kB=2048
7f2000000000 default
"""


def test_parse_cpulist():
  assert procfs.parse_cpulist("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
  assert procfs.parse_cpulist("") == set()


def test_private_bytes_are_clean_plus_dirty_private():
  assert procfs.private_bytes(SMAPS_ROLLUP) == (24288 + 400000) * 1024


def test_numa_bytes_weight_pages_by_their_size():
  assert procfs.numa_bytes(NUMA_MAPS) == {0: 700 * 4096, 1: 400 * 4096 + 2 * 2048 * 1024}


def test_remote_ratio_is_the_share_outside_the_local_nodes():
  by_node = procfs.numa_bytes(NUMA_MAPS)
  total = sum(by_node.values())
  assert procfs.remote_ratio(by_node, {0}) == by_node[1] / total
  assert procfs.remote_ratio(by_node, {0, 1}) == 0.0
  assert procfs.remote_ratio({}, {0}) is None and procfs.remote_ratio(by_node, set()) is None


def test_readers_record_nothing_off_linux(monkeypatch):
  monkeypatch.setattr(procfs, "LINUX", False)
  assert procfs.read_private_bytes() is None and procfs.read_numa_remote_ratio([0]) is None
