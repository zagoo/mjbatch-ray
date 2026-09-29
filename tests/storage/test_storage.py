# SPDX-License-Identifier: Apache-2.0
"""Durable storage against a real PostgreSQL and S3-compatible server
(tools/local_services.sh start; eval "$(tools/local_services.sh env)")."""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

from raysimbatch.protocol import Artifact
from raysimbatch.storage.blob import BlobCorrupt, FileBlobStore, S3BlobStore
from raysimbatch.storage.db import Database, DigestConflict, LeaseHeld

DSN = os.environ.get("RSB_TEST_DSN")
S3 = os.environ.get("RSB_TEST_S3_ENDPOINT")


@pytest.fixture(scope="module")
def db():
  if not DSN:
    pytest.skip("RSB_TEST_DSN not set (start tools/local_services.sh)")
  d = Database(DSN, timeout_s=5)
  d.migrate()
  yield d
  d.close()


def new_run(db) -> str:
  rid = uuid.uuid4().hex
  db.create_run(rid, "c" * 64, "eval", {}, idempotency_key=rid, caller="test")
  return rid


pytestmark = pytest.mark.postgres


def test_migrations_are_idempotent(db):
  assert db.migrate() == []


def test_run_submission_is_idempotent(db):
  key = uuid.uuid4().hex
  a, created = db.create_run(uuid.uuid4().hex, "x" * 64, "eval", {}, key, "c")
  b, again = db.create_run(uuid.uuid4().hex, "x" * 64, "eval", {}, key, "c")
  assert created and not again and a == b


def test_leases_fence_owners_and_expire_on_server_time(db):
  key = f"lease/{uuid.uuid4().hex}"
  a = db.acquire_lease(key, "A", 0.5)
  with pytest.raises(LeaseHeld):
    db.acquire_lease(key, "B", 5)
  assert db.renew_lease(key, "A", a.epoch, 0.5) is not None
  time.sleep(0.8)
  assert db.renew_lease(key, "A", a.epoch, 5) is None  # expired: renewal refused
  b = db.acquire_lease(key, "B", 5)
  assert b.epoch == a.epoch + 1  # a new owner fences the old one
  assert db.renew_lease(key, "A", a.epoch, 5) is None


def units(n, prefix="u"):
  return [{"unit_id": f"{prefix}{i:03d}", "payload_hash": "p", "payload": {"i": i}} for i in range(n)]


def test_unit_leasing_never_double_leases_under_concurrency(db):
  rid = new_run(db)
  assert db.insert_units(rid, units(60)) == 60
  assert db.insert_units(rid, units(60)) == 0  # enumeration is idempotent
  got: list[list[str]] = [[] for _ in range(6)]

  def lease(i):
    while True:
      rows = db.lease_units(rid, f"owner{i}", 3, 60, 3)
      if not rows:
        return
      got[i] += [r["unit_id"] for r in rows]

  ts = [threading.Thread(target=lease, args=(i,)) for i in range(6)]
  for t in ts:
    t.start()
  for t in ts:
    t.join()
  flat = [u for g in got for u in g]
  assert sorted(flat) == sorted(u["unit_id"] for u in units(60))  # each exactly once


def test_commit_fencing_dedup_and_digest_conflict(db):
  rid = new_run(db)
  db.insert_units(rid, units(1))
  [u] = db.lease_units(rid, "A", 1, 60, 3)
  rows = [{"ordinal": k, "digest": f"d{k}", "object_uri": "file:///x", "outcome": "success", "row": {"k": k}}
          for k in range(3)]
  ack = db.commit_episodes(rid, u["unit_id"], u["attempt"], u["lease_epoch"], rows[:2], expected_count=3)
  assert ack.ok and len(ack.committed) == 2 and ack.unit_state == "RUNNING"
  # Retried delivery of the same rows: deduplicated.
  ack = db.commit_episodes(rid, u["unit_id"], u["attempt"], u["lease_epoch"], rows[:2], expected_count=3)
  assert ack.committed == () and len(ack.duplicates) == 2
  # Same identity, different content: corruption.
  with pytest.raises(DigestConflict):
    db.commit_episodes(rid, u["unit_id"], u["attempt"], u["lease_epoch"],
                       [{**rows[0], "digest": "other"}], expected_count=3)
  # A stale attempt cannot commit.
  ack = db.commit_episodes(rid, u["unit_id"], u["attempt"] - 1, u["lease_epoch"], rows[2:], expected_count=3)
  assert ack.rejected == "stale_attempt"
  ack = db.commit_episodes(rid, u["unit_id"], u["attempt"], u["lease_epoch"], rows[2:], expected_count=3)
  assert ack.unit_state == "SUCCEEDED"
  assert db.committed_ordinals(rid, u["unit_id"]) == {0, 1, 2}


def test_expired_leases_retry_until_exhausted(db):
  rid = new_run(db)
  db.insert_units(rid, units(1))
  attempts = []
  for _ in range(3):
    rows = db.lease_units(rid, "A", 1, 0.2, 3)
    assert len(rows) == 1
    attempts.append(rows[0]["attempt"])
    time.sleep(0.4)  # the lease expires without commit
  assert attempts == [1, 2, 3]
  assert db.lease_units(rid, "A", 1, 0.2, 3) == []
  assert db.unit_counts(rid) == {"EXHAUSTED": 1}


def test_old_attempt_cannot_overwrite_a_newer_one(db):
  rid = new_run(db)
  db.insert_units(rid, units(1))
  [old] = db.lease_units(rid, "A", 1, 0.2, 5)
  time.sleep(0.4)
  [new] = db.lease_units(rid, "B", 1, 60, 5)
  assert new["attempt"] == old["attempt"] + 1
  row = [{"ordinal": 0, "digest": "d", "object_uri": "u", "outcome": "success", "row": {}}]
  assert db.commit_episodes(rid, old["unit_id"], old["attempt"], old["lease_epoch"], row, 1).rejected
  assert db.fail_unit(rid, old["unit_id"], old["attempt"], old["lease_epoch"], {"x": 1}, 5) is None
  assert db.commit_episodes(rid, new["unit_id"], new["attempt"], new["lease_epoch"], row, 1).unit_state == "SUCCEEDED"


def test_outbox_dedup_and_id_blocks(db):
  rid = new_run(db)
  a = Artifact("file:///f", "a" * 64, 10, "application/x")
  frag = f"frag-{uuid.uuid4().hex}"  # fragment ids are globally unique; the test DB persists across runs
  assert db.outbox_insert(frag, rid, "d1", a, 32) == "inserted"
  assert db.outbox_insert(frag, rid, "d1", a, 32) == "duplicate"
  with pytest.raises(DigestConflict):
    db.outbox_insert(frag, rid, "d2", a, 32)
  s1 = db.allocate_episode_ids(rid, 100)
  s2 = db.allocate_episode_ids(rid, 100)
  assert s2 == s1 + 100


def test_keyed_commit(db):
  a = Artifact("file:///f", "b" * 64, 10, "x")
  key = f"k/{uuid.uuid4().hex}"
  assert db.commit(key, 1, "d", a).committed == (key,)
  assert db.commit(key, 1, "d", a).duplicates == (key,)
  with pytest.raises(DigestConflict):
    db.commit(key, 2, "e", a)
  assert db.commit(key, 0, "e", a).rejected == "stale_epoch"


def _blob_roundtrip(store):
  a = store.put(b"hello world", "text/plain")
  assert store.get(a) == b"hello world"
  assert store.put(b"hello world", "text/plain") == a  # content-addressed, idempotent
  with pytest.raises(BlobCorrupt):
    store.get(Artifact(a.uri, a.sha256, a.byte_length + 1, a.media_type))
  assert a.sha256 in store.list_hashes()


def test_file_blob_store(tmp_path):
  store = FileBlobStore(tmp_path)
  _blob_roundtrip(store)
  f = store._file(store.put(b"x" * 10, "b").sha256)
  f.write_bytes(b"y" * 10)  # corrupt at rest
  with pytest.raises(BlobCorrupt):
    store.get(Artifact("", "".join(__import__("hashlib").sha256(b"x" * 10).hexdigest()), 10, "b"))


@pytest.mark.s3
def test_s3_blob_store():
  if not S3:
    pytest.skip("RSB_TEST_S3_ENDPOINT not set")
  store = S3BlobStore(f"rsb-test-{uuid.uuid4().hex[:8]}", "run", S3)
  _blob_roundtrip(store)


def test_migrate_refuses_non_utf8_database(db):
  """Found in kind qualification: the postgres image initialised under LC_ALL=C gives a
  SQL_ASCII database, which stores unvalidated bytes; migrate must refuse it loudly
  rather than half-apply the schema."""
  import psycopg

  from raysimbatch.storage.db import StorageError

  name = f"rsb_ascii_{uuid.uuid4().hex[:8]}"
  with psycopg.connect(DSN, autocommit=True) as c:
    c.execute(f"CREATE DATABASE {name} ENCODING 'SQL_ASCII' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0")
  try:
    other = Database(psycopg.conninfo.make_conninfo(DSN, dbname=name), timeout_s=5)
    with pytest.raises(StorageError, match="SQL_ASCII"):
      other.migrate()
    other.close()
  finally:
    with psycopg.connect(DSN, autocommit=True) as c:
      c.execute(f"DROP DATABASE {name} WITH (FORCE)")


def test_controller_keeps_its_run_lease_while_the_main_thread_blocks(db):
  """Found in kind: a gang reservation blocked the controller longer than its lease and
  the controller fenced itself. Renewal runs on a keeper thread; a lease taken over by
  another owner is still detected and surfaced to the main thread."""
  from raysimbatch.config import ConfigurationSystem
  from raysimbatch.control.controller import RunController
  from raysimbatch.storage.blob import FileBlobStore
  from raysimbatch.storage.store import DurableStore

  cfg = ConfigurationSystem().parse({"failure": {"lease_s": 1.5, "lease_renew_s": 0.3}})
  rid = new_run(db)
  ctl = RunController(rid, cfg.model_dump_json())
  ctl.store = DurableStore(db, FileBlobStore("/tmp"))
  ctl.lease = db.acquire_lease(f"run/{rid}", ctl.name, 1.5)
  ctl._renewed = time.monotonic()
  threading.Thread(target=ctl._lease_keeper, daemon=True).start()
  time.sleep(4.5)  # three lease lifetimes with the "main thread" blocked
  ctl._renew()  # raises if the lease was lost
  with pytest.raises(LeaseHeld):
    db.acquire_lease(f"run/{rid}", "intruder", 1.5)  # still held by the controller
  # Another owner takes the lease over (as a replacement controller would after expiry).
  db.tx(lambda cur: cur.execute("UPDATE leases SET owner='successor', epoch=epoch+1 WHERE key=%s", (f"run/{rid}",)))
  deadline = time.monotonic() + 5
  while ctl._lease_lost is None and time.monotonic() < deadline:
    time.sleep(0.1)
  with pytest.raises(RuntimeError, match="another controller owns this run"):
    ctl._renew()
  ctl.state = "COMPLETED"  # stop the keeper


def test_resumed_controller_waits_out_a_dying_owner_but_not_a_live_one(db):
  """Found in kind: the force-deleted release's controller kept renewing for ~90 s and the
  resume on the rollback cluster failed the run with LeaseHeld instead of waiting."""
  from raysimbatch.config import ConfigurationSystem
  from raysimbatch.control.controller import RunController
  from raysimbatch.storage.blob import FileBlobStore
  from raysimbatch.storage.store import DurableStore

  cfg = ConfigurationSystem().parse({"failure": {"lease_s": 1.0, "lease_renew_s": 0.2},
                                     "cluster": {"termination_grace_s": 1}})
  rid = new_run(db)
  db.acquire_lease(f"run/{rid}", "dying-owner", 1.0)  # stops renewing: lapses within 1 s
  ctl = RunController(rid, cfg.model_dump_json())
  ctl.store = DurableStore(db, FileBlobStore("/tmp"))
  t0 = time.monotonic()
  lease = ctl._acquire_run_lease()
  assert lease.owner == ctl.name and time.monotonic() - t0 < 2.5

  rid2 = new_run(db)
  stop = threading.Event()
  alive = db.acquire_lease(f"run/{rid2}", "live-owner", 1.0)

  def renew():
    while not stop.is_set():
      db.renew_lease(f"run/{rid2}", "live-owner", alive.epoch, 1.0)
      time.sleep(0.2)

  threading.Thread(target=renew, daemon=True).start()
  ctl2 = RunController(rid2, cfg.model_dump_json())
  ctl2.store = ctl.store
  with pytest.raises(LeaseHeld, match="its owner is alive"):
    ctl2._acquire_run_lease()
  stop.set()
