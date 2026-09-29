# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL persistence: run state, leases, evaluation units, episode commits,
outbox, routes, id blocks, gaps, quarantine and audit (design §6.6, §8).

Every call has a statement deadline and bounded retries on transient errors. Leases
use database server time; completion compares attempt and lease epoch, so an old
attempt can never overwrite a newer one. There is no distributed transaction with
object storage: blobs are written content-addressed *before* the transaction that
references them (a crash in between leaves a collectable orphan).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from raysimbatch.protocol import Artifact, Lease


class StorageError(RuntimeError):
  pass


class StorageUnavailable(StorageError):
  """Deadline exceeded or the database is unreachable after retries."""


class DigestConflict(StorageError):
  """Same logical identity, different content: corruption; the partition must stop."""


class LeaseHeld(StorageError):
  pass


@dataclass(frozen=True)
class CommitAck:
  committed: tuple[str, ...]
  duplicates: tuple[str, ...] = ()
  rejected: str | None = None
  generation: int = 0
  unit_state: str | None = None

  @property
  def ok(self) -> bool:
    return self.rejected is None


UNIT_TERMINAL = ("SUCCEEDED", "EXHAUSTED", "CANCELED")


@dataclass
class Database:
  dsn: str
  timeout_s: float = 5.0
  retry_attempts: int = 3
  backoff: tuple[float, ...] = (0.1, 0.5, 2.0)
  max_size: int = 4
  _pool: ConnectionPool | None = field(default=None, init=False, repr=False)

  def __post_init__(self) -> None:
    ms = int(self.timeout_s * 1000)
    self._pool = ConnectionPool(
      self.dsn,
      min_size=1,
      max_size=self.max_size,
      timeout=self.timeout_s,
      kwargs={"autocommit": False, "row_factory": dict_row, "connect_timeout": max(1, int(self.timeout_s)),
              "options": f"-c statement_timeout={ms} -c lock_timeout={ms} -c client_encoding=UTF8"},
      open=True,
    )

  def close(self) -> None:
    if self._pool is not None:
      self._pool.close()

  # ------------------------------------------------------------------ plumbing

  def tx(self, fn, *args, **kw):
    """Run ``fn(cur, ...)`` in one transaction with bounded retries."""
    from raysimbatch import telemetry

    m = telemetry.exporter(actor_role="storage")
    last: Exception | None = None
    pool = self._pool
    assert pool is not None  # set in __post_init__
    for i in range(self.retry_attempts + 1):
      t0 = time.perf_counter()
      try:
        with pool.connection() as conn:
          with conn.transaction():
            with conn.cursor() as cur:
              result = fn(cur, *args, **kw)
        m.observe("storage_commit_seconds", time.perf_counter() - t0)  # after COMMIT returned
        return result
      except DigestConflict:
        m.count("storage_errors_total", outcome="digest_conflict")  # pages (design §11.3)
        raise
      except LeaseHeld:
        raise
      except (psycopg.OperationalError, psycopg.errors.QueryCanceled, psycopg.errors.SerializationFailure,
              psycopg.errors.DeadlockDetected, psycopg.errors.LockNotAvailable) as exc:
        last = exc
      except Exception as exc:  # pool timeout, connection refused
        if type(exc).__name__ in ("PoolTimeout", "TooManyRequests"):
          last = exc
        else:
          raise
      m.count("storage_errors_total", outcome="retry" if i < self.retry_attempts else "unavailable")
      if i < self.retry_attempts:
        time.sleep(self.backoff[min(i, len(self.backoff) - 1)])
    raise StorageUnavailable(f"database operation failed after retries: {last}") from last

  def migrate(self) -> list[str]:
    files = sorted(
      (p for p in resources.files("raysimbatch.storage").joinpath("migrations").iterdir() if p.name.endswith(".sql")),
      key=lambda p: p.name)

    def run(cur):
      # Records hold UTF-8 JSON and names; a SQL_ASCII database (e.g. initdb under
      # LC_ALL=C without --encoding) would store unvalidated bytes. Refuse it.
      cur.execute("SHOW server_encoding")
      enc = cur.fetchone()["server_encoding"]
      if enc.upper() not in ("UTF8", "UTF-8"):
        raise StorageError(f"database encoding is {enc}; RaySimBatch requires a UTF8 database")
      cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())")
      cur.execute("SELECT pg_advisory_xact_lock(7201)")
      cur.execute("SELECT name FROM schema_migrations")
      done = {r["name"] for r in cur.fetchall()}
      applied = []
      for f in files:
        if f.name in done:
          continue
        cur.execute(f.read_text())
        cur.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))
        applied.append(f.name)
      return applied

    return self.tx(run)

  def now_ns(self) -> int:
    return self.tx(lambda cur: cur.execute("SELECT (extract(epoch from clock_timestamp())*1e9)::bigint AS t").fetchone()["t"])

  # ---------------------------------------------------------------------- runs

  def create_run(self, run_id: str, config_hash: str, mode: str, payload: dict, idempotency_key: str,
                 caller: str) -> tuple[str, bool]:
    def run(cur):
      cur.execute("SELECT run_id FROM runs WHERE idempotency_key=%s", (idempotency_key,))
      row = cur.fetchone()
      if row:
        return row["run_id"], False
      cur.execute(
        "INSERT INTO runs (run_id, config_hash, mode, state, idempotency_key, caller, payload) "
        "VALUES (%s,%s,%s,'NEW',%s,%s,%s)",
        (run_id, config_hash, mode, idempotency_key, caller, json.dumps(payload)),
      )
      cur.execute("INSERT INTO id_blocks (run_id) VALUES (%s) ON CONFLICT DO NOTHING", (run_id,))
      return run_id, True

    return self.tx(run)

  def get_run(self, run_id: str) -> dict | None:
    return self.tx(lambda cur: cur.execute("SELECT * FROM runs WHERE run_id=%s", (run_id,)).fetchone())

  def set_run_state(self, run_id: str, state: str, expected: Sequence[str] | None = None,
                    payload_patch: dict | None = None, bump_attempt: bool = False) -> bool:
    def run(cur):
      q = "UPDATE runs SET state=%s, updated_at=now()"
      args: list[Any] = [state]
      if payload_patch:
        q += ", payload = payload || %s::jsonb"
        args.append(json.dumps(payload_patch))
      if bump_attempt:
        q += ", attempt = attempt + 1, epoch = epoch + 1"
      q += " WHERE run_id=%s"
      args.append(run_id)
      if expected:
        q += " AND state = ANY(%s)"
        args.append(list(expected))
      cur.execute(q, args)
      return cur.rowcount == 1

    return self.tx(run)

  def list_runs(self, states: Sequence[str] | None = None) -> list[dict]:
    def run(cur):
      if states:
        return cur.execute("SELECT * FROM runs WHERE state = ANY(%s) ORDER BY created_at", (list(states),)).fetchall()
      return cur.execute("SELECT * FROM runs ORDER BY created_at").fetchall()

    return self.tx(run)

  # -------------------------------------------------------------------- leases

  def acquire_lease(self, key: str, owner: str, ttl_s: float) -> Lease:
    """Take or renew a fencing lease; each change of owner bumps the epoch."""
    def run(cur):
      cur.execute(
        "INSERT INTO leases (key, owner, epoch, lease_until) VALUES (%s,%s,1, now() + %s * interval '1 second') "
        "ON CONFLICT (key) DO UPDATE SET "
        " epoch = CASE WHEN leases.owner = EXCLUDED.owner THEN leases.epoch ELSE leases.epoch + 1 END,"
        " owner = EXCLUDED.owner, lease_until = EXCLUDED.lease_until "
        "WHERE leases.owner = EXCLUDED.owner OR leases.lease_until < now() "
        "RETURNING epoch, (extract(epoch from lease_until)*1e9)::bigint AS until",
        (key, owner, ttl_s),
      )
      row = cur.fetchone()
      if row is None:
        raise LeaseHeld(f"lease {key!r} is held by another owner")
      return Lease(owner, int(row["epoch"]), int(row["until"]))

    return self.tx(run)

  def renew_lease(self, key: str, owner: str, epoch: int, ttl_s: float) -> Lease | None:
    def run(cur):
      cur.execute(
        "UPDATE leases SET lease_until = now() + %s * interval '1 second' "
        "WHERE key=%s AND owner=%s AND epoch=%s AND lease_until >= now() "
        "RETURNING (extract(epoch from lease_until)*1e9)::bigint AS until",
        (ttl_s, key, owner, epoch),
      )
      row = cur.fetchone()
      return None if row is None else Lease(owner, epoch, int(row["until"]))

    return self.tx(run)

  def release_lease(self, key: str, owner: str, epoch: int) -> None:
    self.tx(lambda cur: cur.execute("UPDATE leases SET lease_until = now() - interval '1 second' "
                                    "WHERE key=%s AND owner=%s AND epoch=%s", (key, owner, epoch)))

  def lease_epoch(self, key: str) -> int:
    row = self.tx(lambda cur: cur.execute("SELECT epoch FROM leases WHERE key=%s", (key,)).fetchone())
    return 0 if row is None else int(row["epoch"])

  def bump_generation(self, key: str, owner: str) -> int:
    """Atomically allocate the next generation for ``key`` (a new actor incarnation or
    a fence); older generations are rejected by every consumer that checks it."""
    return int(self.tx(lambda cur: cur.execute(
      "INSERT INTO leases (key, owner, epoch, lease_until) VALUES (%s,%s,1, now() + interval '100 years') "
      "ON CONFLICT (key) DO UPDATE SET epoch = leases.epoch + 1, owner = EXCLUDED.owner RETURNING epoch",
      (key, owner)).fetchone()["epoch"]))

  def lease_state(self, key: str) -> dict | None:
    """owner, epoch, lease_until and whether it has expired (database time), or None."""
    return self.tx(lambda cur: cur.execute(
      "SELECT owner, epoch, lease_until, lease_until < now() AS expired FROM leases WHERE key=%s", (key,)).fetchone())

  def expired_leases(self, prefix: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute(
      "SELECT key, owner, epoch FROM leases WHERE key LIKE %s AND lease_until < now()", (prefix + "%",)).fetchall())

  # ------------------------------------------------------------------- commits

  def commit(self, key: str, epoch: int, digest: str, artifact: Artifact) -> CommitAck:
    """Epoch-fenced keyed commit: same digest acknowledges, a different digest is
    corruption, a lower epoch is rejected."""
    def run(cur):
      cur.execute("SELECT epoch, digest FROM commits WHERE key=%s FOR UPDATE", (key,))
      row = cur.fetchone()
      if row is None:
        cur.execute("INSERT INTO commits (key, epoch, digest, artifact_uri, artifact_hash) VALUES (%s,%s,%s,%s,%s)",
                    (key, epoch, digest, artifact.uri, artifact.sha256))
        return CommitAck((key,), generation=1)
      if row["digest"] == digest:
        return CommitAck((), duplicates=(key,), generation=1)
      if epoch < row["epoch"]:
        return CommitAck((), rejected="stale_epoch")
      raise DigestConflict(f"commit {key!r}: different digest for the same identity")

    return self.tx(run)

  # ------------------------------------------------------------ evaluation units

  def insert_units(self, run_id: str, units: Iterable[dict]) -> int:
    rows = [(run_id, u["unit_id"], u["payload_hash"], json.dumps(u["payload"]), int(u.get("wave", 0))) for u in units]

    def run(cur):
      n = 0
      for r in rows:
        cur.execute(
          "INSERT INTO units (run_id, unit_id, state, payload_hash, payload, wave) VALUES (%s,%s,'PENDING',%s,%s,%s) "
          "ON CONFLICT (run_id, unit_id) DO NOTHING", r)
        n += cur.rowcount
      return n

    return self.tx(run)

  def lease_units(self, run_id: str, owner: str, limit: int, lease_s: float, max_attempts: int) -> list[dict]:
    def run(cur):
      cur.execute(
        "UPDATE units SET state='RETRYABLE', fault=jsonb_build_object('code','LEASE_EXPIRED','owner',lease_owner), "
        "updated_at=now() WHERE run_id=%s AND state IN ('LEASED','RUNNING','COMMITTING') AND lease_until < now()",
        (run_id,),
      )
      expired = cur.rowcount
      cur.execute("UPDATE units SET state='EXHAUSTED', updated_at=now() "
                  "WHERE run_id=%s AND state='RETRYABLE' AND attempt >= %s", (run_id, max_attempts))
      cur.execute(
        "WITH cand AS (SELECT unit_id FROM units WHERE run_id=%s AND state IN ('PENDING','RETRYABLE') "
        "  ORDER BY wave, unit_id LIMIT %s FOR UPDATE SKIP LOCKED) "
        "UPDATE units u SET state='LEASED', attempt=u.attempt+1, lease_epoch=u.lease_epoch+1, lease_owner=%s, "
        "  lease_until=now() + %s * interval '1 second', updated_at=now() "
        "FROM cand WHERE u.run_id=%s AND u.unit_id=cand.unit_id "
        "RETURNING u.unit_id, u.attempt, u.lease_epoch, (extract(epoch from u.lease_until)*1e9)::bigint AS lease_until_ns, "
        "  u.payload, u.payload_hash",
        (run_id, limit, owner, lease_s, run_id),
      )
      rows = cur.fetchall()
      for r in rows:
        r["expired_before"] = expired
      return sorted(rows, key=lambda r: r["unit_id"])

    return self.tx(run)

  def renew_units(self, run_id: str, owner: str, units: Sequence[tuple[str, int, int]], lease_s: float) -> list[str]:
    def run(cur):
      ok = []
      for unit_id, attempt, epoch in units:
        cur.execute(
          "UPDATE units SET lease_until = now() + %s * interval '1 second' WHERE run_id=%s AND unit_id=%s "
          "AND attempt=%s AND lease_epoch=%s AND lease_owner=%s AND state IN ('LEASED','RUNNING','COMMITTING') "
          "AND lease_until >= now()", (lease_s, run_id, unit_id, attempt, epoch, owner))
        if cur.rowcount:
          ok.append(unit_id)
      return ok

    return self.tx(run)

  def mark_running(self, run_id: str, unit_id: str, attempt: int, lease_epoch: int) -> bool:
    return self.tx(lambda cur: cur.execute(
      "UPDATE units SET state='RUNNING', updated_at=now() WHERE run_id=%s AND unit_id=%s AND attempt=%s "
      "AND lease_epoch=%s AND state IN ('LEASED','RUNNING')", (run_id, unit_id, attempt, lease_epoch)).rowcount == 1)

  def commit_episodes(self, run_id: str, unit_id: str, attempt: int, lease_epoch: int,
                      rows: Sequence[dict], expected_count: int) -> CommitAck:
    """rows: ordinal, digest, object_uri, outcome, row (JSON-able)."""
    from raysimbatch.telemetry import tracing

    with tracing.span("durable_commit", unit_id=unit_id, rows=len(rows)):
      return self._commit_episodes(run_id, unit_id, attempt, lease_epoch, rows, expected_count)

  def _commit_episodes(self, run_id: str, unit_id: str, attempt: int, lease_epoch: int,
                       rows: Sequence[dict], expected_count: int) -> CommitAck:
    def run(cur):
      cur.execute("SELECT state, attempt, lease_epoch FROM units WHERE run_id=%s AND unit_id=%s FOR UPDATE",
                  (run_id, unit_id))
      u = cur.fetchone()
      if u is None:
        return CommitAck((), rejected="unknown_unit")
      if u["state"] == "SUCCEEDED":
        # Late duplicate delivery after success: acknowledge identical rows only.
        pass
      elif u["attempt"] != attempt or u["lease_epoch"] != lease_epoch or u["state"] not in (
        "LEASED", "RUNNING", "COMMITTING"
      ):
        return CommitAck((), rejected="stale_attempt", unit_state=u["state"])
      committed, dup = [], []
      for r in rows:
        key = f"{unit_id}/{r['ordinal']}"
        if u["state"] == "SUCCEEDED":
          cur.execute("SELECT digest FROM episodes WHERE run_id=%s AND unit_id=%s AND ordinal=%s",
                      (run_id, unit_id, r["ordinal"]))
          ex = cur.fetchone()
          if ex is None:
            return CommitAck((), rejected="unit_already_complete", unit_state="SUCCEEDED")
          if ex["digest"] != r["digest"]:
            raise DigestConflict(f"episode {key}: different digest for a committed identity")
          dup.append(key)
          continue
        cur.execute(
          "INSERT INTO episodes (run_id, unit_id, ordinal, digest, object_uri, outcome, attempt, row_data) "
          "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id, unit_id, ordinal) DO NOTHING",
          (run_id, unit_id, r["ordinal"], r["digest"], r["object_uri"], r["outcome"], attempt, json.dumps(r["row"])),
        )
        if cur.rowcount:
          committed.append(key)
        else:
          cur.execute("SELECT digest FROM episodes WHERE run_id=%s AND unit_id=%s AND ordinal=%s",
                      (run_id, unit_id, r["ordinal"]))
          if cur.fetchone()["digest"] != r["digest"]:
            raise DigestConflict(f"episode {key}: different digest for a committed identity")
          dup.append(key)
      state = u["state"]
      if state != "SUCCEEDED":
        cur.execute("SELECT count(*) AS n FROM episodes WHERE run_id=%s AND unit_id=%s", (run_id, unit_id))
        n = cur.fetchone()["n"]
        state = "SUCCEEDED" if n >= expected_count else "RUNNING"
        cur.execute("UPDATE units SET state=%s, updated_at=now() WHERE run_id=%s AND unit_id=%s",
                    (state, run_id, unit_id))
      return CommitAck(tuple(committed), tuple(dup), generation=1, unit_state=state)

    return self.tx(run)

  def fail_unit(self, run_id: str, unit_id: str, attempt: int, lease_epoch: int, fault: dict,
                max_attempts: int) -> str | None:
    def run(cur):
      cur.execute(
        "UPDATE units SET state = CASE WHEN attempt >= %s THEN 'EXHAUSTED' ELSE 'RETRYABLE' END, fault=%s, "
        "updated_at=now() WHERE run_id=%s AND unit_id=%s AND attempt=%s AND lease_epoch=%s "
        "AND state IN ('LEASED','RUNNING','COMMITTING') RETURNING state",
        (max_attempts, json.dumps(fault), run_id, unit_id, attempt, lease_epoch),
      )
      row = cur.fetchone()
      return None if row is None else row["state"]

    return self.tx(run)

  def release_units(self, run_id: str, owner: str) -> int:
    """Voluntary release (drain/cancel of an owner): leased units become PENDING again
    without consuming an attempt only if nothing was committed for them."""
    return self.tx(lambda cur: cur.execute(
      "UPDATE units u SET state='RETRYABLE', updated_at=now() WHERE run_id=%s AND lease_owner=%s "
      "AND state IN ('LEASED','RUNNING','COMMITTING')", (run_id, owner)).rowcount)

  def cancel_units(self, run_id: str) -> int:
    return self.tx(lambda cur: cur.execute(
      "UPDATE units SET state='CANCELED', updated_at=now() WHERE run_id=%s AND state NOT IN %s",
      (run_id, UNIT_TERMINAL)).rowcount)

  def unit_counts(self, run_id: str) -> dict[str, int]:
    rows = self.tx(lambda cur: cur.execute(
      "SELECT state, count(*) AS n FROM units WHERE run_id=%s GROUP BY state", (run_id,)).fetchall())
    return {r["state"]: int(r["n"]) for r in rows}

  def units(self, run_id: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute("SELECT * FROM units WHERE run_id=%s ORDER BY unit_id", (run_id,)).fetchall())

  def committed_ordinals(self, run_id: str, unit_id: str) -> set[int]:
    rows = self.tx(lambda cur: cur.execute(
      "SELECT ordinal FROM episodes WHERE run_id=%s AND unit_id=%s", (run_id, unit_id)).fetchall())
    return {int(r["ordinal"]) for r in rows}

  def episode_rows(self, run_id: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute(
      "SELECT unit_id, ordinal, digest, object_uri, outcome, attempt, row_data FROM episodes WHERE run_id=%s "
      "ORDER BY unit_id, ordinal", (run_id,)).fetchall())

  def episode_summary(self, run_id: str, valid_outcomes) -> dict[str, Any]:
    """Suite-level progress of committed episodes (the eval_* gauges): valid episodes,
    successes among them, and their mean return."""
    return self.tx(lambda cur: cur.execute(
      "SELECT count(*) FILTER (WHERE outcome = ANY(%(v)s)) AS valid, "
      "count(*) FILTER (WHERE outcome = ANY(%(v)s) AND (row_data->>'success')::boolean) AS successes, "
      "avg((row_data->>'return_sum')::float8) FILTER (WHERE outcome = ANY(%(v)s) AND row_data->>'return_sum' IS NOT NULL)"
      " AS return_mean FROM episodes WHERE run_id=%(r)s", {"v": sorted(valid_outcomes), "r": run_id}).fetchone())

  def episode_count(self, run_id: str) -> int:
    return int(self.tx(lambda cur: cur.execute(
      "SELECT count(*) AS n FROM episodes WHERE run_id=%s", (run_id,)).fetchone())["n"])

  # -------------------------------------------------------------------- outbox

  def outbox_insert(self, fragment_id: str, run_id: str, digest: str, artifact: Artifact, transitions: int) -> str:
    def run(cur):
      cur.execute(
        "INSERT INTO outbox (fragment_id, run_id, digest, object_uri, byte_length, transitions, delivery_state) "
        "VALUES (%s,%s,%s,%s,%s,%s,'PENDING') ON CONFLICT (fragment_id) DO NOTHING",
        (fragment_id, run_id, digest, artifact.uri, artifact.byte_length, transitions),
      )
      if cur.rowcount:
        return "inserted"
      cur.execute("SELECT digest FROM outbox WHERE fragment_id=%s", (fragment_id,))
      if cur.fetchone()["digest"] != digest:
        raise DigestConflict(f"fragment {fragment_id}: repeated id with a different digest")
      return "duplicate"

    return self.tx(run)

  def outbox_pending(self, run_id: str, limit: int = 100) -> list[dict]:
    return self.tx(lambda cur: cur.execute(
      "SELECT * FROM outbox WHERE run_id=%s AND delivery_state='PENDING' ORDER BY created_at LIMIT %s",
      (run_id, limit)).fetchall())

  def outbox_mark(self, fragment_id: str, state: str, sink_offset: str | None = None) -> None:
    self.tx(lambda cur: cur.execute(
      "UPDATE outbox SET delivery_state=%s, sink_offset=%s, attempts=attempts+1, "
      "delivered_at = CASE WHEN %s='DELIVERED' THEN now() ELSE delivered_at END WHERE fragment_id=%s",
      (state, sink_offset, state, fragment_id)))

  def outbox_counts(self, run_id: str) -> dict[str, int]:
    rows = self.tx(lambda cur: cur.execute(
      "SELECT delivery_state, count(*) AS n, coalesce(sum(transitions),0) AS t FROM outbox WHERE run_id=%s "
      "GROUP BY delivery_state", (run_id,)).fetchall())
    return {r["delivery_state"]: int(r["t"]) for r in rows}

  # ----------------------------------------------------------------------- ids

  def allocate_episode_ids(self, run_id: str, count: int) -> int:
    def run(cur):
      cur.execute("INSERT INTO id_blocks (run_id) VALUES (%s) ON CONFLICT DO NOTHING", (run_id,))
      cur.execute("UPDATE id_blocks SET next_episode_id = next_episode_id + %s WHERE run_id=%s "
                  "RETURNING next_episode_id - %s AS start", (count, run_id, count))
      return int(cur.fetchone()["start"])

    return self.tx(run)

  def allocate_env_ids(self, run_id: str, count: int) -> int:
    def run(cur):
      cur.execute("INSERT INTO id_blocks (run_id) VALUES (%s) ON CONFLICT DO NOTHING", (run_id,))
      cur.execute("UPDATE id_blocks SET next_env_id = next_env_id + %s WHERE run_id=%s "
                  "RETURNING next_env_id - %s AS start", (count, run_id, count))
      return int(cur.fetchone()["start"])

    return self.tx(run)

  # ------------------------------------------------------ routes, gaps, audit

  def upsert_routes(self, run_id: str, rows: Sequence[tuple[int, int, dict]]) -> None:
    def run(cur):
      for env_id, epoch, dest in rows:
        cur.execute(
          "INSERT INTO routes (run_id, env_id, epoch, destination) VALUES (%s,%s,%s,%s) "
          "ON CONFLICT (run_id, env_id) DO UPDATE SET epoch=EXCLUDED.epoch, destination=EXCLUDED.destination "
          "WHERE routes.epoch < EXCLUDED.epoch", (run_id, env_id, epoch, json.dumps(dest)))

    self.tx(run)

  def routes(self, run_id: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute("SELECT * FROM routes WHERE run_id=%s ORDER BY env_id", (run_id,)).fetchall())

  def record_gap(self, run_id: str, env_id: int, episode_id: int, committed_prefix: int, lost_lower: int,
                 lost_upper: int, reason: str, generation: int) -> None:
    self.tx(lambda cur: cur.execute(
      "INSERT INTO gaps (run_id, env_id, episode_id, committed_prefix, lost_lower, lost_upper, reason, generation) "
      "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
      (run_id, env_id, episode_id, committed_prefix, lost_lower, lost_upper, reason, generation)))

  def gaps(self, run_id: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute("SELECT * FROM gaps WHERE run_id=%s ORDER BY id", (run_id,)).fetchall())

  def quarantine_event(self, run_id: str, bin_key: str, fault_code: int, *, native: bool, health: bool,
                       native_limit: int, health_limit: int, window: int) -> bool:
    """Count a failure (or a healthy episode when both flags are false) for a bin;
    returns whether the bin is quarantined."""
    def run(cur):
      cur.execute(
        "INSERT INTO quarantine (run_id, bin_key, fault_code) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
        (run_id, bin_key, fault_code))
      cur.execute(
        "UPDATE quarantine SET native_count = native_count + %s, health_count = health_count + %s, "
        "episodes = episodes + 1, updated_at=now() WHERE run_id=%s AND bin_key=%s "
        "RETURNING native_count, health_count, episodes, quarantined",
        (int(native), int(health), run_id, bin_key))
      r = cur.fetchone()
      q = r["quarantined"] or r["native_count"] >= native_limit or (
        r["health_count"] >= health_limit and r["episodes"] <= window)
      if r["episodes"] >= window and not q:
        cur.execute("UPDATE quarantine SET episodes=0, health_count=0 WHERE run_id=%s AND bin_key=%s", (run_id, bin_key))
      if q and not r["quarantined"]:
        cur.execute("UPDATE quarantine SET quarantined=true WHERE run_id=%s AND bin_key=%s", (run_id, bin_key))
      return bool(q)

    return self.tx(run)

  def quarantined_bins(self, run_id: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute(
      "SELECT * FROM quarantine WHERE run_id=%s AND quarantined", (run_id,)).fetchall())

  def audit(self, run_id: str | None, event: str, payload: dict | None = None) -> None:
    self.tx(lambda cur: cur.execute("INSERT INTO audit (run_id, event, payload) VALUES (%s,%s,%s)",
                                    (run_id, event, json.dumps(payload or {}, default=str))))

  def audit_events(self, run_id: str) -> list[dict]:
    return self.tx(lambda cur: cur.execute(
      "SELECT ts, event, payload FROM audit WHERE run_id=%s ORDER BY id", (run_id,)).fetchall())

  def register_artifact(self, a: Artifact) -> None:
    self.tx(lambda cur: cur.execute(
      "INSERT INTO artifacts (hash, uri, bytes, media_type) VALUES (%s,%s,%s,%s) ON CONFLICT (hash) DO NOTHING",
      (a.sha256, a.uri, a.byte_length, a.media_type)))

  def artifact(self, sha: str) -> Artifact | None:
    r = self.tx(lambda cur: cur.execute("SELECT * FROM artifacts WHERE hash=%s", (sha,)).fetchone())
    return None if r is None else Artifact(r["uri"], r["hash"], int(r["bytes"]), r["media_type"])

  def referenced_hashes(self) -> set[str]:
    def run(cur):
      out = {r["hash"] for r in cur.execute("SELECT hash FROM artifacts").fetchall()}
      out |= {r["artifact_hash"] for r in cur.execute("SELECT artifact_hash FROM commits").fetchall()}
      for r in cur.execute("SELECT DISTINCT object_uri FROM episodes").fetchall():
        out.add(r["object_uri"].rsplit("/", 1)[-1])
      for r in cur.execute("SELECT object_uri FROM outbox").fetchall():
        out.add(r["object_uri"].rsplit("/", 1)[-1])
      return out

    return self.tx(run)
