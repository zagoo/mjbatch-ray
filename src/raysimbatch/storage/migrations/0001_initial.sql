-- SPDX-License-Identifier: Apache-2.0
-- RaySimBatch durable schema v1 (design §6.6, §6.5, §8.3). Migrations are additive for
-- one release window; destructive cleanup happens only after retention expiry.

CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  config_hash   TEXT NOT NULL,
  mode          TEXT NOT NULL,
  state         TEXT NOT NULL,
  epoch         BIGINT NOT NULL DEFAULT 0,
  attempt       INTEGER NOT NULL DEFAULT 0,
  lease_owner   TEXT,
  lease_until   TIMESTAMPTZ,
  idempotency_key TEXT UNIQUE,
  caller        TEXT,
  payload       JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS units (
  run_id        TEXT NOT NULL REFERENCES runs(run_id),
  unit_id       TEXT NOT NULL,
  state         TEXT NOT NULL,
  attempt       INTEGER NOT NULL DEFAULT 0,
  lease_epoch   BIGINT NOT NULL DEFAULT 0,
  lease_owner   TEXT,
  lease_until   TIMESTAMPTZ,
  payload_hash  TEXT NOT NULL,
  payload       JSONB NOT NULL,
  fault         JSONB,
  wave          INTEGER NOT NULL DEFAULT 0,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, unit_id)
);
CREATE INDEX IF NOT EXISTS units_state ON units (run_id, state);

CREATE TABLE IF NOT EXISTS episodes (
  run_id        TEXT NOT NULL,
  unit_id       TEXT NOT NULL,
  ordinal       INTEGER NOT NULL,
  digest        TEXT NOT NULL,
  object_uri    TEXT NOT NULL,
  outcome       TEXT NOT NULL,
  attempt       INTEGER NOT NULL,
  row_data      JSONB NOT NULL,
  committed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, unit_id, ordinal),
  FOREIGN KEY (run_id, unit_id) REFERENCES units(run_id, unit_id)
);

CREATE TABLE IF NOT EXISTS outbox (
  fragment_id   TEXT PRIMARY KEY,
  run_id        TEXT NOT NULL,
  digest        TEXT NOT NULL,
  object_uri    TEXT NOT NULL,
  byte_length   BIGINT NOT NULL,
  transitions   INTEGER NOT NULL,
  delivery_state TEXT NOT NULL,
  attempts      INTEGER NOT NULL DEFAULT 0,
  sink_offset   TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  delivered_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (run_id, delivery_state);

CREATE TABLE IF NOT EXISTS routes (
  run_id        TEXT NOT NULL,
  env_id        BIGINT NOT NULL,
  epoch         BIGINT NOT NULL,
  destination   JSONB NOT NULL,
  PRIMARY KEY (run_id, env_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
  hash          TEXT PRIMARY KEY,
  uri           TEXT NOT NULL,
  bytes         BIGINT NOT NULL,
  media_type    TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Generic fencing leases (controllers, coordinators, assemblers) on server time.
CREATE TABLE IF NOT EXISTS leases (
  key           TEXT PRIMARY KEY,
  owner         TEXT NOT NULL,
  epoch         BIGINT NOT NULL,
  lease_until   TIMESTAMPTZ NOT NULL
);

-- DurableStore.commit: epoch-fenced, digest-deduplicated keyed commits.
CREATE TABLE IF NOT EXISTS commits (
  key           TEXT PRIMARY KEY,
  epoch         BIGINT NOT NULL,
  digest        TEXT NOT NULL,
  artifact_uri  TEXT NOT NULL,
  artifact_hash TEXT NOT NULL,
  generation    BIGINT NOT NULL DEFAULT 1,
  committed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Durable episode-id blocks: crashes skip unused ids rather than reuse them.
CREATE TABLE IF NOT EXISTS id_blocks (
  run_id        TEXT PRIMARY KEY,
  next_episode_id BIGINT NOT NULL DEFAULT 1,
  next_env_id   BIGINT NOT NULL DEFAULT 0
);

-- Explicit uncommitted-tail gaps (Mode A) with conservative bounds.
CREATE TABLE IF NOT EXISTS gaps (
  id            BIGSERIAL PRIMARY KEY,
  run_id        TEXT NOT NULL,
  env_id        BIGINT NOT NULL,
  episode_id    BIGINT NOT NULL,
  committed_prefix BIGINT NOT NULL,
  lost_lower    BIGINT NOT NULL,
  lost_upper    BIGINT NOT NULL,
  reason        TEXT NOT NULL,
  generation    BIGINT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quarantine (
  run_id        TEXT NOT NULL,
  bin_key       TEXT NOT NULL,
  fault_code    INTEGER NOT NULL,
  native_count  INTEGER NOT NULL DEFAULT 0,
  health_count  INTEGER NOT NULL DEFAULT 0,
  episodes      INTEGER NOT NULL DEFAULT 0,
  quarantined   BOOLEAN NOT NULL DEFAULT false,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, bin_key)
);

CREATE TABLE IF NOT EXISTS audit (
  id            BIGSERIAL PRIMARY KEY,
  run_id        TEXT,
  ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
  event         TEXT NOT NULL,
  payload       JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS audit_run ON audit (run_id, ts);
