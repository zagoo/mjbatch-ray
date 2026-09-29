#!/usr/bin/env bash
# Local qualification services: PostgreSQL (project-local cluster) and MinIO (Docker).
# Usage: tools/local_services.sh start|stop|status|env
# Credentials are generated test values stored in var/services.env (git-ignored).
set -euo pipefail
# PostgreSQL on macOS aborts under some locales ("postmaster became multithreaded").
export LC_ALL=C LANG=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAR="$ROOT/var"
PGBIN="${PGBIN:-$(brew --prefix postgresql@17 2>/dev/null || echo /usr)/bin}"
PGDATA="$VAR/pg"
PGPORT="${RSB_PG_PORT:-55432}"
MINIO_PORT="${RSB_MINIO_PORT:-59000}"
# Upstream MinIO images are no longer publicly pullable (Docker Hub/Quay deny access as
# of 2026-09); a locally cached image is used by digest when present, otherwise the
# still-published SeaweedFS S3 gateway (pinned by digest).
MINIO_IMAGE="minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
SEAWEED_IMAGE="chrislusf/seaweedfs@sha256:ce9e796f1fe6f06968f4c04bdaf8f678dad9c8acdfef3d244133d71bfa6bf882"
ENVF="$VAR/services.env"
mkdir -p "$VAR"

gen_env() {
  if [[ ! -f "$ENVF" ]]; then
    umask 077
    {
      echo "RSB_PG_PASSWORD=$(openssl rand -hex 16)"
      echo "RSB_MINIO_USER=rsbminio$(openssl rand -hex 4)"
      echo "RSB_MINIO_PASSWORD=$(openssl rand -hex 16)"
    } > "$ENVF"
  fi
  # shellcheck disable=SC1090
  source "$ENVF"
}

start() {
  gen_env
  if [[ ! -d "$PGDATA" ]]; then
    pwfile="$(mktemp)"; echo "$RSB_PG_PASSWORD" > "$pwfile"
    "$PGBIN/initdb" -D "$PGDATA" -U rsb --pwfile="$pwfile" -A scram-sha-256 --locale=C --encoding=UTF8 >/dev/null
    rm -f "$pwfile"
  fi
  if ! "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    "$PGBIN/pg_ctl" -D "$PGDATA" -l "$VAR/pg.log" -o "-p $PGPORT -k $VAR -c listen_addresses=127.0.0.1" start >/dev/null
  fi
  for _ in $(seq 50); do
    PGPASSWORD="$RSB_PG_PASSWORD" "$PGBIN/psql" -h 127.0.0.1 -p "$PGPORT" -U rsb -d postgres -qtc "select 1" >/dev/null 2>&1 && break
    sleep 0.2
  done
  PGPASSWORD="$RSB_PG_PASSWORD" "$PGBIN/psql" -h 127.0.0.1 -p "$PGPORT" -U rsb -d postgres -qtc \
    "select 1 from pg_database where datname='raysimbatch'" | grep -q 1 || \
    PGPASSWORD="$RSB_PG_PASSWORD" "$PGBIN/createdb" -h 127.0.0.1 -p "$PGPORT" -U rsb raysimbatch
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    if ! docker ps --format '{{.Names}}' | grep -qx rsb-minio; then
      docker rm -f rsb-minio >/dev/null 2>&1 || true
      if docker image inspect "$MINIO_IMAGE" >/dev/null 2>&1; then
        docker run -d --name rsb-minio -p "127.0.0.1:$MINIO_PORT:9000" \
          -e "MINIO_ROOT_USER=$RSB_MINIO_USER" -e "MINIO_ROOT_PASSWORD=$RSB_MINIO_PASSWORD" \
          "$MINIO_IMAGE" server /data >/dev/null
      else
        cfg="$VAR/s3.json"
        printf '{"identities":[{"name":"rsb","credentials":[{"accessKey":"%s","secretKey":"%s"}],"actions":["Admin","Read","Write","List","Tagging"]}]}' \
          "$RSB_MINIO_USER" "$RSB_MINIO_PASSWORD" > "$cfg"
        docker run -d --name rsb-minio -p "127.0.0.1:$MINIO_PORT:8333" -v "$cfg:/etc/s3.json:ro" \
          "$SEAWEED_IMAGE" server -s3 -s3.config=/etc/s3.json >/dev/null
      fi
    fi
    for _ in $(seq 100); do curl -s -o /dev/null "http://127.0.0.1:$MINIO_PORT/" && break; sleep 0.3; done
  else
    echo "docker unavailable: MinIO not started (S3 tests will be skipped)" >&2
  fi
  status
}

stop() {
  "$PGBIN/pg_ctl" -D "$PGDATA" stop -m fast >/dev/null 2>&1 || true
  docker rm -f rsb-minio >/dev/null 2>&1 || true
}

status() {
  gen_env
  "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1 && echo "postgres: up on 127.0.0.1:$PGPORT" || echo "postgres: down"
  curl -s -o /dev/null "http://127.0.0.1:$MINIO_PORT/" 2>&1 && echo "s3: up on 127.0.0.1:$MINIO_PORT" || echo "s3: down"
}

env_out() {
  gen_env
  echo "export RSB_TEST_DSN='postgresql://rsb:${RSB_PG_PASSWORD}@127.0.0.1:${PGPORT}/raysimbatch'"
  # Only a reachable endpoint: without one, the S3 tests skip instead of failing.
  if curl -s -o /dev/null "http://127.0.0.1:$MINIO_PORT/"; then
    echo "export RSB_TEST_S3_ENDPOINT='http://127.0.0.1:${MINIO_PORT}'"
    echo "export AWS_ACCESS_KEY_ID='${RSB_MINIO_USER}' AWS_SECRET_ACCESS_KEY='${RSB_MINIO_PASSWORD}' AWS_DEFAULT_REGION=us-east-1"
  else
    echo "unset RSB_TEST_S3_ENDPOINT"
  fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  env) env_out ;;
  *) echo "usage: $0 start|stop|status|env" >&2; exit 2 ;;
esac
