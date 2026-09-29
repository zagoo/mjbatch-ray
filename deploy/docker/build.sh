#!/usr/bin/env bash
# Build the production CPU image from the release lock.
# Usage: deploy/docker/build.sh <tag> [extra docker build args...]
# Always builds --target runtime (never a test/sanitizer stage), stamps the RaySimBatch
# and MJBatch revisions, and appends the image id to var/images.log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:?usage: build.sh <tag>}"; shift
GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] || GIT_SHA="$GIT_SHA-dirty"
MJBATCH_SHA="$(git -C "$ROOT/third_party/mjbatch" rev-parse HEAD)"
docker build --target runtime -f "$ROOT/deploy/docker/cpu.Dockerfile" \
  --build-arg GIT_SHA="$GIT_SHA" --build-arg MJBATCH_SHA="$MJBATCH_SHA" \
  -t "raysimbatch-cpu:$TAG" "$@" "$ROOT"
ID="$(docker image inspect "raysimbatch-cpu:$TAG" --format '{{.Id}}')"
mkdir -p "$ROOT/var"
echo "$(date -u +%FT%TZ) raysimbatch-cpu:$TAG $ID git=$GIT_SHA mjbatch=$MJBATCH_SHA" | tee -a "$ROOT/var/images.log"
