#!/usr/bin/env bash
# Validate and unit-test the Prometheus rules with promtool from a digest-pinned image.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG="prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996"
run() { docker run --rm --entrypoint promtool -v "$ROOT/deploy/observability:/rules:ro" -w /rules "$IMG" "$@"; }
run check rules alerts.yaml
run test rules alerts.test.yaml
# Dashboard queries (tools/gen_dashboards.py): each as a recording rule, so promtool
# parses every PromQL expression the panels use.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"${PYTHON:-python3}" "$ROOT/tools/gen_dashboards.py" --rules-out "$TMP/dashboard-queries.yaml"
docker run --rm --entrypoint promtool -v "$TMP:/q:ro" -w /q "$IMG" check rules dashboard-queries.yaml
