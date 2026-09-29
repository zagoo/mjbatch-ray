#!/usr/bin/env bash
# Blue/green release operations on a KubeRay deployment (design §11.5,
# docs/deployment/kuberay.md). A release is a RayCluster named rsb-<color>.
#
#   release.sh --context <kube-context> --namespace <ns> <command> ...
#     smoke <color> [smoke args]  run raysimbatch.deployment.smoke in that color's head
#     promote <color>             route new admissions to <color>
#     rollback                    route new admissions back to the previous color
#     active                      print the pointer
#     retire <color>              delete a drained color; never the active one
#
# Routing: `promote` and `rollback` point the Service raysimbatch-active (Ray Client
# 10001, dashboard and Jobs API 8265) at the color's head pod and record active/previous
# in the ConfigMap raysimbatch-active. Clients in the namespace submit through
# ray://raysimbatch-active:10001 and so always reach the active release.
#
# --context and --namespace are required: a release command never acts on whichever
# kubectl context happens to be current. This script uses the caller's KUBECONFIG
# (deploy/kind/release.sh sets the kind one).
set -euo pipefail
CTX="" NS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --context) CTX="${2:?--context needs a value}"; shift 2 ;;
    --namespace) NS="${2:?--namespace needs a value}"; shift 2 ;;
    *) break ;;
  esac
done
if [ -z "$CTX" ] || [ -z "$NS" ]; then
  echo "usage: release.sh --context <kube-context> --namespace <ns> <smoke|promote|rollback|active|retire> ..." >&2
  exit 2
fi
k() { kubectl --context "$CTX" -n "$NS" "$@"; }
pointer() { k get cm raysimbatch-active -o jsonpath="{.data.$1}" 2>/dev/null || true; }
state() { k get raycluster "rsb-$1" -o jsonpath='{.status.state}' 2>/dev/null || true; }
require_ready() {
  local s
  s="$(state "$1")"
  [ "$s" = ready ] || { echo "rsb-$1 is not ready (state '${s:-absent}'); not routing admissions to it" >&2; exit 1; }
}
route() {  # route <active> <previous>
  k apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Service
metadata:
  name: raysimbatch-active
  labels: {app.kubernetes.io/part-of: raysimbatch}
  annotations: {raysimbatch.io/active: "$1"}
spec:
  selector: {ray.io/cluster: rsb-$1, ray.io/node-type: head}
  ports:
  - {name: client, port: 10001, targetPort: 10001}
  - {name: dashboard, port: 8265, targetPort: 8265}
EOF
  k create cm raysimbatch-active --from-literal=active="$1" --from-literal=previous="$2" \
    --dry-run=client -o yaml | k apply -f - >/dev/null
  k annotate cm raysimbatch-active "raysimbatch.io/promoted-at=$(date -u +%FT%TZ)" --overwrite >/dev/null
  echo "active=$1 previous=$2"
}
cmd="${1:-active}"
case "$cmd" in
  smoke)
    color="${2:?smoke <color>}"
    pod="$(k get pod -l "ray.io/cluster=rsb-$color,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')"
    k exec -c ray-head "$pod" -- python -m raysimbatch.deployment.smoke --mode "${3:-eval}" "${@:4}" ;;
  promote)
    color="${2:?promote <color>}"
    require_ready "$color"
    prev="$(pointer active)"
    if [ "$prev" = "$color" ]; then  # idempotent: re-promoting must not erase the rollback target
      prev="$(pointer previous)"
    fi
    route "$color" "$prev" ;;
  rollback)
    prev="$(pointer previous)" cur="$(pointer active)"
    [ -n "$prev" ] || { echo "no previous release to roll back to" >&2; exit 1; }
    require_ready "$prev"
    route "$prev" "$cur" ;;
  active) k get cm raysimbatch-active -o jsonpath='{.data}'; echo ;;
  retire)
    color="${2:?retire <color>}"
    if [ "$(pointer active)" = "$color" ]; then
      echo "rsb-$color is the active release; promote another color first" >&2
      exit 1
    fi
    k delete raycluster "rsb-$color" --wait=true ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
