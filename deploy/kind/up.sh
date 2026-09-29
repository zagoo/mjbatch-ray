#!/usr/bin/env bash
# Create the local KubeRay qualification environment and deploy one release color.
# Usage: deploy/kind/up.sh <color> <image-tag>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tools/env.sh"
COLOR="${1:-blue}" TAG="${2:-dev}"
NS=raysimbatch-qual
if ! kind get clusters | grep -qx rsb-qual; then
  kind create cluster --config "$ROOT/deploy/kind/cluster.yaml" --kubeconfig "$KUBECONFIG" --wait 120s
  helm install kuberay-operator kuberay/kuberay-operator --version 1.7.0 --namespace kuberay-system \
    --create-namespace --wait --timeout 300s
fi
# Only production runtime images may be deployed (a test/sanitizer stage has /src/tests).
if docker run --rm --entrypoint test "raysimbatch-cpu:$TAG" -e /src/tests; then
  echo "raysimbatch-cpu:$TAG is not a runtime-stage image; build it with deploy/docker/build.sh" >&2; exit 1
fi
kind load docker-image --name rsb-qual "raysimbatch-cpu:$TAG" >/dev/null
# Third-party images are pinned by digest and loaded under a local tag (pods use
# imagePullPolicy: Never, so nothing is pulled at runtime).
for img in "postgres@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3" \
           "minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"; do
  ref="${img%@*}:rsb-pinned"; docker tag "$img" "$ref" && kind load docker-image --name rsb-qual "$ref" >/dev/null
done
kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"
if ! kubectl -n "$NS" get secret rsb-db >/dev/null 2>&1; then
  PW="$(openssl rand -hex 16)"; AK="rsb$(openssl rand -hex 4)"; SK="$(openssl rand -hex 16)"
  kubectl -n "$NS" create secret generic rsb-db --from-literal=password="$PW" \
    --from-literal=dsn="postgresql://rsb:$PW@postgres.$NS.svc:5432/raysimbatch" >/dev/null
  kubectl -n "$NS" create secret generic rsb-s3 --from-literal=AWS_ACCESS_KEY_ID="$AK" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$SK" --from-literal=AWS_DEFAULT_REGION=us-east-1 >/dev/null
fi
kubectl apply -f "$ROOT/deploy/kind/infra.yaml" >/dev/null
kubectl -n "$NS" rollout status deploy/postgres --timeout=180s >/dev/null
kubectl -n "$NS" rollout status deploy/s3 --timeout=180s >/dev/null
INV="$(mktemp)"; sed "s/\${TAG}/$TAG/g" "$ROOT/deploy/kind/inventory.yaml" > "$INV"
"$ROOT/.venv/bin/raysimbatch" render --config "$ROOT/deploy/kind/qual-config.yaml" --inventory "$INV" --tenant qual \
  --name "rsb-$COLOR" --no-gpu --out "$ROOT/var/rendered-$COLOR.yaml"
kubectl apply -f "$ROOT/var/rendered-$COLOR.yaml" >/dev/null
echo "waiting for rsb-$COLOR head"
# The CR status and label selectors can still see a previous incarnation's pods; wait
# for the head pod owned by the *current* RayCluster object (owner UID) to be Ready.
UID_="$(kubectl -n "$NS" get raycluster "rsb-$COLOR" -o jsonpath='{.metadata.uid}')"
HEAD=""
for _ in $(seq 120); do
  HEAD="$(kubectl -n "$NS" get pod -l "ray.io/cluster=rsb-$COLOR,ray.io/node-type=head" -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.ownerReferences[0].uid} {.metadata.deletionTimestamp}{"\n"}{end}' \
    | awk -v u="$UID_" '$2 == u && $3 == "" {print $1}')"
  [ -n "$HEAD" ] && break; sleep 3
done
[ -n "$HEAD" ] || { echo "no head pod for rsb-$COLOR ($UID_)" >&2; exit 1; }
kubectl -n "$NS" wait --for=condition=Ready "pod/$HEAD" --timeout=300s
kubectl -n "$NS" get raycluster "rsb-$COLOR"
kubectl -n "$NS" get pods -o wide
