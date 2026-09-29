#!/usr/bin/env bash
# Orphan recovery across a replaced cluster (design §8.1 "RunController loss", janitor):
#   1. a long evaluation runs on <color>;
#   2. the whole <color> RayCluster is deleted: controller and workers are gone, and the
#      run stays live in the durable store; the script waits for its lease to expire;
#   3. <color> is redeployed (current raysimbatch-cpu:<color> image). Its janitor sidecar
#      waits for a worker node, and its first pass finds the orphan (lease expired, no
#      controller) and resumes it; the new controller must run on a worker node;
#   4. the resumed run's per-episode results must equal <reference-run>'s, an
#      uninterrupted run of the same configuration (seeds/steps must match it).
# Usage: deploy/kind/janitor_exercise.sh <evidence-dir> <color> <reference-run> [seeds] [steps]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tools/env.sh"
NS=raysimbatch-qual
OUT="${1:?evidence dir}" COLOR="${2:?color}" REF="${3:?reference run}" SEEDS="${4:-32}" STEPS="${5:-200000}"
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/steps.log"; }
head_pod() { kubectl -n "$NS" get pod -l "ray.io/cluster=rsb-$COLOR,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}'; }
sql() { kubectl -n "$NS" exec deploy/postgres -- psql -U rsb -d raysimbatch -tAc "$1"; }
units() { sql "select state, count(*) from units where run_id='$1' group by state order by 1; select 'run', state, attempt, epoch from runs where run_id='$1';"; }

log "image $(kubectl -n $NS get raycluster rsb-$COLOR -o jsonpath='{.metadata.annotations.raysimbatch\.io/cpu-image}')"
log "xresume-start on $COLOR (seeds=$SEEDS steps=$STEPS)"
kubectl -n "$NS" exec -c ray-head "$(head_pod)" -- python -m raysimbatch.deployment.smoke \
  --mode xresume-start --seeds "$SEEDS" --steps "$STEPS" --handoff-units 2 > "$OUT/xresume-start.json" 2>&1
RID=$(python3 -c "
import json;t=open('$OUT/xresume-start.json').read();d=json.loads(t[t.index('{'):t.rindex('}')+1]);print(d['checks']['xresume_start']['run_id'])")

kubectl -n "$NS" delete raycluster "rsb-$COLOR" --wait=false >/dev/null
kubectl -n "$NS" delete pod -l "ray.io/cluster=rsb-$COLOR" --grace-period=0 --force >/dev/null 2>&1 || true
for _ in $(seq 60); do kubectl -n "$NS" get pods -l "ray.io/cluster=rsb-$COLOR" --no-headers 2>/dev/null | grep -q . || break; sleep 5; done
log "rsb-$COLOR deleted; run $RID durable state:"; units "$RID" | tee -a "$OUT/steps.log"
# Force-deleted pods' processes renew the lease until their grace period ends. Waiting
# for expiry makes the new janitor's *first* pass find the orphan, at cluster start,
# when worker nodes may not have registered yet (the window where a controller used to
# be placed on the head).
until [ "$(sql "select lease_until < now() from leases where key='run/$RID'")" = t ]; do sleep 5; done
log "run lease expired: $(sql "select lease_until from leases where key='run/$RID'")"

"$ROOT/deploy/kind/up.sh" "$COLOR" "$COLOR" > "$OUT/up.log" 2>&1
log "rsb-$COLOR redeployed: $(kubectl -n $NS get raycluster rsb-$COLOR -o jsonpath='{.metadata.annotations.raysimbatch\.io/cpu-image}')"

deadline=$((SECONDS + 3000))
until sql "select state from runs where run_id='$RID'" | grep -qvE '^(VALIDATING|RESERVING|RUNNING|DRAINING|DEGRADED|CANCELING|PAUSED)$'; do
  [ $SECONDS -lt $deadline ] || { log "timed out waiting for $RID"; break; }
  sleep 20
done
log "run $RID durable state:"; units "$RID" | tee -a "$OUT/steps.log"

H=$(head_pod)
kubectl -n "$NS" logs "$H" -c janitor > "$OUT/janitor.log" 2>&1 || true
sql "select ts, event, left(payload::text, 300) from audit where run_id='$RID' order by id" > "$OUT/audit.txt"
# Where each controller of the run ran: RunController worker logs name their run.
for p in $(kubectl -n "$NS" get pod -l "ray.io/cluster=rsb-$COLOR" -o name); do
  t=$(kubectl -n "$NS" get "$p" -o jsonpath='{.metadata.labels.ray\.io/node-type}')
  c=$([ "$t" = head ] && echo ray-head || echo ray-worker)
  kubectl -n "$NS" exec -c "$c" "$p" -- sh -c "cd /tmp/ray/session_latest/logs && for f in \$(grep -l ':actor_name:RunController' worker-*.out 2>/dev/null); do b=\${f%.out}; cat \$b.out \$b.err | grep -q '$RID' && echo \$b; done; true" |
    sed "s|^|$p $t |"
done > "$OUT/controller-placement.txt"
set +e
kubectl -n "$NS" exec -c ray-head "$H" -- python -m raysimbatch.deployment.smoke \
  --mode compare-runs --run-id "$RID" --reference-run "$REF" > "$OUT/compare-runs.json" 2>&1
rc=$?
set -e
on_head=$(grep -c ' head ' "$OUT/controller-placement.txt" || true)
log "compare-runs exit=$rc; controllers of $RID on the head: $on_head"
[ "$rc" -eq 0 ] && [ "$on_head" -eq 0 ]
