#!/usr/bin/env bash
# Release rollback qualification on the kind environment (design §11.5):
#   1. green is active and admits a long evaluation run;
#   2. green fails hard mid-run (all its pods force-deleted: a failed release);
#   3. the release pointer rolls back to blue;
#   4. blue resumes the interrupted run from the durable store (original seeds), and a
#      fresh reference run of the same stored configuration must give identical
#      per-episode results.
# Usage: deploy/kind/rollback_exercise.sh <evidence-dir> [seeds] [steps]
# Both colors must be deployed and healthy. Every step's output lands in <evidence-dir>.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tools/env.sh"
NS=raysimbatch-qual
OUT="${1:?evidence dir}" SEEDS="${2:-32}" STEPS="${3:-200000}"
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/steps.log"; }
head_pod() { kubectl -n "$NS" get pod -l "ray.io/cluster=rsb-$1,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}'; }
units() { kubectl -n "$NS" exec deploy/postgres -- psql -U rsb -d raysimbatch -tAc \
  "select state, count(*) from units where run_id='$1' group by state order by 1; select 'run', state, epoch from runs where run_id='$1';"; }

log "images: blue=$(kubectl -n $NS get raycluster rsb-blue -o jsonpath='{.metadata.annotations.raysimbatch\.io/cpu-image}') green=$(kubectl -n $NS get raycluster rsb-green -o jsonpath='{.metadata.annotations.raysimbatch\.io/cpu-image}')"
"$ROOT/deploy/kind/release.sh" promote green | tee -a "$OUT/steps.log"
log "pointer $(kubectl -n $NS get cm raysimbatch-active -o jsonpath='{.data}')"

log "xresume-start on green (seeds=$SEEDS steps=$STEPS)"
kubectl -n "$NS" exec -c ray-head "$(head_pod green)" -- python -m raysimbatch.deployment.smoke \
  --mode xresume-start --seeds "$SEEDS" --steps "$STEPS" --handoff-units 2 > "$OUT/xresume-start.json" 2>&1
kubectl -n "$NS" delete raycluster rsb-green --wait=false >/dev/null
kubectl -n "$NS" delete pod -l ray.io/cluster=rsb-green --grace-period=0 --force >/dev/null 2>&1 || true
RID=$(python3 -c "
import json;t=open('$OUT/xresume-start.json').read();d=json.loads(t[t.index('{'):t.rindex('}')+1]);print(d['checks']['xresume_start']['run_id'])")
log "green hard-failed; run $RID durable state:"; units "$RID" | tee -a "$OUT/steps.log"

"$ROOT/deploy/kind/release.sh" rollback | tee -a "$OUT/steps.log"
log "pointer $(kubectl -n $NS get cm raysimbatch-active -o jsonpath='{.data}')"

log "xresume-finish on blue"
set +e
kubectl -n "$NS" exec -c ray-head "$(head_pod blue)" -- python -m raysimbatch.deployment.smoke \
  --mode xresume-finish --run-id "$RID" --timeout 3000 > "$OUT/xresume-finish.json" 2>&1
rc=$?
set -e
log "xresume-finish exit=$rc; run $RID durable state:"; units "$RID" | tee -a "$OUT/steps.log"
kubectl -n "$NS" exec deploy/postgres -- psql -U rsb -d raysimbatch -tAc \
  "select ts, event, left(payload::text, 300) from audit where run_id='$RID' order by id" > "$OUT/audit.txt"
exit $rc
