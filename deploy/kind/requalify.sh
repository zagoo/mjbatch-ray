#!/usr/bin/env bash
# Requalify the release image of the current revision on kind (production to-do 1.1):
#   1. build raysimbatch-cpu at HEAD (clean tree only) as both colors;
#   2. redeploy blue and green from it; smoke blue (Mode B) and green (Mode B and A);
#   3. routing: promote green, then roll back to blue; each time the Service
#      raysimbatch-active must select that color's head, and a Ray Client connection
#      through it must land in that cluster;
#   4. CPU admission inside a real worker pod (accept its own exclusive cpuset, refuse a
#      mismatched request);
#   5. the rollback exercise (green fails hard mid-run, blue resumes) and the janitor
#      exercise (blue replaced, its janitor resumes the orphan), both compared with an
#      uninterrupted reference run.
# Usage: deploy/kind/requalify.sh <evidence-dir> [seeds] [steps]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tools/env.sh"
NS=raysimbatch-qual REL="$ROOT/deploy/kind/release.sh"
OUT="${1:?evidence dir}" SEEDS="${2:-32}" STEPS="${3:-200000}"
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] || {
  echo "uncommitted changes: the image must match a revision" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/steps.log"; }
head_pod() { kubectl -n "$NS" get pod -l "ray.io/cluster=rsb-$1,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}'; }
route_check() {  # route_check <color>: the Service and a Ray Client connection reach it
  local want="$1" sel eps host
  sel="$(kubectl -n "$NS" get svc raysimbatch-active -o jsonpath='{.spec.selector.ray\.io/cluster}')"
  for _ in $(seq 30); do
    eps="$(kubectl -n "$NS" get endpoints raysimbatch-active -o jsonpath='{.subsets[*].addresses[*].targetRef.name}')"
    [[ "$eps" == "rsb-$want-head"* ]] && break; sleep 2
  done
  host="$(kubectl -n "$NS" exec -c ray-head "$(head_pod blue)" -- python -c '
import ray
ray.init("ray://raysimbatch-active:10001", log_to_driver=False)
@ray.remote(num_cpus=0)
def where():
  import socket
  return socket.gethostname()
print(ray.get(where.remote()))' 2>/dev/null | tail -1)"
  log "routing: active=$want selector=$sel endpoints=$eps ray-client-landed-on=$host"
  [ "$sel" = "rsb-$want" ] && [[ "$eps" == "rsb-$want-head"* ]] && [[ "$host" == "rsb-$want-"* ]]
}

log "revision $(git -C "$ROOT" rev-parse HEAD)"
"$ROOT/deploy/docker/prune-build-cache.sh" | tee -a "$OUT/steps.log"
for c in blue green; do  # the second build is fully cached: same code, its own record
  "$ROOT/deploy/docker/build.sh" "$c" > "$OUT/build-$c.log" 2>&1
done
# Recorded only after both builds: a tracked file changed in between would stamp the
# second image "-dirty".
for c in blue green; do tail -1 "$OUT/build-$c.log" | tee -a "$OUT/steps.log" "$ROOT/evidence/deploy/images.log"; done
grep -q -- "-dirty" "$OUT/steps.log" && { log "an image is stamped -dirty"; exit 1; }

kubectl -n "$NS" delete raycluster rsb-blue rsb-green --wait=true >/dev/null 2>&1 || true
for _ in $(seq 60); do kubectl -n "$NS" get pods -l ray.io/cluster --no-headers 2>/dev/null | grep -q . || break; sleep 5; done
"$REL" prune-images | tee -a "$OUT/steps.log"
"$ROOT/deploy/kind/up.sh" blue blue > "$OUT/up-blue.log" 2>&1
"$ROOT/deploy/kind/up.sh" green green > "$OUT/up-green.log" 2>&1
"$REL" promote blue | tee -a "$OUT/steps.log"
"$REL" smoke blue eval > "$OUT/smoke-blue-eval.json" 2>&1
log "smoke blue eval: passed"
"$REL" smoke green both > "$OUT/smoke-green-both.json" 2>&1
log "smoke green both: passed"

"$REL" promote green | tee -a "$OUT/steps.log"
route_check green
"$REL" rollback | tee -a "$OUT/steps.log"
route_check blue
if "$REL" retire blue 2> "$OUT/retire-active.txt"; then log "retire of the active color was NOT refused"; exit 1; fi
log "retire of the active color refused: $(cat "$OUT/retire-active.txt")"

W="$(kubectl -n "$NS" get pod -l ray.io/cluster=rsb-blue,ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')"
{
  echo "# $(date -u +%FT%TZ) pod $W"
  echo "## pod request and real CPU set"
  kubectl -n "$NS" exec -c ray-worker "$W" -- sh -c 'echo "RSB_POD_CPUS=$RSB_POD_CPUS"; grep Cpus_allowed_list /proc/self/status'
  echo "## agent with the pod's own request (must accept)"
  kubectl -n "$NS" exec -c ray-worker "$W" -- python -c \
    "from raysimbatch.worker.affinity_agent import AffinityAgent as A; a=A(); print('accepted', a.all_cpus)"
  echo "## agent told the pod requested 8 CPUs (must refuse)"
  if kubectl -n "$NS" exec -c ray-worker "$W" -- env RSB_POD_CPUS=8 python -c \
    "from raysimbatch.worker.affinity_agent import AffinityAgent as A; A()" > "$OUT/refusal.txt" 2>&1; then
    echo "NOT REFUSED"; exit 1
  fi
  grep -E "RuntimeError" "$OUT/refusal.txt"
} > "$OUT/admission-check.txt" 2>&1
log "admission: accepted the pod's own cpuset, refused a mismatched request"

"$ROOT/deploy/kind/rollback_exercise.sh" "$OUT/rollback-exercise" "$SEEDS" "$STEPS"
REF="$(python3 -c "
import json;t=open('$OUT/rollback-exercise/xresume-finish.json').read();d=json.loads(t[t.index('{'):t.rindex('}')+1]);print(d['checks']['xresume_finish']['reference']['run_id'])")"
log "rollback exercise passed; reference run $REF"
"$ROOT/deploy/kind/janitor_exercise.sh" "$OUT/janitor-exercise" blue "$REF" "$SEEDS" "$STEPS"
log "janitor exercise passed"
