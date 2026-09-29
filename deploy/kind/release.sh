#!/usr/bin/env bash
# Release commands for the kind qualification environment: deploy/release.sh on context
# kind-rsb-qual and namespace raysimbatch-qual (with the project's kubeconfig), plus the
# kind-only image pruning.
#   release.sh smoke <color> [eval|rollout|both|xresume-start|xresume-finish --run-id R]
#   release.sh promote <color> | rollback | active | retire <color>   (see deploy/release.sh)
#   release.sh prune-images      # drop superseded `kind load` images from the nodes
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/tools/env.sh"
if [ "${1:-}" != prune-images ]; then
  exec "$ROOT/deploy/release.sh" --context kind-rsb-qual --namespace raysimbatch-qual "$@"
fi
# Remove superseded RaySimBatch images from the nodes: exactly the image IDs that
# deploy/docker/build.sh recorded in var/images.log and that no current
# raysimbatch-cpu tag uses. Never a blanket prune (observed: pruning removed kind's
# local-path-helper, which runs only while provisioning a volume and cannot be pulled
# back), and never by name pattern (kind load leaves untagged import records).
# By image ID (an earlier `ctr images rm <id>` removed nothing, silently: ctr takes
# names, and kind's import records kept every old image); see prune-node.sh.
cur=$(docker images raysimbatch-cpu -q --no-trunc | sort -u)
old=$(grep -ohE "sha256:[0-9a-f]{64}" "$ROOT/var/images.log" | sort -u | grep -vxF "$cur" || true)
for n in $(kind get nodes --name rsb-qual); do
  docker exec -i -e OLD="$old" "$n" bash -s < "$ROOT/deploy/kind/prune-node.sh"
done
