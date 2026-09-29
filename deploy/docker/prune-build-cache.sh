#!/usr/bin/env bash
# Remove reclaimable BuildKit cache records created by this project's Dockerfile steps
# (each rebuild adds ~2.3 GB). Other projects' cache on the same Docker host is untouched.
set -euo pipefail
ids=$(docker buildx du --verbose 2>/dev/null | awk 'BEGIN{RS=""} /Reclaimable:[ \t]*true/ && (/raysimbatch\.cli compat/ || /COPY --from=build \/src\/\.venv/ || /--reinstall-package mjbatch/ || /COPY tests tests/) {for(i=1;i<=NF;i++) if($i=="ID:"){print $(i+1); break}}')
n=0
for id in $ids; do docker buildx prune -f --filter "id=$id" >/dev/null 2>&1 && n=$((n+1)); done
echo "pruned $n project cache records"
