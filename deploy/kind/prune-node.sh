#!/usr/bin/env bash
# Runs inside one kind node (release.sh prune-images feeds it on stdin with OLD set to
# the superseded RaySimBatch image IDs, one per line). Removes exactly those images unless
# a container uses them:
#   - CRI references, by ID (crictl rmi);
#   - the `import-<date>@sha256:<index>` names that `kind load` adds, when their index
#     resolves to one of those IDs (these kept every old image alive before);
#   - containerd leases older than 10 minutes (left by finished or interrupted imports;
#     they stop garbage collection of otherwise unreferenced layers).
set -euo pipefail
ctr() { command ctr -n k8s.io "$@"; }
used=$(crictl ps -a -o json | grep -oE "sha256:[0-9a-f]{64}" | sort -u || true)
is_old() { grep -qxF "$1" <<<"$OLD" && ! grep -qxF "$1" <<<"$used"; }
config_of() {  # index or manifest digest -> image config digest
  local blob; blob=$(ctr content get "$1" 2>/dev/null) || return 0
  if grep -q '"config"' <<<"$blob"; then
    grep -oE '"config":\{[^}]*\}' <<<"$blob" | grep -oE "sha256:[0-9a-f]{64}"
  else
    for m in $(grep -oE '"digest":"sha256:[0-9a-f]{64}"' <<<"$blob" | grep -oE "sha256:[0-9a-f]{64}"); do config_of "$m"; done
  fi
}
rmi=0 names=0 leases=0 in_use=0
for id in $(crictl images -q); do
  if grep -qxF "$id" <<<"$OLD"; then
    if is_old "$id"; then crictl rmi "$id" >/dev/null && rmi=$((rmi + 1)); else in_use=$((in_use + 1)); fi
  fi
done
for name in $(ctr images ls -q | grep '^import-' || true); do
  for cfg in $(config_of "${name#*@}"); do
    if is_old "$cfg"; then ctr images rm "$name" >/dev/null && names=$((names + 1)); break; fi
  done
done
cutoff=$(( $(date +%s) - 600 ))
while read -r lease created _; do
  [ -n "$lease" ] || continue
  if [ "$(date -d "$created" +%s)" -lt "$cutoff" ]; then ctr leases delete "$lease" >/dev/null && leases=$((leases + 1)); fi
done < <(ctr leases ls | tail -n +2)
sleep 2  # containerd collects unreferenced content after deletions
echo "$(hostname) rmi=$rmi import_names=$names stale_leases=$leases in_use=$in_use free=$(df -h / | awk 'NR==2 {print $4}')"
