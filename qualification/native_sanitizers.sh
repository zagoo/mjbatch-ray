#!/usr/bin/env bash
# Native sanitizer qualification (design §13.1): ASan+UBSan and TSan over the MJBatch
# contract suite and the core semantics suite, in the Linux build container.
# Usage: qualification/native_sanitizers.sh <evidence-dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/evidence/sanitizers}"
mkdir -p "$OUT"
cd "$ROOT"
# libstdc++ is preloaded with the sanitizer runtime: python is not a C++ program, so
# without it ASan cannot resolve the real __cxa_throw that MJBatch's error path uses.
run() {
  local san="$1" tag="$2" preload="$3"
  shift 3
  local envs=()
  for e in "$@"; do envs+=(-e "$e"); done
  docker build -f deploy/docker/cpu.Dockerfile --target sanitize --build-arg SAN="$san" -t "raysimbatch-cpu:san-$tag" . \
    > "$OUT/build-$tag.log" 2>&1
  set +e
  # seccomp=unconfined: TSan disables ASLR via personality(), which Docker's default
  # profile blocks. Only this qualification container gets it.
  docker run --rm --cpus 8 --security-opt seccomp=unconfined "${envs[@]}" "raysimbatch-cpu:san-$tag" bash -c \
    "LD_PRELOAD=\$(gcc -print-file-name=$preload):\$(gcc -print-file-name=libstdc++.so.6) /src/.venv/bin/python -m pytest tests/upstream tests/core -q -p no:cacheprovider -x" \
    > "$OUT/run-$tag.log" 2>&1
  local rc=$?
  set -e
  echo "{\"sanitizer\": \"$san\", \"exit_status\": $rc, \"log\": \"run-$tag.log\"}" > "$OUT/result-$tag.json"
  echo "$tag exit=$rc"; tail -3 "$OUT/run-$tag.log"
  return $rc
}
rc=0
run "address,undefined" asan "libasan.so" \
  "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1" "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1" || rc=1
run "thread" tsan "libtsan.so" "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1" || rc=1
exit $rc
