# Troubleshooting

Every entry below was observed while building and qualifying RaySimBatch. Each gives
the symptom as you would see it, the cause, and the fix or workaround now in place.

## Local environment

### PostgreSQL does not start: "postmaster became multithreaded during startup"
**Cause:** the workstation's default locale is not UTF-8 (observed with `zh_CN`).
**Fix:** `tools/local_services.sh` sets `LC_ALL=C` and runs `initdb --locale=C
--encoding=UTF8`.

### `StorageError: database encoding is SQL_ASCII; RaySimBatch requires a UTF8 database`
**Cause:** the database was initialized under `LC_ALL=C` without `--encoding`
(observed with the official postgres image). A SQL_ASCII database stores unvalidated
bytes, so `migrate` refuses it.
**Fix:** re-initialize with `--encoding=UTF8`. The kind manifests set
`POSTGRES_INITDB_ARGS`.

### Exit code 134 after all tests pass
**Symptom:** pytest prints `N passed`, then `logging.cc … recursive_mutex lock failed`
with a stack in `onnxruntime_pybind11_state.so`
(`Microsoft::Applications::Events::…HttpResponseDecoder`), and the process exits
with code 134.
**Cause:** onnxruntime 1.30 bundles a 1DS telemetry client whose worker thread races
static destruction at interpreter exit. `onnxruntime.disable_telemetry_events()`
(called by `LocalPolicyRuntime`) does not stop that thread.
**Workaround:** `tests/conftest.py` exits with pytest's own status once results are
reported. `RSB_TEST_NATURAL_EXIT=1` restores normal interpreter exit.

The same client is present in the Linux wheel. No outbound connections from onnxruntime
were observed on this workstation, but that is not proven in general. Restrict egress
from simulation pods if that matters in your environment.

### `Input should be a valid number … input_value='1e-6'` when loading a config
**Cause:** PyYAML implements YAML 1.1, which reads `1e-6` (no decimal point) as a string.
**Fix:** configuration loading resolves YAML 1.2 floats and reads `.json` files as JSON.

### Ray State API calls fail: `ServerUnavailable … 127.0.0.1:8265`
**Cause:** the Ray dashboard (which serves the State API) is not running, for example in
local sessions started with `include_dashboard=False`.
**Fix:** RaySimBatch's control paths (the janitor, helper discovery) use named actors and
`ray.nodes()`, not the State API. Only the deployment smoke report (affinity leases) uses
the State API, which runs in-cluster where the dashboard exists.

## Images and builds

### A plain `docker build` produced a sanitizer image (`ASan runtime does not come first…`)
**Cause:** without `--target`, Docker builds the Dockerfile's *last* stage, which used to
be `sanitize`.
**Fix:** `runtime` is now the last stage, and `deploy/docker/build.sh` always passes
`--target runtime`. `deploy/kind/up.sh` refuses images that contain `/src/tests`.

### Build fails resolving `docker.io/docker/dockerfile:1.7` (EOF)
**Cause:** a `# syntax=` directive makes every build fetch the frontend image from Docker Hub.
**Fix:** the directive was removed. The Dockerfiles use only built-in frontend features.

### Sanitizer runs: TSan `personality` failure; ASan `__cxa_throw` interceptor CHECK
**Fix:** `qualification/native_sanitizers.sh` runs the container with
`--security-opt seccomp=unconfined` (TSan) and preloads the sanitizer runtime together
with `libstdc++`.

### Docker VM disk fills up (`No space left on device` in `/tmp/ray`, Postgres lock file)
**Cause:** each image build adds about 2.3 GB of BuildKit cache. Every `kind load`
keeps the previous image on every node through an `import-<date>@sha256:…` record, and
interrupted imports leave containerd leases that block garbage collection. When the
disk is full, PostgreSQL stops accepting writes and runs fail with `StorageUnavailable
… No space left on device`.
**Fix:** `deploy/docker/prune-build-cache.sh` removes only this project's cache records.
`deploy/kind/release.sh prune-images` removes superseded RaySimBatch images by ID (the
IDs `build.sh` recorded in `var/images.log`), the import records that resolve to them,
and leases older than 10 minutes. It prints per-node counts and free space; check them,
because an earlier version removed nothing, silently. **Never** run a blanket
`crictl rmi --prune` on kind nodes (next entry).

### `check_mjbatch_clean.py`: `working tree is not pristine … !! .DS_Store`
**Cause:** macOS Finder wrote its view-settings file into `third_party/mjbatch` when the
folder was browsed. The gate reports every ignored file in the pinned checkout, because
build artifacts are ignored files too.
**Fix:** after confirming that the listed file is only `.DS_Store`, delete it
(`rm third_party/mjbatch/.DS_Store`). Investigate any other listed file as a possible
build or tooling leak.

### PVC stuck Pending; `helper-pod-create-pvc-*` in `ErrImagePull`
**Cause:** a blanket image prune removed kind's `local-path-helper`, which runs only
while a volume is being provisioned. kind nodes behind a host proxy cannot pull it back.
**Fix:** `kind load docker-image docker.io/kindest/local-path-helper:<tag>` (and
`local-path-provisioner`, `coredns`) from the host.

## KubeRay

### RayCluster rejected: invalid label value
**Cause:** the image reference (`registry:5000/x:tag`) was used as a label value.
**Fix:** `release_label()` sanitizes it. The full reference is in annotations.

### Head fails: `ray start` receives mangled `--system-config` JSON
**Cause:** KubeRay 1.7.0 splices `rayStartParams` values unquoted into `bash -c`.
**Fix:** JSON values are rendered compact and single-quoted (AM-14). A test replays the
bash splicing.

### Workers fail: "System config parameters can only be set on the head node"
**Fix:** `system-config` is rendered on the head only.

### Autoscaler sidecar crash-loops without an API token
**Cause:** the head used the restricted ServiceAccount with
`automountServiceAccountToken: false`.
**Fix:** the head uses the ServiceAccount that KubeRay creates for the in-tree
autoscaler. Workers keep the restricted, token-less one.

### `Permission denied: /var/run/secrets/raysimbatch/<secret>/dsn`
**Cause:** Secret files mounted `0400 root` are unreadable by the image's uid 1000.
**Fix:** pods run with `fsGroup: 1000`, and secrets are mounted `0440`.

### The autoscaler never scales worker nodes down
**Cause (Ray 2.58):** a node with any live actor, even one with no CPU, or with pinned
objects, is never idle. Finished run controllers, per-node asset caches and affinity
agents kept the nodes busy.
**Fix:** controllers exit after a linger once results are durable; caches release a
run's holds at cleanup and exit when empty; agents exit without live leases (AM-13). In
kind, blue scaled from 2 to 1 worker about 10 minutes after idling.

### Deleted release still acts for ~90 s; resume reports `LeaseHeld … its owner is alive`
**Cause:** `kubectl delete pod --force --grace-period=0` removes only the API object. The
kubelet still stops the containers gracefully, and the Ray head runs until SIGKILL at
`terminationGracePeriodSeconds`.
**Behavior:** fencing keeps the dying owner and the new one from both acting. A resume
waits up to `cluster.termination_grace_s + 2 × failure.lease_s` for the lease to lapse.
If you see this error after that, the owner really is alive: cancel the run there
first.

### Run controller OOM-killed; run stuck in VALIDATING or RUNNING with no controller
**Cause:** Ray's memory monitor killed controllers on a small head
(`… node running low on memory … 1.68GB / 1.76GB`).
**Behavior:** `status` reports `orphaned`. The cluster janitor (head sidecar, every
60 s) resumes orphaned runs of its own cluster, bounded by `failure.max_restarts`, then
fails them with the cause. `raysimbatch resume` does the same on demand.

Controllers are pinned (hard node affinity) to the alive non-head node with the most
memory. They land on the head only when no worker node is alive. A soft preference was
not enough: a worker raylet seconds after joining handed the lease back and the
controller went to the head. At startup the janitor waits up to `--startup-wait-s`
(default 120) for a worker
node before its first pass; its log shows `{"janitor_worker_node": false}` when none
joined in time. The head still needs memory for GCS, the dashboard, the Ray Client
server and any driver run there. On kind, the 2 GiB head reached Ray's 95% threshold
with two concurrent runs and one smoke driver.

### `GangTimeout: gang … not schedulable within …s: deferred wave`
**Cause:** the wave's placement group needs more CPUs than the cluster can provide. On
kind this happened while blue and green shared one node pool. Every model × policy
runtime needs at least one actor.
**Fix:** scale capacity or reduce the plan. Validation refuses plans larger than the
configured maximum (`CAPACITY`), but it cannot know that another release currently holds
nodes.

### Controller fails with `run lease lost` while waiting for placement
**Cause:** the gang wait (up to `gang_timeout_s`) blocked lease renewal.
**Fix:** the run lease is renewed on a keeper thread, and the gang wait checks for lease
loss between slices.

## Workloads

### A worker command raises; the run stays RUNNING forever
**Cause (fixed):** an application error inside a worker command killed the coordinator,
and the controller replaced it without bound.
**Now:** it is a worker fault (restart, then quarantine), and coordinator replacement is
bounded by `failure.max_restarts`. The run ends FAILED with the root cause, or COMPLETED
with `complete=false` under `allow_partial`.

### Mode A stalls: hundreds of thousands of `run_local` commands, no new fragments
**Cause (fixed):** when every fragment credit was in flight, slots filled their buffers
to the hard cap. After the credits returned, `run_local` checked fullness before
flushing, so it never flushed again.
**Now:** ready buffers flush before the fullness check (`run_local` and `advance`), with
regression tests.

### Worker initialization fails: `FileNotFoundError: /proc/self/task/<tid>/status`
**Cause (fixed):** a transient thread exited while affinity verification was inspecting it.

### Metrics missing for short runs
**Cause:** Ray exports custom metrics every 10 s, and a process's series disappear when
it exits. A run shorter than one interval exports nothing.
**Use:** the run manifest and the database for run-level numbers. Tests set
`metrics_report_interval_ms` to 1000.

### `ActorUnavailableError` in a coordinator after a worker is killed
**Cause (fixed):** a synchronous `ack` to a worker that died mid-envelope escaped and
killed the coordinator. Acks now tolerate a dead worker; its loss is handled when its
next command fails.
