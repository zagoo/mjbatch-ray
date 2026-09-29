# KubeRay deployment

RaySimBatch runs on a KubeRay `RayCluster` (operator 1.7.0, Ray 2.58.0). The manifests
are rendered from two inputs: the run configuration (cluster sizing, worker shape,
failure and telemetry settings) and a **platform inventory** (images, pools, secrets,
service endpoints). Nothing is installed at runtime; images come from one release lock.

## Render

```bash
raysimbatch render --config cluster.yaml --inventory inventory.yaml --tenant prod --name rsb-blue --production --out rsb-blue.yaml
```

This produces a namespace (`<cluster.namespace>-<tenant>`), a restricted worker
ServiceAccount (no API token), a ResourceQuota and LimitRange, a NetworkPolicy (ingress
from the namespace, plus metrics from `monitoring`), a ConfigMap with the run
configuration, and the `RayCluster`. The RayCluster:

| Part | Rendered as |
|---|---|
| Head | `num-cpus: 0` (no simulation on the head). Dashboard and metrics ports. `system-config` with the spill path (head only; Ray 2.58 rejects it on workers). The in-tree autoscaler uses the ServiceAccount that KubeRay manages for it. |
| Janitor sidecar (head pod) | `raysimbatch janitor --loop 60`: reconciles controllers, placement groups, cache holds and orphaned runs against database leases every minute (design §7). Before its first pass it waits up to `--startup-wait-s` (120) for a worker node, so resumed controllers are not placed on the head. |
| `sim-general` workers | Guaranteed pods (requests = limits), `num-cpus = cluster.ray_cpus`, custom resource `sim_general: simulation_cpus / worker.num_cpus`, object store, `/dev/shm`, spill volume, preStop `raysimbatch drain-node`, restricted ServiceAccount without a token |
| `sim-contact` workers | the same shape, starting at 0 replicas, for the heavy-contact pool |
| `gpu` workers | only with a GPU image (`--no-gpu` omits them); not qualified here |
| All containers | non-root uid 1000 with `fsGroup` 1000 (secrets are readable through the group), `RuntimeDefault` seccomp, all capabilities dropped except `SYS_NICE`, thread caps of 1, and `RSB_CLUSTER_NAME` |

`rayStartParams` values that contain JSON are compact and single-quoted, because KubeRay
1.7.0 splices them unquoted into `bash -c "ray start …"` (AM-14).
`tests/deployment/test_render.py` validates every rendered RayCluster against the pinned
KubeRay 1.7.0 CRD schema, reports fields the API server would silently prune, and
replays the bash splicing.

### Platform inventory

```yaml
name: prod-us-east
head_image: registry.example/raysimbatch-cpu@sha256:<digest>
cpu_image: registry.example/raysimbatch-cpu@sha256:<digest>
image_pull_policy: IfNotPresent
pools:
  head: {node_selector: {raysimbatch.io/pool: head}}
  sim_general: {node_selector: {raysimbatch.io/pool: sim}, tolerations: [...]}
database_secret: "k8s:rsb-db/dsn"   # Secret rsb-db, key dsn (mounted read-only)
s3_secret: "k8s:rsb-s3"             # AWS_* keys as environment
s3_endpoint: "https://s3.example"   # omit for AWS
spill_volume: {hostPath: {path: /var/lib/raysimbatch-spill, type: DirectoryOrCreate}}
```

`--production` refuses images without digests, pools without node selectors, and a
missing database secret. Production validation also refuses an unqualified MJBatch
candidate (`CANDIDATE_BINARY`) and `worker.affinity: none`.

## Node requirements

- **Kubelet static CPU Manager** (`cpuManagerPolicy: static`) with reserved system CPUs,
  so Guaranteed worker pods get exclusive cpusets. The node affinity agent leases
  disjoint core sets from the pod's actual cpuset.
- **SMT off, or explicitly qualified** (`worker.smt` needs a qualification artifact).
- **Memory:** the worker pod's memory must hold its actors' `worker.memory_gib` plus the
  object store (validated as `MEMORY_BUDGET`), and `/dev/shm` must hold the object store
  (`SHM_LT_STORE`).
- **Spill:** a local NVMe hostPath or ephemeral volume. Spill files are disposable.

The reference sizing (design §9.1) is a 128-core, 512 GiB node running 28 four-core
simulation actors. That shape was **not available** for qualification; see the
qualification report.

## Durable services

- **PostgreSQL:** version 17, database in **UTF8**. The DSN comes from the database
  secret. `raysimbatch db migrate --config <cfg>` applies the additive migrations;
  controllers also migrate on start.
- **S3-compatible blob store:** the artifact root is
  `s3://bucket/prefix[?endpoint=<url-encoded>]`. Credentials come from the S3 secret.
- **Both must persist.** In the kind environment, an OOM kill restarted an ephemeral
  PostgreSQL and lost the run state; the manifests there now use PVCs and Guaranteed
  QoS. Production uses managed or replicated services with backups (§11.5 restore drills).

## Release: blue/green (design §11.5)

A release is a whole RayCluster (`rsb-<color>`) with a pinned image. Two releases run
side by side. New admissions go to the *active* one, and runs in progress finish where
they started.

`deploy/release.sh --context <kube-context> --namespace <ns> <command>` runs the release
commands. Both flags are required, so a command never acts on whichever kubectl context
happens to be current. On kind, `deploy/kind/release.sh <command>` passes
`--context kind-rsb-qual --namespace raysimbatch-qual`.

**Routing.** `promote` and `rollback` point the Service `raysimbatch-active` (Ray Client
port 10001, dashboard and Jobs API 8265) at the chosen color's head, and record
`active`/`previous` in the ConfigMap `raysimbatch-active`. Clients submit through
`ray://raysimbatch-active:10001`, so they always reach the active release. The rendered
NetworkPolicy admits only pods of the namespace, so clients run there (e.g. as a Job).

1. Deploy the new color from its image: `deploy/kind/up.sh green <tag>` (kind), or render
   and apply.
2. Smoke it in-cluster: `release.sh smoke green both`. This runs a Mode B evaluation
   (must be COMPLETED, complete, coverage 1.0, LQR success 1.0 and zero-policy success
   0.0) and a Mode A rollout (5,000 transitions, no gaps) through the normal driver.
3. Promote it: `release.sh promote green`. It refuses a RayCluster that is not `ready`.
   The pointer records the previous color, and re-promotion is idempotent.
4. **Roll back:** `release.sh rollback` routes admissions to the previous color (refused
   if that cluster is not `ready`, e.g. already retired). Durable work of a failed
   release is resumed on the rollback cluster from the original seeds:
   - automatically, by that cluster's janitor, for runs it owned;
   - explicitly, with `raysimbatch resume <run_id> --config <cfg>`, for runs of the failed
     cluster.
5. Retire the old color after its runs drain: `release.sh retire blue`. The active color
   is never retired.

**Timing to know about.** Deleting a release's pods, even with `--force
--grace-period=0`, only removes the API objects. The kubelet still stops the
containers gracefully, and the Ray head keeps running until SIGKILL at
`terminationGracePeriodSeconds` (observed: ~91 s). A resume waits up to
`cluster.termination_grace_s + 2 × failure.lease_s` for the dying owner's run lease to
lapse, and fencing keeps both owners from acting at once. If the owner is still
renewing after that, the resume is refused explicitly.

`deploy/kind/rollback_exercise.sh` automates the rollback qualification:

1. promote green and start a long evaluation on it;
2. force-delete every green pod mid-run;
3. roll back to blue;
4. resume the run on blue; and
5. compare every episode with a fresh reference run of the same stored configuration.

The results are in the [qualification report](../qualification/report.md#release-and-rollback).

Upgrades never mix Ray, MuJoCo or MJBatch versions inside one run. Binary snapshots cross
versions only after explicit certification; otherwise work replays from its seeds.
Database migrations are additive for one release window.

## Local qualification environment (kind)

```bash
source tools/env.sh
```

```bash
deploy/docker/build.sh blue
```

```bash
deploy/kind/up.sh blue blue
```

```bash
deploy/kind/release.sh smoke blue both
```

`deploy/kind/cluster.yaml` defines three nodes: a control plane, a `head` pool node and a
`sim` pool node. The kubelet runs the static CPU Manager. `up.sh`:

1. creates the cluster and installs the KubeRay operator (Helm chart 1.7.0);
2. loads the images; third-party images are pinned by digest and loaded under local tags
   (pods never pull);
3. generates the database and S3 credentials into Secrets (values are never printed);
4. applies PostgreSQL and MinIO on PVCs; and
5. renders and applies the color, then waits for the head pod owned by the *current*
   RayCluster object.

`release.sh prune-images` removes superseded RaySimBatch images from the nodes (never a
blanket prune).

The environment is sized for an 8 GiB Docker VM, and the qualification config shrinks
everything (one-CPU actors, 2 GiB head, 2.4 GiB workers). It proves wiring, fencing,
recovery and release mechanics. It proves **nothing about production capacity**.
