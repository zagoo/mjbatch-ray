# Mode A: rollouts for training

Mode A generates transitions for a learner. It can run several model groups with
different observation and action widths at once, under a curriculum. Results arrive as
columnar **fragments** in a durable replay sink or a learner queue, exactly once by
fragment digest.

## Run the example

```bash
eval "$(tools/local_services.sh env)"
```

```bash
python examples/mode_a_rollout.py
```

Output on the qualification workstation:

```
run 2e9bdc736bde4726bced45264e43e524: COMPLETED; gaps=0
fragments hold 20594 transitions; rows by observation width: {4: 11346, 36: 9248}
```

The two schemas are carried separately. The cart-pole observation width is 4 and the
quadruped's is 36, and nothing is padded to the larger width.

## Inference placement

| `inference.placement` | Behavior |
|---|---|
| `local` (default) | Each simulation actor runs the policy in-process (single-threaded ONNX or TorchScript) and advances up to `worker.local_control_steps` per command (`run_local`). This has the lowest latency and no inference service. |
| `central` | Workers publish observation frames. `PolicyInferenceService` batches compatible requests (same policy, schema and recurrent layout) up to `max_batch_rows`, or for `flush_ms`, runs the model on `inference.device` (`cpu`, `mps` or `cuda`), and returns action frames. Requests past their deadline are rejected (`INFERENCE_DEADLINE`), never substituted. |
| `auto` | Resolved at admission (the `resolved_placement` option). |

CUDA inference is implemented but not qualified here (no CUDA host).

## Fragments

A fragment holds up to `rollout.fragment_steps` transitions per slot, grouped into
episode segments, with these fields:

| Group | Fields |
|---|---|
| Per step | `obs`, `action`, applied action, reward and components, flags, `control_seq`, behavior log-probability, policy and curriculum versions, `dt` |
| Per segment | env and episode IDs, first sequence, initial recurrent state, terminal observation (terminated) or bootstrap observation (truncated or cut) |

Invariants that the tests check on every fragment:

- sequences are contiguous within a segment;
- a fragment never mixes behavior policies (a policy change cuts the open fragments);
- a terminated segment carries its terminal observation;
- an interrupted episode ends with an infrastructure truncation, and the next episode's
  observation is never its bootstrap.

`to_rllib_batch(fragment)` converts a fragment to an RLlib-style sample batch (`obs`,
`new_obs`, `actions`, `rewards`, `terminateds`, `truncateds`, `action_logp`, ...).

## Delivery guarantees

- **Exactly once.** The assembler deduplicates by fragment ID and digest. A repeated
  ID with a different digest stops the assembler (`DigestConflict`, which pages). With
  `rollout.durable_outbox`, the fragment blob and an outbox row are written before
  delivery to the sink, and `redeliver_pending()` re-sends undelivered rows after a
  restart.
- **Backpressure, not loss.** Worker output credits
  (`flow.fragment_slabs_per_worker`, `flow.inflight_frames_per_worker`) and assembler
  byte credits (`flow.assembler_bytes`) make producers wait. A slot whose fragment buffer
  is full stops advancing until a credit returns, and full buffers flush before the slot
  steps again. A regression test covers the livelock that the original order allowed.
- **Policy lag.** Fragments whose behavior policy is more than
  `rollout.max_policy_lag_versions` (or `max_policy_lag_s`) behind the learner's version
  (`set_learner_version`) are rejected and counted, never relabeled.
- **Worker loss.** Committed fragments stay valid. The uncommitted tail of each
  interrupted episode is recorded as a `gaps` row with conservative bounds, and the
  run's `lost_upper_bound` is their sum.

## Controlling a running rollout

```python
orch = RolloutOrchestrator(run_id, namespace="raysimbatch")
await orch.update_policy({"groups": ["quadruped"], "manifest": bundle.to_json().decode(),
                          "exe_ref": ray.put(np.frombuffer(exe, np.uint8)), "hash": bundle.policy_hash,
                          "version": 10})
await orch.update_curriculum(version=2, weights={"cartpole": 0.2, "quadruped": 0.8})
await orch.stop(drain=True)   # COMPLETED after draining; drain=False cancels (CANCELED)
```

- Policy updates take effect at command boundaries. Workers that restart later load
  the updated policy; the coordinator keeps its own copy of the executable.
- Curriculum weights control episode starts through deficit round robin.
- `target_transitions` ends the run when reached. An open-ended run (for example
  `10**9`) runs until `stop()`.

## Sinks

| Sink | Semantics |
|---|---|
| `durable_replay` (default) | The fragment index is in PostgreSQL and the bytes are in the blob store. It is idempotent and readable with `DurableReplaySink(store, run_id).read(offset, limit)`. |
| `learner_queue` | Delivered to a learner actor's queue. Idempotency depends on the learner; `rollout.require_idempotent_sink` refuses non-idempotent sinks. |
