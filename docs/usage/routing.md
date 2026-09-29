# Heterogeneous batches and explicit stepping

Modes A and B drive simulation actors themselves. A caller that owns its own control
loop, such as an RL library's environment runner or a custom evaluator, can instead
step a **heterogeneous logical batch** with `HeteroBatchRouter`. The batch is a set of
environments spread over simulation actors whose models may have different observation
and action widths and different control rates.

## Run the example

```bash
python examples/router_stepping.py
```

Output:

```
groups: [(8, 4), (8, 36)]
stale actions accepted: 0, rejected: 16
```

The first line shows the two groups, 8 cart-pole environments (width 4) and 8 quadruped
environments (width 36). The second shows that replaying old actions is rejected before
any mutation.

## Starting a fleet

`routing.fleet.start_fleet(run_id, [FleetMember(spec, mjb, count), ...])` starts
`count` simulation actors per member, initializes them, and builds routes, one per
environment slot. It returns the router. `stop_fleet(router)` kills the actors.
`FleetMember.policy` optionally loads a local policy for `run_local`.

## Asynchronous stepping (`send` / `recv`)

```python
frame = await router.reset(env_ids, episode_ids)        # one HeteroFrame, one group per schema
submission = await router.send(action_frame)             # routes rows by env id
frame, timed_out = await router.recv(min_ready=1, timeout_s=0.01)
```

- `send` rejects stale or invalid rows **before any mutation**: a wrong routing epoch or
  slot generation, a mismatched episode or control sequence, a duplicate row, a wrong
  schema or action width, or a non-finite action. Rejections come back per row with a
  `FaultCode`. At most one action is outstanding per environment and at most one
  mutating call per actor; further rows queue.
- `recv` returns whatever rows are ready, possibly fewer than `min_ready` with
  `timed_out=True`. A slow actor or group never gates the others; `tests/ray/test_router.py`
  checks this with a deliberately heavy model. Per-environment control sequences
  arrive in order.
- Frames are immutable snapshots published through the object store and read zero-copy
  on the same node.

## Barrier stepping (`barrier_step`)

`barrier_step(actions, epoch, deadline_s)` steps every requested environment and returns
a row, or an explicit timeout or fault, for **every** environment. It pins the policy
and inputs and performs no hidden autoreset. Groups with different timesteps still do
not represent equal simulated time; the barrier aligns calls, not physics clocks.

## Migration

`migrate(env_id, dest_actor, dest_slot)` moves a quiescent environment to another actor
of the same group. The environment's complete slot state (integration state including
warmstart, randomized parameter rows, kernel and task state, and slot bookkeeping) is
snapshotted, installed, then committed through
`PLANNED → DRAINING → SNAPSHOTTED → INSTALLED → COMMITTED → RETIRED`. A failure before
`COMMITTED` aborts the migration, and the old route stays authoritative. Routes carry
per-environment epochs, and actors are told their routing epoch, so an action routed
by a stale epoch is rejected.

## Curriculum and fairness

`DeficitRoundRobin(weights)` apportions episode starts to groups by weight. A group more
than one quantum ahead of its weighted share waits, and no fictitious padding is added.
`rebalance_plan(weights, group_of_actor, version)` proposes actor reassignments at
episode boundaries. Mode A uses the same mechanism for its curriculum.
