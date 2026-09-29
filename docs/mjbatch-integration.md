# MJBatch integration boundary

RaySimBatch uses [MJBatch](https://github.com/kevinzakka/mjbatch) as an unmodified
upstream dependency. This document states the boundary, how it is enforced, and which
upstream semantics RaySimBatch depends on. The upgrade and rollback procedure is in
[operations/mjbatch-upgrade.md](operations/mjbatch-upgrade.md).

## The boundary

| Rule | Where it holds |
|---|---|
| Upstream source is never edited, patched, vendored-and-modified or monkey-patched at runtime | `tools/check_mjbatch_clean.py` (pristine checkout, no assignments or `setattr` on mjbatch objects, no copies of upstream sources elsewhere) |
| Exactly one module imports MJBatch | `src/raysimbatch/compat/mjbatch_adapter.py` (enforced by the checker). `tests/upstream` imports it directly to pin upstream behavior. |
| Only the public API is used: `Batch(...)`, `bind`, `expand`, `step`, `forward`, `reset`, `set_const`, `nstate`, `num_threads` | enforced by the checker (no private attribute reads on mjbatch objects) |
| The pin is one revision everywhere | the superproject gitlink = the checkout HEAD = `QUALIFIED[version]["revision"]` = the installed distribution's `direct_url` source (checker) |
| Build artifacts never land inside upstream's tree | `[tool.uv] config-settings-package = { mjbatch = { "build-dir" = "../../build/mjbatch/{wheel_tag}" } }`. The checker fails on ignored or untracked files in the checkout. |
| Behavior is re-verified on the running binary | `check_environment()` runs the behavior probes in every process. An unqualified version needs `RSB_MJBATCH_CANDIDATE=1`, which production validation rejects (`CANDIDATE_BINARY`). |

Run the checks:

```bash
python tools/check_mjbatch_clean.py
```

```bash
python -m raysimbatch.cli compat
```

The first prints `OK mjbatch checkout … pristine at <sha>; boundary respected`. The
second prints the version pair, binary SHA-256, ABI fingerprint, and every probe result.

## Pinned revision

| Item | Value |
|---|---|
| Repository | `https://github.com/kevinzakka/mjbatch.git`, submodule `third_party/mjbatch` |
| Revision | `2f169146b643adda65e195d8ad13d997abdd2a2e` (tag `v0.1.1`) |
| MuJoCo | `3.13.0` (MJBatch raises at import if `mj_version()` differs from the header it was built against, M18) |
| Build | scikit-build-core + nanobind, from the pinned source by `uv sync` (build directory outside the checkout) |

## Semantics RaySimBatch depends on

The full evidence list, with file and symbol citations, is in
[ledgers/mjbatch-source-findings.md](ledgers/mjbatch-source-findings.md) (M1–M20). The
ones that shape RaySimBatch's design:

| Upstream fact | Consequence in RaySimBatch | Guard |
|---|---|---|
| A failed call raises one error naming the first failing sim in time; failed sims keep pre-call state and pending writes (M8) | per-slot outcomes via the sentinel (AM-1); `ctrl` neutralization (AM-4) | probes `first_error_attributed`, `failed_slots_exact`, `failed_keep_state`, `others_advanced`, `neutralize_cancels_pending` |
| `step(nstep=1)` k times equals `step(nstep=k)` bitwise (M13) | the per-substep composition keeps parity, and fused intervals equal per-substep ones (AM-2) | `tests/core` bitwise parity; probe `fused_substeps_equal` |
| `step(ids, nstep, history=H)` fills `H[j, k]` with selected sim j's state row after substep k; a failed sim's rows are undefined (M17) | fused intervals finish the controller's per-substep bookkeeping and find terminations from the rows (AM-2) | probe `history_rows_are_states`; `tests/upstream` `test_history_rows_are_the_state_after_each_substep` |
| `warning` is an input field carried per sim across calls, like the state rows (`IsInput` in `batch.h`) | a replayed slot's warning counters are rewound by writing its rows | `tests/upstream` `test_written_warning_counters_are_applied_at_the_next_call` |
| `reset` applies pending field writes; integration state includes warmstart (M10, M12) | exact reset through state rows (AM-6) | probe `reset_applies_pending_field_writes`, probe `state_layout` |
| `forward` is a pure query (M11) | observations need no step | probe `forward_pure` |
| `set_const` may expand derived fields for all sims (M14) | closure guard, and re-run for isolation (AM-3) | `tests/upstream` closure tests |
| `sleep` is rejected (M20) | scalar lane (AM-9) | `tests/core/test_scalar_lane.py` |
| `ids` must be writable C-contiguous arrays (M7) | `_as_ids` copies read-only arrays (for example, arrays from the Ray object store) | `tests/ray` |

## What the design asked of MJBatch and how it is met

The design planned a small maintained MJBatch extension (§1). The assignment forbids
one. [design-amendments.md](design-amendments.md) AM-0 through AM-7 record the
substitute for each piece, and AM-2 records the measured cost: very light models lose
throughput to per-substep calls.

## Stepping: per substep or fused

By default a policy interval is `decimation` calls of `step(ids, nstep=1)`, with the
controller, health and termination kernels between calls (AM-2). One call per interval,
`step(ids, nstep=decimation, history=H)`, is used instead when it gives the same results:

- `models.<group>.health.check_each_substep` is `false`: health is checked once per
  interval, on each slot's last substep (the one it terminated on, or the final one),
  before its termination counts. Transient excursions inside an interval are not
  checked, and `qacc` and warning counters are not recorded per substep;
- the adapter's actuator is `torque` or `position`, with no physics-clock delay, so ctrl
  is constant over the interval (`pd` and `dc_motor` read the state every substep);
- `worker.fuse_substeps` is `true` (the default), and the Batch lane is used (the
  scalar lane keeps its own stepping).

The fused path then completes each slot's controller bookkeeping (energy, saturation
and evaluation counters) from the history rows in per-substep order, and scans them with
the task's termination predicate. A slot that terminated before the last substep is
rewound to its pre-interval state and stepped again to its terminal substep in one call.
A slot that failed natively is replayed substep by substep to find its failing substep.
Results are identical to per-substep stepping with the same configuration: every result
field and every per-slot state array are compared bitwise in `tests/core/test_fused.py`.
`MJBatchCore.stepping` reports which path a core uses. The measured effect is in
[qualification/report.md](qualification/report.md) §5.

