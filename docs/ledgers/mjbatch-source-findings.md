# MJBatch source findings (revision `088e2d4cd0d5878ea184010446c058710f23a671`, v0.1.4)

First recorded at `2f169146…` (v0.1.1); reviewed for the 0.1.4 upgrade on 2026-10-09.
Entries that changed in that upgrade say what changed.

Each finding cites the file/symbol and, where behaviour matters, the executable test in
`tests/upstream/test_mjbatch_contract.py` (UC-*) that pins it.

## Construction, ownership, memory

- **M1 Construction** (`batch.h` `Batch::Batch`): copies the template with
  `mj_copyModel`; rejects `mjENBL_SLEEP`; builds field tables from `MJDATA_POINTERS`,
  `MJMODEL_POINTERS`, `MJOPTION_FIELDS`; creates `min(num_threads, num_sims)` pool
  threads (`num_threads<=0` → `hardware_concurrency`); one `mjData` per pool thread;
  per-sim `mjSTATE_INTEGRATION` vector and `mjNWARNING` warning stats. UC
  `test_threads_are_clamped_to_num_sims`, `test_sleep_is_rejected`.
- **M2 Model copies**: a per-thread `mjModel` copy is created on the first `expand`
  (`Expand`), so a randomized actor holds template + T copies. No model sharing across
  processes exists or is possible via this API.
- **M3 Destruction** (`~Batch`): stops the pool, deletes data/models. Bound arrays hold
  an owning reference (`nb::find(this)`), so the Batch lives until the last view dies.
  UC `test_bound_views_keep_the_batch_alive`.

## Execution and synchronization

- **M4 Serialization**: every public call takes `mu_`; `Run` releases the GIL first
  (`nb::gil_scoped_release`), so calls from several Python threads serialize natively.
  `bind`/`expand` take `mu_` *without* releasing the GIL: calling them concurrently
  with a running step blocks the calling Python thread while holding the GIL.
  RaySimBatch binds/expands only at construction, on the mutation thread.
- **M5 Pool** (`threadpool.h`): blocking parallel-for over `max(1, min(threads, sims))`
  workers. Since 0.1.4 worker 0 is the calling thread (`Run` does its share with
  `Work(0)`) and only the other workers are `std::thread`s; before, every worker was a
  pool thread and the caller waited. Idle workers spin 50 µs before sleeping. Item `j`
  (the j-th selected id, not the sim id) starts in slice `j*T/n`; finished workers steal
  from other slices (atomic `fetch_add`); `Run` returns only when all items finish. No
  per-item completion is published. A single-worker pool runs on the caller.
  UC `test_step_releases_the_gil`.
- **M6 Per-call sim execution** (`RunSim`): apply expanded rows to the thread model →
  `mj_setState(INTEGRATION)` (or `mj_resetData[Keyframe]` for reset) → copy changed
  input-field elements (element-wise `memcmp` against the mirror) → op → `mj_getState` →
  copy every bound field out and refresh mirrors.
- **M7 Ids**: int32/int64 sorted unique in-range, or a bool mask of length N; empty
  selection is a no-op. The nanobind signature (`nb::ndarray<nb::ndim<1>, nb::c_contig>`,
  no `nb::ro`) rejects read-only arrays, e.g. ones deserialized from the Ray object store;
  the adapter copies those. UC `test_ids_contract`.

## Semantics RaySimBatch depends on

- **M8 Errors** (`Guarded`, `LogTrap`): a thread-local `setjmp` + `mju` log handler
  trap turns `mjLOG_ERROR` into a `longjmp`. Only one failing sim's message is kept
  (`error_` set once under `changed_mu_`, "sim i: msg"): the first to fail *in time*,
  which with several pool threads is nondeterministic and not the lowest index; other
  failing sims are silent. A failed sim's
  state, warning row and bound views are not written back (pre-call state kept), and
  its pending writes stay pending. The worker's `mjData` is `mj_resetData`'d after the
  jump (scratch reset already upstream). UC
  `test_failed_sims_keep_pre_call_state_and_are_identifiable`,
  `test_error_message_names_only_the_first_failure`,
  `test_with_several_threads_the_named_sim_is_first_in_time_not_lowest_index`.
- **M9 Pending writes**: input fields (`IsInput`: time, qpos, qvel, act, history,
  qacc_warmstart, ctrl, qfrc_applied, xfrc_applied, eq_active, mocap_pos, mocap_quat,
  userdata, plugin_state, warning) are mirrored; only elements differing from the
  mirror are copied in. Writing back the last copied-out value cancels a write. Since
  0.1.4 (`f188344`) the fields in `mjSTATE_INTEGRATION` (time, qpos, qvel, act, ctrl, …)
  are views into the state rows, `nstate` elements apart, so a state-row write is a field
  write; RaySimBatch's native kernels accept such row-strided views. UC
  `test_failed_sim_keeps_pending_write_and_restoring_the_old_value_cancels_it`.
- **M10 Reset** (`Op::Reset`): `mj_resetData`/`mj_resetDataKeyframe`, then every pending
  element write is applied on top, then `mj_forward`. Since 0.1.4 that includes a
  state-row write, element by element (M9); before, a pending state-row write was
  discarded. RaySimBatch never leaves a write pending into a reset (AM-4, AM-6). UC
  `test_pending_field_and_state_row_writes_survive_reset_element_by_element`.
- **M11 forward** is a pure query: it restores the integration state, runs
  `mj_forward`, and writes back an identical state (MuJoCo copies
  `qacc`→`qacc_warmstart` only in integration). UC
  `test_forward_leaves_integration_state_and_trajectory_bitwise_unchanged`.
- **M12 Integration state** includes `mjSTATE_WARMSTART`; restoring a row replays
  bit-for-bit, and zeroing only the warmstart segment changes a contact trajectory.
  Component order is `mjtState` bit order. UC
  `test_state_rows_include_warmstart_and_restore_replays_bitwise`.
- **M13 Substeps**: k calls of `step(nstep=1)` equal one `step(nstep=k)` bitwise. UC
  `test_single_substep_calls_equal_one_multi_substep_call`.
- **M14 set_const**: recomputes selected sims; any not-yet-expanded field that differs
  from the template is expanded and then *all* sims are recomputed (closure growth).
  Afterwards subsets stay subsets. mjModel scalars written by `mj_setConst`
  (`ngravcomp`, flags, `stat`) are kept per sim. UC `test_set_const_subset_semantics`.
- **M15 Derived-field freshness**: after `step`, derived fields lag one substep unless
  the Batch was built with `forward=True`; `reset` and `forward` leave them current.
- **M16 Asset fields** (`mesh_`, `hfield_`, `tex_`, `skin_`, `bvh_`, `oct_`) cannot be
  expanded. UC `test_asset_fields_cannot_be_expanded`.
- **M17 history**: optional caller-allocated `(sims, nstep, nstate)` float64 array; rows
  of a failing sim are undefined. Row k is `mj_getState(mjSTATE_INTEGRATION)` after
  substep k (`batch.h` `RunSim`). Used by fused intervals (AM-2). UC
  `test_history_rows_are_the_state_after_each_substep`. `warning` is an input field
  (`IsInput`): each call starts from the sim's saved counters and applies a written row.
  UC `test_written_warning_counters_are_applied_at_the_next_call`.

## Build, ABI, platform

- **M18 ABI check**: the module raises at import when `mj_version() != mjVERSION_HEADER`
  (headers come from the `mujoco==3.14.0` wheel used at build time).
- **M19 Build**: scikit-build-core + nanobind + CMake; links `libmujoco` from the
  installed wheel with rpath `../mujoco`; macOS rewrites the install name. Default
  `build-dir = build/{wheel_tag}` is *inside* the source tree; RaySimBatch overrides it
  per package (`pyproject.toml` `config-settings-package`).
- **M20 Unsupported**: sleep (rejected); Windows (upstream builds there since 0.1.3, but
  RaySimBatch does not support it); plugins are not specially
  handled (plugin state is part of `mjSTATE_INTEGRATION` only via `plugin_state`), and
  installing another MuJoCo log handler disables the error trap.

## Extension mechanisms

MJBatch exposes no hooks: no per-substep callback, no per-sim status API, no mirror
reset, no affinity control, no Jacobian/diagnostic queries. Its public surface is
`Batch(model, num_sims, num_threads, forward)`, `bind`, `expand`, `step(ids, nstep,
history)`, `forward(ids)`, `reset(ids, keyframe)`, `set_const(ids)`, `nstate`,
`num_sims`, `num_threads`, `model` and the named-view helpers. RaySimBatch composes on
that surface only (see `src/raysimbatch/compat/mjbatch_adapter.py`).

## longjmp audit (design §13.1)

`Guarded` → `RunSim` frames alive across the jump hold only trivially destructible
objects at the points where MuJoCo can raise (range-for iterators over
`std::vector<std::unique_ptr<Slot>>`, raw pointers, ints); the only RAII lock
(`changed_mu_` in `RunSim`'s `SetConst` branch) is taken *after* `mj_setConst` returns.
`LogTrap` assigns `tls_error` (std::string) before jumping, outside any destructor
sequence. MuJoCo's own frames are C. The jump target resets the worker `mjData`
(stack/arena). Conclusion: no destructor or lock is skipped at this revision. The audit
is repeated for every upgrade candidate (`tools/mjbatch_upgrade.py` prints the diff of
`batch.h`) and exercised under sanitizers (`qualification/native_sanitizers.sh`).
