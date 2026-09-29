# Extension points

Everything a worker executes at physics rate is either MuJoCo, reached through MJBatch,
or compiled C++ in `raysimbatch._native`. Extensions that change physics-rate behavior
are therefore native code with a declarative binding. Free-form executable strings in a
configuration are never accepted (design B.2).

## Models

A `ModelGroup` needs the following. Register each artifact with
`raysimbatch artifacts put`, or with `testing.runs.register` in Python.

- an MJCF plus an asset archive, or a precompiled MJB (compiled once at admission and
  identified by content hash);
- physics settings: `timestep_ns`, `decimation`, solver, integrator, cone, iterations,
  tolerance and initial keyframe;
- an **adapter contract** (next section), a **task kernel** document, randomization
  rules, and a health configuration.

Features MJBatch rejects (`sleep`) or cannot represent (stateful plugins) run on the
scalar lane (`worker.native_backend: scalar`, one environment per actor).

## Adapter contracts (observations and actions)

`AdapterContract` (`config/models.py`) is compiled by `adapter.compile_adapter` into the
native observation and controller tables.

- **Observation terms.** Each term has a source, and selectors that resolve by name
  against `ModelMetadata`: joints, bodies and sites. The sources are `joint_pos`,
  `joint_vel`, `base_lin_vel`, `base_ang_vel`, `gravity`, `command`, `previous_action`,
  `phase`, `rot6d`, `sensor`, `custom`, `applied_action` and `base_quat`. A term can
  also set a frame, offsets, signs, `history` (with `history_order`), scale, clip and
  noise.
- **Order of operations.** Normalization (`normalization_clip`) and the per-term
  operations follow `term_op_order`, as fixed by design §6.7 steps 1–8.
- **Actuators.** `ActuatorContract` gives the actuator names in policy-action order,
  `kind` (`torque`, `position`, `pd` or `dc_motor`), gains and limits, an optional
  torque-speed envelope, an EMA filter, and policy and physics delays.

A new observation **source** or controller **kind** is a native change:

1. add it to `native/src/kernels.h` (`Observation` or `Controller`);
2. add its enum value to the `Literal` in `config/models.py` and its resolution in
   `adapter.py`;
3. add a parity test against a numpy reference (`sim2sim/reference.py` shows the pattern)
   and a golden case in `tests/sim2sim`.

## Task kernels

A task kernel document is `{"name": ..., "version": ..., "spec": {...}}`. `tasks.bind_task`
resolves named parameters (joints, bodies, thresholds) into the native kernel's ordered
`params`/`iparams`. The kernels are `balance`, `velocity_tracking` and `reach`. Each
reports physics-rate termination, a reward with named components, and task metrics
(fell, tracking RMSE, ...).

To add a task:

1. implement it in `TaskKernel` (`native/src/kernels.h`) with its metrics;
2. register its parameter schema in `tasks.py` (`_native.task_spec`);
3. add a fixture in `testing/fixtures.py` and tests in `tests/core`.

## Policies

`policy/bundle.py` `make_bundle(name, version, runtime, executable, obs_dim, action_dim,
...)` describes an ONNX or TorchScript executable. The description includes explicit
tensor names and shapes, a deterministic flag, action distribution parameters, recurrent
state shapes, and certified batch sizes. `policy/export.py` shows how to export MLP,
recurrent and linear policies with batch-dynamic tensors. Bundles are verified by digest
at every load, and a CUDA request without a CUDA provider is an error.

## Replay sinks (Mode A)

Implement the `ReplaySink` protocol (`experience/sinks.py`):

```python
async def capabilities(self) -> SinkCapabilities   # idempotent_append, ordered, ...
async def append(self, fragment: Fragment, idempotency_key: str) -> SinkAck
async def flush(self, run_id: str) -> SinkOffset
```

`append` must be idempotent per `idempotency_key` (the fragment ID) to keep exactly-once
delivery. Otherwise set `rollout.require_idempotent_sink: false` and accept at-least-once.
Raising `DigestConflict` for a repeated key with a different digest stops the assembler.

## Source engines (Mode C)

A source engine is supported by producing the `rsb-source-export/1` format
(`sim2sim/source.py`): resolved property tables, observation fixtures, actuator traces,
contact fixture curves, initial-state replay and policy goldens, plus `SourceConventions`
(quaternion order, axis mapping, units). The reference exporter
(`export_reference`) and `quadruped_conventions()` are the worked example. No Isaac
exporter is included, because no Isaac installation was available.

## Configuration

`RunConfig` is closed: unknown keys are errors. A new option needs:

- a field with a type, a default and constraints in `config/models.py`;
- cross-field rules in `config/system.py` (validate), plus the field's change
  classification (hot, boundary, new run or forbidden) in `_classify`;
- a regenerated reference: `python tools/gen_config_reference.py` (CI checks it with
  `--check`).
