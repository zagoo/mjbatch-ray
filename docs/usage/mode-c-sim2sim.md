# Mode C: Sim2Sim transfer (e.g. Isaac to MuJoCo)

Mode C admits a policy trained in another engine and measures how it transfers. A
**transfer bundle** (canonical MJCF, source conventions, source export, policy) must pass
twelve automated gates. The policy is then evaluated on the aligned model and on
single-axis perturbations as paired Mode B units, and the result is a scorecard.

## Run the example

```bash
eval "$(tools/local_services.sh env)"
```

```bash
python examples/mode_c_sim2sim.py
```

Output on the qualification workstation:

```
gates: {'XML_ABI_MISMATCH': 'pass', 'JOINT_MAPPING': 'pass', 'OBS_CONTRACT': 'pass', 'NORMALIZATION': 'pass', 'STATE_LEAK': 'pass', 'POLICY_EXPORT': 'pass', 'CONTROL_RATE': 'pass', 'ACTUATOR': 'pass', 'INERTIAL': 'pass', 'ASSET_FRAME': 'pass', 'CONTACT': 'pass', 'PHYSICS_CONFIG': 'pass'}
target success 1.0, source success 1.0, conservative transfer ratio 1.0
  axis contact  paired coverage 1.00
  axis filter   paired coverage 1.00
  axis inertia  paired coverage 1.00
```

> **Scope of what is verified here.** No Isaac installation or real Isaac traces were
> available. The example and the tests use a reference quadruped and a *reference source
> simulator* (`sim2sim/source.py`) that exports the `rsb-source-export/1` format with
> deliberately different conventions: quaternion order, axis flips, degrees versus
> radians, and so on. Real Isaac-trace qualification is BLOCKED; see the qualification
> report.

## The transfer bundle

| Part | Content |
|---|---|
| Canonical MJCF | the target model (named bodies and joints are required; the exporter rejects unnamed ones) |
| Source export (`rsb-source-export/1`) | resolved property tables (masses, inertias, joint limits and axes, actuators), observation fixtures, actuator traces, contact fixture curves, initial-state replay, policy input/output goldens |
| Adapter contract | observation terms (sources, selectors, offsets, signs, frames, history order, normalization), actuator mapping and conventions |
| Policy | ONNX or TorchScript bundle with explicit tensor contracts |

## The gates (design §6.7)

Each gate is blocking. Tolerances come from the versioned `sim2sim.parity`
configuration; loosening them inside a run is a `forbidden` change.

| Code | Checks | Default criterion |
|---|---|---|
| `XML_ABI_MISMATCH` | compiles in the pinned MuJoCo; no unsupported features; metadata | exact native version, no compiler error |
| `JOINT_MAPPING` | bijective name/sign/address mapping; single-joint positive perturbations | every actuated DOF mapped once; response signs derived from resolved properties agree |
| `OBS_CONTRACT` | the export's source state fixtures (1,000 in the reference exporter), compared per term before the policy; sequences check history order | float32 tolerance 1e-5; frames and history exact |
| `NORMALIZATION` | graph inspection plus raw and normalized goldens | exactly one normalization; finite statistics |
| `STATE_LEAK` | two interleaved environments, one reset, against isolated references (history and recurrent state) | no cross-environment influence |
| `POLICY_EXPORT` | source policy goldens (1,000 in the reference exporter) against the exported runtime; every certified batch size | max action error ≤1e-4; static batch shapes valid |
| `CONTROL_RATE` | 100 policy updates, all substeps | exact decimation, no phase drift |
| `ACTUATOR` | swept targets and velocities, impulse, saturation, delay | effort error ≤ max(0.01 Nm, 1% of range); filter impulse ≤1e-6 |
| `INERTIAL` | resolved body tables in a common frame | mass ≤1%, COM ≤1 mm, inertia eigenvalues ≤2% |
| `ASSET_FRAME` | known configurations; link and site poses | translation ≤1 mm, rotation ≤0.1° |
| `CONTACT` | drop, sliding, resting and friction-ramp fixtures | slide threshold ≤5%; settling and penetration ≤10% normalized error |
| `PHYSICS_CONFIG` | dt, gravity, limits, armature, damping, solver budget; initialization replay | exact declared configuration; state ≤1e-6 |

`tests/sim2sim/test_gates.py` injects a golden mismatch for each class (reversed joint
sign, swapped history order, degree/radian mix-ups, wrong solver iterations, mass and
inertia errors, and so on) and requires the specific gate to fail.

Gate reports are cached per (store, bundle digest, parity profile) as durable commits,
so validating the same bundle again does not re-run the gates.

## Contact alignment

Contact parameters (for example PhysX offsets against MuJoCo `solref`/`solimp`) have no
one-to-one conversion. `AssetAlignmentHarness.fit_contact` fits target values against
the declared fixture curves with a bounded deterministic grid search, freezes them, and
validates them on held-out fixtures. A failed fit is reported, never forced.

## Evaluation and scorecard

`Sim2SimOrchestrator.evaluate(bundle, policy, seeds, axes=(...), source_success=...)`:

1. validates the bundle (cached);
2. registers the bundle manifest and the source export as artifacts;
3. submits a `sim2sim` run with the aligned model plus one model group per diagnostic
   axis (for example `inertia`, `contact`, `filter`), with common seed IDs; and
4. returns a scorecard (`build_scorecard`, rendered by `scorecard_markdown`).

The scorecard contains:

| Section | Content |
|---|---|
| Gates | per-gate pass/fail and metrics |
| Target | success rate and CI, coverage |
| Source | the source engine's outcomes, as provided |
| Transfer ratio | target/source, including a conservative variant |
| Diagnostics | per-axis paired effects versus the aligned baseline, with paired coverage |

`evaluate(..., exploratory=True)` runs a bundle whose gates failed. The run is marked
`exploratory=true` (the design's "unvalidated" run class) and is excluded from release
comparisons. Without it, failed gates refuse admission.
