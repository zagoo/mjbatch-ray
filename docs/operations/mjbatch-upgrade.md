# MJBatch upgrade and rollback

The production pin (submodule gitlink, `QUALIFIED` table, `pyproject.toml`/`uv.lock`)
changes only after a candidate passes every gate. The candidate is built and tested in
isolation: nothing in `third_party/mjbatch`, the project `.venv`, or any other checkout
of MJBatch is touched until `promote`. All steps use `tools/mjbatch_upgrade.py`.

## 1. Inspect the current pin

```bash
python tools/mjbatch_upgrade.py status
```

This prints the committed pin, the checkout HEAD, the `QUALIFIED` table, the installed
mjbatch and mujoco versions, and any prepared candidates.

## 2. Prepare a candidate

```bash
python tools/mjbatch_upgrade.py prepare origin/<branch-or-tag> [--mujoco X.Y.Z] [--no-fetch]
```

`prepare`:

- fetches the ref into the pinned checkout's repository (refs only; the working tree and
  the pin do not move), and falls back to refs already present if the fetch fails;
- creates a **detached git worktree** of the candidate under `var/upgrade/<sha12>/mjbatch`;
- creates a separate virtualenv `var/upgrade/<sha12>/venv`, installs the release lock
  minus mjbatch and mujoco, then MuJoCo, then the candidate MJBatch, then RaySimBatch;
- builds everything outside the candidate tree (`-C build-dir=var/upgrade/<sha12>/build/...`),
  with `--no-config` so the project's own build settings don't merge in, and with
  `PYTHONDONTWRITEBYTECODE=1`;
- records `candidate.json` (sha, declared version, MuJoCo version, paths).

## 3. Qualify the candidate

```bash
python tools/mjbatch_upgrade.py qualify <sha12>
```

It runs with `RSB_MJBATCH_CANDIDATE=1`, which admits an unqualified version only if every
behavior probe passes. The gates are recorded in `var/upgrade/<sha12>/qualification.json`:

| Gate | What must hold |
|---|---|
| `behaviour_probes` | every runtime probe of the upstream semantics RaySimBatch relies on (first-error attribution, exact failed slots, failed sims keep state, others advance, pending-write neutralization, pure forward, reset applies pending writes, state layout) |
| `upstream_contract_and_parity_tests` | `tests/upstream` (the MJBatch contract) and `tests/core` (bitwise parity with scalar MuJoCo, resets, snapshots, kernels) against the candidate build |
| `performance_regression` | wrapper throughput of the candidate within 5% of the pinned build (design §11.5). Both builds are measured **in the same session, interleaved A-B-A-B**, with the same benchmark code. |
| `candidate_tree_pristine` | the candidate worktree has no modified, untracked or ignored files after the build |

Two things were learned while building this workflow:

- A stored baseline file is not a valid comparison. Host load (another workload on the
  machine) and a change in the benchmark's methodology both moved the numbers by 5–12%
  with no candidate change at all. The gate therefore always measures A/B in one session.
- Candidate `121e033` (`origin/mujoco-versions`) adds a `mujoco_metadata.py` build hook
  that left `__pycache__` inside the upstream tree. The build now runs without bytecode,
  and the pristine gate caught the problem.

## 4. Promote (only with a passing qualification)

```bash
python tools/mjbatch_upgrade.py promote <sha12>
```

`promote` refuses without a passing `qualification.json`. It then:

1. checks out the candidate revision in `third_party/mjbatch` (the gitlink changes);
2. adds the candidate's version to `QUALIFIED` in `compat/mjbatch_adapter.py`;
3. updates the mjbatch and mujoco pins in `pyproject.toml`;
4. runs `uv lock` and `uv sync`; and
5. runs `tools/check_mjbatch_clean.py --no-linux-gate` (the Linux gate comes next).

The changes are left **uncommitted** for review. If the MuJoCo version changed,
regenerate its type stubs (`uv run --with pybind11-stubgen==3.0.0 python
tools/gen_mujoco_stubs.py`; `tests/tooling/test_mujoco_stubs.py` fails until you do) and
run `python tools/pyright_ratchet.py`.

## 4a. Linux gate (required before commit)

```bash
python tools/mjbatch_upgrade.py linux <sha12>
```

The gates of step 3 run on macOS, which has no CPU affinity. MJBatch 0.1.4 passed all of
them while its thread-pool change (the calling thread became pool worker 0) broke CPU
pinning on every Linux worker; only the Linux suite showed it. `linux`:

1. requires the promoted pin (the checkout at `<sha12>`, its version in `QUALIFIED`);
2. builds the `test` image from the working tree and uses it by image ID, so a failed
   build cannot leave an old image to be tested by mistake;
3. checks that the image has the promoted mjbatch and MuJoCo versions;
4. runs `tools/linux_suite.sh` (Linux-only tests, `tests/core`, `tests/upstream`); any
   failure *or skip* fails the gate; and
5. writes `evidence/mjbatch-upgrade/<sha7>/linux-gate.json` and `linux-gate.log`.

**Enforcement:** `tools/check_mjbatch_clean.py` (CI `static` job, every commit) fails
unless the pinned revision has a passing record, so a pin change cannot pass CI without
this gate. CI's `image-and-linux` job also runs the same suite on x86-64 for every
commit. Both must be required status checks on `main` (production to-do 1.4).

Then run the full suite and commit the gitlink, lock, table, stubs and the gate record
together. Rebuild the images and requalify that commit on kind
(`deploy/kind/requalify.sh`, which refuses uncommitted changes), and fix forward if it
fails.

A promotion is a release: it goes through blue/green like any other image change. Runs
never mix MJBatch versions, and binary snapshots do not cross versions without explicit
certification (otherwise work replays from its seeds).

## 5. Roll back

```bash
python tools/mjbatch_upgrade.py rollback
```

This restores the committed pin: it checks the submodule out at the superproject's
gitlink and restores `pyproject.toml`, `uv.lock` and the `QUALIFIED` table from `HEAD`,
then runs `uv sync` and the clean check. In a deployed environment, rolling back the
*release* (the previous image) is the blue/green rollback.

## 6. Clean up

```bash
python tools/mjbatch_upgrade.py cleanup <sha12>
```

This removes the candidate worktree and virtualenv.

## Automated boundary checks

```bash
python tools/check_mjbatch_clean.py
```

The checker runs in CI and must pass at every commit (see
[mjbatch-integration.md](../mjbatch-integration.md) for the rules it enforces). It also
requires the pinned revision's passing Linux gate record (step 4a).
`tests/tooling/test_noninvasive.py` proves it detects each class of violation.
