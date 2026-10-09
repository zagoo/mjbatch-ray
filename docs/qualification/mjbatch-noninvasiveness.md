# MJBatch non-invasiveness and upgradeability report

**Question:** does RaySimBatch use MJBatch without modifying it, in a way that allows
straightforward upstream upgrades?

**Answer:** yes (verified). The pinned checkout is pristine, and the boundary is enforced
and tested. A real upstream candidate (`121e033`) passed every upgrade gate. Promote and
rollback were exercised end to end in a disposable clone. The one real cost of not
modifying MJBatch is measured and recorded (design amendment AM-2).

## 1. Non-invasiveness: observed facts

| Check | Result | Evidence |
|---|---|---|
| Pinned checkout `third_party/mjbatch` pristine (no modified, untracked or ignored files) at `088e2d4cd0d5878ea184010446c058710f23a671` | OK | `evidence/mjbatch-upgrade/088e2d4/check-clean.txt` (`tools/check_mjbatch_clean.py`; at `2f169146…`: `evidence/mjbatch-upgrade/10-check-clean-pinned.txt`) |
| Superproject gitlink = checkout HEAD = `QUALIFIED["0.1.4"]` | OK | same |
| Installed `mjbatch` built from that checkout (`direct_url.json` → `third_party/mjbatch`) | OK | same |
| Only `raysimbatch/compat/mjbatch_adapter.py` imports mjbatch | OK | same (AST scan of `src/`) |
| No attribute assignment or `setattr` on mjbatch objects (no monkey-patching); no private-attribute reads | OK | same |
| No MJBatch source files outside the checkout (no vendor-and-modify) | OK | same |
| Build artifacts outside the checkout (`build/mjbatch/<wheel_tag>` via `[tool.uv] config-settings-package`); `PYTHONDONTWRITEBYTECODE=1` in `tools/env.sh` | OK | `pyproject.toml`, `tools/env.sh` |
| The checker detects each violation class: an import outside the boundary, monkey-patching, a private read, edits, untracked files, build artifacts, a vendored copy | OK | `tests/tooling/test_noninvasive.py` (5 tests) |
| The user's own MJBatch checkout (`~/Workspace/Develop/GitHub/mjbatch`) is untouched: branch `main` at `088e2d4` (the owner pulled it; the upgrade only fetched from it), `git status` empty (the only ignored entry is the environment's `.claude/` worktree directory), no stash entries. | OK | rechecked 2026-10-09 (first on 2026-09-28 at `2f16914`) |

What RaySimBatch depends on is only MJBatch's **public** API (`Batch`, `bind`,
`expand`, `step` including its documented `history` argument, `forward`, `reset`,
`set_const`, `nstate`, `num_threads`) and the
**documented behavior** listed in `docs/ledgers/mjbatch-source-findings.md` (M1–M20).
Those behaviors are re-established on the running binary by runtime probes in every
process, and pinned by `tests/upstream` (18 tests).

## 2. What replaced the planned MJBatch extension

The design (§1) planned a maintained MJBatch extension. Each piece was replaced by
composition over the public API: AM-1 (per-slot outcomes via a sentinel in a bound
buffer), AM-2 (per-substep `step(nstep=1)` with compiled kernels between calls), AM-3
(`set_const` isolation), AM-4 (pending-write neutralization), AM-5 (affinity through
thread inheritance and per-TID pinning), AM-6 (exact reset through state rows), and AM-7
(diagnostic queries on a private scalar model). See `docs/design-amendments.md`.

**Measured cost** (AM-2, Apple arm64, `qualification/bench_throughput.py`):

- For the contact model (quadruped), the wrapper spends 84–92% of wall time inside
  MJBatch and meets the ≥90%-of-upstream target at 1, 2 and 4 threads (0.93–1.09×).
- For the lightest models (cartpole, reacher, ≈1 µs physics steps), the wrapper reaches
  0.26–0.65× of upstream `nstep=d`. Per-call overhead dominates when physics is that
  cheap. An in-MJBatch substep hook, as the design assumed, would avoid it.
- Fused intervals (AM-2) recover most of the light-model loss through public API only
  (`step(..., history=H)`, M17), with results identical to per-substep stepping. With
  RaySimBatch's own bookkeeping moved to native code, the lightest models reach
  0.75–0.93×.
- Figures and method: [report.md §5](report.md).

## 3. Upgradeability

**Where version-specific knowledge lives:**

- The `QUALIFIED` table and the behavior probes are in `compat/mjbatch_adapter.py`.
- The semantics ledger (M1–M20) and `tests/upstream` pin what the rest of the code
  relies on.

A new upstream version that keeps those semantics needs a new `QUALIFIED` entry and
nothing else. One that changes them fails a probe or a contract test *before* any pin
moves.

**Workflow:** `tools/mjbatch_upgrade.py` (`docs/operations/mjbatch-upgrade.md`) runs
prepare (an isolated worktree and venv), then qualify (gates), then promote (the pin,
uncommitted for review), with rollback and cleanup.

**Exercise on a real upstream candidate:** `origin/mujoco-versions` at `121e033`
("Support MuJoCo 3.10 through 3.13 with one wheel per release"). The candidate declares
version `0.1.0`, but its build produces `0.1.0.post31300`.

| Gate | First attempt | Final (`evidence/mjbatch-upgrade/05-qualification.json`) |
|---|---|---|
| behaviour_probes | pass | **pass**: every probe true in candidate mode |
| upstream_contract_and_parity_tests (`tests/upstream`, `tests/core`) | pass | **pass**: 43 passed |
| performance_regression | fail (−5% to −12%). **Invalid:** a stored baseline under different host load and benchmark methodology; upstream MJBatch itself moved −6.5%. | **pass**: interleaved A-B-A-B in one session. Wrapper throughput change vs pinned: cartpole T1 −0.2%, T2 −0.2%; quadruped T1 −0.4%, T2 +0.2%. |
| candidate_tree_pristine | fail. The checker, pointed at the candidate, flagged the pinned checkout (a tool bug). After that was fixed, it caught a **real** artifact: the candidate's build hook wrote `__pycache__` into the upstream tree. | **pass**, with bytecode disabled for builds |

**Promote and rollback, exercised in a disposable clone** (the production pin did not move):

- `promote` checked out `121e033` and added `QUALIFIED["0.1.0.post31300"]`. It set
  `mjbatch==0.1.0.post31300`, then `uv lock` and `uv sync` succeeded
  (`22-clone-promote.log`).
- In the promoted clone, *without* candidate mode, the pair reports `qualified: true`,
  every probe passes, and the 43 contract and parity tests pass
  (`23-clone-promoted-verify.log`).
- `rollback` restored pin `2f169146`, `pyproject.toml`, `uv.lock` and the table. The clone
  is clean, `0.1.1` is qualified again, and the 43 tests pass
  (`24-clone-rollback.log`, `25-clone-after-rollback.log`).
- The exercise found one more tool defect, now fixed. Promote's final check compared the
  checkout with the *committed* gitlink and exited 1 for a correct promotion. Promote now
  stages the pin change as one unit, and the checker compares the index. The re-run
  exited 0 for both promote and rollback (`26-clone-promote-2.log`,
  `27-clone-rollback-2.log`).

Defects found in the workflow and fixed along the way:

- the project's build settings merged into the candidate build (`--no-config`);
- the fetch refspec;
- the pristine check's scope;
- A/B measurement replaced a stored baseline;
- the built version is recorded instead of the declared one;
- promote's final check compared against the committed gitlink.

**No newer upstream release exists.** `main` is at the pinned `2f16914`. The candidate
branch is unreleased and older-numbered, so the production pin stays at `2f16914`.

After qualification the candidate's manifest and the tool's own logs were copied to
`evidence/mjbatch-upgrade/06-*`. The candidate was then removed with
`tools/mjbatch_upgrade.py cleanup 121e0331323e`: its worktree was deregistered from the
pinned repository, `status` lists no candidates, and the pinned checkout still passes
`check_mjbatch_clean.py`.

### Upgrade to `088e2d4` (v0.1.4, MuJoCo 3.14.0), 2026-10-09

Upstream `main` moved from `2f16914` (0.1.1) through 0.1.2 and 0.1.3 to `088e2d4` (0.1.4),
which tracks MuJoCo 3.14.0. The candidate came from the owner's local checkout (`main`
fetched into the pinned repository, then `prepare 088e2d4… --mujoco 3.14.0 --no-fetch`).

**First qualification: failed, and the pin did not move.** All ten probes passed, but 65
of 131 contract and parity tests failed, and the throughput gate measured nothing (the
benchmark hit the same error). Two upstream changes broke RaySimBatch:

- `f188344` makes the integration-state fields (time, qpos, qvel, act, ctrl, …) views
  into the state rows, `nstate` elements apart. RaySimBatch's native kernels took MJBatch
  views only when C-contiguous (`qpos: must be a C-contiguous CPU array`).
- In the same change a pending state-row write became a field write, so `reset` now
  applies it element by element instead of discarding it (M10). One contract test pinned
  the old behavior; RaySimBatch itself never leaves a write pending into a reset.

A third change was invisible to every gate, because they run on macOS without CPU
affinity. Since `b6613c0` the calling thread is pool worker 0, and a pool of T workers
creates only T−1 threads (M5). `worker/affinity.py` pinned "T new threads" and would
have reported `found 1 physics threads, expected 2` on every Linux worker, leaving
worker 0's physics unpinned. Only the Linux tests show this
(`088e2d4/affinity-old-code-on-0.1.4.txt`). Since `03102f1` the Linux suite is a
required gate: `tools/mjbatch_upgrade.py linux` after promote, and
`tools/check_mjbatch_clean.py` (CI, every commit) refuses a pin without its passing
record (`docs/operations/mjbatch-upgrade.md` step 4a). `088e2d4` passed it
(`088e2d4/linux-gate.json`: 153 passed).

**Port (RaySimBatch only; MJBatch untouched):**

- `native/src/arrays.h`: a view's rows must be contiguous, but rows may be any distance
  apart (`Mat::ld`). One-column arrays are indexed through `Mat::operator[]`, which
  honors the stride. `tests/core/test_strided_views.py` drives the adapter and health
  kernels with row-strided views on any MJBatch version.
- `worker/affinity.py`: the calling (`mutate`) thread is pinned to the lease's first
  core and the T−1 pool threads to the rest (AM-5). `tests/linux/test_affinity.py`
  checks it.
- `tests/upstream`: the reset contract now pins element-by-element application (passes on
  0.1.4, fails on 0.1.1).
- `QUALIFIED` lists only 0.1.4. The affinity code is correct only for the 0.1.4 pool, so
  rolling back means the previous release image, which carries its own table. MuJoCo
  stubs were regenerated, and the default `AssetContract.target_version` is 3.14.0.

**Second qualification: passed** (`088e2d4/qualification.json`, `qualify.log`):

| Gate | Result |
|---|---|
| behaviour_probes | pass: all ten |
| upstream_contract_and_parity_tests | pass: 134 passed |
| performance_regression (A-B-A-B, one session, macOS arm64) | pass. Candidate vs pinned wrapper throughput: cartpole +13.6% (1 thread), +35.3% (2); quadruped +6.6%, +11.3% (`throughput-*.json`) |
| candidate_tree_pristine | pass |

After `promote`: ruff clean, pyright 0 errors, `check_mjbatch_clean.py` OK. The full
local suite gave 302 passed, 7 skipped (Linux-only) and 1 failed: Ray refused to spill
because the host disk was 98% full (`pytest-full.log`). That test passed when rerun alone
with Ray's disk threshold raised to 0.99 (`pytest-spill-rerun.log`). The Linux tests,
`tests/core` and `tests/upstream` in the `test` image with MJBatch 0.1.4 and MuJoCo
3.14.0: 153 passed (`linux-container.log`). The 100,000-step bitwise parity: 12 passed
(`parity-100k.log`). The kind requalification at `8c46586` passed (report §6).

