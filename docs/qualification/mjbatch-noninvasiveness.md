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
| Pinned checkout `third_party/mjbatch` pristine (no modified, untracked or ignored files) at `2f169146b643adda65e195d8ad13d997abdd2a2e` | OK | `evidence/mjbatch-upgrade/10-check-clean-pinned.txt` (`tools/check_mjbatch_clean.py`) |
| Superproject gitlink = checkout HEAD = `QUALIFIED["0.1.1"]` | OK | same |
| Installed `mjbatch` built from that checkout (`direct_url.json` → `third_party/mjbatch`) | OK | same |
| Only `raysimbatch/compat/mjbatch_adapter.py` imports mjbatch | OK | same (AST scan of `src/`) |
| No attribute assignment or `setattr` on mjbatch objects (no monkey-patching); no private-attribute reads | OK | same |
| No MJBatch source files outside the checkout (no vendor-and-modify) | OK | same |
| Build artifacts outside the checkout (`build/mjbatch/<wheel_tag>` via `[tool.uv] config-settings-package`); `PYTHONDONTWRITEBYTECODE=1` in `tools/env.sh` | OK | `pyproject.toml`, `tools/env.sh` |
| The checker detects each violation class: an import outside the boundary, monkey-patching, a private read, edits, untracked files, build artifacts, a vendored copy | OK | `tests/tooling/test_noninvasive.py` (5 tests) |
| The user's own MJBatch checkout (`~/Workspace/Develop/GitHub/mjbatch`) is untouched: branch `main` at `2f16914`, `git status` empty (the only ignored entry is the environment's `.claude/` worktree directory), no stash entries. Its only other worktree is the one the session environment created (also clean). | OK | rechecked at the end of qualification, 2026-09-28 |

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
