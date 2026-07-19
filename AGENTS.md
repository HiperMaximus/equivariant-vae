# Repository Instructions

This repo is the paper/research repository for the equivariant VAE work.

## Project Boundaries

- Main thesis repo:
  `/home/maximus/Documents/Tesis/Tesis`
- This repo:
  `/home/maximus/Documents/Tesis/equivariant-vae`
- SIPAIM paper subtree:
  `paper/sipaim2026`
- Overleaf Git remote:
  `https://git.overleaf.com/69c614433cbc9e46cf226d24`
- Historical working FSQ reference: read the verbatim extract
  `kaggle/fsq_train_reference.py` (plain `.py`, no notebook JSON to parse) — the proven
  dual-T4 fast-path recipe and the MINIMUM efficiency floor to match and ideally beat
  (rule 30). `kaggle/train_runs` is only the raw notebook JSON; do NOT hand-parse it.
  Use the `.py` as architecture/runtime evidence for the broad ResNet-like
  autoencoder shape, Kaggle DDP/AMP/compile efficiency tactics, and training
  behavior. Do not carry forward FSQ quantization, codebooks, rounding, or
  discrete latent telemetry into the equivariant VAE; quantization does not mix
  well with the continuous `SO(2)` equivariance target.

## Hard Rules

1. Before architecture, evaluation, paper, or workflow changes, read the
   canonical landing sequence:
   `AGENTS.md`, `CURRENT.md`, `GOAL.md`,
   `docs/repo_goal_and_requirements.md`,
   `docs/issue_image_inventory.md`,
   `docs/equivariant_vae_transition_plan.md`,
   `docs/kaggle_cli_workflow.md`,
   `docs/behavior_inventory_kaggle.md`,
   `docs/overleaf_sync_workflow.md`,
   `docs/agentic_review_workflow.md`,
   `docs/spec_driven_development.md`, `docs/specs/README.md`, active specs
   linked from that index, and `docs/decisions/README.md`.
2. Do not push this whole repo to Overleaf.
3. Do not add Overleaf as `origin`.
4. Do not run plain `git push overleaf`.
5. Sync Overleaf only through:

   ```bash
   ./scripts/sipaim_overleaf_sync.sh
   ```

6. The active paper source lives in `paper/sipaim2026`.
7. The tracked advisor-facing PDF should be `paper/sipaim2026/sipaim2026.pdf`.
   Keep it updated so the current paper can be viewed from both GitHub and
   Overleaf. Keep `main.pdf`, logs, aux files, and other LaTeX build artifacts
   ignored.
8. Pull Overleaf edits before local paper edits when the professor may have
   changed the project.
9. Commit local paper changes before pushing the subtree to Overleaf.
   Overleaf remote reads, `pull`, and `push` require explicit user permission
   and must be run with `OVERLEAF_SYNC_CONFIRMED=1` only after that permission.
10. The architecture target is continuous `SO(2)` steerability, not a
   discrete-group implementation.
11. GitHub issue updates intended for the thesis professor should be written in
    Spanish unless the user asks otherwise.
12. GitHub issue updates should say what changed, where it lives, and what is
    still pending. Do not close issues unless the user explicitly asks.
13. Never store, print, or commit Overleaf tokens or other credentials.
14. GitHub issue images are requirements evidence. Inspect them before deriving
    plans, figures, metrics, or deliverables from issue comments.
15. Keep the repo LEAN and LIVE as a standing discipline, not a later cleanup.
    Non-current content is a DEFECT, not history: anything that no longer reflects CURRENT
    INTENT must be DELETED outright — never superseded, banner-flagged, or left as a
    "historical"/"superseded" note — repo-wide, specs, decisions, `GOAL.md`, and `AGENTS.md`
    included. Contradictory old text is how agents keep falling into already-corrected traps.
    Every doc, spec, `CURRENT.md`, decision note, plan, `README.md`, `GOAL.md`,
    `AGENTS.md`, workflow doc, memory, and file must earn its place by being
    currently useful. Delete deprecated, superseded, dead, purely-historical, or
    prose-only artifacts outright instead of appending to or around them; replace
    stale/bad/incorrect information rather than leaving contradictory historical
    notes; and write compactly the FIRST time so the repo never re-accumulates
    append-only history that costs thousands of lines to trim (as the 2026-07
    doc-trim campaign did). Git history, `docs/decisions/`, and superseded specs
    already preserve the past — live files carry only what a reader needs now:
    one fact, one home; one line, one hook (point at where detail lives, do not
    duplicate state). This does NOT license deleting things that merely LOOK
    unused: apply rule 29 first — verify the premise, since symptoms of a
    deliberate policy are not rot. Use the state-file policy in
    `docs/agentic_review_workflow.md`.
16. Keep `CURRENT.md` updated after meaningful shifts in active work, blockers,
    or next steps.
17. Before handing work back or stopping at a partial state, update the repo
    memory/handoff files. At minimum, record what changed, what is currently in
    progress, exactly where the agent left off, the next concrete action,
    blockers, and verification status in `CURRENT.md` and any affected active
    plan/spec.
18. Claude-specific instructions live in `CLAUDE.md` but are adapters only.
    Canonical facts belong in `AGENTS.md`, `CURRENT.md`, `GOAL.md`, and docs.
19. Before substantial work, run:

   ```bash
   ./scripts/agent_preflight.sh
   ```
20. For substantial workflow, architecture, evaluation, or paper-claim changes,
    use independent clean-context adversarial subagent reviews when the tooling
    is available. Follow `docs/agentic_review_workflow.md`. A green gate proves
    internal consistency, not correctness — only a skeptic checks whether the
    premise was worth acting on.
21. Use spec-driven development for substantial implementation, experiment,
    evaluation, paper, or workflow changes. Write or update the relevant spec in
    `docs/specs/` before coding, then verify against its acceptance criteria.
22. For Python changes, run `./scripts/python_quality.sh` before handing work
    back. The script intentionally uses the existing repo-local `.venv` and does
    not run `uv sync` or download dependencies. If the environment needs to be
    created or refreshed, ask the user first, then run
    `uv sync --locked --python 3.12 --group dev`. The gate now takes ~10 min, so
    a foreground run may exceed an agent's tool timeout and get backgrounded
    anyway — where it is reaped mid-pytest (observed twice at ~63%, matching the
    long-standing ~57% report). The reaper kills the caller's process group, so
    DETACH the gate instead (verified 2026-07-15: survived three poller kills):

    ```bash
    setsid bash -c 'cd <repo>; unset TMPDIR; ./scripts/python_quality.sh \
      > .agent_tmp/gate.log 2>&1; echo "GATE_EXIT=$?" >> .agent_tmp/gate.log' \
      < /dev/null > /dev/null 2>&1 &
    # then poll: until grep -q GATE_EXIT .agent_tmp/gate.log; do sleep 30; done
    ```

    Always capture the exit code as above. Do NOT pipe the gate into `tail`/`head`
    to read it: the pipeline's status is the LAST command's, so a failing gate
    reports success (this masked 13 ruff errors on 2026-07-15). `eqvae` is
    editable-installed as of
    2026-07-15, so a bare `.venv/bin/python -m pytest` no longer needs
    `PYTHONPATH=src`. Prefer NOT to set it: `PYTHONPATH=src` makes `eqvae`
    resolvable from any interpreter, which masks a torch-less one until the
    import that needs torch. Always use the venv interpreter: bare `python3` is
    the system one and has neither torch nor `eqvae`.
23. Python quality is intentionally strict: Ruff selects `ALL`, BasedPyright is
    strict, no global ignores are allowed, and tests may ignore only Ruff `S101`
    for bare `assert`.
24. Local repo tests use CPU-only PyTorch. GPU training belongs to Kaggle.
25. Python dependency truth lives in `pyproject.toml` for direct dependencies and
    `uv.lock` for the resolved local environment. A root `requirements.txt` is
    not allowed; pip requirements files may only be generated, context-specific
    exports such as a future Kaggle bootstrap file.
26. Kaggle is a remote execution surface, not a Git remote. Use
    `./scripts/kaggle_kernel.sh` and `docs/kaggle_cli_workflow.md`; do not use a
    GitHub-linked Kaggle notebook as the source of truth. Kaggle remote writes
    require explicit user permission and `KAGGLE_PUSH_CONFIRMED=1`.
27. Do not wait in-turn for long-running Kaggle kernels. After a Kaggle push or
    status check shows a run is still `RUNNING` and it is likely to take more
    than about 5 minutes, stop active waiting, tell the user a concrete local
    time to prompt with `continue`, and record the suggested cadence in
    `CURRENT.md` or `docs/kaggle_cli_workflow.md` when relevant.
28. Agent temporary and scratch files go in `.agent_tmp/` (on-disk, gitignored),
    not the OS `/tmp` scratchpad — `/tmp` here is a small tmpfs, so large writes
    there trigger `EDQUOT` and cause fake test failures. Leave `TMPDIR` unset for
    `./scripts/python_quality.sh` so its heavy pytest temp redirects to
    `runs/local_tmp` on disk. Clean up `.agent_tmp/` when done.
29. Before de-pinning a validator literal, relaxing a constraint, or deleting
    something that looks unused, VERIFY THE PREMISE: read what PRODUCES the value,
    and what the repo has already DECIDED about it. A plausible name or a config
    block is not evidence a feature exists; symptoms of a deliberate policy
    (nothing imports it, the gate excludes it) are not evidence of rot; a
    convenient property enforced only by a comment is not a constraint; and a
    green gate proves internal consistency, not correctness. See
    `docs/decisions/0010-verify-the-premise-before-changing-a-pin.md`, which
    records three same-day cases where the answer was already written down
    in-repo and contradicted the change in flight.
30. Speed over declared don't-cares (the user has stated this MANY times; agents keep
    violating it). The paper-promotable run is judged on wall-clock TIME PER EPOCH and model
    quality, NOT on reproducibility or tail-completeness. Treat as a hard default:
    - Exact bit/numerical REPRODUCIBILITY is not a goal — small drift is fine. Prefer the
      faster option: `cudnn.benchmark=True`, never `torch.use_deterministic_algorithms(True)`,
      `cudnn.deterministic=False`, the fastest RNG (drop per-sample / blake2b deterministic
      seeding), fp16-first (fp32 ONLY where numerically REQUIRED, not "to be safe"), and
      latest/beta torch features. A single global `set_seed` IS fine and expected — FSQ's
      `set_seed` (`kaggle/fsq_train_reference.py:112`) seeds torch/cuda/numpy for identical
      DDP-rank init (a real correctness need) while explicitly keeping
      `cudnn.deterministic=False`/`benchmark=True`; seeding is NOT determinism. What to avoid
      is deterministic ALGORITHMS, per-sample deterministic seeding, deterministic collate,
      or sorted reductions added "to be safe".
    - The dataset TAIL does not matter — use `drop_last=True`; add NO remainder /
      partial-batch / padding logic (a fixed batch shape is also what CUDA graphs require).
    Add reproducibility or tail-handling machinery ONLY for a real correctness reason
    (NaN/divergence), never out of caution. The proven dual-T4 fast-path recipe is captured
    verbatim in `kaggle/fsq_train_reference.py` (a plain `.py`, no notebook JSON to parse) —
    it is the MINIMUM efficiency bar we aim at, the floor to MATCH and ideally BEAT. Read it
    before making any runtime/speed decision; do not ship something slower than it without a
    measured reason. Tune for the REAL dual-T4: single-GPU / `max-autotune` kernel tuning
    misses the DDP all_reduce-overlap cost, so always search the compute-comm-overlap knobs
    (`compiled_autograd` / `optimize_ddp` / a comm hook) and measure on 2 GPUs.

## Safe Paper Workflow

```bash
./scripts/sipaim_overleaf_sync.sh check
./scripts/sipaim_overleaf_sync.sh setup
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull

# edit paper/sipaim2026
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Update SIPAIM paper"

OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

See:

- `CURRENT.md` for active status and next concrete steps.
- `GOAL.md` for the repo north star.
- `docs/repo_goal_and_requirements.md` for issue-derived deliverables.
- `docs/issue_image_inventory.md` for inspected issue screenshots.
- `docs/kaggle_cli_workflow.md` for CLI-managed Kaggle script kernels.
- `docs/behavior_inventory_kaggle.md` for historical Kaggle data, training,
  resume, metric, and artifact behavior.
- `docs/overleaf_sync_workflow.md` for the full workflow and failure modes.
- `docs/decisions/README.md` for settled project decisions.
- `docs/open_follow_ups.md` for the known-issues backlog to revisit.
- `docs/agentic_review_workflow.md` for independent adversarial review.
- `docs/spec_driven_development.md`, `docs/specs/README.md`, and active specs
  in `docs/specs/` for implementation contracts.
- `docs/specs/0002-strict-python-quality-gate.md` for Python quality rules.
