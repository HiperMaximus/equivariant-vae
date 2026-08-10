# Repository Instructions

This repo is the paper/research repository for the equivariant VAE work.

## Project Boundaries

- Main thesis repo:
  `/home/n00b1337/Documents/Max/Tesis/Tesis`
- This repo:
  `/home/n00b1337/Documents/Max/Tesis/equivariant-vae`
- SIPAIM paper subtree:
  `paper/sipaim2026`
- Overleaf Git remote:
  `https://git.overleaf.com/69c614433cbc9e46cf226d24`
- Historical working FSQ reference: read the verbatim extract
  `kaggle/fsq_train_reference.py` (plain `.py`, no notebook JSON to parse) — a successful
  dual-T4 run and the MINIMUM efficiency floor to match and ideally beat, NOT an optimized
  recipe (its flags/parameters were chosen directly, without a comparative search)
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
    quality, NOT on reproducibility or tail-completeness. This is a ONE-OFF research
    experiment comparing two architectures, not production software or a user-facing runtime.
    Optimize the code we will actually execute on the dual T4s; do not spend experiment time on
    general compatibility, defensive abstractions, exhaustive malformed-input handling, or
    historical artifacts that do not protect this run. Treat as a hard default:
    - Exact bit/numerical REPRODUCIBILITY is not a goal — small drift is fine. Prefer the
      faster option: `cudnn.benchmark=True`, never `torch.use_deterministic_algorithms(True)`,
      `cudnn.deterministic=False`, the fastest RNG (drop per-sample / blake2b deterministic
      seeding), fp16-first (fp32 ONLY where numerically REQUIRED, not "to be safe"), and
      latest/beta torch features. Do not impose deterministic/global training or search RNG;
      DDP's constructor synchronization supplies identical initial parameters. Fixed seeds are
      allowed only in construction/unit tests or statistical artifact analysis where stable
      expected values are the thing being tested. Avoid deterministic algorithms, per-sample
      deterministic seeding, deterministic collate, or sorted reductions added "to be safe".
    - The dataset TAIL does not matter — use `drop_last=True`; add NO remainder /
      partial-batch / padding fallback logic (a fixed batch shape is also what CUDA graphs
      require). If a proposed batch cannot yield a full shard batch, reject that candidate;
      never switch the paid run to a variable tail shape to preserve patches.
    Add reproducibility or tail-handling machinery ONLY for a real correctness reason
    (NaN/divergence), never out of caution. The successful but UNOPTIMIZED dual-T4 reference is captured
    verbatim in `kaggle/fsq_train_reference.py` (a plain `.py`, no notebook JSON to parse) —
    it is the MINIMUM operational bar to MATCH and ideally BEAT: steady-state throughput /
    time-per-epoch plus stable quality on shared reconstruction metrics. FSQ-only losses and
    discrete-latent metrics are not comparison targets. Read it before making any runtime/speed
    decision; do not ship something slower or materially worse on shared quality evidence
    without a measured reason. Compile/startup time is a NON-COST for the long run; score the
    settled step and projected epoch, so expensive autotuning is in-bounds. Tune for the REAL
    dual-T4: single-GPU / `max-autotune` kernel tuning
    misses the DDP all_reduce-overlap cost, so always search the compute-comm-overlap knobs
    (`compiled_autograd` / `optimize_ddp` / a comm hook) and measure on 2 GPUs.
    The complete FSQ fast path is a required CONTROL, not an optimum: reproduce its
    compiled-autograd/DDP pair, read-only mmap + `MADV_SEQUENTIAL`, uint8 H2D,
    channels-last, pinned loader, static graph, and on-device metric accumulation as
    candidates, but also test current alternatives for every performance-relevant flag/parameter.
    The FSQ run performed no ablation or optimization search; never infer that its batch 60,
    LR 5e-4, one worker/rank, mmap wrapper/advice, channels-last, static graph, compile defaults,
    AMP details, DDP mode, optimizer form, warmup, or telemetry cadence is best for either VAE.
    FSQ is not the boundary of the search. Inventory the newest installed PyTorch/Inductor/
    Dynamo/DDP/optimizer/communication, CUDA/cuDNN, and relevant NCCL performance surfaces.
    Every plausibly speed-affecting option, including experimental/internal ones absent from the
    repo, must be measured or have a concrete recorded exclusion reason; then measure important
    finalist interactions rather than assuming independent main effects compose.
    Experimental/beta PyTorch features are TRUSTED candidates for this simple model: feature-
    detect them on the installed latest release, execute them, and keep them if the bounded
    correctness smoke passes and measured epoch throughput wins. Do not penalize a candidate
    merely for being experimental.
    Keep the timed train step free of `.item()`, `.cpu()`, printing, Python decisions on CUDA
    tensors, and other device-to-host synchronizations except an unavoidable optimizer/AMP or
    DDP primitive. Accumulate telemetry on-device and materialize it only outside timing or at
    an infrequent artifact boundary.
    Do NOT use zero graph breaks as a universal performance gate. Use a single-GPU
    `fullgraph=True` diagnostic to find accidental breaks, then measure the real DDP execution:
    stable DDP graph partitions/breaks that enable earlier gradient synchronization are allowed
    and may beat a break-free graph. Rank the exact executed recipe by settled end-to-end
    dual-T4 step time/projected epoch time, not by graph aesthetics.
    Kaggle is the primary test surface for CUDA, Inductor, dual-T4 DDP, VRAM, transfer overlap,
    and throughput; test those directly there instead of simulating them on the CPU laptop.
    Runtime/compile/DDP correctness and compute-timing kernels use generated uint8 tensors and
    attach NO real dataset; reproduce the real shape/layout in memory. Use a tiny generated file
    in the same binary format for mmap/loader mechanics. Pay the ~30-minute real-dataset mount
    only after narrowing the compute recipe, specifically for final loader-starvation evidence,
    learning-rate/quality tuning, or the real run.
    Keep searches staged and purposeful, and retain the explicit permission guard for every
    remote write.
31. Every test's docstring must state its INTENT — the why and the justification, not the
    mechanics. Ruff already forces a docstring to EXIST; no linter can check that it says
    anything useful, so this is a review standard. A test encodes a claim about what is
    CORRECT, and a test that encodes the wrong claim is worse than no test: when the real fix
    lands the test fails, and the next agent "fixes" the code to satisfy the test or concludes
    the fix was wrong. A green gate proves internal consistency, not correctness (rule 20).
    Every test docstring states:
    - **The invariant it pins** — the property that must hold, not the steps it performs.
      "Asserts X == 3" is mechanics; "the schedule is derived, so a non-dividing batch floors
      rather than ceils" is an invariant.
    - **Why that invariant matters** — what actually breaks in the real run if it regresses.
    When a test pins a non-obvious literal, fixture, tolerance, or external artifact, its
    docstring also says what KIND of expected value it asserts, because this tells a future
    agent whether a failure means "fix the code" or "update the test":
      * a DELIBERATE POLICY / safety guard (fail-closed parser, hardware anchor) — keep it;
      * a MEASURED value or a CROSS-CHECK of what some producer currently emits — expected to
        change when the producer or the measurement changes; say so explicitly;
      * a DERIVED relationship — the durable form; prefer it to a frozen literal (rule 29).
    During review, the reviewer must be able to name a concrete source mutation the test
    catches. The docstring need not recite that mutation when it is already obvious from the
    invariant and assertion; requiring ritual prose creates churn. If no mutation exists, the
    test is vacuous — that is the bug, not a documentation gap. Tests whose names promise an
    invariant ("applied", "records", "before", "uses") are the usual offenders: they assert
    presence where they claim ordering or causation.
    A test must also be the cheapest strong proof of a UNIQUE failure. Delete a weaker test
    when another test already fails on the same mutation; do not preserve redundant tests by
    inventing different prose. Parameter cases are distinct only when they protect distinct
    branches or failure modes. When behavior is retired, delete its tests in the same change
    unless a live artifact consumer still requires compatibility.
    Never write a docstring that over-claims what the assertions check; the docstring is the
    contract a reviewer reads first, and an over-claiming one hides the gap it describes.
32. Benchmark and execute the runtime on the SAME latest PyTorch/CUDA-compatible stack.
    Every Kaggle kernel, including pretests, runtime selection, compile probes, debug, and
    the full run, must call its stdlib-only `_ensure_latest_torch` upgrade to the newest
    available PyPI release before importing `eqvae`/torch and must record the resolved
    torch/CUDA versions. Kaggle's preinstalled torch is never authoritative. Do not infer
    current compatibility from old PyTorch issues or old agent knowledge: check current
    upstream source/issues against the upgraded version, feature-detect experimental modes,
    and measure them on the real dual-T4. Decision 0012 is canonical.
33. Agent-program memory is never a state of record. Any Claude, Codex, Gemini, IDE, or
    other tool-local memory or plan that affects this repo must also be written into tracked
    `AGENTS.md`, `CURRENT.md`, `GOAL.md`, or the active spec before an agent relies on it or
    hands work off. A fresh laptop or different agent must be able to resume from the repo
    alone; never point a canonical handoff at an untracked program-memory entry.

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
