# Repository Instructions

This is the research, experiment and working-paper repository for the matched
normal versus continuous-`SO(2)` VAE comparison.

## Boundaries

- Repository: `/home/n00b1337/Documents/Max/Tesis/equivariant-vae`.
- Separate thesis repository: `/home/n00b1337/Documents/Max/Tesis/Tesis`.
- Working paper subtree: `paper/sipaim2026`.
- Overleaf remote: `https://git.overleaf.com/69c614433cbc9e46cf226d24`.
- Final professor report: `reports/professor/informe_final_experimento_eqvae.{docx,pdf}`.
- SIPAIM 2026 was not submitted; do not present it as an active venue.

## Operating Rules

1. Before architecture, evaluation, paper or workflow work, read:
   `AGENTS.md`, `CURRENT.md`, `GOAL.md`,
   `docs/repo_goal_and_requirements.md`,
   `docs/issue_image_inventory.md`,
   `docs/equivariant_vae_transition_plan.md`,
   `docs/kaggle_cli_workflow.md`,
   `docs/behavior_inventory_kaggle.md`,
   `docs/overleaf_sync_workflow.md`,
   `docs/agentic_review_workflow.md`,
   `docs/spec_driven_development.md`, `docs/specs/README.md`, relevant specs and
   `docs/decisions/README.md`.
2. Keep live docs compact. Delete stale status, completed to-do lists and run
   narration; Git preserves history. Retain only current contracts, evidence,
   limitations and next boundaries.
3. Verify producers, consumers, guards and settled decisions before deleting a
   file or literal that appears unused.
4. `CURRENT.md` is the current handoff, not a diary. Update it after a real
   state, blocker or authorization change.
5. Agent-local memory is not a state of record. Material state belongs in
   tracked repo docs or the active spec.
6. Preserve unrelated modified and untracked work. Never reset or clean the
   research worktree casually.
7. Use spec-driven development for substantial implementation, experiment,
   evaluation, paper or workflow changes.
8. Use independent clean-context adversarial review for substantial
   architecture, workflow, evaluation or claim changes when available.
9. Run `./scripts/agent_preflight.sh` before substantial work and before
   handoff.
10. GitHub professor updates are in Spanish unless requested otherwise. State
    what changed, where it lives and what remains. Do not close issues without
    an explicit request.
11. Treat GitHub issue images as requirements evidence; inspect them before
    deriving deliverables or claims.
12. Do not alter frozen checkpoints, sealed predictions or scored packages in
    place. New versions require a new explicit contract and authorization.
13. Sealed-test results cannot drive training, checkpoint selection, tuning,
    thresholds, retries or architecture changes.
14. No WSI attention/attribution map may be inferred from slide logits alone.
    Patch-level attribution requires instrumented inference and a separate
    scope.
15. Never store, print or commit Kaggle, Overleaf or other credentials.
16. Kaggle remote reads and writes require explicit user permission and the
    confirmation variables enforced by `scripts/kaggle_kernel.sh`.
17. The authenticated Kaggle account owns only newly created resources.
    Preserve each input/output under its exact canonical owner, slug and
    version; never rewrite cross-owner provenance.
18. Do not wait in-turn through a long Kaggle job. After confirming `RUNNING`,
    give the user a concrete local time to prompt again and stop polling.
19. Sync Overleaf only with `scripts/sipaim_overleaf_sync.sh`. Never push the
    whole repository, never make Overleaf `origin`, and never use a plain
    `git push overleaf`.
20. Overleaf remote reads, pulls and pushes require explicit permission and
    `OVERLEAF_SYNC_CONFIRMED=1`. Compile and refresh
    `paper/sipaim2026/sipaim2026.pdf` before a paper sync.
21. The thesis repository is edited only under an explicit thesis request.
22. For production Python changes, run `./scripts/python_quality.sh` with the
    existing `.venv`. The script does not install dependencies; ask before
    `uv sync --locked --python 3.12 --group dev`.
23. Python quality is strict: Ruff `ALL`, strict BasedPyright, no broad global
    ignores, and focused tests proportionate to the change. Each test docstring
    states the invariant, why it matters and whether pinned values are policy,
    measurement or derivation.
24. Local repo tests use CPU-only PyTorch. CUDA, Inductor, DDP, VRAM and
    throughput validation belong on Kaggle.
25. Dependency truth is `pyproject.toml` plus `uv.lock`. Do not add a root
    `requirements.txt`; pip files are explicit context-specific exports only.
26. Put agent scratch in `.agent_tmp/`, not `/tmp`. Leave `TMPDIR` unset for
    `scripts/python_quality.sh`, which uses `runs/local_tmp`.
27. `git diff --check` is required before handoff. A green automated gate does
    not replace claim or premise review.

## Scientific Invariants

- Target symmetry is continuous `SO(2)`, not a discrete rotation group.
- Both VAEs are frozen at 60,000 updates and use matched data, latent shape,
  optimization budget and evaluation.
- The latent is Gaussian posterior `mu/logvar`, shape `(B,16,32,32)`; no FSQ,
  codebook, rounding or discrete telemetry belongs in the comparison models.
- The decoder has no final `tanh`; raw output is used for reconstruction loss
  and clamped projection only for image-domain metrics/artifacts.
- No PixelShuffle or nearest-neighbor equivariant upsampling.
- `SO(2)` operations preserve field structure. Do not apply arbitrary channel
  operations to geometric fields.
- Frozen latent binaries remain remote and immutable; logical task/split views
  reference physical parts without copying them.
- Supplemental masks are non-exhaustive; unannotated pixels are not a negative
  tissue class.
- Report validation, sealed-test and exploratory evidence separately.

## Workflows

- Kaggle: `scripts/kaggle_kernel.sh` and `docs/kaggle_cli_workflow.md`.
- Python quality: `scripts/python_quality.sh`.
- Overleaf: `scripts/sipaim_overleaf_sync.sh` and
  `docs/overleaf_sync_workflow.md`.
- Open actionable debt: `docs/open_follow_ups.md`.
- Exact experiment/report state: `CURRENT.md`.

The FSQ reference at `kaggle/fsq_train_reference.py` is retained only as a
non-authoritative runtime/macro-architecture reference. Notebook exports and
`reference/` are not executable sources of truth.
