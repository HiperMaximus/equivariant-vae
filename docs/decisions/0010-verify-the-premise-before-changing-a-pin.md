# Decision 0010: Verify the Premise Before Changing a Pin

Date: 2026-07-15

Decision: before de-pinning a validator literal, relaxing a constraint, or
deleting something that looks unused, prove the premise first — by reading what
PRODUCES the value and what the repo has already DECIDED about it. Three
different changes on 2026-07-15 rested on premises that were false and already
contradicted in-repo. This note records the checks that would have caught each,
so a future agent does not re-derive them the expensive way.

## Audit the producer, not just the readers

Ask first: *can anything emit a value other than this literal?* If the emitter
stamps a constant, the pin is not a stale preference from one lucky run — it is a
cross-check that the plan matches what the harness actually ran. De-pinning buys
zero capability and deletes a real invariant.

Worked case: `_dataloader_errors` in `src/eqvae/training/selected_runtime.py` was
de-pinned to a torch-coherence model, gated green, and reverted. The dataloader is
not a searched axis — `runtime_matrix` has no dataloader term, and
`runtime_selection_executor.py:1186` stamps all five cells from
`real_data_runtime_pretest.DEFAULT_DATALOADER_*`, value-equivalent to the deleted
pins. See Spec 0011 S17d for the real work and the two traps that manufactured the
false premise.

Related check: a pin may be the ONLY real check. `_application_mismatches` compares
the dataloader cells against a plan echo (`selected_runtime_runner.py:5084-5088`),
so it can never fail until S17c lands. Never remove a pin whose replacement is a
tautology.

## A plausible name or a config block is not evidence a feature exists

Both are write-only artifacts until a consuming code path is found. Grep the config
key for a reader; grep the constant for a writer.

- `runtime_schema.DATALOADER_MATRIX_COLUMNS` is the CSV **record** schema of a
  measurement row. Every consumer is `write_csv(...)` or a header assert. The name
  reads as "search axes". It is not.
- `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` has a
  `dataloader.candidates` grid that **nothing reads**: its only would-be reader
  (`dataloader_pretest.py:175`) wants a `dataloader_pretest` key that config does
  not define. It is an unbuilt intention.

Same failure, other costumes: a datapoint without its context is not evidence
either (`docs/behavior_inventory_kaggle.md:194` `num_workers: 1` sits under
`accelerator: NVIDIA Tesla T4` and `torchrun --nproc_per_node=2` two lines above —
it is the dual-T4 DDP case, so it argues FOR a tight CPU budget, not against).

## Check whether the repo already anticipated you

Three times in one day the answer was already written down and contradicted the
change in flight:

- **Spec 0001 package/import policy** already said `PYTHONPATH=src` was the
  interim contract "until a packaging backend is explicitly added", and required
  updating that spec when one was. The backend was added without reading the
  policy that governed it.
- **`docs/open_follow_ups.md` DO NOT DROP** listed `src/nn/` as a retained **user
  decision**, and Spec 0002 gives the rationale ("intentionally left on disk for
  reference while the comparable VAE implementation is built"). It was one
  approval away from `git rm`.
- **Spec 0011 S17b-3** described baking identity at build time, justified by a
  "the BUILD must stay torch-less" premise that was itself an accident (see below).

Before contradicting a doc, read it. Before deleting, read what it is for.

## Symptoms of a deliberate policy are not evidence of rot

`src/nn` looked dead by every measure: nothing imports it, ruff AND basedpyright
exclude it, the Kaggle payload does not ship it. All three are Spec 0002 working as
designed — "not a supported import target, not part of the quality gate" is what
that policy literally says. Reference material parked on purpose looks identical to
abandoned code from the outside. Check for the decision before diagnosing decay.

## A convenient property is not a constraint

`scripts/kaggle_kernel.sh` invoked the kernel builder with bare `python3` (the
system interpreter: no torch, no `eqvae`). A comment then rationalised this as "the
torch-less kernel BUILD", and that phrase went on to shape design — it nearly forced
a SECOND copy of the identity-composition formula into the builder, the exact
duplication `benchmarking/row_id.py` single-sourcing exists to prevent. Nothing
required a torch-less build. Someone typed `python3`.

Ask what enforces a constraint. If the answer is "a comment", it is not a
constraint.

## A green gate proves internal consistency, not correctness

The reverted dataloader de-pin passed 566 tests, ruff, and basedpyright. Tests
prove the code does what it says; only a skeptic checks whether what it says is
worth doing. Every real defect found on 2026-07-15 came from clean-context
adversarial review (two fleets, 165 agents, 3 self-introduced regressions), none
from the suite, which was green throughout. Verify your own fix's FAILURE path, not
just its happy path — the `build_kernel_py` guard probed `import eqvae`, which is
torch-free, so it passed on interpreters the build then died on.

See `docs/agentic_review_workflow.md`. Reviewers on uncommitted work must be
read-only.
