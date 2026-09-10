# Spec 0054: Spec 0053 Numerical And Resume Preflight Amendment

Status: locked / implementation-ready; result-blind private execution pending
Owner/workstream: Spec 0053 bounded numerical and resumability preflight
Last updated: 2026-09-10

## Purpose

Execute the first and only result-blind gate before any scientific stage of
Spec 0053. This amendment measures numerical viability, fixed-shape runtime,
memory, compilation closure, and authentic Kaggle predecessor mounting. It
does not evaluate, rank, compare, or disclose scientific behavior of either
VAE.

This amendment supersedes none of Specs 0038, 0050, 0051, or 0052. It adds no
model training, checkpoint change, data publication, sealed-data access, report,
paper, thesis, Overleaf, GitHub, commit, or push scope.

## Fixed Boundary

- One private parent kernel, exactly
  `maximshtefan/eqvae-functional-geometry-preflight-01a08ab5`.
- At most one private child kernel, exactly
  `maximshtefan/eqvae-functional-geometry-preflight-01a08ab5-resume`.
- The parent creates a deliberate, synthetic pending resume work unit. The
  child is authorized only to authenticate and complete that unit; it is not a
  retry and cannot perform additional benchmarks.
- Both kernels are private T4 scripts with internet disabled. They attach only
  the existing frozen-input dataset and the existing pre-shuffled validation
  source. No dataset is written.
- The fixed input is selector rank 0. The only non-fixture model work is one
  exact C4 orbit, one 36-angle padded compatibility batch, a 17-knot decoder
  path iteration, and the progressive JVP microbatch ladder `1,2,4,8,16,32`.
- Model parameters are loaded frozen, pre/post state-dict hashes are asserted,
  no optimizer is imported or created, and no backward/update path is allowed.
- Runtime records use only anonymous labels `branch_a` and `branch_b`; no
  output may contain model names, paired differences, rankings, medians, a
  scientific endpoint, decoded comparison, or a decision.

## Inputs And Provenance

The exact input dataset, state files, selector, state-dict hashes, source
checkpoint hashes, normalization, C4/D4 convention, and 25-patch selector are
inherited from `docs/data/spec0051_decoded_transform_contract.json`. This
amendment validates the inherited hashes but uses only rank 0. Its immutable
machine contract is `docs/data/spec0053_preflight_contract.json`; generated
parent and child wrappers embed the SHA-256 of this amendment and that contract.

Before the parent claim is consumed, a read-only authenticated Kaggle listing
must establish that its exact owner/slug does not already exist. The resulting
launch receipt must be exactly version `1` of that owner/slug; any other
canonical reference fails closed before a receipt is written. The parent launch
creates an immutable receipt under
`runs/local/kaggle_launches/`. Before creating the child package, the local
preparer must require that receipt's canonical owner/slug/version and download
the parent output under explicit remote-read confirmation. It must consume that
download directory's `kaggle_output_receipt.json`, requiring the exact parent
`/1` resource reference and byte/hash inventory before inspecting its manifest.
It derives the
child's predecessor binding from the downloaded parent `manifest.json`; no
literal version, mutable slug reference, base64 state, or output dataset is
accepted.

## Result-Blind Numerical Contract

### Analytic fixtures

The parent must pass each fixture before any model telemetry is accepted:

1. asymmetric scalar and F1 vector exact-C4/D4 array actions, full group table,
   identity, inverse, and composition: bit-exact;
2. scalar and vector Lie generators, including F1 internal rotation sign,
   against central angle differences at `1`, `0.5`, and `0.25` degrees:
   final relative error `<= 1e-2` and monotone contraction after the first
   step;
3. L2/H1/disk quadrature, projectors, Fourier coefficients, generalized PCA,
   posterior W2/Fisher, path energy, quotient alignment, and graph gauge
   covariance on analytic FP64 fixtures: relative error `<= 1e-10` unless the
   exact quantity is zero, where absolute error `<= 1e-12`;
4. randomized range finder against a small exact SVD: residual and trace
   estimates within the fixed 99% certificate bound. The machine contract fixes
   a rank-2 Rademacher sketch of a rank-4 fixture, exhaustive 256-sketch
   residual coverage, and a 32-probe Rademacher trace estimator with an explicit
   Hoeffding 99% bound;
5. deterministic work-ID framing: two formerly colliding positional examples
   must produce distinct SHA-256 IDs.

### Model-mechanics gates

No model-derived quantity is compared across labels. Each anonymous branch must
independently pass finite/output-shape checks, state hash preservation, and:

| Check | Acceptance tolerance |
| --- | --- |
| JVP finite difference | relative L2 `<= 1e-2` at central epsilon `1e-3` |
| VJP adjoint | relative scalar discrepancy `<= 2e-3` |
| HVP finite difference | relative L2 `<= 5e-2` at epsilon `2e-3` |
| FP32 repeat closure | relative L2 `<= 1e-6` |
| compiled closure | relative L2 `<= 5e-3`; compile failure is recorded and forces eager-only later work |
| native/padded/path compatibility | expected tensor shapes, all finite, no parameter mutation |

The parent records the peak allocated/reserved bytes and synchronized elapsed
seconds for each declared workload. It tries each direction microbatch in
ascending order only and stops before a launch would leave less than 25% free
VRAM. The common later-stage cap is not chosen here; the output supplies only
the masked evidence needed for a separately reviewed amendment.

## Immutable Resume Proof

The parent atomically writes `preflight_parent_v1/` with a redacted binding,
runtime identity, JSONL work ledger, anonymous partial telemetry, `status.json`,
and a manifest that hashes every payload except itself. It never writes the
full input contract, payload manifest, branch permutation, or model identity.
Its two predeclared work
units are `fixtures_and_telemetry` (complete) and `authenticated_resume_fixture`
(pending). It intentionally reports `partial` only because the latter is
reserved for the authorized child.

The child package must bind its own unique slug, exact canonical predecessor
owner/slug/version, parent contract SHA, parent manifest SHA, and the singleton
pending ID. At runtime it discovers exactly one mounted predecessor, validates
the complete parent manifest and ledger, rejects a completed/unknown/extra
pending set, and atomically writes `preflight_resume_v1/` with a full ancestor
DAG and that unit marked complete. It is forbidden from importing models,
benchmarking, creating optimizers, or reading any other input.

Failure to mount or authenticate the predecessor is an accepted stop. It does
not permit a dataset write, mutable source reference, a different child slug,
or another continuation.

## Remote Guard And Outputs

The parent push requires all of:

```bash
KAGGLE_PUSH_CONFIRMED=1
KAGGLE_FULL_DATASET_CONFIRMED=1
KAGGLE_FUNCTIONAL_GEOMETRY_PREFLIGHT_CONFIRMED=1
```

and the unconsumed canonical guard phrase
`spec0053_numerical_resume_preflight_authorized`. The child requires the same
two push confirmations plus
`KAGGLE_FUNCTIONAL_GEOMETRY_PREFLIGHT_RESUME_CONFIRMED=1`; its builder refuses
to run without an authenticated parent receipt/output pair. Both launch claims
are atomic and each records the upload inventory before remote mutation.

The metadata title is exactly the parent slug, so a title/slug normalization
cannot silently target another kernel. The parent has no automatic continuation. After a terminal parent status and
receipt-bound output download, the user must be shown the parent receipt and
the child preparation/validation result before the separately bounded child
push occurs. There is no wait loop in this task.

Expected parent output is under `preflight_parent_v1/`; expected child output
is under `preflight_resume_v1/`. No artifact is a scientific output. The parent
output must not be inspected for cross-model comparison before the later
unmasking protocol is independently locked.

## Verification

Before the parent push:

```bash
.venv/bin/pytest -q tests/test_functional_geometry_preflight_kernel.py
.venv/bin/ruff check scripts/build_functional_geometry_preflight_continuation.py tests/test_functional_geometry_preflight_kernel.py
.venv/bin/ruff format --check scripts/build_functional_geometry_preflight_continuation.py tests/test_functional_geometry_preflight_kernel.py
.venv/bin/basedpyright scripts/build_functional_geometry_preflight_continuation.py tests/test_functional_geometry_preflight_kernel.py
./scripts/kaggle_kernel.sh build kaggle/kernels/functional_geometry_preflight
./scripts/kaggle_kernel.sh validate kaggle/kernels/functional_geometry_preflight
./scripts/kaggle_kernel.sh check kaggle/kernels/functional_geometry_preflight
./scripts/agent_preflight.sh
git diff --check
```

Fresh clean-context reviewers must find no unresolved P0/P1 in numerical
contracts, provenance/resume behavior, and scientific blindness before the
parent push. A child requires the same applicable local checks after its
receipt-derived package is generated.

## Completion Rule

This amendment is complete only after the parent has a canonical receipt and
the child either authenticates/completes its exact singleton unit or records a
receipt-bound mount failure. Completion establishes only numerical and resume
viability. It cannot unlock a scientific stage; a later Spec 0053 amendment
must select limits and methods from the blind evidence, receive fresh review,
and name its own guard and launch ceiling.
