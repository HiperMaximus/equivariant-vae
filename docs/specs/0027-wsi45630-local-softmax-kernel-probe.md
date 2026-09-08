# Spec 0027: WSI45630 Local-Softmax Kernel Probe

Status: executed / closed without a backend selection
Implementation readiness: evidence retrieved; rerun requires a revised spec and fresh authorization
Owner/workstream: Spec 0026 local-attention backend selection
Last updated: 2026-09-01

## Purpose

Measure whether PyTorch scaled-dot-product attention is a correct and efficient
implementation of the proposed radius-2 local-softmax replacement in Spec 0026. Use the
real 32,595-patch WSI45630 coordinate graph, generated width-192 tokens and one
static zero-value null key. Select an implementation backend by measured
forward/backward time and peak VRAM rather than transferring the old full-
attention result by assumption.

## Non-Goals

- No latent payload read, CNN, complete MIL model, optimizer step, classifier
  learning, validation/test access, architecture-quality claim or ablation.
- No new Kaggle dataset, public visibility change, latent producer mount,
  FlexAttention implementation, custom CUDA/Triton kernel or automatic retry.
- No claim that a passing local kernel proves the full Spec 0026 model fits.

## Inputs And Data Contract

- Reuse only private dataset
  `maximusshtefan/eqvae-wsi45630-capacity-inputs`, version 1.
- Authenticate `wsi45630_capacity_input.json` SHA-256
  `99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660`
  and `probe/pointers.csv` SHA-256
  `08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c`.
- Read only `(wsi_id,x,y)` from the pointer CSV. Require 32,595 unique WSI45630
  coordinates, all divisible by 256 and ordered by numeric `(y,x)`.
- Construct the exact Spec 0026 symmetric Chebyshev-radius-2 graph with self
  edges, maximum 25 patch keys, holes absent and neighbours ordered by
  `(delta_gx,delta_gy)`.
- Generate FP16 `X[32595,192]` on GPU from seed 1701. The seed stabilizes the
  correctness cross-check only; deterministic CUDA algorithms are forbidden.
- Runtime must upgrade `torch`, `torchvision` and `torchaudio` before importing
  Torch, then record resolved Torch/CUDA/cuDNN/device identities.

## Probe Algebra

Use six heads of width 32. Each candidate owns byte-identical bias-free Q/K/V
and biased output projections. Every query attends its valid patch neighbours
plus one static null slot:

```text
patch_score_ijh = dot(q_ih,k_jh) / sqrt(32) + B_h[radial_code_ij]
null_score_ih   = dot(q_ih,k_null_h) / sqrt(32) + gamma_h
v_null_h        = 0
weights         = softmax(valid patch scores + one null score)
output          = OutProj(sum(weights * values))
```

`B[6,6]`, `k_null[6,32]` and `gamma[6]` are learnable probe parameters and
initialize to zero. Invalid patch slots are `-inf`; the null slot is always
valid. There is no `-log(degree)`, common head offset or local register residual
token because each would be meaningless or a different architecture under
softmax.

## Candidate Contract

Benchmark identical algebra for these implementations:

1. explicit gathered FP32 score/softmax/value reference;
2. gathered `torch.nn.functional.scaled_dot_product_attention` with automatic
   backend selection;
3. the same SDPA call forced to `EFFICIENT_ATTENTION`;
4. the same SDPA call forced to `FLASH_ATTENTION`.

Forced backends that the upgraded T4 runtime rejects are recorded as unsupported,
not probe failures. Test chunk sizes `2048`, `8192` and `32595` queries. A
candidate may not construct an `N x N` tensor. The explicit implementation is
the unfused performance candidate, not automatically the winner.

Before full-graph timing, compare explicit and every supported SDPA backend
against an independently assembled per-query oracle on a deterministic
257-query slice. Compare output, input gradient and every
projection/relative/null parameter gradient using `rtol=5e-3, atol=5e-3` plus
relative L2 at most `1e-2`; require all values finite and every oracle gradient
norm above `1e-6`. Use one shared initialized state with deliberately nonzero
radial/null parameters copied independently per candidate and a deterministic
summed upstream gradient.

For each supported/correct full-graph row, run two untimed warmups and five
measured forward/backward iterations. Record mean/min/max milliseconds and peak
allocated/reserved VRAM. Synchronize outside each measured region. The selected
row is the lowest mean time among correctness-passing rows; peak allocation must
remain below device capacity.

## Outputs And Acceptance Artifacts

- Kernel source:
  `kaggle/kernels/wsi45630_local_attention_probe/run.py`.
- Private output: `spec0027_local_attention_probe.json` plus the Kaggle log,
  retrieved only through the dedicated non-overwriting output action.
- Retrieved evidence root:
  `runs/kaggle/wsi45630_local_attention_probe_v1/`.
- Artifact records input/graph hashes, degree histogram, candidate status,
  backend errors, correctness deltas, timing, VRAM and selected row.
- A failed artifact is still published with phase and traceback.

## Remote Authorization

The user's 2026-09-01 instruction “ok do the probe” authorizes exactly one push,
the status/output reads needed to retrieve it, and no rerun, for private kernel
`maximusshtefan/eqvae-wsi45630-local-attention-probe`. The guarded push requires:

```text
KAGGLE_PUSH_CONFIRMED=1
KAGGLE_FULL_DATASET_CONFIRMED=1
KAGGLE_LOCAL_ATTENTION_PROBE_CONFIRMED=1
```

The route accepts no passthrough overrides or blocking wait flag. It recognizes
success only from Kaggle's explicit accepted-version message, then writes a
local push receipt containing that version and the uploaded source/metadata
hashes. The receipt blocks every later push. This authority is consumed by the
first successful push. Any repair push after a remote failure requires fresh
user authorization.

Guard authorization token: `spec0027_local_softmax_probe_authorized`.

Exact authorized execution bytes:

- `run.py` SHA-256:
  `99bf923da39820591b3d5ec388c86fc76a050c931e5b7a0c902982e5dac29f5c`;
- `kernel-metadata.json` SHA-256:
  `9350828a9d59fb7ea0307310c875734e4f8528aa973ed2f6af5f31a4eb1db867`.

The guard must reject any later edit rather than launch bytes not reviewed here.

## Acceptance Criteria

1. Local tests pin input hashes, graph semantics, null-key algebra, candidate
   allow-list, exact metadata and the dedicated push guard.
2. Kernel source is below 1 MB, compiles locally and upgrades Torch before its
   first Torch import.
3. No dataset/kernel/model sources other than the one existing private input
   dataset are attached.
4. Every supported candidate passes the small-slice output/gradient comparison
   before full-graph timing.
5. The artifact contains all 12 backend/chunk rows or an explicit unsupported/
   failed status for each, and selects only a correct successful row.
6. Focused tests, `./scripts/python_quality.sh`, repository/workspace preflight
   and `git diff --check` pass before push.
7. The dedicated retrieval route obtains both JSON and log, refuses overwrite,
   and records both hashes in a retrieval receipt and `CURRENT.md`; Spec 0026
   changes only after interpreting measured evidence.

## Tests And Verification Commands

```bash
.venv/bin/python -m pytest -q tests/test_spec0027_local_attention_probe.py \
  tests/test_kaggle_torch_upgrade.py tests/test_kaggle_oauth_exec.py
./scripts/python_quality.sh
./scripts/agent_preflight.sh
git diff --check
```

## Implementation Blockers

The one authorized execution is consumed. It did not reach timing because the
locked elementwise `allclose` gate rejected both K/V projection-weight gradients
for explicit, automatic SDPA and forced efficient SDPA. A revised numerical
acceptance contract and fresh user authorization are required for any rerun.

## Executed Result

Private Kaggle kernel version 1 ran on a Tesla T4 with Torch `2.13.0+cu130` and
authenticated the 32,595-node, 793,081-edge graph. It settled `ERROR` after
145.06 seconds because no candidate passed the complete locked correctness
gate, so no timing row or backend winner exists.

- Explicit attention failed only `grad:k_proj.weight` and
  `grad:v_proj.weight`; maximum relative L2 discrepancy across all comparisons
  was `9.56e-4`.
- Automatic and forced-efficient SDPA failed the same two elementwise checks;
  maximum relative L2 discrepancy was `1.047e-3`.
- Flash SDPA was unsupported because this T4 path rejects the non-null additive
  attention mask.
- The explicit implementation itself failed against the independently assembled
  oracle, so this artifact does not show that SDPA is uniquely inaccurate. It
  shows that the predeclared elementwise `rtol=atol=5e-3` gate was stricter than
  the observed FP16 projection-gradient rearrangement.

Evidence:

- artifact SHA-256:
  `299dcbfbf03b2301d03cd7fedd109c7833039c5a4a307f00fd56c55d593e7753`;
- normalized retrieved `kaggle.log` SHA-256:
  `37677e479e4269cc9b9996eece7388d96029171cc8271b12ff5c4567e901fdaa`;
- raw Kaggle-downloaded log SHA-256:
  `21cd8874542144ccdc23c4fbffb423ec38d2334a2a0916631fccbc4aadc9a53e`;
- retrieval receipt SHA-256:
  `94e0dca8c5eea5efaf80f2178a2b43d2be5aa25201e14e90c6d37c779163a4bf`.

Acceptance criteria 1–3, 6 and 7 passed. Criteria 4–5 did not; Spec 0026 remains
blocked on the local activation/backend decision.

## Known Risks

- SDPA fused kernels may reject an additive learned bias, query length one or
  T4 capability and fall back or report unsupported.
- Gather/scatter and saved backward tensors may dominate runtime despite only
  26 keys per query.
- A local-kernel winner may still be a poor full-model choice once the CNN and
  two complete graph blocks are present.
- Five measured iterations estimate engineering throughput, not statistical
  model performance.

## Adversarial Checks

- Change one coordinate, pointer hash, dataset source or WSI identity and require
  local/remote validation to fail before timing.
- Duplicate the null slot through padding or give it a nonzero value and require
  algebra tests to fail.
- Remove the radial bias from one candidate or let invalid keys receive mass and
  require the reference comparison to fail.
- Permit an `N x N` allocation, silently fall back from a forced backend or
  select a numerically failing row and require structural/artifact checks to fail.
- Attempt a second push without renewed authorization and require the operational
  handoff to identify it as unauthorized.

## Open Questions

The execution established that Flash SDPA cannot implement this masked form on
the target stack. It did not establish an explicit-versus-SDPA speed winner.
Full-model capacity remains a later gate regardless of any future result.

## Related Files

- `docs/specs/0026-local-global-sigmoid-mil.md`
- `docs/kaggle_cli_workflow.md`
- `runs/local/wsi45630_capacity/input_receipt.json`
- `scripts/kaggle_kernel.sh`
