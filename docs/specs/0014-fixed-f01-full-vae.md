# Spec 0014: Fixed F01 Full VAE Assembly

Status: implemented / locally verified
Implementation readiness: complete; selected-runtime readiness is next and full training remains unauthorized
Owner/workstream: matched continuous-`SO(2)` VAE assembly
Last updated: 2026-08-13

## Purpose

Assemble the one fixed 43-convolution continuous-`SO(2)` VAE from the accepted
Spec 0013 primitives while mirroring the completed normal-VAE macro topology.
This slice proves the executable model locally. It does not tune or train it.

## Non-Goals

- No full training, real-data run, checkpoint migration, paper claim, or issue update.
- No radial/profile/F2/layout/multiplicity/mechanics/runtime alternatives.
- No generic equivariance library, arbitrary layouts, dynamic architecture,
  fallbacks, portability layer, or production configuration surface.
- No new resampling, normalization, gate, bias, initialization, AMP, latent,
  output, loss, or cache policy.
- No runtime `escnn`; it remains a focused test oracle only.

## Locked Inputs And Data Contract

- RGB input/output: scalar NCHW tensors normalized to `[-1,1]`.
- Deployment input: `(B,3,256,256)`; fixed local mechanics tests may use a
  smaller spatial grid while preserving all 43 learned positions.
- Latent: scalar `(B,16,32,32)` at deployment; `mu`, `logvar`, `z`, and `eps`
  share that shape.
- Hidden layouts: `A=16F0+16F1`, `B=24F0+24F1`, `C=32F0+32F1`,
  `D=48F0+48F1`; packed widths `48,72,96,144`.
- Stem: selected `9-low`; all remaining learned convolutions: selected `7-low`.
- Downsample: fixed fieldwise 5x5 binomial blur plus stride-two decimation.
- Upsample: fieldwise bilinear x2 with `align_corners=False`.
- Posterior clamp and reparameterization exactly match the normal VAE.

## Exact 43-Convolution Topology

Every row below is a learned steerable convolution and executes exactly one
dense `conv2d`. Fixed grouped resampling convolutions are not learned positions.

| # | normal-VAE position | fixed map |
| ---: | --- | --- |
| 1 | stem | `R->A` |
| 2-3 | encoder block 0 main 1/2 | `A->A`, `A->A` |
| 4-5 | encoder block 1 main 1/2 | `A->A`, `A->A` |
| 6-8 | encoder transition 2 main 1/2, skip | `A->B`, `B->B`, `A->B` |
| 9-10 | encoder block 3 main 1/2 | `B->B`, `B->B` |
| 11-13 | encoder transition 4 main 1/2, skip | `B->C`, `C->C`, `B->C` |
| 14-15 | encoder block 5 main 1/2 | `C->C`, `C->C` |
| 16-18 | encoder transition 6 main 1/2, skip | `C->D`, `D->D`, `C->D` |
| 19-20 | encoder block 7 main 1/2 | `D->D`, `D->D` |
| 21-22 | posterior mean/log-variance heads | `D->L`, `D->L` |
| 23 | latent projection | `L->D` |
| 24-25 | decoder block 0 main 1/2 | `D->D`, `D->D` |
| 26-27 | decoder block 1 main 1/2 | `D->D`, `D->D` |
| 28-30 | decoder transition 2 main 1/2, skip | `D->C`, `C->C`, `D->C` |
| 31-32 | decoder block 3 main 1/2 | `C->C`, `C->C` |
| 33-35 | decoder transition 4 main 1/2, skip | `C->B`, `B->B`, `C->B` |
| 36-37 | decoder block 5 main 1/2 | `B->B`, `B->B` |
| 38-40 | decoder transition 6 main 1/2, skip | `B->A`, `A->A`, `B->A` |
| 41-42 | decoder block 7 main 1/2 | `A->A`, `A->A` |
| 43 | raw RGB output head | `A->R` |

The derived signature/count cross-check is locked:

| signature | occurrences | coefficients each | subtotal |
| --- | ---: | ---: | ---: |
| `R->A` | 1 | 624 | 624 |
| `A->A` | 7 | 7,680 | 53,760 |
| `A->B` | 2 | 11,520 | 23,040 |
| `B->B` | 6 | 17,280 | 103,680 |
| `B->C` | 2 | 23,040 | 46,080 |
| `C->C` | 6 | 30,720 | 184,320 |
| `C->D` | 2 | 46,080 | 92,160 |
| `D->D` | 7 | 69,120 | 483,840 |
| `D->L` | 2 | 7,680 | 15,360 |
| `L->D` | 1 | 7,680 | 7,680 |
| `D->C` | 2 | 46,080 | 92,160 |
| `C->B` | 2 | 23,040 | 46,080 |
| `B->A` | 2 | 11,520 | 23,040 |
| `A->R` | 1 | 480 | 480 |

This gives exactly `1,172,304` coefficients. The full learned total must be
`1,172,304 + 3,600 norm + 4,096 gate + 35 scalar bias = 1,180,035`.
If the instantiated model differs, diagnose construction/counting rather than
changing any locked width, support, layer, bias, norm, or gate.

## Architecture Contract

- Add one `SO2VAE` with the baseline-compatible public contract:
  `latent_channels=16`, `encode`, `decode`, `reparameterize`, and
  `forward(inputs, eps=None) -> VaeForwardOutput`.
- Use eight encoder and eight decoder residual blocks in the exact order above.
- Identity skips occur only within one layout. Stage changes use the accepted
  branch order and one learned projection on the skip.
- Reuse the accepted Spec 0013 modules and pair banks. F01 hidden maps retain
  padded `torch.bmm` plus direct quadrant assembly; scalar-boundary maps retain
  fixed `torch.mm`; every learned map calls one dense `conv2d`.
- Preserve FP32 master parameters/buffers, FP32 norm/radial statistics, and the
  selected autocast behavior. Do not call `.half()` or convert buffers in forward.
- Keep all 40 normalization and 34 gate positions. Hidden learned convolutions
  have no bias; the two scalar posterior heads have 16 biases each; the final
  RGB head has three zero-initialized biases and zero coefficients.
- Decoder output is raw RGB with no final `tanh` or clamp.
- The three encoder transitions contain two branch-local downsamplers each and
  the three decoder transitions contain two branch-local upsamplers each:
  exactly six fixed downsamplers and six fixed upsamplers, before the learned
  main/skip maps in the Spec 0013 order. No shared or post-add resampling is legal.
- Extend the existing semantic optimizer grouping for `FixedF01RadialGate` and
  the fixed SO(2) coefficient modules. All four `a,b` vectors per gate use zero weight decay
  and the locked `0.5` LR multiplier. Every learned SO(2) coefficient tensor
  receives the same ordinary AdamW decay as a baseline `Conv2d.weight`; norm
  and scalar bias parameters remain no-decay. Verify both F0 and F1 families
  directly in this slice; selected-runtime gate-health telemetry remains part
  of the later training-readiness integration because that runner is still
  specific to the normal VAE.
- Keep implementation private and direct. Construction helpers may accept only
  the already locked stage layouts needed to avoid duplicating block code.

## Outputs And Acceptance Artifacts

- `src/eqvae/models/so2_vae.py`: the complete fixed model and builder.
- `tests/test_so2_vae.py`: focused topology/count/shape/numerical/compile tests.
- Existing Spec 0013 primitive source remains the singular mechanics implementation.
- `CURRENT.md`, this spec, and `docs/specs/README.md` record final evidence and next step.

## Acceptance Criteria

1. The instantiated model has exactly 43 learned steerable convolutions with the
   signature occurrences above and exactly `1,180,035` learned parameters.
2. The coefficient/norm/gate/bias partition is exactly
   `1,172,304/3,600/4,096/35`.
3. Encoder/decoder stage order and deployment shapes are
   `256->128->64->32` and `32->64->128->256`; deployment posterior/output
   shapes are `(B,16,32,32)` and `(B,3,256,256)`.
   The instantiated model contains exactly six branch-local fixed downsamplers
   and six branch-local fixed upsamplers in the locked pre-convolution order.
4. `encode`, controlled-epsilon reparameterization, `decode`, and full forward
   match the normal-VAE external contract; raw/clamped log-variance semantics
   and clamp count are preserved.
5. Reduced fixed-shape full-model SO(2) checks report full-frame, cropped, and
   raw-transform-floor errors at fixed cardinal and non-cardinal angles for
   encoder `mu`/raw `logvar`, controlled transformed-epsilon sampling, decoder,
   and reconstruction. The accepted downsample phase error is reported
   separately; tolerances are frozen from the correctly wired implementation
   and must detect F1 sign/order or stage-composition mutations.
6. A reduced fixed-shape eager CPU full-model forward/backward and optimizer
   proof completes with finite loss, gradients, and parameters. Because the
   zero RGB head blocks first-step upstream reconstruction gradients, the proof
   must use zero weight decay and demonstrate named nonzero gradient-driven
   updates in the output head first, then in representative decoder, posterior,
   encoder, and stem coefficient/gate parameters after the head becomes nonzero.
   Every radial-gate parameter is covered once by the gate no-decay group at
   the locked `0.5` LR multiplier, and direct bounded checks cover both its F0
   and F1 gate families in FP32.
7. CPU autocast-facing execution remains finite and keeps FP32 master
   parameters/buffers plus FP32 norm/radial arithmetic.
8. `torch.compile(backend="eager", fullgraph=True, dynamic=False)` captures a
   reduced fixed-shape full model-plus-loss forward; backward is invoked outside
   that graph. Repeat the identical fixed contract and fail on a recompile.
   When CPU autocast is supported, the same fullgraph invocation is exercised
   under explicit autocast while master parameters/buffers remain FP32. The
   selected CUDA deployment dtype remains FP16 and is not relabeled as BF16.
9. Focused tests preserve Spec 0013 buffer, dtype, padded-`bmm`, direct
   assembly, scalar-`mm`, and one-learned-`conv2d` contracts across the full model.
10. Required Ruff, BasedPyright, full Python quality, repo/workspace preflight,
   `git diff --check`, and two fresh read-only adversarial reviews pass.
11. Run at most a narrow generated-data dual-T4 compile/VRAM/settled-execution
    check if local evidence leaves a concrete hardware-dependent question. Do
    not create a tuner or start training.

## Tests And Verification Commands

```bash
.venv/bin/pytest -q tests/test_so2_architecture_probe.py tests/test_so2_architecture_probe_kernel.py tests/test_so2_vae.py
.venv/bin/ruff check src/eqvae/models/so2_architecture_probe.py src/eqvae/models/so2_vae.py tests/test_so2_vae.py
.venv/bin/ruff format --check src/eqvae/models/so2_architecture_probe.py src/eqvae/models/so2_vae.py tests/test_so2_vae.py
.venv/bin/basedpyright src/eqvae/models/so2_architecture_probe.py src/eqvae/models/so2_vae.py tests/test_so2_vae.py
./scripts/python_quality.sh
./scripts/agent_preflight.sh
../agent_preflight.sh
git diff --check
```

## Verification Evidence

- The implementation starts from HEAD `c110a7d`; that commit is the current
  HEAD and therefore an ancestor. The initially clean worktree contained no
  unrelated changes to preserve.
- The live model contains exactly 43 learned convolutions and
  `1,180,035` parameters, partitioned as `1,172,304` coefficients, `3,600`
  normalization parameters, `4,096` gate parameters, and `35` scalar biases.
  It contains 40 norms, 34 gates, six branch-local downsamplers, and six
  branch-local upsamplers. Meta execution proves deployment shapes
  `(B,3,256,256) -> (B,16,32,32) -> (B,3,256,256)`.
- A reduced full forward performs exactly 43 learned dense `conv2d` calls, 38
  padded `torch.bmm` expansions, 10 scalar-boundary `torch.mm` contractions,
  and six fixed grouped downsampling convolutions. Spec 0013 tests remain the
  direct buffer/dtype/assembly oracle.
- The zero-head proof uses zero weight decay: step one updates the output head;
  step two produces finite nonzero gradients and parameter changes in named
  decoder, posterior, encoder, stem, and F0/F1 gate parameters. CPU BF16
  autocast keeps all master parameters/buffers FP32. Both base-FP32 and
  autocast full-model loss forwards capture with `fullgraph=True`, invoke
  backward outside the graph, and repeat with `error_on_recompile=True`.
- Frozen sampled scalar endpoint evidence is:

  | angle | `mu` full/crop | raw `logvar` full/crop | controlled `z` full/crop | chained decoder full/crop | independent decoder full/crop | raw floor | downsample phase full/crop |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 30 deg | `0.38441 / 0.26962` | `0.27565 / 0.20070` | `0.32820 / 0.25596` | `0.47516 / 0.44568` | `0.43155 / 0.40574` | `0.03089` | `0.10082 / 0.09005` |
  | 90 deg | `0.43311 / 0.44324` | `0.33012 / 0.32356` | `0.35546 / 0.36669` | `0.44575 / 0.45799` | `0.0000031 / 0.0000032` | `0.0` | `0.22181 / 0.21777` |

  Full-forward reconstruction equals the chained controlled-epsilon decoder
  values. The independent decoder isolates scalar-latent composition; the raw
  floor and accepted 5x5 decimation phase are bounded separately.
- Focused verification passes 84 tests across the analytic basis, primitive,
  kernel, full-model, and optimizer suites; the final full-model file passes 11
  tests. Fresh mathematical/topology and compile/performance/scope reviews
  found no model defect after their acceptance-test findings were corrected.
- Final `./scripts/python_quality.sh` passes Ruff, 767 tests with one expected
  GPU-only skip, and BasedPyright with zero errors. Repo/workspace preflights
  and `git diff --check` pass.
- No Kaggle probe was launched. Local evidence leaves no unresolved assembly
  question; a meaningful dual-T4 measurement first requires the separately
  scoped selected-runtime registration and SO(2) gate-health integration.

## Implementation Blockers

None for local assembly. Before any training-readiness or remote execution
slice, the fixed model must be registered in the selected runtime and its F0/F1
families added to gate-health telemetry. Full training remains separately
unauthorized.

## Known Risks

- The 4.383x learned-convolution MAC cost may make deployment batch/VRAM or
  settled throughput differ materially from the normal VAE.
- A locally compiled reduced spatial shape does not prove dual-T4 deployment
  memory or performance; answer only concrete remaining questions remotely.
- Reusing private Spec 0013 primitives is intentionally one-off. Duplicating or
  generalizing them would create a second mechanics path and is forbidden.
- The retained downsampler has the accepted sampled-grid phase error.

## Adversarial Checks

- Recount the 43 positions independently from `NonEquivariantVAE`, including
  all six transition skip projections and excluding fixed resampling.
- Recount live parameter objects by role and ensure no tied coefficient rows or
  duplicate/extra affine parameters.
- Inspect every full-model learned map for exactly one dense `conv2d`, and every
  hidden/scalar-boundary map for the locked contraction path.
- Verify stage-changing branch ordering and identical-layout residual adds.
- Verify zero raw RGB start, no final bounding function, controlled epsilon,
  logvar clamp, finite gradients, and an actual optimizer update.
- Review the diff read-only for generalized configuration, fallback, runtime,
  training, or portability scope creep.

## Open Questions

None for local assembly. Selected-runtime model/telemetry integration and actual
dual-T4 full-model batch size and settled epoch time remain later narrow Spec
0011 readiness work before training.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`
- `docs/specs/0012-continuous-so2-vae-architecture.md`
- `docs/specs/0013-fixed-f01-architecture-probe.md`
- `docs/decisions/0004-so2-gaussian-ring-kernel-basis.md`
- `src/eqvae/models/non_equivariant_vae.py`
- `src/eqvae/models/so2_architecture_probe.py`
