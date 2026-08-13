# Spec 0015: Fixed SO2 Selected-Runtime Readiness

Status: complete / private Kaggle v1 passed
Implementation readiness: fixed model ready at the single batch-1 coordinate; full training remains unauthorized
Owner/workstream: fixed continuous-`SO(2)` selected-runtime integration
Last updated: 2026-08-13

## Purpose

Register the one fixed Spec 0014 `SO2VAE`, make it selectable by the existing
Spec 0011 runner, add architecture-specific F0/F1 radial-gate evidence, and run
one private generated-data dual-T4 readiness coordinate. This slice answers
whether the already selected runtime can execute the fixed model safely; it
does not select a new runtime or train on UBC data.

## Locked Scope

- Add exactly one registry kind, `so2_vae_fixed`, whose builder accepts no
  architecture options and returns the exact Spec 0014 model.
- The selected runner reads `model.kind`; the existing normal kind and its
  artifacts, telemetry, configuration, and execution semantics stay unchanged.
- Fail closed on model kind, concrete class, latent width, 43 learned
  convolutions, 34 radial gates, and `1,180,035` learned parameters.
- Reuse the selected FP16, Inductor, compiled-autograd, Python-reducer DDP,
  channels-last, fused-AdamW, clipping, scaler, zero-grad, loss, KL, and
  reparameterization implementation. Do not create another training recipe.
- Preserve the established optimizer partition: SO2 coefficient tensors use
  ordinary AdamW decay; every F0/F1 radial-gate `a,b` tensor uses zero decay and
  `0.5x` learning rate; normalization and scalar biases remain no-decay.
- Emit two evidence rows for every `FixedF01RadialGate`, one for the scalar F0
  family and one for the vector-radius F1 family: exactly 68 rows. Existing
  normal `GatedScalarActivation` telemetry is untouched.
- The remote coordinate is fixed at per-device batch 1, two T4 ranks, generated
  device-resident `B,3,256,256` tensors, and no dataset source. No sweep,
  fallback, tuner, alternate model, or alternate runtime is permitted.

## Runtime And Evidence Contract

The readiness executable must prove all of the following or fail:

1. Requested and effective runtime settings match the selected Spec 0011
   bundle, including rank-local CUDA device assignment, FP16 autocast, FP32
   master parameters/buffers, FP32 normalization/radial arithmetic, compiled
   autograd, and the selected DDP flags.
2. The actual selected model-plus-corruption-plus-VAE-loss closure compiles and
   performs scaled backward, unscale, foreach clipping, fused AdamW step,
   scaler update, skip accounting, and `set_to_none` zeroing.
3. Rank-distinct inputs produce differing local pre-reduction gradients; DDP
   gradients match an explicit cross-rank mean and parameters remain
   synchronized after updates.
4. The zero RGB head updates first. A subsequent successful update changes
   named decoder, posterior, encoder, stem, F0-gate, and F1-gate parameters.
5. Compilation settles, followed by a single diagnostic fixed-coordinate
   measurement with zero post-settlement graph breaks and recompiles. Record
   diagnostic settled step time plus peak allocated/reserved VRAM and headroom;
   do not extrapolate a training claim.
6. The artifact contains exactly 68 finite radial-family rows with actual gate,
   saturation, input/output RMS, gradient, update, and dtype evidence.

## Outputs

- Minimal registry/runner integration and focused tests.
- `src/eqvae/benchmarking/so2_runtime_readiness.py` with its one-use CLI entrypoint.
- One private no-dataset Kaggle kernel and guarded workflow route.
- One rank-zero JSON readiness artifact and one 68-row gate-health CSV.
- `CURRENT.md` and this spec record the exact result and next separately
  authorized action.

## Acceptance Criteria

1. Registry tests prove the fixed identity and reject unknown SO2 options.
2. Runner tests prove explicit model-kind selection and no normal-path telemetry
   change.
3. Optimizer tests cover all coefficient and both radial-gate families exactly
   once with the locked decay/LR policy.
4. Readiness-unit tests prove identity/policy rejection, 68-row coverage,
   zero-head then upstream update checks, and artifact verdict fail-closedness.
5. The kernel has no data sources, embeds the exact committed tree, and is
   guarded separately from prior architecture probes and training kernels.
6. Focused tests, `./scripts/python_quality.sh`, repo/workspace preflights,
   `git diff --check`, and two fresh independent read-only adversarial reviews
   pass after findings are resolved and rechecked.
7. Launch at most one private dual-T4 run. One corrected rerun is allowed only
   for a narrow implementation defect; otherwise stop and report the blocker.
8. Do not read UBC data, begin real training, alter architecture/search choices,
   mutate historical normal artifacts, update paper claims, or close issues.

## Remote Stop Rule

If the guarded Kaggle run remains active beyond the normal observation window,
record its kernel/version/status and exact continuation command in `CURRENT.md`,
stop polling, and ask the user to say `continue`. A completed failure is not
permission to broaden or tune the probe.

## Verification Commands

```bash
.venv/bin/pytest -q tests/test_model_registry.py tests/test_so2_runtime_readiness.py tests/test_selected_runtime_runner.py tests/test_so2_runtime_readiness_kernel.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/basedpyright src tests
./scripts/python_quality.sh
./scripts/agent_preflight.sh
../agent_preflight.sh
git diff --check
```

## Hardware Result

Private Kaggle v1 from clean commit `6cdccb0` passed the strict remote and local
validators. On two Tesla T4s with Torch `2.13.0+cu130` / CUDA `13.0`, the exact
selected compiled runtime completed batch 1 per rank with zero AMP skips,
nonfinite losses/parameters, settled graph breaks, recompiles, buffer drift,
DDP mean error, or parameter drift. The zero head and subsequent named
decoder/posterior/encoder/stem/F0/F1 updates pass. All 68 actual gate-family
rows pass with positive finite gradient/update evidence and no dead channels.
Diagnostic settled step median is `132.285 ms`; peak allocated/reserved memory
is `410.016/538 MiB`, leaving `96.392%` reserved headroom. This answers only
the bounded readiness question and is not a runtime-selection or training claim.

## Local Verification Result

The implemented slice passes the post-review repository gate: Ruff
format/check, 780 tests with one expected GPU-only skip, and BasedPyright with
zero errors. The embedded private dual-T4 kernel preflight and `git diff
--check` pass. Two independent clean-context reviews found and drove fixes for
complete selected-plan pinning, exact executed F1 radius/dtype semantics,
per-family positive finite gradients and updates, two-rank graph/recompile
aggregation, strict proof/rank/module artifact validation, generated-launcher
ignore policy, and adversarial mutation coverage. Both reviewers rechecked the
final fixes with zero unresolved findings.

The compact hash-bound evidence is
`docs/data/spec0015_so2_runtime_readiness_v1.json`; raw downloaded artifacts
remain ignored under `runs/kaggle/so2_runtime_readiness_v1`. No corrected rerun
was needed and no training run was launched.

## Related Files

- `docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`
- `docs/specs/0013-fixed-f01-architecture-probe.md`
- `docs/specs/0014-fixed-f01-full-vae.md`
- `src/eqvae/models/registry.py`
- `src/eqvae/training/selected_runtime_runner.py`
- `src/eqvae/training/optim.py`
