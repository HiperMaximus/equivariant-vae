# Spec 0051: Decoded Latent Transform Consistency

Status: locked / implementation-ready
Implementation readiness: remote execution authorized; no result has been viewed
Owner/workstream: fixed-validation post-hoc decoder-side transformation audit
Last updated: 2026-09-09

## Purpose

Test the user's result-independent observation that spatially transforming the
final posterior mean may decode cleanly in the frozen SO(2) VAE even though
Spec 0050 did not demonstrate a shared low-dimensional action in raw `mu`.
The experiment separates two questions:

1. **Decoded action consistency:** does
   `D(rho_T E(x)) ~= T D(E(x))`?
2. **Decoded inverse canonicalization:** does
   `D(rho_T^-1 E(Tx)) ~= D(E(x))`?

Here `rho_T` is the prescribed sampled-grid spatial candidate on the
`16x32x32` final F0 fields; only the exact D4 controls are literal group
actions on the discrete arrays.
This is not a new fitted action. Both frozen VAEs receive the identical action;
for SO(2) rotations it is architecture-prescribed, while for the normal VAE
and every reflection it is a prespecified control.

## Hard Boundaries

- Use only the accepted frozen 60,000-update checkpoints and SHA-pinned fixed
  validation 25. Do not train, tune, sample `z`, mutate checkpoints, or access
  sealed test data.
- Do not update the professor report, paper, thesis, Overleaf, GitHub issue,
  commit, or push Git without separate authorization.
- Do not reinterpret decoder agreement as raw-latent equivariance, a shared
  learned action, disentanglement, factorization of `M`, or a global product.
- Ranks 0 and 12 are the only qualitative examples and are fixed before output.
- This spec owns one private inference-only Kaggle launch. It does not reuse
  the consumed Spec 0038 or Spec 0050 authority and permits no automatic retry.

## Fixed Inputs And Operators

The machine contract is
`docs/data/spec0051_decoded_transform_contract.json`. It pins the checkpoint,
state bundle, selector and ordered patch bytes, operator, metric,
threshold, output and remote-execution identities.

- Input: FP32 `3x256x256`, normalized as `uint8/255*2-1`.
- Posterior: deterministic raw FP32 `mu`, shape `16x32x32`.
- Dense rotations: `0,5,...,355` degrees through Spec 0050's corrected uniform
  bilinear zero-padded operator. No cardinal `rot90` splice occurs in this path.
- Exact controls: the complete eight-element D4 group
  `{e,r,r²,r³,s,rs,r²s,r³s}`, with `r=torch.rot90(+1)` and
  `s=flip_h`. The identity is an exact sanity check rather than a ratio endpoint;
  the seven nonidentity elements are measured on input, latent, and
  reconstruction tensors. In named tensor operations, `rs=flip_diag`,
  `r²s=flip_v`, and `r³s=flip_anti_diag`. The identity
  `flip_v(flip_h(x)) == rot180(x)` is a required algebraic check.
- Reflections are O(2)/D4 controls, not guarantees of an SO(2) architecture.
- Primary image mask: centered radius 112 disk. Full-image results, a centered
  224-square view and a radius-110 eroded disk for finite differences are
  secondary boundary/interpolation controls.
- Raw-latent comparison mask: centered radius-14 disk on `32x32` `mu`.

For a transform `T`, define `z0=E(x)`, `y0=D(z0)`, `zT=E(Tx)`:

- action output `y_action(T)=D(rho_T z0)`;
- action target `y_action_ref(T)=T y0`;
- transformed-input reconstruction `y_input(T)=D(zT)`;
- canonical output `y_can(T)=D(rho_T^-1 zT)`;
- canonical target `y0`;
- output-space control `y_oracle(T)=T^-1 y_input(T)`.

The oracle is only an image-domain reference/control, not a bound; it is not latent
canonicalization. For rotations, wrong-sign controls replace `rho_T` with
`rho_T^-1`. For reflections, the inverse is the same operation, so instead of
an artificial sign control the identity/no-action baselines are used.
The end-to-end path `D(E(Tx))` is also compared directly with `T D(E(x))`,
especially for every exact reflection, to answer whether the input transform
itself changes autoencoder behavior before any latent intervention.

## Metrics And Statistical Unit

For each model and patch, first aggregate squared error across all non-identity
angles and masked scalars, then take the square root. The primary ratios are:

`A_i = RMS(y_action-y_action_ref) / (RMS(y0-y_action_ref)+1e-8)`

`C_i = RMS(y_can-y0) / (RMS(y_input-y0)+1e-8)`.

The descriptive end-to-end commutation ratio is
`I_i = RMS(y_input-Ty0) / (RMS(y0-Ty0)+1e-8)`.

Thus `1` is the prespecified identity/no-action baseline and lower is better.
A patch is action-nondegenerate only when `RMS(y0-Ty0)>=0.02`, and is
canonicalization-nondegenerate only when `RMS(y_input-y0)>=0.02`, in the
normalized `[-1,1]` image domain. Degenerate patches remain in every table and
figure but do not support a ratio claim. Dense action and canonicalization
claims each require at least 18 valid patches and apply all medians, success
fractions and model comparisons to their corresponding SO(2)-valid ranks.
An exact-D4 transform is assessable only when at least 18 patches are valid for
both denominators; its rules apply to that intersection.
The aggregate exact-quarter corroborators likewise compute their own action and
canonicalization denominator signals over 90/180/270 degrees. H1/H2 require at
least 18 ranks in the intersection of the corresponding dense-valid set and
this exact-quarter joint-valid set; otherwise the corroborator is unresolved,
not a successful near-zero ratio.

Also report disk/full RMS and MAE, centered-crop SSIM on values clamped to
`[-1,1]`, pre-clamp out-of-range fraction and unconditional mean excess beyond
`[-1,1]` (zeros retained for in-range scalars), eroded-disk
gradient discrepancy to the corresponding target, metrics by angle, exact D4
controls, and dense results under 10/20-degree subsampling and after excluding
angles within 5 degrees of a cardinal. Raw latent inverse-action RMS is retained
to test whether decoder agreement can coexist with a raw-latent mismatch.

The 25 patches come from 16 WSIs, so patches are fixed-set reduction units, not
independent biological samples; angles are repeated measurements. Report all
patch pairs, medians and favorable counts. For uncertainty, first reduce each
metric's paired SO(2)-minus-normal patch differences to one median per WSI,
then use a fixed-seed 10,000-draw percentile bootstrap over the 16 WSI medians.
These are descriptive fixed-validation intervals, not population confidence
intervals or p-values.

An artifact contradiction is operational rather than visual: among the valid
ranks, median candidate-minus-target out-of-range fraction must be `<=0.01`,
median candidate-minus-target mean overshoot must be `<=0.002`, and median
gradient-error RMS ratio versus the matching identity/no-action baseline must
be `<=1.0`. H1/H2 cannot pass if their corresponding artifact checks fail.

## Preregistered Hypotheses

| Hypothesis | Support rule |
| --- | --- |
| H1 decoded rotational action | At least 18 action-valid patches; on those ranks, disk `A_i` median `<=0.50`, at least 75% `<=0.75`, SO(2) median at least 20% below normal, and SO(2) favorable in at least 75%. On at least 18 dense/exact jointly valid ranks, the exact-quarter aggregate median must be `<=0.75`; dense action SSIM gain must be strictly positive in at least 75% of the action-valid ranks; all action artifact gates must pass. |
| H2 decoded rotational canonicalization | At least 18 canonicalization-valid patches; on those ranks, SO(2) median `C_i<=0.50`, at least 75% `<=0.75`, SO(2) median at least 20% below normal, and SO(2) favorable in at least 75%. On at least 18 dense/exact jointly valid ranks, the exact-quarter aggregate median must be `<=0.75`; dense canonicalization SSIM gain must be strictly positive in at least 75% of the canonicalization-valid ranks; all canonicalization artifact gates must pass. |
| H3 decoded residual suppression | Descriptive support requires H2, a median raw-latent canonicalization ratio `>0.75` on the same H2-valid ranks, and at least 75% of those ranks jointly having raw ratio `>0.75` and decoded ratio `<=0.50`; this means the decoded outputs suppress substantial finite residual differences, not that a decoder null space was identified or that raw `mu` factorizes. |
| H4 reflection robustness | Reported separately for both models and every exact reflection. No architecture-specific claim is allowed. A transform is assessable with at least 18 jointly valid patches and is robust for a model only if its end-to-end input, latent-action and canonical ratios each have median `<=0.50` and at least 75% of valid patches `<=0.75`. Classify both-model success as generic, not SO(2)-specific. |

If fewer than 18 denominators clear the threshold, the corresponding hypothesis
is declared unresolved due to insufficient pose signal even if images look stable.
If both models pass similarly, the result is generic decoder robustness rather
than an SO(2)-specific advantage. If only selected examples look clean, no
population claim passes. If image-domain agreement is good but overshoot or
gradient discrepancy is worse, the qualitative claim is explicitly qualified.

## Geometric Interpretation Contract

- **Guaranteed by architecture:** the SO(2) hidden field types and candidate
  spatial F0 action; not empirical decoder commutation and not reflections.
- **Empirically verified here:** only fixed-25 decoded action/canonicalization
  endpoints that pass the locked rules.
- **Potentially emergent after training:** decoder insensitivity to directions
  in which raw inverse-action latents disagree.
- **Transfer across patches:** these formulas contain no fitted per-patch
  parameters; population consistency is assessed on all 25, but this fixed set
  cannot establish external generalization.
- **Local bundle interpretation:** successful decoded canonicalization alone is
  compatible with removal of pose at the decoder output, but does not establish
  a latent section, shared effective generator, or content-space quotient.
- **Speculative/impossible here:** causal architectural attribution, training-
  seed robustness, population generalization, topology, or global product.

## Outputs And Acceptance

Persist a compact package under `decoded_latent_transform_v1` containing:

1. machine-readable summary with all 25 paired rows, exact D4 rows, hashes,
   code/payload/runtime identity and preregistered decisions;
2. population action/canonicalization ratio figure;
3. all-angle median/IQR figure;
4. rank-0/rank-12 decoded-action grid;
5. rank-0/rank-12 inverse-canonicalization grid;
6. exact-D4 summary figure;
7. raw-latent-versus-decoded discrepancy figure and selected compact arrays.

Before launch: focused unit tests, Ruff/format/BasedPyright on changed Python,
kernel build/verify, metadata/source/hash checks, shell syntax, diff checks and
repo preflight must pass. The remote kernel must require a CUDA T4, be private,
disable internet, verify every input hash, remain inference-only, and claim its
one-shot authority before the push. After launch, confirm `RUNNING` once and
stop; do not poll a long job in-turn. Accepted outputs require hash verification,
schema/finite/count checks, visual inspection, focused verification and
independent clean-context scientific/statistical review.

Remote authorization guard:
`spec0051_decoded_latent_transform_authorized`.
