# 0009: Fixed-25 embedding-equivariance evaluation proxy (rot90), decoupled from training

Status: settled (2026-07-01)

## Context

A past agent conflated two orthogonal things and it caused real confusion. This
note pins the distinction so it does not recur.

## Decision

1. The fixed-25 rotated-input / rotated-embedding protocol is an
   **evaluation / inspection tool only**. It is NOT part of training: no loss
   term, no data augmentation, no effect on the objective. It is measurement run
   at eval points (e.g. half-epoch boundaries, as in the FSQ reference
   `kaggle/train_runs:1010-1122`, and re-runnable post-hoc on any checkpoint).

2. Its purpose is to measure **embedding-space equivariance under rotation as a
   proxy for embedding-space smoothness / structure** (the EQ-VAE insight,
   arXiv:2502.09509: a latent space that transforms consistently under spatial
   transforms is smoother and better structured). We use it to **compare** two
   otherwise-matched autoencoders on identical inputs:
   - the non-equivariant, non-steerable baseline AE (built first), and
   - the future `SO(2)`-steerable, equivariant AE.
   The hypothesis is that the steerable model has a more equivariant (smoother,
   more structured) embedding — the "good change" the paper motivates.

3. **`SO(2)` steerability is an architectural property of the convolutions of the
   equivariant model. It is NOT a property of the evaluation and NOT an angle
   set.** The evaluation rotation used for the shared, comparable proxy is exact
   `torch.rot90` at `{0, 90, 180, 270}` degrees — discrete, an exact spatial
   permutation, so zero interpolation artifacts (the paper's "multiples of 90
   degrees to avoid corner artifacts", §3.3; the FSQ notebook uses exactly this).
   Using `rot90` for the evaluation does NOT conflict with the continuous-`SO(2)`
   architecture goal (`docs/decisions/0001-continuous-so2-scope.md`); they are
   different layers. This is NOT a deviation from any requirement.

4. The 25 images are a **single frozen canonical selector from the validation
   set** (`configs/spec0001/fixed_25_validation_patches.json`, schema
   `spec0001.fixed_selector.v1`; the FSQ notebook likewise drew its 25 from the
   validation set, `kaggle/train_runs:827-830`). The SAME 25 images are used for
   **every evaluation of BOTH models**, so all visual and numerical comparisons
   are on identical inputs. This canonical per-label selector (5 per label,
   digest-sorted) is stricter and more reproducible than the FSQ "first 25 of the
   val loader".

5. Per fixed image and per rotation `k in {1, 2, 3}`, the protocol saves and
   compares: the clean reconstruction `D(E(x))`, the rotated-input reconstruction
   `D(E(rot90 x))`, and the rotated-embedding reconstruction `D(rot90 E(x))` —
   plus the normalized latent equivariance error and the PCA latent maps. If the
   embedding is equivariant, the last two match; the comparison across the two
   models is the deliverable, not a pass/fail on the baseline.

6. Continuous-angle inspection MAY be added for the `SO(2)` model later (its
   steerable convs make arbitrary-angle rotation meaningful); that is a future
   extension, not required for the shared comparison
   (`docs/issue_image_inventory.md:38`, "continuous angles for the `SO(2)`
   model").

## Consequences

- Spec 0010 implements this protocol for the non-equivariant baseline; the future
  `SO(2)` spec reuses the identical protocol and the identical fixed 25 images.
- Do not describe `rot90` evaluation as the model's symmetry, and do not describe
  it as a training-time transform. Do not frame the baseline's higher equivariance
  error as a bug; it is the expected contrast.
