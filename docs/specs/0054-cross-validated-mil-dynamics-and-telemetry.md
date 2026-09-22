# Spec 0054: Cross-Validated MIL Dynamics And Telemetry Plan

Status: diagnostic A0 v3 running; five-fold training remains pending
Owner/workstream: repeated downstream evaluation of the frozen normal and
continuous-`SO(2)` VAE representations
Last updated: 2026-09-22

## Purpose

Plan a new classifier campaign that reuses the 361 non-TMA WSI used for VAE
development as the supervised classifier development pool, while keeping both
accepted VAE checkpoints frozen. The immediate objective is not to select a new
regularizer. It is to obtain enough paired, cross-validated learning-dynamics
evidence to identify why the current local-global MIL classifier is unstable
and to choose a small, defensible Phase B.

The plan preserves the Spec 0026 classifier topology initially. It adds
cross-validation, repeated trajectories, gradient accumulation probes, EMA and
tiered telemetry. AEM, MIL-Dropout, spatial subsampling, alternate optimizers
and architecture changes remain later experiments whose activation must be
justified by the measurements defined here.

This document is a scientific plan, not the execution contract. The exact
361-WSI identities, five folds, A0 sentinels/cost panel, source hashes, binary
headers and probe fold are frozen in `docs/data/spec0054_a0_contract.json`.
Training horizons, cadence and the full-run artifact budget remain to be frozen
from the bounded A0 measurement before any classifier campaign begins.

## Local Implementation Boundary

The accepted Spec 0026 classifier and whole-bag fixed-25 attention backend have
been restored from the historical executed source and extended without adding
trainable parameters. The ordinary historical `forward` remains unchanged. A
separate state-compatible entry point exposes six semantic terminals to a
compile-friendly T0 wrapper: `X0`, `X1`, `X2`, `T1`, CLS and the normalized bag
embedding. The wrapper returns the ordinary logits plus one fixed `6 x 6`
device tensor containing count, sum, squared sum, maximum magnitude, near-zero
count and nonfinite count. Means, RMS, standard deviations and fractions are
reconstructed after one compact host transfer.

The implemented local surface is deliberately narrower than the complete plan:

- T0 records per-WSI prediction/loss/margin/entropy quantities and measures
  actual unscaled gradient and committed AdamW update norms, energy shares,
  update-to-weight ratios, temporal cosines, sign flips and counterfactual clip
  coefficients for the six semantic blocks;
- T1 runs on a disposable eager checkpoint copy, immediately reduces forward
  activations and transient backward gradients at every named layer, and
  records channel dispersion, dead-channel fraction, gradient amplification,
  parameter/gradient distributions, Adam moments, effective coordinate steps,
  radial/perpendicular gradient energy and the previous actual update;
- T2-lite reconstructs local fixed-25-plus-null probabilities in chunks,
  global sigmoid gates and final CLS-to-register attention, and records
  residual/SwiGLU scales, top global patches and representation rank/collapse
  summaries without retaining full local attention maps;
- scalar telemetry is written column-wise to three small cumulative NPZ tables
  (`t0`, `t1`, `t2`) using a temporary file and atomic replacement. Sparse T2
  rows retain their capture/metric names, exact logits, ordered atlas indices,
  lattice coordinates and mean sigmoid-gate scores. On resume, rows later than
  the update restored from `latest.pt` are simply discarded;
- each run directory has one atomic `latest.pt` and, at completion, one
  `final.pt`. They contain model, AdamW, GradScaler, scheduler, RNG, exact
  order/cursor/exposure progress, WSI learning/forgetting state and the small
  temporal telemetry state. There is no object store, manifest/pointer layer or
  generic shard registry;
- the session guard retains the historical 12-hour ceiling and a fixed
  15-minute margin for writing the cumulative tables and `latest.pt`.

Focused local tests cover compact reductions including nonfinite values,
ordinary-versus-instrumented and full-graph-compiled logits, exact first-step
loss/gradient/AdamW-state equivalence, accepted attention-backend state/output
equivalence, actual AdamW parameter deltas, a real GradScaler overflow followed
by update-one retry, detailed eager non-intrusion, learning/forgetting restore,
self-describing T2-lite attention/rank/top-patch tables, atomic cumulative NPZ
replacement and uninterrupted-versus-resumed next-update equivalence.
Degenerate attention, spectrum and radial quantities carry
explicit validity flags rather than being interpreted as valid zero/one
statistics.

The deterministic five-fold manifest, class-median sentinels, median/P99/max
cost panel and bounded A0 Kaggle runner now exist locally. A0 validates the
330,000 physical patch rows and binary CRCs, reconstructs only its precommitted
bags through both frozen encoders, checks unchanged first-update semantics and
measures the same T0/T1/T2-lite panel in both branches. It does not train a
fold. Its probe update uses fold-0 training-only class weights; the fold-0
holdout cannot influence that update. Cadence, horizon, the full latent store
and five-fold training runner remain prerequisites for A1. Heavy T2
families remain future work: iterative line search, SAM, Hessian/Lanczos,
per-example full gradients, perturbation sweeps, Muon counterfactuals and
PANTHER fitting.

## Scientific Boundary

- The statistical unit is one WSI. Patches are not independent observations.
- The normal and continuous-`SO(2)` VAE checkpoints remain frozen. Only the
  Spec 0026 classifier is trained.
- The candidate development population contains 361 WSI: the 322 VAE-training
  WSI and 39 VAE-validation WSI. Their VAE source role is retained as a
  covariate and fold-balancing variable.
- Reusing these WSI is valid for the supervised classifier because the frozen
  VAE did not consume diagnosis labels. Nevertheless, CV on this population
  estimates classifier generalization conditional on a representation learner
  that had unsupervised/transductive access to the same images; it does not
  estimate a complete VAE-plus-classifier pipeline on wholly unseen WSI.
- The separate 152-WSI masked cohort is excluded from all new fitting, fold
  creation, monitoring, threshold selection, debugging and hyperparameter
  decisions.
- The 152 WSI are a held-out cohort relative to this new classifier fit, but
  they are not a pristine never-observed test population for the research
  program: 106/23/23 of them previously served as classifier train,
  validation and test, and their aggregate outcomes have already been seen.
  A later evaluation on all 152 must therefore be described as a historically
  exposed external comparison, not as a newly sealed confirmatory test. A
  genuinely confirmatory claim requires a new independent cohort.
- Every fold, seed, initialization, WSI order, diagnostic schedule and
  selection rule is shared between the two latent branches. A choice may not
  be made because it favors one representation.
- Any normalization, class weighting, clustering, prototype, noise scale or
  calibration statistic is fitted using only the training portion of the
  relevant fold.

## Questions To Answer Before Phase B

1. How much variability comes from fold membership, initialization, WSI order
   and latent representation?
2. Does accumulation over multiple WSI reduce gradient noise and validation
   volatility, and what effective batch is supported by the observed gradient
   covariance?
3. Are unstable updates caused by a few WSI, classes, bag sizes or spatial
   graph regimes?
4. Does attention collapse onto a small and redundant patch set, or is sparse
   evidence genuinely necessary for some classes?
5. Do patch or register representations collapse, oversmooth or become
   dominated by bag cardinality?
6. Does train classification interpolate while CE, parameter norms or logit
   margins continue increasing without OOF improvement?
7. Are errors driven by poor calibration, class imbalance, representation
   overlap or sensitivity to partial views of the WSI?
8. Which future intervention has a measured target: AEM, MIL-Dropout,
   spatial subsampling, spatial smoothness, SC-MIL, clipping, stronger decay,
   SAM, Muon, PEGR, orthogonal-gradient projection, StableMax, adaptive
   per-step step sizes or a PANTHER-like prototype representation?

## Cohort Audit And Fold Construction

Before model execution, create one immutable cohort table containing at least:

- WSI identity, diagnosis and diagnosis index;
- patient/case identity and acquisition site when available;
- VAE source role (`vae_train` or `vae_validation`);
- image-update/replacement indicator when applicable;
- number of patches, `log(N)`, coordinate bounding box, occupied lattice area
  and graph degree summaries;
- exact latent and logical-dataset source identities for both representations;
- missingness, duplication and cross-cohort overlap checks.

If multiple WSI can belong to one patient or case, all such WSI must remain in
one fold. If no group field exists, the absence must be recorded rather than
silently assuming independence.

Generate folds without using latent values, predictions or learned features.
The primary constraint is diagnosis stratification. Secondary balance targets
are patient/site group, VAE source role, image-update status and quantiles of
`log(N)`. Secondary targets may improve balance but may never break group or
class constraints.

### Primary five-fold partition

Five-fold grouped stratified CV is the principal exploratory design:

- each training split contains approximately 288--289 WSI;
- each holdout contains approximately 72--73 WSI;
- every WSI produces exactly one OOF prediction per seed and checkpoint
  boundary;
- the primary OOF macro-F1 is calculated once over the concatenated 361 OOF
  predictions, not as the unweighted mean of five noisy fold-level macro-F1
  values;
- fold-level scores remain useful for heterogeneity and failure analysis.

The folds must make the minority-class support explicit. If the grouping
constraints prevent every fold from containing all five classes, fold count or
the estimand must be reconsidered before training.

More generally, guaranteeing at least `q` validation WSI from every class in
every `K`-fold partition requires at least `K*q` WSI in the smallest class
before group constraints. Thus five WSI per class per holdout requires at least
25 WSI in every class for five-fold CV and at least 15 for three-fold CV. When
this is not achievable, per-fold minority F1 is only a warning signal; the
primary class metric is computed from pooled OOF predictions.

### Role of three-fold CV

Three-fold CV is not a larger-data training stage. With the same pool it uses
approximately 241 WSI for training and 120 for validation, whereas five-fold
uses approximately 289 for training.

Three-fold CV is allowed in only two roles:

1. **Training-size/validation-support sensitivity.** After a provisional
   recipe is frozen, repeat that recipe with one paired seed to learn whether
   conclusions change when the training set is smaller and the validation set
   has more support per class. It is not independent confirmation because it
   reuses the same 361 WSI.
2. **Inner selection for final contenders.** If Phase B leaves more than one
   plausible recipe, each outer five-fold training set may contain a grouped
   stratified three-fold inner CV. Hyperparameters and stopping duration are
   chosen only inside the outer-training data; the untouched outer fold then
   estimates the selected procedure. This nested 5-by-3 design is expensive
   and is reserved for final contenders, not the telemetry baseline or a large
   grid.

If the actual goal is a learning curve, nested stratified subsets are more
direct than changing the number of folds. Within each five-fold training set,
construct fixed nested subsets such as 25%, 50%, 75% and 100% while preserving
class and group support. Evaluate all subsets on the same outer holdout. This
produces a genuine small-to-medium-to-large sequence without pretending that
the folds are independent datasets.

## Paired Seed Panel

The minimal robust baseline is five folds by three paired trajectories per
representation: 30 fits total. A cheaper first pass may use five folds by one
seed, but any baseline/finalist used to claim stability must be repeated.

To separate two important sources of randomness, prefer these three
trajectories over three opaque global seeds:

- `base`: fixed initialization and fixed epoch orders;
- `init`: alternate initialization with the base epoch orders;
- `order`: base initialization with alternate epoch orders.

An optional fourth trajectory changes both. Both representation branches use
byte-identical classifier initialization and identical WSI order within each
trajectory. Seeds do not multiply the number of independent WSI: never treat
`361 x seeds` predictions as independent samples.

## Stage Plan

### Immediate recommended execution

The first run is an observational anchor, not a candidate-method bakeoff. Use
the unchanged Spec 0026 local-global classifier, AdamW, loss definition,
complete bags and physical/effective batch of one WSI on the new 361-WSI
development population. Recompute only fold-dependent quantities such as class
weights from each fold's training partition. Run the five frozen outer folds
for the paired normal and continuous-`SO(2)` representations with the `base`
initialization/order trajectory and a fixed precommitted horizon.

The first execution changes data allocation and observability, not the training
rule:

- no clipping, EMA-selected checkpoint, dropout, AEM, MIL-Dropout, sampling,
  label smoothing, noise injection or alternate optimizer;
- no online line search or telemetry-driven LR change;
- no architecture replacement or comparison;
- no outer-fold early stopping;
- no use of the historically exposed 152-WSI cohort.

T0 is collected continuously. T1 is collected at its frozen cadence. T2 runs
on frozen checkpoint copies and precommitted sentinel/probe WSI, so line-search,
SAM, Muon, orthogonal-gradient, StableMax, noise and prototype calculations are
counterfactual and cannot change the optimizer state or training RNG stream.
Before the five-fold run, one short deterministic instrumentation check must
show that enabling T0/T1 produces identical logits, losses, gradients and first
update to telemetry disabled, and a cost probe must freeze T1/T2 cadence and
artifact budget.

This anchor answers whether the former instability survives the larger
supervised population and provides the unmodified trajectory needed for a clean
batch comparison. The same first campaign then compares effective batches 4
and 8 with the same architecture, folds, base trajectory, WSI-exposure horizon,
peak LR and exposure-indexed schedule. It therefore does increase effective
batch, but only after preserving one unmodified batch-one reference. The first
campaign ends after selecting an effective batch from 1/4/8; repeated
initialization/order trajectories, data-scale curves, architecture Stage C and
memorization regularizers follow later.

### A0 -- freeze data and observability

1. Audit the 361-WSI cohort and exact class/group support.
2. Freeze the five-fold manifest, optional three-fold manifest, nested
   learning-curve subsets and diagnostic-sentinel identities.
3. Freeze module names and parameter groups used by telemetry.
4. Define compact outputs, sampling frequencies and maximum artifact size.
5. Prove that telemetry summaries do not change logits, losses or gradients in
   a focused deterministic comparison.

### A1 -- instrumented unchanged baseline

Run the existing AdamW recipe and Spec 0026 architecture on all five folds with
the `base` trajectory. Do not add clipping, dropout, AEM, noise or sampling.
Train to a fixed precommitted horizon rather than stopping when an outer-fold
metric peaks; otherwise the learning dynamics after interpolation are lost.

This stage answers whether instability and overfitting recur with more WSI and
larger validation folds. It also measures the overhead of each telemetry tier.

### A2 -- effective-batch mechanism probe

Using the unchanged loss and optimizer, compare effective WSI batches 1, 4 and
8. First hold peak LR and the exposure-indexed schedule fixed to isolate the
effect of averaging. Then, only if necessary, perform a narrow LR response
probe around the baseline for batches 4 and 8. Accumulate `loss / M`, unscale
once, measure/clip at most once, step once and advance the scheduler once.

No clipping is applied in the first A2 comparison. Pre-clipping gradient norms
allow counterfactual clipping rates for several thresholds to be computed
without confounding the batch experiment.

Larger effective batch is not assumed to be monotonically better. It reduces
gradient variance but also reduces committed updates per WSI exposure and may
remove useful stochastic regularization. Batch 16 is admitted only as a
conditional follow-up when the measured gradient-noise scale continues falling
materially from 4 to 8, the LR response remains stable and the resulting number
of optimizer steps still resolves the learning dynamics. Physical batching and
gradient accumulation are scientifically equivalent for the unchanged model
because it has no batch-dependent normalization; runtime differences are
reported separately.

The primary comparison uses all five outer folds and the `base` trajectory, not
one convenient fold. T0/T1 are retained for every arm; the densest T2 probes may
be restricted to a precommitted diagnostic fold and sentinel panel to control
cost. Compare batches at equal WSI exposures and schedule progress. Do not
linearly scale LR with batch in this mechanism comparison.

### A3 -- variance decomposition at the selected batch

After selecting the effective batch from A2, repeat that recipe with the `init`
and `order` trajectories. Report variability from folds and trajectories
separately. If one source dominates, Phase B should target it rather than
averaging it away. The batch-one anchor is not automatically repeated for all
trajectories unless A1/A2 shows that batch one remains competitive.

### A4 -- explicit data-scale diagnostic

For one paired trajectory at the selected effective batch, train on the fixed
nested 25/50/75/100% subsets of each outer-training fold and validate on the
same holdout. Compare by all of:

- epoch;
- cumulative WSI exposures;
- committed optimizer updates;
- normalized schedule progress;
- wall time, reported only as runtime telemetry.

This distinguishes sample-size effects from update-budget effects.

### B1 -- low-cost stabilizers selected by A1/A2

Only measured evidence can activate these arms:

- heavy-tailed or isolated update spikes: calibrated global clipping;
- noisy endpoint with a stable central trajectory: EMA evaluation;
- persistent train/OOF divergence with norm growth: weight-decay or logit
  regularization probe;
- attention concentration and destructive top-patch dependence: AEM or
  MIL-Dropout, one at a time;
- prediction instability under spatial views: spatially stratified bag
  subsampling;
- class-overlap in bag embeddings: SC-MIL or class-proxy loss.

Muon, SAM, PEGR, `perp`-gradient and architectural replacements remain outside
this plan's first execution.

### C -- curated architecture screen after a stable recipe

Architecture comparison begins only after A/B1 has frozen a defensible
effective batch, LR/schedule convention and training horizon. Otherwise an
architecture can appear weak merely because it was evaluated under a noisy or
mis-scaled optimizer regime.

The historical Spec 0023 gated-attention result does not eliminate simple
ABMIL. It used 106 training WSI, quarter-coverage bags, one trajectory and a
short campaign. The primary simple control is therefore a matched gated-ABMIL
aggregator trained on the same 361-WSI folds, complete bags, patch encoder,
loss, effective batch and exposure budget as the current model. The old exact
configuration may be reproduced only as a historical bridge; its metric is not
directly comparable to the new full-data screen.

Use a narrow ladder rather than a state-of-the-art catalogue:

1. mean pooling plus the common patch encoder and linear head;
2. matched gated ABMIL, including class-specific gated attention as a nested
   variant;
3. the current Spec 0026 local-global model;
4. at most two mechanism-selected candidates:
   - CLAM-SB or DSMIL when instance discrimination is weak;
   - AttriMIL-style contribution/spatial/ranking terms when attention is not a
     faithful contribution score;
   - DTFD-MIL when slide count, rather than patch representation, is the
     dominant limitation;
   - PANTHER when stable panoramic prototypes exist.

TransMIL, long-sequence/state-space models, graph replacements and teacher-
student masked frameworks remain later candidates. They are activated only by
evidence that long-range spatial relations, sequence re-embedding or hard-
instance discovery is missing. Full quadratic attention over the largest bags
is not an admissible control.

Every architecture receives the same outer folds, representation branches,
WSI exposures and paired randomness. Report both its native parameterization
and a capacity-aware comparison; do not distort a published mechanism merely
to force an exact parameter match.

## Telemetry Design Principles

“Collect everything” must not mean synchronizing every parameter to CPU after
every WSI. Telemetry is divided into three tiers:

- **T0, continuous and cheap:** scalar/tiny-tensor summaries buffered on-device
  or in memory, written in batches.
- **T1, sampled:** per-module optimizer and representation diagnostics at a
  fixed cadence of committed updates and every evaluation boundary.
- **T2, boundary/heavy:** separate deterministic inference or diagnostic
  passes on a fixed sentinel set at initialization, dense early boundaries,
  regular later boundaries, selected checkpoint and final checkpoint.

The compiled training graph should emit only compact summaries required at T0.
Python hooks, full attention maps, per-parameter `.item()` calls, SVDs,
perturbations and curvature probes do not belong in the hot loop. T2 may use an
eager diagnostic copy of the same checkpoint if necessary; it must never alter
the training optimizer, RNG stream or checkpoint selection.

Counterfactual diagnostics must evaluate frozen copies of the same checkpoint
and explicitly distinguish quantities measured on the effective training batch
from transfer to an independent fold-train probe batch. A candidate step size,
perturbation or gradient transformation that improves only the batch on which
it was constructed is not evidence that it should become part of training.
Outer-fold validation gradients or losses may be inspected descriptively in
Phase A, but may not drive an online update or an unbiased final selection.

Suggested boundary cadence is initialization; epochs 1, 2, 3, 5 and 10; every
five epochs thereafter; and all selected/final checkpoints. Full validation
predictions may be recorded more frequently, but heavyweight probes remain on
this schedule.

### Session and runtime packaging

The reusable historical runner already supplies complete-bag training,
validation, checkpointing and exact resume. The new implementation work is
therefore concentrated in named diagnostic capture, compact aggregation and
the counterfactual evaluators; it is not a new training system. Nevertheless,
code size and runtime cost are different questions. T0 reductions add one
ordinary path, whereas several T2 diagnostics require extra forward/backward
passes, per-WSI gradients, parameter perturbations, line-search trials or
spectral iterations.

Treat one Kaggle session as a resumable compute shard, never as a scientific
unit. The full five-fold campaign does not have to fit in one session and no
metric may depend on where a session boundary occurred. A practical first
shard for one fold and paired representation branches contains:

1. the instrumentation equivalence and cost probe;
2. ordinary training with all T0 and the accepted T1 cadence;
3. the light T2 sentinel suite at initialization and the dense early
   boundaries that occur before the wall guard;
4. an atomic checkpoint containing model, optimizer, scaler, scheduler,
   sampler/order cursor and telemetry accumulator state.

The full T2 catalogue is not run at every boundary. Freeze a light mandatory
subset for each boundary and distribute the expensive families across
precommitted boundaries or replay them from immutable checkpoint copies in a
separate diagnostic shard. In particular, iterative step-size search, SAM
gradient comparison, per-example full-gradient calculations, Hessian/Lanczos
probes and prototype fitting must each have an explicit pass budget. This
preserves the temporal coverage needed to diagnose instability without letting
diagnostics consume most of the training session.

Historical Spec 0036 evidence gives only an initial capacity bound, not a new
runtime promise: a branch completed 2,226 committed WSI updates, or 21 epochs
of the former 106-WSI train split, in about 7.5 hours while repeatedly
validating 23 WSI. A new fold has roughly 289 train and 72 validation WSI, so
the old throughput cannot imply that a long fold, much less five folds, fits in
one session. The A0 cost probe must report observed seconds per train WSI,
validation WSI and each diagnostic family, then freeze a horizon and cadence
that remain exactly resumable. T0/T1 plus a sparse prebudgeted T2 subset is
expected to fit alongside useful training progress in one session; exhaustive
T2 across all boundaries is explicitly out of scope for a single session.

## T0: Every WSI Or Optimizer Step

### Identity, data and runtime

- run, branch, outer/inner fold, seed role, epoch, microbatch and committed
  update;
- WSI, class, VAE source role, position in epoch and retry count;
- bag size, graph edge/degree summaries and coordinate coverage;
- load, host-to-device, forward, backward and optimizer-step time;
- allocated/reserved/peak device memory and compiler specialization count at
  safe boundaries;
- AMP scale before/after, overflow/backoff, skipped update and nonfinite-loss
  status.

### Prediction and example dynamics

- unweighted and class-weighted CE;
- all five FP32 logits and probabilities;
- predicted class, correctness, true-class probability and rank;
- logit L2 norm, maximum absolute logit and logit range;
- true-class margin against the largest incorrect logit;
- predictive entropy and maximum probability;
- cumulative correctness transitions, forgetting events, area under the
  margin and first-learned/last-forgotten boundary for every WSI.

Training-stream losses are associated with different parameter states and are
not a replacement for fixed-boundary inference over the full training split.
Accordingly, update the cumulative learning/forgetting tracker only during
those fixed-boundary full-split inference passes, never after individual WSI
training updates.

### Optimizer-step summaries

- LR and normalized schedule progress for every parameter group;
- effective batch size and partial-final-batch status;
- global gradient norm before any clipping;
- counterfactual clip coefficients for a frozen list of thresholds;
- actual clipping coefficient and post-clip norm if clipping is later enabled;
- global parameter norm, update norm and
  `update_norm / parameter_norm`;
- cosine between aggregate gradient and actual update;
- cosine with the previous committed gradient and update, plus update sign-flip
  rate, so rolling volatility and oscillation can be reconstructed;
- contributions to gradient/update norm from patch CNN, local block 1, local
  block 2, sigmoid summary, final token block and classifier head.

## T1: Sampled Module And Optimizer Telemetry

For each semantic module and optimizer parameter group record:

- parameter, gradient, Adam first-moment, second-moment and actual-update
  L2/RMS/max norms;
- data-gradient and decoupled-weight-decay contributions to the update;
- gradient-to-weight and update-to-weight ratios;
- cosine of weight with raw gradient and with actual update;
- fraction of gradient norm in the radial weight direction;
- ratio of perpendicular to total gradient norm;
- cosine of raw gradient with Adam first moment and final update;
- quantiles of `sqrt(v)`, fraction of coordinates dominated by Adam epsilon,
  and quantiles of the effective coordinate-wise step size;
- zero, nonfinite and extreme-value fractions;
- row-norm coefficient of variation for matrix parameters;
- norms and gradients of normalization gains/biases, attention offsets,
  radial-bias tables, null keys and CLS/REG tokens.

For future batch-size, PEGR and gradient-conflict decisions, estimate on sampled
effective batches:

\[
\bar g=\frac1M\sum_i g_i,
\qquad
G_2=\frac1M\sum_i\|g_i\|^2,
\qquad
V_g=G_2-\|\bar g\|^2.
\]

Record per-WSI gradient norms, pairwise/mean gradient cosine, conflict fraction
and a gradient-noise-scale proxy. Stratify them by class, bag-size quantile,
correctness/forgetting state and persistent WSI identity; also record
within-class and between-class gradient alignment. This separates stochastic
noise from repeatable hard or possibly mislabeled slides. These quantities must
be accumulated on device or evaluated in a diagnostic pass; do not clone every
parameter to CPU for every WSI.

For future Muon decisions, periodically compute for eligible 2D matrices:

- leading singular values of raw gradient and Adam first moment;
- stable rank, entropy/effective rank and top-k spectral energy;
- smallest reliably resolved singular value and condition proxy;
- row/column leverage and row-norm imbalance;
- fraction of trainable parameters and actual Adam update energy contained in
  Muon-eligible matrices;
- counterfactual polar/Muon direction norm, singular spectrum and
  orthogonality residual;
- cosine between Adam and counterfactual Muon directions, normalized predicted
  descent `-g dot d`, and layerwise contribution to that descent;
- stability of the leading gradient/momentum subspaces across adjacent WSI,
  accumulated effective batches and diagnostic boundaries.

Full SVD is unnecessary when randomized or exact small-matrix summaries give
the required spectrum.

## T1/T2: Representation And Activation Telemetry

For the patch encoder outputs `X0`, local blocks `X1/X2`, global token sequence
`T1` and final CLS representation record:

- mean, standard deviation, RMS, L2/max norms and nonfinite fraction;
- channel/unit variance and near-zero/dead-unit fraction;
- effective rank and singular-value entropy of sampled token matrices;
- mean pairwise cosine on a deterministic patch sample;
- residual-update/input norm for every attention and SwiGLU sublayer;
- SwiGLU gate/value/hidden RMS, near-zero fraction and saturation quantiles;
- neighboring versus non-neighboring token similarity;
- graph Dirichlet energy before and after each local block.

Token-collapse evidence consists of falling effective rank, rising pairwise
cosine and vanishing spatial/feature variance. Oversmoothing evidence requires
the same changes to be stronger for neighboring tokens and across successive
local blocks; one statistic alone is insufficient.

### Layer-resolved instability map

Because the current classifier is small, telemetry should cover every semantic
transformation rather than only the six coarse optimizer groups. Freeze named
capture points for:

- every convolution, normalization and activation in the patch encoder;
- the input normalization, query/key/value projections, attention scores and
  probabilities, output projection, residual sum, second normalization,
  SwiGLU gate/value/product and feed-forward residual of each local block;
- the corresponding projections, sigmoid weights, context and residual of the
  global summary;
- every normalization, projection, softmax, residual and SwiGLU component of
  the final CLS/REG block;
- the final normalization, bag embedding and classifier logits.

For each captured forward tensor record count, signed mean, mean absolute
value, standard deviation, RMS, L2 and maximum norms, robust quantiles,
minimum/maximum, nonfinite fraction,
zero/near-zero fraction and per-channel variance summaries. For residual
sublayers also record `||delta|| / ||input||` and cosine between input, delta
and output. For normalized tensors record pre/post RMS and variance; for gates
and probabilities record saturation, entropy and effective support. Means and
standard deviations alone are insufficient because cancellation, a few extreme
tokens or a small set of dead channels can produce an apparently normal pair
of scalars.

For the gradient of each captured output record the same scale/tail/nonfinite
summaries plus its share of the total activation-gradient energy. Where both
input and output gradients are available, record the amplification proxy

\[
A_l=\frac{\|\partial L/\partial x_l\|_2}
          {\|\partial L/\partial y_l\|_2+\epsilon}.
\]

For the trainable parameters belonging to that layer, record weight, raw
unscaled gradient and actual-update RMS/L2/max norms; gradient-to-weight and
update-to-weight ratios; radial-gradient fraction; and cosine with the previous
committed gradient/update. T1 additionally stores channel/row summaries,
gradient quantiles, Adam-moment summaries and exponential moving mean/variance
of these measurements. On sampled effective batches it also estimates
per-WSI gradient variance and conflict by layer. Together these quantities can
distinguish exploding forward scale, backward amplification, optimizer-driven
overshoot, dead/saturated units and conflicting examples.

The default frequency is intentionally asymmetric:

- T0 every committed update: compact forward RMS/standard deviation,
  nonfinite and near-zero fractions at the output of each semantic block, plus
  parameter-gradient norm/share and actual update-to-weight ratio by block;
- T1 at the frozen sampled cadence: every named capture point, tail/channel
  summaries, activation gradients, Adam state and temporal cosines;
- T2 at diagnostic boundaries on the sentinel panel: full sampled
  distributions, spectra/effective ranks and any counterfactual re-evaluation.

Reductions must be computed on device and only compact summaries transferred.
The compiled hot path should use explicit optional diagnostic returns or
stable named capture points; generic Python forward/backward hooks are reserved
for an eager T1/T2 checkpoint copy if they introduce graph breaks. Parameter
gradients are summarized after AMP unscaling and before the optimizer step.
The instrumentation equivalence check must cover the diagnostic-disabled and
enabled paths and confirm identical logits, loss, raw gradients and first
update.

For SC-MIL/proxy decisions, record per-class bag-embedding centroids,
within-class scatter, between-class distances, nearest-centroid errors,
silhouette-like summaries and Fisher separation. These are descriptive on each
fold; centroids used for any learned loss must later be fitted on fold-train
only.

### Architecture-neutral interface

To compare different MIL families, require each model to expose the same small
semantic interface even when its internal attention is different:

- common patch embedding before aggregation;
- final bag/slide embedding;
- five logits and probabilities;
- native instance score, contribution or prototype responsibility when one
  exists;
- a sentinel-only counterfactual contribution obtained by instance deletion or
  gradient-times-input when no native score exists.

Record trainable parameter count by subsystem, activation/optimizer memory,
forward/backward/update time, processed patches per second, retained-token
fraction, update/weight ratio and gradient share by encoder, aggregator and
head. Report effective instance count, spatial coverage and deletion stability
with the same definitions across models. A raw attention coefficient is not
treated as comparable to an AttriMIL contribution, DSMIL critical-instance
score or PANTHER responsibility.

## Dataset Structure And Memorization Telemetry

The classifier traces alone cannot distinguish memorization caused by model
capacity from shortcuts or duplicated structure in the data. Add an offline
fold-aware audit that records:

- exact and near-duplicate WSI/patch-feature neighbours, patient/site overlap
  and nearest-neighbour label agreement across folds;
- label predictability from nuisance-only slide summaries: patch count, tissue
  area/coverage, coordinate extent, graph density, VAE source role and any
  available acquisition/replacement metadata;
- within-slide versus between-slide feature variance, patch-distribution
  entropy, effective morphotype count and class-conditional distribution
  overlap;
- performance of fold-train-fitted linear, nearest-centroid and k-nearest-
  neighbour probes on simple slide summaries, bag embeddings and prototype
  summaries;
- label-prediction change when coordinates are removed, shuffled between
  slides of similar size or replaced by a matched synthetic layout;
- agreement across seeds/folds of which WSI are learned early, forgotten,
  high-gradient, high-loss or high-influence.

At every fixed evaluation boundary, store the exact small classifier-head
gradient for every WSI and a fixed low-dimensional random projection/sketch of
the complete per-WSI gradient. The projection matrix is frozen before labels
or outcomes are inspected. From these artifacts derive:

- self-gradient energy and alignment with the population gradient;
- within-class and between-class gradient similarity;
- checkpoint-summed TracIn-like influence proxies;
- whether a small persistent WSI subset dominates updates;
- gradient-neighbour versus representation-neighbour agreement;
- train confidence and margin unsupported by nearby fold-validation WSI.

Also store per-WSI bag-embedding trajectories on fixed boundaries, their drift
from initialization/previous boundary, nearest-neighbour identity changes and
class-centroid decomposition. Memorization is supported by a joint pattern of
idiosyncratic gradient direction, persistent high self-influence, isolated
representation and rising train confidence without corresponding OOF-neighbour
improvement; train accuracy alone is insufficient.

One optional fold-train-only stress test may corrupt or permute a small
precommitted subset of labels. Its purpose is to compare time-to-fit and
gradient/representation signatures of intentionally arbitrary labels against
the real data. It cannot identify real labels as wrong, select WSI for removal
or contribute to the primary OOF metric.

## MIL Attention Telemetry

### Local softmax-plus-null blocks

For each layer and head, stratified by valid degree and radial code, record:

- normalized entropy over valid neighbours plus null;
- null mass, self-edge mass and non-self mass;
- maximum weight and top-k mass;
- mass by squared radial distance;
- Q/K/V norms and score mean, standard deviation, min/max;
- softmax saturation fraction;
- learned null bias and radial-bias values and gradients;
- head-to-head similarity and top-neighbour overlap.

### Global sigmoid patch-summary read

Because these weights are not normalized, record both magnitude and
concentration. For every query and head:

- pre- and post-`-log(N)` score statistics;
- sigmoid mean, quantiles, min/max and fractions near zero or one;
- total gate mass `sum(w)` and context/output norm;
- diagnostic normalized distribution
  `p_i = w_i / sum_j(w_j)` when the mass is positive;
- normalized entropy of `p`, top-1/top-10/top-1%-mass and
  `exp(H(p))`;
- Gini/concentration coefficient of `p`;
- participation ratio `(sum(w))^2 / sum(w^2)`;
- top-patch coordinate coverage, connected components and pairwise distances;
- overlap/cosine/Jaccard among heads and among CLS/REG queries;
- correlation of gate mass, entropy and context norm with `log(N)`.

This distinction is essential: applying standard softmax AEM directly to the
unnormalized sigmoid weights would change both concentration and total mass.
Telemetry must first show which quantity is failing.

### Final CLS-to-17-token softmax read

- entropy, maximum/top-k mass and CLS self-mass by head;
- mass assigned to each register;
- pairwise cosine/effective rank of the 16 REG tokens;
- head diversity and register-usage diversity;
- residual and FFN update ratios for the final CLS.

Store full attention vectors only for precommitted sentinel WSI at T2
boundaries. For all other WSI store summaries and the top-k patch identities,
coordinates and weights. This retains diagnostic value without producing an
unbounded artifact.

## T2: Counterfactual Optimization, Perturbation And Invariance Diagnostics

Use fixed train and validation sentinel panels stratified by class, VAE source
role and bag-size quantile. At diagnostic boundaries evaluate without changing
training state:

### Per-step step-size oracle

The phrase "optimal LR at a step" is incomplete without a direction, batch and
objective. For a frozen checkpoint, define a candidate data-update direction
`d` before decoupled weight decay and

\[
\phi_B(\alpha)=L_B(\theta+\alpha d).
\]

On a small precommitted set of boundaries and effective fold-train batches:

- evaluate `phi_B(alpha)` on a logarithmic grid around the scheduled step and,
  where useful, refine the best bracket by bounded interpolation or
  backtracking;
- record the scheduled, best-grid and refined step, accepted/rejected trial
  count, boundary-hit status, loss reduction per extra forward/backward pass
  and Armijo/Wolfe or gradient-only line-search conditions;
- record `g dot d`, directional derivatives at candidate points,
  finite-difference directional curvature and the quadratic-model proposal
  `alpha_quad = -(g dot d)/(d^T H d)` when curvature is positive;
- repeat loss evaluation, without adapting the candidate, on an independent
  fold-train probe batch and on the next WSI/effective batch in the frozen
  order;
- compare raw-gradient, Adam pre-decay and actual pre-decay update directions;
- summarize the distribution, temporal smoothness, class/bag-size dependence
  and transfer rank-correlation of selected step sizes.

Also retain the low-cost sufficient histories needed to diagnose non-iterative
adaptive methods: loss above an explicitly assumed floor, gradient squared
norm, displacement from initialization, cumulative path length, cumulative
gradient/update inner products and the running scale proposed by any shadow
SPS, D-Adaptation or Prodigy estimator. A shadow estimator is diagnostic only
and must not alter the baseline optimizer.

This probe distinguishes a locally useful line search from an expensive
procedure that overfits one WSI. It does not define a globally optimal LR.

The baseline does not perform this search before every update. First run a
dense diagnostic on one precommitted fold/trajectory at a fixed `n`-step
cadence plus all interpolation, overflow and gradient-spike events. The
artifact-size and runtime probe freezes `n`. If selected step sizes transfer,
a later optimizer arm compares online search every update against warm-started
search every `n` updates with the accepted step held or smoothly interpolated
between searches. Those are different training algorithms and require fresh
paired runs.

### Evidence-deletion and view stability

- delete top-attended patches at fixed counts and fractions;
- delete equal-size random patch sets with several deterministic draws;
- remove contiguous spatial blocks;
- evaluate spatially stratified 75%, 50% and 25% views;
- duplicate selected patches to test cardinality sensitivity;
- permute patch order as an implementation-invariance control.

Record prediction agreement, true-class margin change, logit L2 change,
Jensen-Shannon divergence, calibration change and attention redistribution.
Top-delete degradation beyond random-delete degradation supports
MIL-Dropout/STKIM; instability across well-covered spatial views supports
subsampling/consistency; isolated spatially incoherent high scores support a
spatial regularizer.

### Spatial diagnostics

- Moran's I or a graph-based analogue for attention/contribution scores;
- connected-component count, area, compactness and diameter of top-score
  regions;
- attention-weighted spatial dispersion and coverage;
- local total variation of score maps;
- agreement of high-score regions across checkpoints, seeds and branches.

### Curvature and radial-growth proxies

At initialization, interpolation onset, selected and final checkpoints record:

- loss under fixed relative random parameter perturbations;
- loss along normalized gradient and actual-update directions;
- finite-difference directional curvature;
- optional Hutchinson Hessian-trace and short power/Lanczos top-eigenvalue
  estimates on a fixed small WSI panel;
- latent sensitivity `||d loss / d mu||` and its spatial/channel distribution;
- weight-gradient and weight-update radial alignment;
- train accuracy/CE, logit norm and margin growth after interpolation.

For a direct SAM decision, at a fixed small radius grid additionally record:

- clean-batch sharpness
  `L(theta + rho*g/||g||) - L(theta)` and its relative version;
- loss of the same perturbation on an independent fold-train probe batch;
- cosine and norm ratio between clean and perturbed gradients;
- modulewise contribution to the perturbation and sharpness;
- whether sharpness and clean/perturbed gradient disagreement predict later
  forgetting, OOF failure or seed instability;
- measured extra forward/backward time and peak memory.

These measurements determine whether SAM, PEGR/F-GR, stronger decay or
orthogonal-gradient projection has a plausible target. They are not themselves
generalization metrics.

### Orthogonal-gradient and StableMax diagnostics

For every eligible weight tensor at the same boundaries, decompose

\[
g_{\parallel}
=\frac{\langle \theta,g\rangle}{\|\theta\|^2+\epsilon}\theta,
\qquad
g_{\perp}=g-g_{\parallel}.
\]

Record radial energy, `||g_perp||/||g||`, the norm-restoration factor used by
the published orthogonal-gradient rule, predicted descent before/after the
projection, and finite-step loss/margin changes under radial-only,
perpendicular-only and original directions. Measure whether a radial step
changes logits primarily by one common multiplicative scale: fit
`delta_z ~= a*z + b*1`, report residual variance, prediction changes and CE
reduction. Orthogonal projection becomes plausible only if the removed radial
component repeatedly reduces CE through near-pure logit scaling without useful
changes in predictions or held-out margins.

For final class logits and every actual softmax, record in FP32 and an FP64
diagnostic reference:

- shifted-logit range, exact-zero exponentials and probabilities exactly zero
  or one;
- `logsumexp`, CE, probabilities and gradient disagreement across precisions;
- true-class/non-true-class gradient components and their smallest nonzero
  magnitudes;
- counterfactual StableMax probabilities, CE, gradient norm/direction,
  calibration and ranking on the sentinel panel.

StableMax is justified only by numerical absorption/collapse or a reproducible
benefit of its changed objective. Large logits or ordinary overconfidence alone
are insufficient, especially with only five output classes.

### Per-example regularization and noise diagnostics

For PEGR/F-GR and possible input, activation, gradient or weight noise, record:

- the distribution and temporal persistence of per-WSI `||g_i||^2`, its class
  contribution and association with forgetting, margin and suspected outliers;
- the counterfactual mean per-example gradient penalty and the gap between
  `mean(||g_i||^2)` and `||mean(g_i)||^2`;
- on a tiny sentinel subset, Hessian-vector products needed to estimate the
  first-order change induced by the gradient penalty;
- loss, logits and gradient cosine under paired zero-mean Gaussian
  perturbations at scales expressed relative to latent, activation, gradient
  or parameter RMS;
- local response slope, curvature, variance across noise draws and whether the
  perturbation suppresses slide-specific gradients more than shared signal.

This separates useful smoothing from merely injecting another source of
variance into an already noisy effective batch.

### Prototype suitability and PANTHER-like diagnostics

PANTHER is not an optimizer. It is an alternative task-agnostic slide
representation that models patch embeddings with morphological Gaussian
mixture prototypes and concatenates per-slide component prevalence, means and
covariances. Its prototypes and all preprocessing must be fitted only on the
training portion of each fold.

Before implementing a PANTHER arm, perform an offline T2 prototype audit on
deterministic patch samples from fold-train and untouched fold-validation WSI:

- held-out patch log likelihood, quantization/responsibility-weighted residual
  and within/between-prototype scatter over a small frozen range of prototype
  counts;
- mixture prevalence, responsibility entropy, effective occupied prototypes,
  empty/collapsed components and covariance eigenvalue/condition summaries;
- prototype/responsibility stability across initialization, patch subsampling,
  WSI views and folds, using matched assignments before comparing components;
- class separation and calibration of fold-train-fitted linear and small-MLP
  probes on `[pi, mu, Sigma]`, evaluated OOF;
- information loss relative to the current bag embedding: prediction agreement,
  class-conditional distances and residual performance when spatial coordinates
  are removed or shuffled;
- whether discriminative evidence is panoramic or needle-like, using the
  existing top-deletion, spatial-view and effective-patch-count diagnostics;
- construction time, peak memory and final representation dimension.

Stable prototypes, broad evidence and competitive low-capacity OOF probes make
PANTHER plausible. Unstable components, rare focal evidence or a large loss
from destroying spatial organization argue against replacing the current
local-global MIL aggregator.

### Rotation/equivariance boundary

Coordinate permutation and exact square-lattice rotations with unchanged token
values may test classifier implementation invariance, but they do not test VAE
equivariance. A scientific end-to-end rotation diagnostic requires rotated RGB
patches to be re-encoded by both frozen VAEs under a matched protocol. Do not
construct an apparently favorable SO(2)-only result by applying a prescribed
latent action without a comparable normal-VAE input transformation.

## Evaluation At Fixed Boundaries

Run deterministic inference over the complete fold-train and fold-validation
sets at the fixed boundary schedule. For both raw and precommitted EMA tracks
record:

- concatenated OOF macro-F1, macro recall, balanced accuracy and accuracy;
- per-class precision, recall, F1, support, one-vs-rest ROC-AUC and PR-AUC;
- mean unweighted CE, Brier score, predictive entropy and calibration error;
- fixed-bin and adaptive-bin reliability summaries, overall and per class;
- confusion matrices and normalized confusion matrices;
- per-WSI logits, margins, bag/graph covariates and complete prediction
  histories;
- train--validation gaps for loss, margin, calibration and classification;
- metric association with class, `log(N)`, graph density, VAE source role and
  image-update status.

Fold-level calibration curves are descriptive because a single holdout is
small. The primary ECE/reliability result is computed from the concatenated OOF
predictions, with the binning rule frozen in advance.

Maintain EMA tracks by half-life in committed optimizer steps or WSI exposures
rather than blindly copying one decay across effective batches. Raw and EMA
weights are both evaluated; EMA does not select or alter the training path.

## Selection And Statistical Reporting

- Exploratory Phase A/B may use five-fold OOF dynamics to choose what to study,
  but OOF performance used for those choices is not an unbiased estimate of the
  selected procedure.
- Final contenders use nested outer-five/inner-three selection or are evaluated
  only as exploratory configurations.
- Folds are correlated because their training sets overlap. Do not use five
  fold scores as five independent samples in a naive t-test.
- For each seed, concatenate exactly one OOF prediction per WSI. Report seed
  mean, standard deviation and range separately from WSI sampling uncertainty.
- Normal-versus-`SO(2)` and configuration comparisons are paired by WSI, fold,
  initialization and order. Use diagnosis-stratified WSI bootstrap for paired
  prediction differences and show per-class effects.
- A hierarchical analysis may resample seeds and WSI, but must not pretend the
  repeated predictions are new patients.
- Hyperparameters shared by the two VAE branches are selected by their
  prespecified average performance and stability, or by a branch-neutral
  criterion. Never tune a shared setting to maximize the observed branch gap.
- Promotion among predeclared contenders is lexicographic: pooled OOF utility
  first, then unweighted NLL as a tie-breaker, no consistent material loss of
  minority recall, lower seed variability and absence of new numerical
  failures. Exact tolerances and metric order belong in the execution contract.

## Full Refit

After architecture, optimizer, effective batch, regularization, EMA rule and
training duration are frozen:

1. refit on all 361 WSI with no validation-dependent early stopping;
2. derive the fixed duration from inner-CV or precommitted normalized exposure,
   not from the external 152 WSI;
3. train a predeclared number of paired seeds, preferably three;
4. if ensembling is used, predeclare probability averaging and include the
   single-seed results;
5. evaluate the 152-WSI external cohort only after all decisions and artifacts
   are frozen;
6. describe that cohort's historical exposure and avoid a fresh-sealed-test
   claim.

## Mapping Measurements To Later Decisions

| Observed pattern | Supported next experiment |
| --- | --- |
| Gradient-noise proxy falls strongly from batch 1 to 4 and little from 4 to 8 | Prefer effective batch 4 |
| Gradient-noise proxy still falls materially from 4 to 8 and learning remains update-resolved | Admit a conditional batch-16 probe |
| Larger batches reduce noise but worsen OOF or minority recall | Retain the smaller batch; noise was useful regularization |
| Rare extreme norms dominate; counterfactual clipping is infrequent | Calibrated global clipping |
| Most steps would be clipped | Reduce LR or diagnose loss scaling before clipping |
| Step-size minima are stable and transfer to an independent train probe batch | Pilot stochastic line search or adaptive step-size method |
| Per-batch step-size minima vary wildly or fail to transfer to the next WSI | Prefer a smoothed schedule; do not run exact line search online |
| SPS/Prodigy shadow scale stabilizes and agrees with transferable line-search scales | Parameter-free optimizer probe becomes plausible |
| Raw endpoints oscillate while EMA trajectory is stable | EMA half-life comparison |
| Normalized patch-attention entropy collapses; top deletion is destructive | AEM or MIL-Dropout, separately |
| Total sigmoid gate mass, not normalized concentration, drifts with bag size | Gate-mass/cardinality correction, not ordinary AEM |
| Spatial views disagree despite broad coverage | Stratified subsampling plus consistency |
| High-score regions are fragmented beyond pathology expectation | AttriMIL-like spatial smoothness probe |
| Bag embeddings show high within-class overlap and stable class confusions | SC-MIL/proxy loss |
| Train interpolates, logits/norms grow and radial alignment rises | decay/logit regularization, then `perp`-gradient probe |
| High sharpness/curvature predicts seed failures | SAM before second-order optimizers |
| SAM perturbations increase loss consistently across independent WSI and rotate gradients materially | SAM has a measured target |
| Sharpness is batch-specific and disappears on independent WSI | Do not prioritize SAM |
| Per-WSI gradient norms/conflict dominate | PEGR/F-GR or robust sampling probe |
| A persistent small WSI subset dominates per-example gradient energy | PEGR/F-GR or robust slide weighting probe |
| Matrix gradients/momenta have useful multi-directional effective rank | Muon hybrid becomes plausible |
| Muon-eligible gradients are persistently low-rank/noisy | Do not prioritize Muon |
| Counterfactual Muon directions have stable subspaces and better normalized predicted descent than Adam | Pilot hybrid Muon on eligible internal matrices |
| Radial gradient mainly rescales logits after interpolation with no held-out margin benefit | Orthogonal-gradient ablation becomes plausible |
| FP32 softmax disagrees with FP64 or loses non-target gradient components | StableMax diagnostic arm becomes plausible |
| No numerical softmax collapse is observed | Do not prioritize StableMax |
| GMM prototypes are stable, broadly occupied and yield competitive OOF probes | PANTHER-like slide representation becomes plausible |
| Prototype assignments are unstable or spatial destruction is costly | Retain spatial local-global MIL rather than PANTHER |
| Mean/gated ABMIL matches the current model under the stable recipe | Prefer the simpler model or require a demonstrated spatial benefit |
| Simple pooling fails but critical-instance/attribute probes are strong | CLAM/DSMIL/AttriMIL mechanism becomes plausible |
| A few WSI have persistent isolated gradient sketches and unsupported train margins | Memorization-focused regularization/stress test becomes plausible |
| Nuisance-only probes predict diagnosis OOF | Treat shortcut control as higher priority than optimizer replacement |
| X1/X2 rank collapses and graph Dirichlet energy vanishes | local oversmoothing/architecture diagnosis |
| REG tokens become nearly identical or unused | reduce/register-diversity ablation |

## Required Plan Artifacts Before Execution

- immutable 361-WSI cohort audit;
- grouped stratified five-fold manifest and optional three-fold/nested manifests;
- nested learning-curve subset manifest;
- branch-paired initialization/order seeds;
- sentinel WSI manifest;
- telemetry field/frequency/cost table and artifact-size projection;
- fixed fold-train probe batches and candidate grids for step-size, SAM,
  orthogonal-gradient, precision and prototype counterfactuals;
- frozen gradient-sketch projection, architecture-neutral output schema and
  nuisance/prototype probe definitions;
- exact fixed training horizon and evaluation-boundary schedule;
- machine-readable configuration defining Phase A and the allowed B1 arms;
- proof that the 152 external WSI are absent from all development inputs;
- focused numerical tests for attention summaries, gradient/update norms,
  accumulation semantics, EMA and OOF assembly.

## Non-Goals

- No VAE retraining or unfreezing.
- No use of the 152 external WSI for tuning or debugging.
- No simultaneous AEM, MIL-Dropout and STKIM arm.
- No broad optimizer or architecture search; Stage C is a precommitted narrow
  mechanism screen.
- No treating folds, patches or repeated seeds as independent patients.
- No conclusion that attention weights are causal pathology explanations.
- No claim that telemetry alone improves generalization.
- No remote execution until the compact machine contract and runner wiring
  receive separate review and authorization.

## Related Specs

- `0025-full-foreground-logical-mil-dataset.md`
- `0026-local-global-sigmoid-mil.md`
- `0032-amp-fixed26-mil.md`
- `0035-largest-class-weighted-amp-probe.md`
- `0036-local-global-mil-training.md`
- `0041-sealed-mil-test-evaluation.md`
