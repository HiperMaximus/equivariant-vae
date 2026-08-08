# Spec 0011: Goal-derived runtime and compiled fast path

Status: v4 locked / implementation-ready; canonical six-file evidence package included
in the focused relock commit; remote execution not authorized
Owner/workstream: selected-runtime speed and reuse
Last updated: 2026-08-08

## Fresh-session start here

Preserve the substantial dirty tree. Do not execute the partial v3 controller, old-v2
`p00310`, or any Kaggle/GitHub/Overleaf/remote command without fresh explicit user
authorization. Read `AGENTS.md`, `CURRENT.md`, `GOAL.md`, the specs index, and this spec
before changing code. The user authorizes targeted removal or reversion of changes proved
to belong only to the failed v3 approach when replacement is cleaner than adaptation.
This is not permission for `git reset`, `git checkout`, blanket restoration, or removal
of unrelated dirty work: inspect each diff, edit surgically, and record every discarded
v3-only behavior/file.

The user approved the v4 direction on 2026-08-08:

1. optimize correct dual-T4 `drop_last=True` projected epoch time, with no penalty for
   enabling extra toggles;
2. start from every installed/source/runtime control with a plausible steady-step
   acceleration mechanism, including experimental and deprecated-but-executable
   features;
3. prove exclusions, construct inclusion-maximal compatible recipes, and test declared
   complex interactions directly;
4. use the 309 immutable v2 rows to avoid repeating old singleton sweeps and batch
   discovery, while requiring current evidence for genuinely new complete recipes;
5. do not minimize Kaggle GPU time or impose the obsolete 118-probe/two-session limit.
   The finite search may span as many resumable sessions as necessary.

This amendment replaces the v3 main-effect, exact-Bmax, 12-recipe, two-interaction,
four-contender, 118-probe, and `<=16 h` contracts. The substantial local v3 code is
fail-closed partial work to refactor, not an approved executable.

## Purpose and objective

V4 selects compute finalists for `non_eq_vae` first. The code is reusable, but the later
continuous-`SO(2)` search is a separate model-digest-bound generation with a fresh
inventory, activation ledger, cover matrix, and measurements. Cross-model rows may order
work only; they never close coverage or count as avoided model-specific measurement.
Loader, real-data quality, LR, promotion, and full training remain later gates.

Let `P = 300000`, hash-bound to
`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` and the repo data constant,
`b = per_device_batch`, `world_size = 2`, and
`G = 2*b`. Rank a complete recipe/batch pair by

```text
floor(P / G) * synchronized_mean_steady_step_wall_time
```

The synchronized mean is the slower-rank measured-block wall duration divided by
successful updates. A hash-bound `fastpath_step` function defines the exact timed body:
`zero_grad(set_to_none=True)`, every device-side transform performed after batch delivery,
forward, loss/reconstruction telemetry, backward, unscale, gradient clipping,
optimizer/scaler update, and DDP communication. Synthetic inputs are already device-
resident; loader, CPU corruption, H2D transfer, validation, schedule updates, logging,
and checkpointing are excluded here and measured by later real-data gates. No hidden
timed-body difference from the paid runner is allowed.

Fixed-batch standings, step latency alone, maximum feasible batch, throughput at an
arbitrary shared batch, compile time, deterministic RNG, and incomplete observations do
not select the runtime. Compile/startup cost is recorded but excluded from the long-run
epoch objective.

The selector has no sparsity or simplicity objective. A fastest correct recipe may keep
neutral, redundant, or individually unnecessary requested toggles. Alias, implied, or
runtime-proven inert requests canonicalize to one effective recipe rather than creating
duplicate evidence identities.

## Mandatory stage order

Each newly authorized remote execution starts in this order:

1. upgrade to latest PyPI Torch before importing Torch or `eqvae`;
2. regenerate the installed inventory, executable core modes, source digests, hardware
   fingerprint, and effective option domains;
3. verify and import the immutable v2 evidence package;
4. run activation/operator/compatibility scouts and seal the v4 acceleration ledger;
5. generate and independently verify the finite maximal-cover recipe matrix and direct
   interaction matrix;
6. publish the exact resumable work plan before the first selectable measurement;
7. screen complete recipe/batch pairs, refine configuration and batch jointly, then run
   untouched confirmation blocks;
8. independently verify the artifact before emitting compute finalists.

No timing result may change stages 1-5 retroactively. An upgraded inventory that exposes
new controls regenerates and reseals the matrix before selectable work.

## Acceleration inventory and exclusion proof

The collector exhaustively walks these upgraded-runtime surfaces rather than a hand list:

| Surface | Required source/domain collector |
| --- | --- |
| Compiler stack | every installed `ConfigModule`/compiler registry found by source walk, explicitly including `torch.compiler.config`, `torch._dynamo.config`, `torch._inductor.config`, `torch._inductor.list_options()`, `list_mode_options()`, `torch._functorch.config`, and `torch.fx.experimental.config` |
| Compile/DDP | signatures and implementation registries for `torch.compile`, DDP, reducer/compiled-autograd modes, bucket lists, delayed reductions, mixed precision, static/bucket-view/batched-copy settings, and installed comm hooks |
| Optimizer/AMP | AdamW, `zero_grad`, `GradScaler`, gradient clipping, fused/foreach/capturable/differentiable/compiled paths |
| Distributed/backends | `torch.distributed.config` including TorchComms, CUDA/cuDNN/cuBLAS/matmul precision, allocator and memory controls, CUDA Graph settings, ProcessGroupNCCL options, every source-read NCCL/environment/API acceleration control, and installed distributed registries |
| Repo path | fastpath recipe/precision/step, runner-owned layout, telemetry, DDP, optimizer, and sealed environment values |

The collector first discovers every installed Torch `ConfigModule`, runtime option
registry, relevant callable signature, and source-read accelerator environment/API
control, then cross-checks the surface set against installed Torch source; the explicit
table entries are mandatory minima, not a whitelist. For every discovered entry, record
stable option ID, source path and digest, owning
surface, declared value type, default, complete finite domain, prerequisites, conflicts,
mechanism, and disposition. Domain enumeration uses the declared type/source schema, not
the runtime default type, so nullable and sentinel Booleans are retained. Every Boolean
target, `Literal`/enum member, public preset, and source-defined finite value is listed.
Open numeric/string domains use only a source-defined preset or a separate hash-bound
`sealed_value_registry.json` whose rationale and values are fixed before timing. No
collector entry may disappear by name filtering.

The PyPI wheel is not sufficient source proof for C++ controls. Before inventory sealing,
read `torch.version.git_version`, acquire the official PyTorch upstream source archive for
that exact commit, record URL/commit/archive SHA-256, and scan its C++/CUDA implementation,
including ProcessGroupNCCL environment reads. The archive and extracted-source digest join
the generation fingerprint. Missing source, commit mismatch, archive mutation, or a wheel
without an exact upstream revision stops before ledger sealing. This acquisition belongs
to a separately authorized internet-enabled runtime setup; it is never an implicit local
or remote action.

This includes reducer families, public and experimental compiler/autotune/fusion/layout/
reduction/codegen/memory controls, CUDA Graphs, DDP scheduling/buckets/batched copy,
complete layouts, optimizer alternatives, backend precision, allocator choices, and
sealed NCCL algorithms/protocols/channels/priorities. Experimental, internal, unstable,
or deprecated-but-executable status is not an
exclusion. An item leaves the selectable acceleration universe only with a verifier-
reproducible disposition:

- alias, implied value, or duplicate effective configuration;
- absent executed operator/branch;
- unsupported hardware/topology/runtime;
- illegal prerequisite/conflict state;
- compile/startup/debug/telemetry-only effect with no steady-step mechanism;
- correctness or required-quality failure;
- source-proven speed-first dominance over the admitted finite domain.

Every non-universal proof is keyed by
`runtime + GPU/topology + model/operator-graph digest + exact batch/shape coordinate (or
sealed enumerated regime ID) + complete requested/effective interacting co-assignment`.
Only a source/effective-path proof valid for every generated context ID may
canonicalize or exclude globally. Context-inert or context-invalid values become
conditional constraints. A dominance certificate must cover every compatible context
in the finite admitted domain and be independently reconstructed; otherwise the value
remains admitted.

Activation scouts may reuse live objects inside a compatibility class with exactly that
context key and may
inspect graph breaks, recompilations, generated kernels, requested/effective settings,
operator reachability, DDP logging, CUDA Graph capture/recapture, optimizer path, VRAM,
and cache hits. Scouts are permanently `selectable=false`; their results may determine
ledger legality, activation, compatibility, and ordering, but never rank a runtime.

Experimental does not mean incorrect. Loss/update finiteness, rank agreement, effective
setting fidelity, stable measured graphs, and the downstream real-data quality gate
remain mandatory. Lossy communication or numerical-semantics changes are separate
direct recipes and cannot bypass that quality gate.

## Maximal-compatible recipe construction

The hash-bound registry uses typed records:

```text
factor_id, value_id, value_kind(required_baseline|optional_atom|categorical_choice),
absence/default semantics, prerequisites, conflicts, implications, context_key,
requested/effective identity, interaction_family
```

Required baseline factors choose exactly one value and do not establish maximality.
Optional acceleration atoms use the partial order `absent < present`; values of one
categorical factor are mutually incomparable. Maximality is evaluated separately within
each complete categorical assignment: a legal assignment is inclusion-maximal when no
absent compatible optional atom can be added. Categorical substitutions generate
incomparable, independently re-maximalized recipes and never invalidate the original
because another value admits more optional atoms. The solver emits a symbolic proof and
the fully enumerated canonical effective assignment. Recipe ID is the SHA-256
of the generation digest plus sorted effective factor/value IDs.

Construct a deterministic maximal-cover seed matrix satisfying all of:

1. every admitted value appears in at least one complete seed recipe;
2. every compatible pair of admitted values from distinct factors appears together in
   at least one seed;
3. source-dependent prerequisite triples are forced atomic direct recipes;
4. mutually exclusive alternatives appear in separate otherwise re-maximalized recipes;
5. eager/FSQ controls are recorded without displacing acceleration coverage;
6. no fixed recipe, root, interaction, or contender cap may silently omit required
   coverage.

The seed-cover solver minimizes seed count, then lexicographically minimizes the sorted
recipe-ID vector. The artifact contains the complete candidate universe, constraint
graph, all legal maximal assignments, selected cover, uncovered set, and an independent
cover proof. If a selected seed fails, each value/pair/triple obligation it carried must
be closed by a passing replacement cover or by a context-scoped proved exclusion. If
required coverage is impossible, the generation stops with a precise scope contradiction.
Coverage certificates are model-generation-specific.

Before timing, expand every interaction family below into exact atom/value tuples with
model, context, prerequisites, conflicts, batch obligations, and stable logical action
IDs. Each
tuple is a direct complete recipe; a convenient representative cannot close a family.
The initial required interaction ledger includes installed/applicable tuples for:

- autotune x CUDA Graphs;
- autotune x layout/padding;
- CUDA Graphs x memory planning;
- CUDA Graphs x fused/foreach/capturable optimizer;
- reducer/compiled-autograd x communication scheduling;
- C++ reducer x bucket policy x batched gradient copy;
- communication hook/NCCL choice x bucket policy.
- precision x compiler/autotune/optimizer/CUDA Graphs;
- compiler mode x reducer/compiled-autograd;
- layout x cuDNN/autotune;
- CUDA Graphs x DDP/NCCL/allocator.

The ledger may add interactions discovered from the upgraded official source or
implementation before the matrix is sealed. Performance additivity is never inferred.

## Positive-interaction search

Start with complete inclusion-maximal recipes. Do not require a toggle to win alone and
do not run singleton ablations merely to explain a valid fast recipe.

A valid maximal recipe remains eligible unchanged. The sealed transformation DAG contains
dependency-closed single-family removals, categorical substitutions followed by
re-maximalization, failure `ddmin` splits, removed-family re-additions, and the exact
direct recipes above. Every node/edge has a logical action ID, prerequisite terminal states,
and deterministic lexicographic order. Descend only when:

- the complete recipe fails compilation, correctness, or activation;
- a matched conditional removal proves a lower projected epoch time; or
- a mutually exclusive alternative must be substituted.

A valid node tests every legal dependency-closed single-family removal while holding the
remaining context fixed. Each endpoint uses its closed batch-policy coordinate with the
lowest screening geometric-mean epoch objective; ties use smaller batch, then recipe ID.
Each parent/child edge receives four untouched paired screening blocks at those endpoint
coordinates. For block `i`, define
`r_i = log(epoch_objective(child,i) / epoch_objective(parent,i))`. Promote when
`exp(mean(r)) < 0.99` and the exponentiated one-sided 90% paired-t upper bound is `< 1.0`.
Unresolved edges receive four more blocks; the decision pools all eight. If still
unresolved, close `no_proved_descent`. Recurse from every promoted child without
revisiting an effective recipe ID. This finite conditional descent is for speed, not
sparsity. It can be large; there is no GPU budget cap.

A compilation, correctness, or activation-invalid maximal recipe triggers deterministic
dependency-aware `ddmin`: split the
sorted removable-group vector into stable complementary halves, test both legal halves,
and recurse until each invalid interaction is isolated or every leaf is terminal. Every
diagnostic child keeps unrelated acceleration atoms enabled. A group failure does not
condemn every member. Passing rescue branches must re-cover the failed seed's obligations.
Every family removed on a terminal retained branch is re-added once in that final context
to detect masking or sign reversal. Diagnostic rows are non-selectable unless their
complete recipe independently satisfies the selectable protocol.

Complex or higher-order interactions in the sealed interaction ledger are measured
directly. Arbitrary undeclared higher-order global optimality is out of scope.

## Joint configuration and batch search

Search `(complete_recipe, per_device_batch)` jointly. Do not certify exact Bmax for every
recipe and do not assume capacity monotonicity: the immutable evidence contains larger
passing batches above smaller OOM coordinates. An OOM invalidates only that exact
recipe/batch coordinate unless a separate proof establishes a bound.

Before timing, compute

```text
b_absolute = min(floor(P / 2), floor(min_rank_total_vram_bytes /
                                     input_tensor_bytes_per_sample))
```

The second term is a proof-only impossibility bound using the exact input dtype/shape;
it does not claim feasibility. The sealed batch domain is every integer `1..b_absolute`.
The bounded policy does not measure all of it. For every complete recipe, mandatory
anchors are the in-domain union of powers of two, `{12,24,48,54,60,64,70,72}`, and two
same-model exact-family old coordinates using a hash-bound family-ID map sealed before
timing. Qualified old rows require `status=pass`, `selectable=true`, `world_size=2`, exact
family recipe/effective identity, and verified measurement protocol. `old_epoch_best` is
the smallest `projected_epoch_seconds` among qualified `probe_phase` values beginning
`epoch_`, ties by smaller batch then probe ID. `old_high` is the largest qualified passing
per-device batch, ties by probe ID. The map stores coordinate/probe IDs; no fuzzy analogue
or other phase adds anchors. Every anchor runs even after a lower OOM; structured capacity
failure is coordinate-local, never invalidates a recipe/atom, and never invokes recipe
`ddmin`.

After anchors, schedule `b +/- {1,2}` around every successful anchor. Then run exactly
three adaptive rounds. In each round, any newly measured successful coordinate whose
objective is lower than the recipe's previous best schedules its in-domain, not-yet-seen
`b +/- {1,2}` neighbors for the next round. Round three adds no successors. Stable action
IDs deduplicate all coordinates. Pass-at-4/OOM-at-5/pass-at-6 is a valid closed trace when
all three actions and triggered neighbors are terminal; no monotone repair is invented.

Use the deterministic constraint-aware transformation DAG above to propose:

- new batch coordinates for an existing complete recipe;
- dependency-closed removal from a maximal recipe;
- mutually exclusive substitution followed by re-maximalization;
- a sealed direct-interaction recipe.

Every promoted configuration is a new complete recipe. Parent and old rows choose its
first coordinates only and never become child evidence. Screening observations guide
adaptation; they are not reused as final confidence replicates.

Before the first selectable row, seal the complete conditional action universe: every
generation/recipe/batch/phase/repetition logical action ID, transformation edge, branch predicate,
retry edge, prerequisite, successor rule, deduplication rule, and lexicographic tie-break.
Results activate presealed branches but cannot invent actions. An action is terminal only
as `pass`, coordinate-local `capacity_failure`, proved scoped exclusion, invalid recipe,
`no_proved_descent`, or verified retry-exhausted `unknown`. The controller completes only
when every mandatory/activated logical action ID is terminal and cover/confirmation obligations
close. A mandatory `unknown` yields `incomplete_no_narrowing`. Historical wall/probe
budgets never terminate the search.

## Immutable v2 evidence and avoided work

The canonical repo evidence root is
`docs/data/spec0011_runtime_recipe_v2/`. It contains four pinned artifacts, preserved
`producer/run.py`, and `evidence_manifest.json`. The v4 builder must read only this path
and fail closed if its checked-in manifest or any payload differs. All six files are
Git-tracked by the focused relock commit; agent preflight enforces this fresh-clone
guarantee. The recovered package contains
exactly 309 contiguous rows `p00001..p00309`, stable
experiment ID `34bf23c5370815c37a2d50cf702609f4e84fcfc99dd16e476f49096f97c67a35`,
and matrix SHA-256
`dee362d15a8c4a324d9c1f10bdec0b9aa8e1fafbf522e90d638fafcbc1571536`.
It consumed 18.19 subprocess-hours, including 6.35 hours of irregular all-batch timing.
Old-v2 `p00310` is a provenance locator only and is permanently unschedulable.

The package is immutable source evidence. The importer verifies its pinned manifest,
producer, payload, rows, raw aggregates, model/step/recipe/protocol/effective settings,
hardware, and runtime identity. Canonical identities are:

```text
executed_semantics_digest = AST/source digests of the worker, timed step, model, and every
                            executed dependency after the single whitelist below
protocol_digest           = ordered warmup/settle/measure/update/rank/timing protocol
recipe_id                 = generation digest + canonical requested/effective assignment
row_role_key              = recipe_id + batch + phase + repetition + protocol_digest
logical_action_id         = generation + recipe + batch + phase + repetition/obligation
attempt_id                = logical_action_id + zero-based attempt_index
confirmation_obligation_id = generation + singleton_or_sorted_pair_id + block_index
block_instance_id        = confirmation_obligation_id + drift_replacement_index(0|1)
block_attempt_id         = block_instance_id + execution_attempt_index(0|1)
child_row_id              = block_attempt_id + role_ordinal
```

Confirmation role ordinals are exactly `anchor_before`, `singleton`, `anchor_after` for
one finalist, or `anchor_before`, `candidate_first`, `candidate_second`, `anchor_after`
for a pair; candidate order is fixed by block parity. Confirmation rows deduplicate/conflict by
`child_row_id`. A block attempt is keyed by `block_attempt_id` plus the ordered tuple of
child-row hashes; missing, duplicate, swapped, or extra roles invalidate the attempt.

The only normalization whitelist is the emitted result dictionary field named exactly
`schema_version` inside `_measure_candidate_worker`: producer `SCHEMA_VERSION` and current
`LEGACY_SCHEMA_VERSION` may canonicalize only when both evaluate to the exact v2 schema
literal and no control/data dependency consumes that field during measurement. Every
other AST, constant, config, dependency, or protocol mutation changes identity.

An exact identity match may close only the identical observation role. The mandatory
Torch upgrade will normally make old timings prior-only. Prior-only rows still:

- order reducer/compiler families and maximal seeds;
- select batch pivots, old wells, local neighborhoods, and VRAM priors;
- prevent repeated standalone sweeps for old eager, no-optimization, lite,
  channels-last, max-autotune, Python-reducer, and C++-DDP recipes;
- calibrate irregular/non-monotone capacity and uncertainty;
- prioritize direct interactions without proving them.

No old row proves a new maximal bundle, child, or current final repetition. V2 checkpoint
position, controller state, standings, frontiers, and results never transfer.

The planner emits an `evidence_reuse_report` listing, for every old row, exact role or
prior use, rejection reason, and every fresh action avoided because the package exists.
The verifier independently builds the sealed evidence-disabled plan and evidence-enabled
plan; `avoided_action_ids` is their exact set difference after removing calibration-only
actions. Only same-model rows may appear in that set. Cross-model rows affect ordering
only and never count as avoided measurement.
With the package present, the plan must not schedule any old singleton all-batch sweep.
Only minimal current-runtime reference coordinates may repeat an old recipe to measure
runtime drift or connect priors; they must be labeled calibration, not rediscovery.

## Selectable-process and compiler-cache reuse

Live-object capsules are diagnostic by default. A selectable capsule class is opt-in and
keyed by runtime/model/recipe plus an exact ordered shape list. Every coordinate restores
the same initial model/optimizer/scaler state and uses its own deterministic batch-sized
input/RNG digest; inputs of different shapes are not called identical. Because allocator,
autotune, graph, and cache history cannot be reset reliably, the class must compare both
forward and reverse coordinate orders against three isolated fresh-process repetitions
per shape. It proves output/loss/gradient/update equivalence, identical effective graph
and settings, no order-dependent graph/recompile activity, safe VRAM, and a simultaneous
95% timing-equivalence interval wholly within `[0.98, 1.02]`. For each
`shape x order x repetition`, pair capsule and isolated wall/update values and define
`q_i = log(capsule/isolated)`. Use two one-sided paired-t equivalence tests against
`log(0.98)` and `log(1.02)` with Bonferroni familywise `alpha=0.05` across every
shape/order comparison in the class; both adjusted bounds must lie inside the margins.
Otherwise every capsule row remains diagnostic. Capacity failures that can poison a
process are always isolated.

Final confirmation blocks are fresh processes with no inherited live model, DDP,
optimizer/scaler, allocator, CUDA Graph, or CUDA-process state.

Cross-process compiler-cache reuse is allowed only for an exact runtime, payload, model,
recipe, batch/shape-regime, compiler, and GPU key. The first required cold and warm
processes must prove output/loss/gradient/update equivalence, requested/effective
settings, rank agreement, graph identity/stability, VRAM safety, actual cache hit, and
bounded path-safe cleanup. Failure disables reuse for that key and schedules a fresh
cold replacement. Each recipe/batch key then seals one confirmation cache regime: all
confirmation blocks use certified warm exact-key caches, or all use private cold caches
if warm proof fails. A final process never inherits live CUDA state even when it reads a
certified disk cache. Cache reuse may reduce startup cost but never changes the objective
or fresh-process confirmation requirement.

## Resumable Kaggle execution and failure semantics

There is no v4 total GPU-hour, probe-count, recipe-count, session-count, or two-session
wall limit. Coverage and correct selection take priority over minimizing Kaggle GPU use.

The finite plan remains operationally bounded:

- every probe has a declared timeout and at most one identical fresh retry;
- every Kaggle session has a platform-aware no-start deadline and reserved shutdown,
  cleanup, hash, verification, and publication margin;
- every row, ledger transition, cache journal, and controller checkpoint is atomic;
- a session may end with a verified resumable partial artifact and no narrowing;
- the next session upgrades/inventories first and resumes only if the performance-
  relevant fingerprint and sealed-plan compatibility rules pass;
- actual subprocess/GPU/wall consumption is recorded, not optimized.

One selectable generation is identified by

```text
generation_digest = SHA256(payload + model + timed-step/protocol + upgraded runtime/GPU
                           fingerprint + inventory + acceleration ledger + constraint/
                           cover/interaction registries + batch domain + policy)
```

Every session writes an append-only `session_manifest.json` containing generation digest,
session ID, exact parent artifact-manifest hash (null only for genesis), completed/pending
logical action IDs, attempt IDs/counts, row and checkpoint hashes, cache journals, wall/GPU
accounting, and terminal state. Ordinary probe rows are keyed by `attempt_id`;
confirmation child rows are keyed by `child_row_id`. Exact retransmission of either key
with the same hash deduplicates, while a differing hash for the same key corrupts the
generation. Confirmation block closure is keyed by `block_attempt_id` plus the ordered
child-hash tuple. The verifier reduces at most two ordered execution attempts per logical
probe or block instance using the exact transition table below. Execution retries never
collide with drift-replacement instances.

The guarded resume transport is: explicitly authorized output download; local independent
verification; builder embedding of the compact verified parent artifact under
`source_evidence/spec0011_runtime_recipe_v4_parent`; separately authorized next push;
and in-kernel verification before import. Publication and import are atomic. Global retry
counts come from the verified parent chain. No session reads an unverified predecessor.

Fingerprint drift never silently mixes timings. Any performance-relevant generation
field drift starts a new generation; old rows become immutable prior-only evidence and
never mix selectably. Non-performance publication metadata may change without changing
the digest. Compatible completed work within one digest remains selectable.

Outcome closure is fixed:

| Outcome | Retry | Obligation effect |
| --- | --- | --- |
| successful qualified pass | none | closes the logical action `pass` |
| structured OOM/allocation | none | closes only that recipe/batch coordinate as capacity failure |
| timeout/generic crash | one identical fresh retry globally | second failure closes `unknown`; mandatory action makes generation incomplete |
| wrong GPU/rank/topology/runtime | none | invalid session; no selectable rows |
| unsupported/inapplicable option proven for context | none | scoped exclusion; re-cover affected obligations |
| requested/effective mismatch | one activation recheck | scoped activation failure; re-cover or incomplete |
| compilation failure | one identical fresh retry, then `ddmin` | recipe invalid; atoms remain admitted until isolated; re-cover |
| NaN/missing update/rank disagreement/unstable graph | none | recipe correctness-invalid; re-cover obligations |
| corrupt handoff/hash/parent conflict | none | generation incomplete; no import or narrowing |

Attempt reduction is deterministic:

| Ordered state | Next/terminal state |
| --- | --- |
| attempt 0 is qualified pass or any non-retry outcome | that outcome is terminal; attempt 1 is forbidden |
| attempt 0 is timeout/crash, compilation failure, or activation mismatch | activate attempt index 1 of the identical logical action |
| attempt 1 is qualified pass | terminal `pass`, superseding the retryable attempt-0 failure |
| attempt 1 is timeout/crash | terminal `unknown` |
| attempt 1 is compilation failure | terminal recipe compilation-invalid; activate sealed `ddmin` and re-cover |
| attempt 1 is activation mismatch | terminal scoped activation failure; re-cover or incomplete |
| attempt 1 is another non-retry outcome | apply that outcome's terminal rule |

Attempt 1 without a retryable attempt 0, any attempt index above one, a retry after a
terminal attempt 0, or an outcome not legal for its phase corrupts the generation. This
reducer is identical across sessions.

Any unverifiable row or unfinished mandatory scope yields `incomplete_no_narrowing` and
no certified winner.

Remote execution still requires fresh explicit user authorization and the guarded Kaggle
workflow. Unlimited search budget is not standing authorization for a push, status read,
output download, or other remote action.

## Final confirmation and claim

After every search obligation closes, each valid recipe contributes its closed-policy
batch with lowest screening objective; ties use smaller batch, then recipe ID. Run exactly
three new fresh-process screening repetitions at each contributed coordinate. Screening
score is the geometric mean of the three exact epoch objectives; its interval is the
ordinary two-sided 95% t-interval on their logs. If no valid recipe contributes, close
`incomplete_no_narrowing`. Otherwise let `s_best` be the lowest score, ties by recipe/
batch ID. The finalist set contains every candidate with score `<= 1.05*s_best` or
whose exponentiated lower interval bound is `<= 1.05*s_best`. Confirm every member; there
is no subset or cardinality cap.

With one finalist, run eight fresh candidate blocks bracketed by the same drift anchor;
successful drift-valid completion emits that sole compute finalist with claim
`only qualifying finalist`, not a pairwise 1%-lead claim. With two or more finalists,
confirmation uses a session-bounded complete pairwise design, not one indivisible
all-finalist block. For every unordered finalist pair in lexicographic recipe/batch order,
run exactly eight within-session paired blocks. Odd blocks measure `a` then `c`; even
blocks reverse the order. Each block measures the fixed drift anchor—upgraded eager fp16
control at `b=12`, exact recipe ID sealed before screening—immediately before and after
the singleton or ordered pair. Singleton and pair blocks use the same block identity,
whole-attempt retry, drift, and no-start rules; the four-process bound conservatively
covers the singleton's three processes. At plan seal, prove
`4 * probe_timeout + publication_margin <= fresh_session_usable_wall`; otherwise stop
incomplete before confirmation. The no-start rule launches a block only if that full
allowance fits the current session; otherwise the block waits for a fresh session. The
entire ordered anchor-before/singleton-or-pair/anchor-after block is one confirmation
obligation with original block instance index 0;
its three or four subprocess rows use the sealed child-row IDs above. Any timeout or
generic crash, compilation failure, or activation mismatch discards the whole block
attempt from selectable/statistical evidence and activates exactly one identical
whole-block execution attempt in a fresh session. The second execution attempt must
independently fit the same bound and remain within one session. A second retryable
failure applies its global terminal outcome and makes confirmation incomplete; any
constituent non-retry terminal failure does likewise. Individual constituent retries are
forbidden, preserving pairing/bracketing.

For unordered pair `(a,c)` and paired block `i`, define
`d_i = log(epoch_objective(a,i) / epoch_objective(c,i))`. Report ordinary two-sided 95%
paired t-intervals on mean `d`. For selection, compute each two-sided paired-t p-value,
sort by `(p_value, pair_id)`, and apply Holm step-down familywise `alpha=0.05`: comparison
rank `i` among `m` is rejected only while `p_i <= 0.05/(m-i+1)`; stop at the first failure
and retain all later hypotheses. The proposed unique winner is the finalist with
lowest across-block geometric mean, ties by recipe/batch ID. It wins only if, against
every competitor, `exp(mean(d_w,c)) <= 0.99`, Holm rejects equality in the winner's
faster direction, and `d_w,c < 0` in at least six of eight blocks.

Reference drift first applies the local rule: one original block is locally invalid when
its after/before anchor ratio leaves `[0.98,1.02]`. From all locally valid original blocks,
compute each session's mean anchor log value. If none is locally valid, terminate
`timing-incomplete` with no narrowing. Otherwise freeze the generation baseline as the
median of those session means (for even count, the arithmetic mean of the two middle
sorted values). Classify originals once: discard every original block from a session
whose mean differs from the frozen baseline by more than `log(1.02)`, plus every locally
invalid block. Each discarded original block instance activates exactly one presealed
whole-block drift replacement with instance index 1 in a later session. That replacement
has its own execution attempts 0/1 under the same obligation and may use the whole-block
execution retry above without ID collision. A replacement must pass the local rule and its session
mean must be within `log(1.02)` of the frozen baseline; a second failure is terminal
timing-incomplete. The baseline is never recomputed. Because every candidate pair is
within-block paired, valid blocks from accepted session strata pool directly. If no unique
winner satisfies every comparison, the complete tied set contains every finalist not
uniquely beaten by another under the same pair rule; cycles/unresolved comparisons retain
all involved finalists.

The admissible claim is:

> Fastest statistically confirmed recipe/batch among the executed maximal-cover matrix
> and its sealed direct-interaction, substitution, failure-rescue, and adaptive batch
> neighborhood, with complete admitted-value and declared interaction coverage.

Do not claim a global optimum over open numeric domains, every undeclared higher-order
interaction, or unmeasured integer batches. A complete no-dataset selector writes only
`compute_finalists.json`, never `selected_runtime.json`.

## Artifact and independent verifier

The artifact contains upgraded inventory/fingerprint, immutable evidence/import/reuse
reports, acceleration ledger, constraint graph, maximal assignments and cover proof,
direct-interaction ledger, sealed finite work plan, raw probe matrix, atomic checkpoints,
cache certificates, adaptive trace, confirmation blocks, results, and hashes binding the
policy, source, payload, settings, and evidence.

The read-only verifier lives in a separate `runtime_recipe_verify.py` module/CLI and
consumes only raw/hash-bound artifacts. It may share schema/data parsing but must not
import controller frontier, proposal, acceptance, completeness, or winner helpers. It
independently reconstructs effective option dispositions,
constraints, cover completeness, recipe identities, evidence reuse, avoided work,
adaptive legality, exact objective values, confirmation statistics, session handoffs,
cache/process proofs, completeness, and hashes. It must not trust controller-derived
acceptance, frontier, winner, or certificate fields. The artifact preserves compact raw
source excerpts/digests and activation probes needed to reconstruct source-based
exclusions.

`complete=true` requires every sealed coverage and interaction obligation to close.
Incomplete artifacts may retain ranked diagnostic telemetry and
`best_observed_noncertified`, but have `certified_winner=null`, no usable frontier, and
`selected_runtime_written=false`.

## Implementation order and files

Implement only after this amended text passes independent clean-context review and is
relocked implementation-ready.

1. Replace v3 pure policy in
   `src/eqvae/benchmarking/runtime_recipe_search.py` with inventory constraints,
   maximal-cover generation, evidence-aware planning, joint recipe/batch proposals, and
   confirmation policy.
2. Refactor `src/eqvae/benchmarking/runtime_recipe_bakeoff.py` for the v4 schema,
   activation scouts, immutable importer, recipe capsules/fresh confirmation,
   resumable multi-session controller, and raw artifacts. Implement the independent
   verifier in `src/eqvae/benchmarking/runtime_recipe_verify.py`.
3. Update `kaggle/kernels/runtime_recipe_bakeoff/run_template.py`,
   `scripts/build_kaggle_embedded_kernel.py`, and `scripts/kaggle_kernel.sh` for the
   sealed v4 payload and resumable guarded workflow.
4. Replace obsolete v3 expectations in
   `tests/test_runtime_recipe_search.py`, `tests/test_runtime_recipe_bakeoff.py`,
   `tests/test_kaggle_embedded_kernel.py`, and `tests/test_kaggle_torch_upgrade.py`.
5. Keep `CURRENT.md`, `GOAL.md`, the specs index, Kaggle behavior inventory, and CLI
   workflow synchronized; never restore stale v2/v3 Bmax, 118-slot, or two-session text.

Preserve unrelated dirty work and reuse sound partial helpers. A v3-only helper, test,
artifact, or workflow change may be surgically deleted or reverted after its provenance
and lack of v4 value are verified. Do not use bulk Git restoration, broadly rewrite
unrelated files, or execute old-v2 `p00310`. Record targeted removals in the active spec
and `CURRENT.md` before handoff.

## Acceptance criteria

1. Latest-Torch inventory gives every finite option value a verifier-rederived inclusion
   or proved exclusion disposition; experimental status alone never excludes it.
2. The constraint solver and independent verifier prove complete admitted-value,
   required compatible-pair, prerequisite-triple, and sealed direct-interaction coverage
   without arbitrary recipe/interaction caps.
3. Complete recipes begin inclusion-maximal; removal is conditional on failure or lower
   projected epoch time, never on a minimal-toggle objective.
4. The 309-row package is hash-bound, semantically imported, materially changes planning,
   prevents old singleton sweeps, and produces an auditable avoided-work report.
5. Configuration and batch are ranked jointly by the exact floored synchronized epoch
   objective; OOM is coordinate-local absent proof, and no exact-Bmax/monotone assumption
   prunes a recipe.
6. New promoted recipes have independent current evidence. Screening/capsule/cache reuse
   satisfies its proof boundary; final confirmation remains untouched and fresh-process.
7. Resumable sessions can continue until finite scope closure without a total GPU/time
   cap, while per-probe/session timeout, deadline, cleanup, checkpoint, fingerprint, and
   remote-authorization guards remain enforced.
8. Independent verification recomputes raw evidence, cover/adaptive legality, statistics,
   completeness, and failure semantics. Incomplete/corrupt/unresolved work cannot narrow.
9. A complete result emits only verified compute finalists and accurately scopes its
   claim; loader, quality/LR, promotion, and full training remain later gates.
10. Focused tests, semantic mutation tests, clean-context reviews, repo quality gates,
    build preflight, Bash syntax, diff checks, and handoff memory pass locally.

## Focused verification

- Inventory mutation tests for omitted Boolean/enum values, aliases, prerequisites,
  conflicts, activation, operator reachability, hardware applicability, experimental
  controls, upstream C++ source commit/archive mismatch, and omitted source-read env/API.
- Constraint/cover tests for maximality, every admitted value, every compatible pair of
  admitted values from distinct factors, forced triples, direct interactions,
  determinism, arbitrary-pair omission, impossible coverage, and no silent caps.
- Import/reuse mutations for every pinned hash, semantic-digest normalization, runtime,
  recipe/effective setting, raw evidence, role, avoided action, and `p00310` exclusion.
- Positive-interaction tests for valid maximal winners, conditional removal, failure
  `ddmin`, alternative substitution/re-maximalization, re-addition, and direct recipes.
- Joint batch tests for irregular/non-monotone capacity, coordinate-local OOM, old priors,
  local neighbors, edge expansion, and exact floored objective replay.
- Process/cache tests for capsule restoration, contamination rejection, cold/warm proof,
  fresh confirmation, cleanup, crash recovery, and path safety.
- Multi-session tests for atomic checkpoints, clean resume, fingerprint drift,
  timeout-to-pass, timeout-to-timeout, compile-fail-to-pass/ddmin, activation recheck,
  cross-session retry, exact retransmission, no-start/block-fit deadline, partial evidence,
  scope completion, and no-winner incomplete states.
- Confirmation/verifier tests for zero/one/two/many finalists, balanced pair blocks,
  end-of-session failure at each of four block positions with whole-block retry, equal-p
  Holm tie-break/stop, paired statistics, frozen/even-session drift baseline, zero valid
  original blocks, one replacement, singleton three-role blocks, swapped/missing/
  duplicate child roles, exact block retransmission, tied finalists, winner mutations,
  complete coverage, and controller-independent reconstruction.

Run from the repo:

```bash
.venv/bin/ruff format --check <touched Python files>
.venv/bin/ruff check <touched Python files>
CUDA_VISIBLE_DEVICES="" .venv/bin/pytest \
  tests/test_runtime_recipe_search.py \
  tests/test_runtime_recipe_bakeoff.py \
  tests/test_kaggle_embedded_kernel.py \
  tests/test_kaggle_torch_upgrade.py
.venv/bin/basedpyright
bash -n scripts/kaggle_kernel.sh
./scripts/kaggle_kernel.sh preflight-runtime-recipe-bakeoff
git diff --check
./scripts/python_quality.sh
```

CUDA/DDP performance and activation remain Kaggle observations requiring separate
explicit permission.

## Independent review

Before relock, independent clean-context reviewers audit:

- toggle completeness, source/activation exclusions, compatibility, maximal-cover and
  direct-interaction scope;
- evidence identity/reuse, joint batch policy, statistics, process/cache boundaries,
  resumable execution, verifier independence, and failure semantics.

After implementation, repeat adversarial reviews for controller/verifier correctness,
test soundness, dirty-tree safety, and stale-v3 cleanup.

Two independent xhigh clean-context review tracks completed on 2026-08-08. Their
inventory/context/maximality/descent/batch and evidence/lineage/statistics/process/verifier
blocker/high findings were integrated through repeated narrow re-review. Both final
verdicts report no remaining blocker/high contract finding. The focused local relock
commit tracks all six canonical evidence files and closes their only condition.

## Non-goals

- Minimal enabled-toggle count or attribution of individual speedups.
- Exhaustive power-set or undeclared arbitrary higher-order interaction search.
- Global optimization over open numeric domains.
- Deterministic training or exact RNG resume.
- Tail preservation or partial-batch fallback.
- Selecting loader/LR/quality/full training from generated tensors.
- Any remote action without fresh explicit user permission.

## Related sources

- `AGENTS.md`, `CURRENT.md`, `GOAL.md`
- `docs/spec_driven_development.md`, `docs/agentic_review_workflow.md`
- `docs/decisions/0012-kaggle-runtime-torch-upgrade.md`
- `docs/kaggle_cli_workflow.md`, `docs/behavior_inventory_kaggle.md`
- `kaggle/fsq_train_reference.py`
