# Current Repository Status

Last updated: 2026-10-26

The two VAE checkpoints remain frozen at 60,000 updates. Scientific boundaries
and foreground limitations are in [AGENTS.md](AGENTS.md) and [GOAL.md](GOAL.md).
The 152 external WSI remain reserved for final evaluation.

## WSI classification — Spec 0054

[Scientific spec](docs/specs/0054-cross-validated-mil-dynamics-and-telemetry.md)
· [classical contract](docs/data/spec0054_classical_wsi.json).

The five-fold comparison of six fixed representations is complete. PANTHER
full16 has mean validation macro-F1 0.5545 normal / 0.5587 SO(2); mean-only
SO(2) has CE 1.0206. Normal mean/std retains its highest mean F1, 0.5612.
All PANTHER fits converged, but the train/validation gap remains substantial.
Full results, protocol and audit:
[professor report](reports/professor/spec0054_panther_20261022.md).
Published in commit `75ebc76` and
[MIDL issue update](https://github.com/HiperMaximus/equivariant-vae/issues/8#issuecomment-6350435031).

PANTHER execution: `maximshtefan/eqvae-spec0054-panther/1`, complete.
Source `45f64fe`; entry `48b8667`. Downloaded outputs:
`runs/kaggle/spec0054_panther_v1_outputs/`; inspection tables:
`runs/kaggle/spec0054_panther_v1_review/`. Both construction and classification
ran on Kaggle CPU. Use Kaggle for subsequent experiments, including CPU work.
No follow-up classical-head regularization recipe has been selected.

The CLS + gated-ABMIL dropout/smoothing/C4 candidate (`7ad0601`) started from
the original classifier initialization in Kaggle v13 on2026-10-22 at16:15 Bogota.
Inspection on2026-10-23 confirms `maximshtefan/eqvae-spec0054-mil-fold0/13`
COMPLETE, both workers exit0 and pause normally at the deadline. Both latest
states have128 updates/512 presentations, epoch1 cursor224 (1.7778 epochs).
Last complete and best-by-CE assessment is epoch1: normal validation
CE/F1 1.60605/.14074; SO2 1.60574/.11461. Both remain in warmup; later
generalization and benefits of the combined recipe are unmeasured. D4 is deferred.

The authorized next training session resumes these same v13 states via the new
private `maximshtefan/eqvae-spec0054-cls-c4-fold0-resume/1` dataset, status ready.
Both latest checkpoints retain128 updates/512 presentations/epoch1 cursor224,
optimizer/scaler/RNG and last completed assessment288. All14 uploaded resume
files were downloaded individually and match v13 byte-for-byte; the whole ZIP
endpoint returned404. Receipt:
`runs/local/spec0054_cls_fold0_c4_v13_review/resume_dataset_v1.json`.
The entry restores source `7ad06010416a789a6b96e612919203c162adc23e`, RESUME=True,
READ_PROBE=False, full original12-shard metadata and direct input reads.
Normal/SO2 use separate physical GPUs0/1 on the original T4 allocation. Recipe,
105-epoch horizon and144-presentation checkpoint cadence remain unchanged.
Submission cutoff2026-10-26T11:31:37-05:00 reserves15min of the12-hour session.
Pinned runner compilation, recipe parse, kernel build/validation and entry Ruff
pass. Local entry commit `c0d7e8a`; submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/18` on2026-10-25 around23:47 Bogota.
API confirms RUNNING. Initial logs confirm source7ad0601, the new private
resume root and independent normal/SO2 Tesla T4 workers on physical GPUs0/1,
with all probes disabled. Both `resume_done` logs confirm update128,
exposures512, epoch1/cursor224 and last assessment288. Initial paired WSI40079
subset/C4 hashes agree. No new source publication was required:
the unchanged training code/config are already published at the v13 pin.

Live inspection on2026-10-26 around09:38 Bogota confirms v18 still RUNNING.
Latest visible progress is update348/exposures1392 for both branches, epoch4
cursor240 (4.8333 epochs);220 updates/880 presentations have advanced in this
session. Latest visible saved checkpoint is update324/exposures1296. Last
complete assessment is epoch3: validation CE1.60842 normal/1.57188 SO2. Best
CE events select normal epoch2 (1.57755) and SO2 epoch3 (1.57188); macro-F1
is unavailable from the live event stream. Both are still before the end of
the five-epoch warmup. All880 consumed WSI/subset/C4 records agree across
branches; no AMP skipped attempts/backoffs are visible.

Full-atlas subsequent reads remain slow: median22.15/21.88s, mean39.14/39.07s,
versus matched v10 medians4.46/4.55s and means5.16/5.19s. All880 WSI requests
have prior v13 training history; these are not initial dataset accesses.
Mean step89.66/89.46s versus matched v10 12.66/12.81s; input waits consume
97.6--97.7% of step time. Shard1 median37.94/39.78s and mean95.13/96.08s,
with slow reads also on the other shards. The fast six-WSI probe did not
establish full-working-set performance. Exact storage/cache cause is unmeasured.
Live snapshot only, not final checkpoints/metrics. Review:
`runs/local/spec0054_cls_fold0_c4_v18_live_review_20261026/summary.json`.
The session is unchanged and retains cutoff11:31:37 Bogota; inspect after11:45
for completed artifacts rather than repeatedly polling unchanged live state.

Followup after the deadline confirms v18 COMPLETE, worker exits[0,0]; both
latest states are update360/exposures1440/epoch5 cursor0. The epoch5 assessment
paused at the session deadline and must replay on continuation. Saved outputs:
`runs/kaggle/spec0054_cls_fold0_c4_v18/`. A future training session must package
these new latest states; private C4 resume dataset version1 still contains v13.

Final v18 review confirms both latest states at360 updates/1440 presentations,
with all1440 cumulative WSI/subset/C4/LR records paired and the v13 telemetry
prefix unchanged. No AMP skips, nonfinite losses or nonfinite-gradient counters.
Best-by-CE checkpoints: normal epoch2, validation CE1.57755/macroF1.17354;
SO2 epoch3, CE1.57188/macroF1.25139. Latest complete assessment is epoch3
(normal CE1.60842/F1.12982; SO2 CE1.57188/F1.25139). Epoch5 assessment remains
pending, with100 of105 training epochs left. This session added232 updates;
mean step90.12/90.01s, median read22.17/21.95s and queue waits97.7--97.8%.
Numerical review: `runs/local/spec0054_cls_fold0_c4_v18_review/summary.json`.

Matched early-epoch comparison on2026-10-26 is mixed: at epoch3, old/new
validation CE is1.59161/1.60842 normal and1.57693/1.57188 SO2; macroF1
is.19500/.12982 and.21244/.25139 respectively. Epoch2 normal favored the
new recipe. Both have completed only warmup, with the new epoch5 assessment
pending; no post-warmup evidence yet. Prior fresh LR confirmation favored
5e-4 over1e-3 under the older recipe. Recommendation is to assess epoch5 and
observe post-warmup training before deciding on a narrow LR comparison;
no LR search or further training session was launched by this review.
Evidence: `runs/local/spec0054_cls_fold0_c4_v18_review/matched_early_assessments.json`.

External I/O context reviewed2026-10-26: a Kaggle staff member reproduced
fast2GB/slow30GB reads in an older `/kaggle/tmp` report, a different mount from
this experiment ([thread](https://www.kaggle.com/discussions/product-feedback/443920)).
Kaggle documents29GB RAM for T4x2; the v17 panel touches3.17GB paired versus
200.33GB for the atlas. Linux page-cache reclamation provides a plausible
small-panel/full-atlas mechanism, not proof of this session's cause or the
v10 regression. Actual input filesystem/cache counters are unmeasured.
No matching current incident was confirmed; no running-session change.

The authorized CPU followup measures the same six training WSI with the
unchanged region reader versus full-bag reading followed by the identical60%
subset. For each WSI/method it requests bag-range POSIX_FADV_DONTNEED, measures
actual page residency before/after via mincore, then immediately repeats the
read. Two readers per branch and counterbalanced method order give48 reads.
Record actual mount type, UID, working/scratch space/device, memory availability,
process I/O and host reclaim/refault snapshots alongside timing and value hashes.
The advice may only partially discard pages; no global cache drop, local file
copy, forced memory allocation or training change is involved. Process/host
counters are shared and mincore is only a point-in-time kernel-cache snapshot,
not a measurement of backend caching. The30-minute probe does not reproduce
the entire atlas working set. A48-read synthetic check confirms independent
content parity, page counts/residency and mount metadata; seven reader tests,
compilation, Ruff and targeted BasedPyright pass. Published source `4e3c6fc`,
entry `d7afccb`; submitted as `maximshtefan/eqvae-spec0054-mil-fold0/19`
on2026-10-26 around12:20 Bogota, API status RUNNING. Shared cutoff is
2026-10-26T12:49:52-05:00. Inspection around13:25 Bogota still reports RUNNING,
but a45-second live stream returns no bytes; no cache or timing measurement is
available. Authenticated metadata confirms latest version19, source4e3c6fc and
GPU disabled. The explicit numeric-version stream returns404, which does not
by itself diagnose the session phase. Platform setup, in-flight I/O and log
delivery cannot be distinguished from this evidence. The cutoff is checked
inside the code between reads; it cannot force setup or a blocked call to end.
No cancellation, relaunch or source change. Status receipt:
`runs/local/spec0054_io_cache_v19_review/status_20261026.json`. Inspect after
14:00 Bogota or once logs/artifacts become available. V18 is preserved separately.

Reinspection around15:35 Bogota still reports v19 RUNNING, about3h15 after
submission, with no data in a30-second stream or static logs. The file listing
exposes only previous CLS training outputs dated04:47UTC, not v19 probe results.
No code-start, mount, cache or timing result is available. Treat this as an
unusable pending session rather than measured I/O. A controlled relaunch should
calculate its read budget from actual code startup so platform preparation does
not consume the entire absolute deadline. No cancellation, rerun or edit has
been performed. Receipt: `runs/local/spec0054_io_cache_v19_review/status_1535_20261026.json`.

At17:17 Bogota v19 is COMPLETE, workers[0,0], but both probe results have
completed=false and0 reads: the absolute deadline had expired before any
read could start. The outputs do identify both inputs as NFS, device0:52,
owner UID65534 with process euid0. Working free space is20.94GB/device1794;
scratch reports1125.98GB/device53, a different filesystem. Scratch type,
quota and actual performance remain unmeasured; reported free space alone
does not establish a fast local cache. Outputs: `runs/kaggle/spec0054_io_cache_v19/`;
review: `runs/local/spec0054_io_cache_v19_review/summary.json`.

To complete the authorized measurement, one controlled repeat was submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/20` around17:23 Bogota on2026-10-26.
The source4e3c6fc and48-read comparison are unchanged. Entry `ad1f43d` computes
the30-minute read budget when the code starts, rather than before Kaggle setup;
in-flight I/O can still overrun it. Build, validation and entry Ruff pass.
API confirms RUNNING. No GPU, training change or new source publication.
Inspect around18:00 Bogota, or once measurements are available.

Inspection around18:14 Bogota confirms v20 COMPLETE and both branches
completed=true,24 reads each. All request/subset keys pair across branches;
all four method/repeat value hashes agree within each branch/WSI. After
DONTNEED, measured client-cache residency is0--0.128%; pooled median
read-plus-subset is3.839s for sampled regions versus1.680s for full-bag then
the identical subset (means3.823/2.304s). Immediate-repeat medians are.0525s
and.1664s respectively. Both shards show the effect; neither reproduces the
training median22s or historical shard1 extremes. Both input types are NFS.
Backend cache is uncontrolled, and full-read is slower for cached bags; do
not assume the panel speedup transfers to training. No measured reclaim in
the short panel; this does not establish full-atlas cache behavior.
Outputs: `runs/kaggle/spec0054_io_cache_v20/`; review:
`runs/local/spec0054_io_cache_v20_review/summary.json`.
The slowdown remains input-bound with an unresolved full-working-set cause.
Next useful mitigation measurement is scratch filesystem/write/read behavior
and a small content-identical input-versus-scratch comparison, before staging
the full atlas or changing the training reader. No new run was launched.

The next authorized average-throughput measurement is prepared in the existing
read-probe runner. An independent subagent recommended all49 fold-training
WSI from shards1/2, natural cache replay and staging-cost amortization. The
probe copies33.474GB paired to scratch with fsync/frozen SHA verification,
then compares input/scratch × sampled/full reads over two random passes with
new paired subsets:784 reads, two readers per branch, CPU-only and90-minute
runtime budget. Branches synchronize after staging and before each condition;
condition order reverses on pass2. Actual bag contents are27.633GB paired
for the train population; this remains smaller than the full atlas. Record
per-read cache residency, arithmetic mean/p95, per-condition elapsed and copy
cost before estimating amortization. No forced eviction or classifier change.
The concurrent synthetic48-read check against independent arrays, seven
reader tests, compilation/static checks and kernel build/validation pass;
subagent review found no substantive issue. Source `242cf6f` and thin entry
`a4a46cf` are published. Submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/21` around18:28 Bogota on2026-10-26;
API confirms RUNNING. Budget begins at actual entry startup, not submission;
blocked I/O/copy may overrun it. Next useful result inspection around20:15
Bogota, allowing for platform preparation, or when artifacts become available.
No further polling or classifier continuation accompanies the launch.
Initial logs confirm source242cf6f,49 train WSI per branch, GPU disabled,
input NFS/device0:51 and scratch on the root overlay/device0:53. Both branches
have begun copying shard1. Overlay identifies the filesystem layer, not the
underlying medium or achieved throughput. Initial stream:
`runs/local/spec0054_io_scratch_v21_review/initial_stream.txt`.

Inspection around19:51 Bogota confirms v21 COMPLETE, workers[0,0] and all784
reads (392 per branch). All four frozen file SHA checks match; requests/subsets
pair across branches, and all196 branch/epoch/WSI groups preserve identical
selected values across the four paths. Pooled mean read+subset seconds are
7.922 input-sampled,3.326 input-full,4.467 scratch-sampled,2.613 scratch-full;
p95 is20.247/8.061/12.114/6.327s. Mean paired pass wall is219.89/107.51/
141.26/91.56s respectively. Full input reduces observed pass wall51.1%
without staging; full scratch saves another15.95s/pass on this panel but
costs628.23s upfront including fsync/SHA, a descriptive39.4-pass amortization
against input-full. Scratch is usable for33.474GB, not established as fast
SSD or sufficient for the full atlas. Natural cache turnover is observed:
on pass2, mean input residency falls from~98.9% after full reads to~5.8%
before later sampled reads; all49 WSI per branch lose over half residency.
CPU/two shards/two passes and order/backend warming limit transfer to full
training. Recommend evaluating full contiguous reads on the training working
set before full-atlas staging; no reader change or new session launched.
Outputs: `runs/kaggle/spec0054_io_scratch_v21/`; numerical review:
`runs/local/spec0054_io_scratch_v21_review/summary.json`.

The authorized next CPU-only probe restores all12 original input mounts and
compares only direct sampled-region versus full-bag-then-subset reads. It uses
the288 fold-training WSI, exact next-training epoch orders/subsets (indices5/6),
two readers per branch and counterbalanced72-WSI windows. Both branches run
the same method together; partial output retains paired comparisons from
finished windows if the90-minute runtime budget is reached. No scratch copies,
cache eviction, classifier, GPU or resume states. Record read+subset means/p95,
window wall, cache residency and process RSS/high-water mark before/after reads
and with full+selected arrays alive. Peak RSS is cumulative per process and
includes both readers, rather than an isolated per-method allocation peak.
All24 synthetic reads preserve independent array values and exact paired
training order; memory fields are populated. Seven reader tests, compilation,
Ruff, targeted BasedPyright and kernel build/validation pass. Publication and
submission are pending.

All512 WSI/subset/C4 records match across branches; initial parameters and reader
match the prior recipe. No AMP backoffs or nonfinite losses/gradients. Mean steps
143.96/144.64s versus14.94/14.71s over the matched first128 v10 updates. Input
queue waits consume97.7% of training-step time; mean reads56.05/56.29s per WSI
versus0.20/0.23s host preparation including C4. Three complete assessments take
another~4.96h of the10.33h kernel. Storage-side cause remains unmeasured; inspect
I/O before another full session. Outputs: `runs/kaggle/spec0054_cls_fold0_c4_v13/`;
review: `runs/local/spec0054_cls_fold0_c4_v13_review/summary.json`.

Matched-read followup identifies shard1 as the largest hotspot:318.38/320.90s
per training read in v13 versus4.29/4.61s in v10; other shards average25--36s
in v13 versus4--7s previously. The inherited v12 and submitted v13 launchers
both mount shard1 as the same private dataset; this launch only removed the old
resume mount. A historical notebook-to-dataset change exists, but actual v10
input metadata could not be retrieved, so it cannot yet explain the difference.
Reader code, matched512 subset hashes/sizes, two-reader/two-future prefetch and
latent extraction/cohort identities agree. C4 runs after the measured read.
Detailed evidence: `runs/local/spec0054_cls_fold0_c4_v13_review/read_comparison.json`.

A bounded CPU-only I/O followup completed in the existing read-probe mode.
It mounts only shards1/2 and compares complete contiguous reads plus the same
60% subset against the unchanged sampled-region reader, counterbalancing method
order on two fixed training WSI per shard and repeating each method twice.
Each completed read is persisted with timing, physical path and value hash;
no GPU, classifier or training state is used. Seven reader tests, a synthetic
32-read content-parity check, compilation, Ruff and targeted BasedPyright pass.
Published source `338ff7d`, entry `0b3ee59`; submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/14` on2026-10-23 at10:11 Bogota.
Inspection around10:25 Bogota confirms COMPLETE, worker exits[0,0], all32
reads and exact within-branch selected-value/subset hashes across methods and
repeats for all8 branch/WSI groups. Four logical CPUs, no GPU; expected shard1
dataset/shard2 notebook paths (devices54/53). First-script full reads cost
366.19--367.59s on shard1 and274.25--275.08s on shard2; first sampled reads cost
374.41--376.23s and278.25--279.32s, respectively. Same-method second reads cost
0.0466--0.2282s. Reading a full bag after its sampled regions still costs
24.44--25.23s, then0.16--0.23s on repeat. First-read thread CPU is only
0.33--0.61s. Storage/cache behavior is implicated even for contiguous reads and
both input types; its exact backend cause and the v10/v13 difference remain
unidentified. Different WSI supply the two first-method orders, and the small
panel does not establish full-atlas throughput. Outputs:
`runs/kaggle/spec0054_io_probe_v14/`; review:
`runs/local/spec0054_io_probe_v14_review/summary.json`. No further launch or
training-reader change. This inspection preserved v13; continuation must restore its own
states from source7ad0601 and use full training metadata rather than this probe.

The authorized next CPU probe keeps only shard1 and its same two training WSI.
Each branch first runs the existing8 input reads, then streams its entire
8,340,340,736-byte binary to `/kaggle/working/spec0054_io_cache/<branch>/`,
records copy wall/CPU time, bytes, free space, destination device and the
streamed SHA256 against the frozen result hash. After fsync it issues advisory
POSIX_FADV_DONTNEED on that copy before the same8 local reads. Both branches
run concurrently:32 reads total plus two full copies. Incremental logs persist
progress; temporary binaries are removed on exit. The planned one-hour shared
cutoff applies between reads/copy chunks; in-flight I/O may finish later.
This measures copy cost and local access on a small panel; it does not guarantee
a cold cache or establish capacity/throughput for all12 paired shards.
Seven reader tests, a32-read synthetic independent-value/copy-hash/cleanup
check, compilation, Ruff and targeted BasedPyright pass. Source `f4c29f4`;
the CPU entry mounts only the unchanged shard1 dataset and uses the shared
cutoff2026-10-23T11:32:01-05:00. Build and kernel validation pass. Published entry
`dfc0f7d`; submitted as `maximshtefan/eqvae-spec0054-mil-fold0/15` on2026-10-23
at10:32 Bogota after explicit publication/launch authorization. Inspection
on2026-10-25 confirms COMPLETE, worker exits[0,0], all32 reads, and both full
binary hashes equal their frozen authoritative hashes. All4 branch/WSI groups
retain the same selected values across input/local methods/repeats and v14.
Sourcef4c29f4, four logical CPUs, GPU unused and expected shard1 paths agree.

The concurrent8.34GB copies cost141.90s normal/133.41s SO2, including SHA256
and fsync (141.90s elapsed across both). First input reads cost759.27--784.45s;
all local reads cost0.0746--0.5020s after advisory cache discard. Copy destination
device1794 differs from input53. Copies follow initial input reads, so their
cost is not a cold-start preparation estimate. Observed free space before
copying was20.94GB and after both copies4.26GB; the paired shard1 uses16.68GB,
while all12 pairs total200.33GB and do not fit in this session. Local staging
of shard1 is promising; full-atlas/GPU training throughput and cold-copy cost
remain unmeasured. No further launch, reader change or training continuation.
Outputs: `runs/kaggle/spec0054_io_local_copy_v15/`; numerical review:
`runs/local/spec0054_io_local_copy_v15_review/summary.json`.
The v13 saved artifacts remain preserved; active continuation is v18 above.

The user requested a new direct-input measurement on2026-10-25 to check whether
the earlier slowdown persists. Reuse published source `338ff7d` and the v14
CPU entry/input metadata, changing only the submission-based30-minute cutoff
to2026-10-25T14:22:00-05:00. This preserves the exact two WSI per shard1/2,
two readers per branch, two repetitions, full/sampled method orders and32-read
budget. No local copy/cache step, GPU or classifier training. Build, validation
and entry Ruff checks pass. Submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/16` on2026-10-25 around13:53 Bogota.
Inspection around14:22 Bogota confirms COMPLETE, worker exits[0,0] and all32
reads. Source, reader and contract hashes equal v14; all32 request/content
records match v14 exactly. Both CPU branches, four logical CPUs, GPU unused
and original input paths/devices54/53 agree. First accesses on shard1 take
163.40--167.53s versus366.19--376.23s in v14 and759.27--784.45s in v15; the
mean drops55.4% relative to v14. Shard2 stays near v14:269.13--274.99s versus
274.25--279.32s, mean down1.7%. Same-method second reads take0.0471--0.2285s;
first-read thread CPU is0.33--0.57s. Session variation is observed, but slow
direct access persists; no specific backend incident or full training speed
is established. No further launch or reader change. Outputs:
`runs/kaggle/spec0054_io_direct_repeat_v16/`; review:
`runs/local/spec0054_io_direct_repeat_v16_review/summary.json`.
No new source publication was required; the entry loads the published v14
contract rather than the local-copy contract at the research checkout's HEAD.

Interpretation correction: first-access cost alone does not diagnose persistent
training slowdown or justify local staging. Immediate repeats in v16 are fast.
Historical v13 telemetry, restricted to512 consumed presentations (excluding
6 drained unused prefetch reads), contains224 WSI revisited in epoch2. Their
second training-read median is32.55s normal/32.42s SO2 versus4.45/4.58s for
the matching WSI/subset requests in v10. On the20 shard1 revisits, medians are
203.02/200.72s versus4.11/3.70s; the other204 revisits take29.86/29.78s versus
4.54/4.68s. Subsets change between epochs and full-bag assessments intervene;
these are realistic training revisits rather than identical immediate rereads.
This confirms the historical subsequent-read issue, not its persistence in the
new session. V16 does not measure revisits after cycling through the full
working set. Review: `runs/local/spec0054_cls_fold0_c4_v13_review/subsequent_training_reads.json`.
No further run was launched.

The next authorized CPU probe is deliberately small: six training WSI across
shards1/2, one full-bag warmup pass followed by three random passes using the
unchanged60% sampled-region reader, with a new epoch-keyed subset each pass.
Both branches share deterministic orders/subsets and two reader threads each.
Each pass visits every panel WSI once; round-boundary order avoids an immediate
same-WSI repeat. This is24 reads per branch (6 initial/18 subsequent),48 total,
with no local copies, forced eviction, augmentation, GPU or classifier training.
Per-read logs retain round, planned request index/gap, bytes, wall/thread-CPU
time and hashes. The30-minute cutoff bounds new reads. This small working set
addresses interleaved revisits only and cannot establish full-atlas behavior.
The existing probe function/contract are updated; v14--16 and the copy probe
remain reproducible from their original source pins. Seven reader tests,
48-read independent synthetic value/order/subset checks, compilation, Ruff and
targeted BasedPyright pass. Source `03dfbb2`; build and kernel validation pass.
The refreshed shared cutoff is2026-10-25T15:14:01-05:00. The user explicitly authorized
publishing commits03dfbb2/c057bb3 and launching the probe after automatic
review requested confirmation. Published entry `0bdc98a`; submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/17` on2026-10-25 around14:44 Bogota.
Inspection on2026-10-25 confirms COMPLETE, both workers exit0 and all48 reads
finish. The12 initial full reads take1.76--6.65s. The36 random subsequent
sampled reads take0.0532--0.2404s, pooled median0.0975s and mean0.1098s;
per-pass means are0.1082/0.1061/0.1054s normal and0.1389/0.0948/0.1052s SO2.
All planned paired orders/request/subset records agree, with2--9 intervening
requests and three distinct subsets per WSI; all8 overlapping value checks
agree with v16. No slow subsequent read is observed in this current small panel,
without copying files. This supports a direct-input continuation measurement;
full-atlas/GPU training throughput remains unmeasured. No further launch or
training-reader change. Outputs: `runs/kaggle/spec0054_io_random_revisit_v17/`;
review: `runs/local/spec0054_io_random_revisit_v17_review/summary.json`.
The training continuation launched after this probe is v18 above.


The previous CLS + gated-ABMIL fold0 run is paused after
`maximshtefan/eqvae-spec0054-mil-fold0/12`. Resume input is
`maximshtefan/eqvae-spec0054-cls-regularized-fold0-resume/3`.

| Branch | Updates | WSI presentations | Epoch / cursor |
| --- | ---: | ---: | --- |
| normal_vae | 5228 | 20,912 | 72 / 176 |
| so2_vae | 5182 | 20,728 | 71 / 280 |

Both completed epoch70 assessment; both best-by-CE checkpoints are epoch55.
Best validation CE/F1: normal 0.96126/0.60816, SO(2) 0.85010/0.61773.
These are exploratory fold0 results, distinct from the five-fold fixed-head
comparison. Review evidence: `runs/local/spec0054_cls_fold0_regularized_v12_review/`;
[professor update](https://github.com/HiperMaximus/equivariant-vae/issues/8#issuecomment-6149719046).

The v13 submitted entry (retained in commitac196da) was pinned to
`7ad06010416a789a6b96e612919203c162adc23e` (entry `ac196da`, `RESUME=False`),
with normal/SO2 on physical GPU0/1, confirmed in logs. Its cutoff is now expired.
Any candidate continuation must restore the v13 latest states and cursor from
a candidate-specific resume input, enable resume and set a fresh deadline;
never resume the previous-recipe v12 weights. The separate CPU probe v14 does
not advance training. Numeric recipe: `docs/data/spec0054_fold0_run.json`; scientific
definition: Spec0054. The105-epoch horizon and five-fold evaluation remain unfinished.

## Functional geometry — Spec 0053

[Scientific spec](docs/specs/0053-functional-riemannian-latent-geometry.md).
Stage A1 is accepted. Stage A2d already continued all four bridge states to
step4096; do not repeat the older step2048-to4096 plan. Source: `b630b6e`.
Outputs and saved optimizer states:
`runs/kaggle/functional_geometry_stage_a2b_v5/functional_geometry_stage_a2d/`.

All four saved best states are at step4096 with unchanged LR; traces still
descend. SO(2) bridge E/endpoint-RMS-squared is 19.002/20.986 (ranks0/12),
and decoded bottlenecks are 1.796/1.879 times the endpoint gap. Convergence and
a connected decoder fiber have not been established. No step6144 execution
was identified in the latest review; extending only the SO(2) bridges remains
a proposal. K64 reoptimization and decoded-curve stability follow convergence.

The existing restricted-chart shooting used the original 512-step sides and
an initial tangent without endpoint fitting. Its rank losses and sensitivity
to time refinement do not establish a closed geodesic cycle. Full-latent
shooting, continuous prescribed-orbit analysis and fiber continuation remain
pending; holonomy interpretation requires the relevant validated structure.

## Paper handoff

Latest published manuscript sync: `bbe8fa0`, imported main.tex v963 and
references.bib v5 on 2026-10-16. Source/PDF and current writing handoff:
[manuscript README](paper/midl2027/README.md).
The 22-page PDF compiled successfully; all 19 professor threads retain their
states (7 open, 12 resolved). Only the professor closes comments.

Read live Overleaf before further manuscript changes, preserving professor
edits. Import/build/publication procedure:
[Overleaf handoff](docs/overleaf_sync_workflow.md). The manuscript still has
pending 152-WSI results; the historical 23-WSI evaluation remains separate.
