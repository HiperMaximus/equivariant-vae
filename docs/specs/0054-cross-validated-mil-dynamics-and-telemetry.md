# Spec 0054: Cross-Validated MIL Dynamics And Telemetry Plan

Status: dropout/smoothing/C4 v18 paused at5 epochs; CPU v24/v1 reviewed; gap1 queue/cache CPU v2 RUNNING
Owner/workstream: repeated downstream evaluation of the frozen normal and
continuous-`SO(2)` VAE representations
Last updated: 2026-10-26

## Accepted dropout, smoothing and C4 candidate (2026-10-22)

Train a new paired fold0 candidate from the original classifier initialization.
Preserve v12 latest/best checkpoints, telemetry and its version3 resume dataset
as the previous recipe. The VAE weights, frozen historical Otsu bags, folds,
patch retention, AEM, effective batch, LR schedule and horizon remain unchanged.
Exact numerical settings reside in `docs/data/spec0054_fold0_run.json`.

Apply channel dropout immediately after the spatial encoder's convolution,
activation dropout inside both residual MLPs, and feature dropout on the WSI
embedding before its linear classifier. Keep parameter names and initialization
unchanged so the initial learned tensors match the prior recipe. Evaluation
disables all dropout and retains full untransformed bags.

The training objective for original diagnosis y is
`w_y * CE((1-epsilon)*one_hot(y) + epsilon/5, p) - lambda_AEM*H(attention)`.
The original class weight multiplies the whole smoothed CE. Retain ordinary CE
and its class-weighted value as distinct telemetry; record smoothed CE and AEM
separately. Select checkpoints by minimum ordinary unweighted validation CE.

For each training presentation, first retain the existing paired patch subset,
then rotate each retained posterior-mu map independently by a uniformly sampled
element of C4 on its last two spatial axes, identically across all16 channels.
Use a separate deterministic RNG stream keyed by seed/epoch/WSI so branches
and resumed sessions share rotations without changing subset selection or the
model RNG. Record the sampled rotation-code hash. Rotate owned input buffers
in the reader before the existing host preparation and transfer; cached shards
and coordinates stay unchanged. Exact quarter turns introduce no interpolation.

This is exploratory latent-input regularization, not evidence that rotating
stored mu equals encoding rotated RGB. The fixed25 evaluation population and
its outcomes do not select augmentation settings. D4/reflections remain a
separate future comparison. The combined candidate cannot identify individual
regularizer effects. Preserve the exploratory status of fold0 and the external
152-WSI boundary.

Submitted on2026-10-22 at16:15 Bogota as
`maximshtefan/eqvae-spec0054-mil-fold0/13`, initial API status RUNNING.
Source `7ad06010416a789a6b96e612919203c162adc23e`, entry `ac196da`.
Both branches start fresh on separate GPUs, without the prior resume mount.
Shared cutoff2026-10-23T04:00:03-05:00 reserves15min of the session budget.
Inspect initial GPU/throughput evidence around17:00 Bogota; subsequent sessions
must restore this candidate's own saved state rather than any v12 checkpoint.

Inspection on2026-10-23 confirms both workers exit0 and pause at128 updates,
512 presentations, epoch1/cursor224. Last complete and best assessment is epoch1:
normal validation CE/F1 1.60605/.14074; SO2 1.60574/.11461. Initialization equals
the prior recipe and all512 ordered WSI/subset/C4 records match across branches.
No AMP backoffs or committed nonfinite loss/gradient observations were found.
Mean updates cost143.96/144.64s versus14.94/14.71s for the matched first128 v10
updates. Queue waits consume97.7% of training-step time; mean reads are56.05/56.29s
per bag while host preparation including C4 is0.20/0.23s. The three full-bag
assessments consume~4.96h of the10.33h kernel. The reader is unchanged; the
storage-side cause of the slowdown has not been measured. These warmup-only
results cannot establish the candidate's generalization. Downloaded artifacts:
`runs/kaggle/spec0054_cls_fold0_c4_v13/`; review and matched comparisons:
`runs/local/spec0054_cls_fold0_c4_v13_review/summary.json`. No further launch,
upload or recipe change accompanied this inspection.

The authorized next training session restores the v13 candidate rather than
the previous-recipe v12 run. The private continuation dataset is
`maximshtefan/eqvae-spec0054-cls-c4-fold0-resume/1`, status ready. All14 files
(latest/best, identities and cumulative tables for both branches) were downloaded
individually and match their source bytes; the whole-archive endpoint returned404.
Both latest states preserve128 updates/512 presentations, epoch1 cursor224,
last completed assessment288 and optimizer/scaler/RNG. Use the original published
source7ad06010416a789a6b96e612919203c162adc23e to preserve exact resume identity;
the current checkout's probe/config changes do not enter this run.

Restore full original12-shard metadata and direct input reading, enable resume,
disable all smoke/read/LR probes and assign normal/SO2 to physical GPUs0/1.
Keep the105-epoch horizon, unchanged C4/dropout/smoothing/decay recipe and
144-presentation checkpoint cadence. Shared cutoff2026-10-26T11:31:37-05:00
reserves15min of the12-hour submission budget. Inspect around00:20 Bogota on
2026-10-26 for restored progress and actual full-atlas read/queue-wait timing.

Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/18` on2026-10-25 around
23:47 Bogota, local entry `c0d7e8a`. API confirms RUNNING; initial logs confirm
the original source7ad0601, private C4 continuation input and independent Tesla
T4 workers on physical GPU0/1 with all probes disabled. Both `resume_done`
messages confirm update128/exposures512, epoch1/cursor224 and last assessment288.
The first completed paired WSI40079 subset/C4 hashes agree. No new training source or
configuration publication is needed because this session uses the original
published v13 commit. Inspect around00:20 Bogota on2026-10-26.

Live inspection around09:38 Bogota on2026-10-26 confirms RUNNING, last visible
update348/exposures1392 (epoch4 cursor240) on both branches. This session adds
220 updates/880 presentations; latest visible saved checkpoint is update324/
exposures1296. Last complete assessment is epoch3 with validation CE1.60842
normal/1.57188 SO2; best-CE events select normal epoch2 (1.57755) and SO2 epoch3
(1.57188). Live events do not provide macro-F1. The model remains before the
end of five-epoch warmup, and these metrics do not establish recipe benefit.

All880 consumed WSI/subset/C4 records agree across branches, with no visible
AMP skips/backoffs. Training read medians are22.15/21.88s and means39.14/39.07s,
versus4.46/4.55s and5.16/5.19s on matching v10 WSI/subset requests. All have prior
v13 training presentations, so the delay is not limited to initial access.
Step means are89.66/89.46s versus12.66/12.81s on matched v10 updates; queue
waits consume97.6--97.7% of step time. Shard1 median reads are37.94/39.78s,
means95.13/96.08s. The small six-WSI probe cannot predict performance over the
full working set; exact cache/storage cause remains unidentified. This is a
partial live snapshot, not the final exported checkpoint/metric state. No
intervention or additional execution accompanies the inspection. Cutoff remains
11:31:37 Bogota; inspect completed artifacts after11:45. Numerical review:
`runs/local/spec0054_cls_fold0_c4_v18_live_review_20261026/summary.json`.

External context reviewed2026-10-26: an older
[Kaggle report](https://www.kaggle.com/discussions/product-feedback/443920)
has staff reproduction of fast2GB and slow30GB reads, but concerns `/kaggle/tmp`
and explicitly distinguishes `/kaggle/input`. A first-hand
[PyTorch forum report](https://discuss.pytorch.org/t/data-loading-time-becomes-unstable-jumping-to-high-values-when-dataset-is-large/218042)
also describes random binary-file access becoming variable on a larger dataset;
it is not a confirmed Kaggle diagnosis.

[Kaggle documentation](https://www.kaggle.com/docs/notebooks) lists29GB RAM
for T4x2. V17's full paired panel totals3,170,172,928 bytes versus200,332,607,488
bytes for the atlas. The [Linux memory-management documentation](https://docs.kernel.org/admin-guide/mm/concepts.html)
explains that file reads populate the page cache and cached pages can be
reclaimed. Cache residency is therefore a plausible explanation for fast small
panels and slower later reads after traversing more data. This remains an
inference; it does not explain the change from v10 under the same data/reader.
Filesystem type, actual cache pressure and backend behavior are unmeasured;
neither NFS nor GCS FUSE is established for these mounts. No matching current
Kaggle incident or corrective setting was confirmed. No session mutation follows
from this literature check.

The authorized cache/read-method followup remains a CPU-only change to the
existing probe function and contract. On the same six fixed training WSI,
compare the unchanged sampled-region reader with one full-bag read followed
by the identical seed/epoch0 sorted60% subset. Each WSI runs both methods in
counterbalanced order. Before each method request POSIX_FADV_DONTNEED for only
its bag range, measure actual mincore residency, read, then repeat immediately.
There are24 reads per branch,48 total, using two readers and a30-minute cutoff.
Selected-value hashes must be reported for direct content-equivalence inspection;
no classifier, augmentation, input rewrite or local-copy mitigation is applied.

Record mount type/device/owner UID, process UID, free working/scratch space and
whether their devices differ. Each read retains bag-page residency before advice,
before reading and afterward; memory availability/cache, process I/O and host
reclaim/refault counters are recorded as snapshots. mincore records errors as
undefined residency observations. Kernel visibility may be restricted, pages can
change after sampling, and backend caches are not measured. POSIX_FADV_DONTNEED
is nonbinding and can discard only part of a region; use observed residency
rather than labeling the first read cold. Process counters include both reader
threads and host counters may include unrelated work, so their differences are
not exclusive per-request I/O. No global drop_caches or memory-pressure workload
is used. Compare read-plus-subset time: the full-read candidate transfers about
1/0.6 times the bytes but reduces region calls to one without changing tensors.
Cache capacity alone does not explain the v10 regression; this controlled panel
tests a potential mitigation rather than establishing whole-atlas throughput.

V18 was confirmed COMPLETE before preparing the CPU entry, with both workers
exit0 and latest states at360 updates/1440 presentations/epoch5 cursor0. Its
epoch5 assessment was interrupted by the cutoff and remains pending. Outputs
are saved in `runs/kaggle/spec0054_cls_fold0_c4_v18/`; future continuation must
use these states rather than the v13 states in private resume dataset version1.

The final v18 artifact review confirms all1440 cumulative pairing records and
exact v13 telemetry prefixes, with no AMP skips or recorded nonfinite training
loss/gradient values. Best-by-CE validation results are normal epoch2 CE1.57755/
macroF1.17354 and SO2 epoch3 CE1.57188/F1.25139. Latest complete assessment is
epoch3: normal CE1.60842/F1.12982, SO2 CE1.57188/F1.25139. The epoch5 assessment
must replay and100 training epochs remain. Session step means90.12/90.01s,
read medians22.17/21.95s and input-wait fractions97.7--97.8% confirm continued
full-working-set I/O cost. Numerical review:
`runs/local/spec0054_cls_fold0_c4_v18_review/summary.json`.

Matched early assessments against v10 use the same full train/validation bags
and ordinary unsmoothed, unweighted CE. At epoch2 old/new validation CE is
1.58450/1.57755 normal and1.57120/1.57496 SO2; macroF1 is.15385/.17354 and
.21462/.19487. At epoch3 CE is1.59161/1.60842 normal and1.57693/1.57188 SO2;
macroF1 is.19500/.12982 and.21244/.25139. Thus early changes are mixed rather
than a uniform degradation. New epoch3 train CE1.61276/1.58331 is also high;
these observations do not establish overfitting or excessive LR. Both latest
states only reach the end of warmup, with no completed post-warmup epoch.
Historical v23 fresh confirmation favored5e-4 over1e-3 under an older recipe;
its result is a useful starting point, not a new-recipe optimum. Assessment5
and post-warmup behavior should precede a decision on another narrow LR
comparison. No search or training launch follows from this interpretation.
Evidence: `runs/local/spec0054_cls_fold0_c4_v18_review/matched_early_assessments.json`.

The cache/read-method probe was submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/19` on2026-10-26 around12:20 Bogota.
Published source `4e3c6fc075d032c6dc39c83fd6317a8088163c94`, entry `d7afccb`;
initial API status RUNNING. Shared cutoff2026-10-26T12:49:52-05:00. Inspect
around12:35 Bogota or after completion. It mounts only the original shard1
dataset and shard2 notebook output, runs both CPU branches without GPU and
does not mount or alter training continuation states.

Inspection around13:25 Bogota on2026-10-26 still reports RUNNING after the
12:49:52 cutoff, with no output in a45-second live-stream window. Authenticated
metadata confirms latest version19 and the intended source/GPU setting; a
numeric-version stream request returns404 and does not establish session phase.
No per-read timing, mount or residency observations are available, so neither
cache behavior nor mitigation efficacy can be assessed. The internal deadline
is checked before reads and cannot interrupt platform startup or blocked I/O.
No cancellation, relaunch or source change accompanies this inspection. Status:
`runs/local/spec0054_io_cache_v19_review/status_20261026.json`. Next useful
inspection is after14:00 Bogota or when logs/artifacts become available.

At15:35 Bogota, status remains RUNNING roughly3h15 after submission. Both static
logs and a30-second live stream provide no diagnostic events. The files API
lists prior CLS training outputs dated04:47UTC rather than v19 measurements.
This cannot establish code-start, session phase or any cache/read result. No
cancellation, relaunch or source edit was performed. A subsequent controlled
attempt should compute the diagnostic read budget at code startup rather than
before platform preparation. Status receipt:
`runs/local/spec0054_io_cache_v19_review/status_1535_20261026.json`.

At17:17 Bogota v19 is COMPLETE with worker exits[0,0], but both results report
completed=false and0 reads. The expired absolute cutoff prevented measurements;
the source's only early read return is the deadline condition. Environment
metadata identifies both inputs as NFS/device0:52, file owner UID65534 and
process euid0. Free working space is20.94GB/device1794; scratch reports1125.98GB/
device53. The latter's filesystem type, quota and performance are not established,
so this is not evidence of usable fast local capacity. There is no read-method
or cache-efficacy comparison from v19. Outputs: `runs/kaggle/spec0054_io_cache_v19/`;
review: `runs/local/spec0054_io_cache_v19_review/summary.json`.

A single controlled repeat v20 was submitted around17:23 Bogota on2026-10-26,
reusing published source4e3c6fc and the same48-read CPU-only experiment. Only
the thin entry changes (local commitad1f43d): compute the30-minute deadline at
code startup so platform preparation cannot consume the read budget. The budget
is still checked between calls, not an interrupt for blocked I/O. API status
RUNNING; no GPU or classifier continuation. Inspect around18:00 Bogota or when
measurements become available. No new experiment-source publication is needed.

Inspection around18:14 Bogota confirms v20 COMPLETE, both branch results
completed=true and24 rows each. All48 requests preserve branch-paired
WSI/subset/method/repeat keys. Per branch/WSI, selected-value hashes agree
across both methods and repeats; JSONL rows equal the final result rows.
Mincore succeeds throughout and reports0--0.128% client-cache residency
after bag-range DONTNEED. Pooled read-plus-subset medians are3.8387s sampled
regions and1.6797s full-bag then identical subset; means3.8228/2.3036s.
Immediate-repeat medians are.05250/.16635s, showing the opposite method
ranking once cached. After the fragmented read, about60--62% of bag pages
are resident; after full read,100%. Thread read CPU medians are.1645/.1583s
after advice, much smaller than elapsed read time. These observations locate
the panel's extra delay outside thread CPU execution and show a client-cache
effect, not the state of NFS/backend caches.

Both dataset shard1 and notebook shard2 are NFS/device0:52. Branch shard1
sampled-read medians after advice are3.486/3.718s versus3.988/4.036s on
shard2; shard1 is not the hotspot in this session. The six-WSI panel's read
window spans23.23s and MemAvailable remains30.59--32.30GB, with recorded
reclaim/refault counters unchanged at0. This does not reproduce the200.33GB
training working set, new subsets across epochs or the T4 training allocation.
Backend caching and per-WSI method order remain possible influences despite
counterbalancing. The training slowdown and earlier-session regression remain
unexplained; the cold-client-cache panel does not reproduce22s training-read
medians. Full reads are a candidate mitigation to measure on representative
training access, not an established replacement. Scratch free space is still
only a capacity report; filesystem type, practical quota and throughput need
measurement before full-atlas staging. No new run or reader change accompanies
this review. Outputs: `runs/kaggle/spec0054_io_cache_v20/`; numerical review:
`runs/local/spec0054_io_cache_v20_review/summary.json`.

## Average-throughput scratch mitigation probe (2026-10-26)

The authorized followup reuses the existing read-probe entry and the two
unchanged mounted shards. All49 fold0-training WSI in these shards form the
measurement population. Actual frozen bag lengths give27.633GB paired full
contents; historical cohort patch counts are not the storage byte counts.
The contract defines two random passes with epoch-dependent paired60% subsets,
two readers per branch and four conditions: input/scratch crossed with sampled
regions/full-bag then subset. Both branches run the same condition together.
Condition order reverses on the second pass; WSI order and subsets are shared
across conditions within each pass. Total planned reads are784, CPU-only with
a90-minute runtime budget. No classifier or frozen representation changes.

Before measurement, each branch copies both complete binaries to scratch:
33.474GB paired. Record filesystem metadata, copy+fsync time, separate full
destination SHA verification against frozen result checksums, stage timestamps
and free space afterward. Shared readiness markers synchronize the branches
after staging and before every condition, with waits recorded separately and
bounded by the same deadline. Copying and checksum verification may warm both
client/backend caches; this is a natural post-staging replay, with no forced
eviction. Capture residency before/after each read to describe the observed
cache mixture. Record condition/pass wall time and persist every completed
read, including read+subset time, CPU time, bytes and value/subset hashes.
Wall time includes snapshots, verification hashes and output overhead; keep
these distinct from read+subset measurements. An in-flight copy/read can
overrun the deadline.

Review arithmetic means, p95 and concurrent elapsed per condition/pass. Compare
the observed paired stage cost S plus R scratch passes against R direct
passes, rather than quoting a ratio of medians. Where direct elapsed D exceeds
scratch elapsed L, report the descriptive amortization point S/(D-L), retaining
the measured pass/cache context. Two shards on CPU do not establish the
training cache mixture or throughput on all12 shards/T4. Input and scratch
touch distinct file pages, with paired full contents totaling about55GB across
the two locations. Scratch filesystem type, practical capacity and performance
are observations of this experiment, not prerequisites assumed from free space.

Focused verification exercises both actual branch probes concurrently on two
synthetic shards:48 reads total and eight blocks per branch preserve exact values
against independent source arrays and exclude the held-out fold. The seven
existing shard-reader tests, compilation, Ruff, targeted BasedPyright and
kernel build/validation pass. Independent subagent review found no substantive
runner/config issue. Source `242cf6f0afc25d44cd6a90453a606eddb2b06382`
and thin entry `a4a46cf` are published. Submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/21` on2026-10-26 around18:28 Bogota;
API confirms RUNNING. Next useful inspection is around20:15 Bogota, allowing
for platform preparation, or when completed artifacts become available.
No training continuation or additional poll follows from this launch.
Initial logs confirm the intended source,49 train WSI per branch, CPU-only
workers, NFS inputs/device0:51 and scratch on root overlay/device0:53. Both
branches begin copying shard1. Overlay alone does not identify the physical
storage medium, quota or speed. Initial stream:
`runs/local/spec0054_io_scratch_v21_review/initial_stream.txt`.

V21 inspection around19:51 Bogota confirms COMPLETE, workers[0,0], both
branches completed=true and392 rows each, with all eight blocks per branch
complete. All four copied binaries match frozen SHA256; all196
branch/epoch/WSI groups preserve subset and selected-value hashes across the
four conditions. Branch request/subset keys agree; JSONL matches final rows
and mincore succeeds throughout. Outputs:
`runs/kaggle/spec0054_io_scratch_v21/`; reproducible numerical review:
`runs/local/spec0054_io_scratch_v21_review/{analyze_v21.py,summary.json}`.

Pooled mean read+subset seconds (196 reads/condition) are7.92231 input-sampled,
3.32629 input-full,4.46680 scratch-sampled and2.61320 scratch-full. The p95
values are20.24684/8.06062/12.11370/6.32702s. Corresponding mean concurrent
pass wall is219.89295/107.50579/141.25929/91.55712s. Input-full reduces
observed average read time58.0% and pass wall51.1% relative to input-sampled.
It improves both passes: paired wall143.82 to91.81s on pass1,295.97 to123.20s
on pass2. Pooled read means improve on both shards:9.15054 to4.04632s on
shard1 and6.53388 to2.51235s on shard2.

Paired staging costs628.22574s including fsync and destination SHA verification;
free scratch afterward is~1092.51GB. This demonstrates33.474GB usable here,
not full-atlas capacity or physical SSD identity. Using measured mean pass
walls, scratch-sampled amortizes after7.99 passes relative to input-sampled;
scratch-full after4.90. Relative to the faster input-full alternative,
scratch-full saves only15.94867s/pass and amortizes after39.39 passes.
Scratch-sampled is slower than input-full before adding stage cost. These
are descriptive extrapolations of two nonstationary passes, not predictions
of later epochs or an end-to-end training speedup.

Natural cache turnover is directly observed without DONTNEED. During pass2,
per-branch mean residency after input-full is.98893/.98951; before later
input-sampled reads it is.05869/.05692, with all49 WSI per branch losing
over half their bag residency. Pooled input-sampled mean read time rises
from4.73525 to11.10936s between passes while mean pre-read residency falls
from.51285 to.05780. Host reclaim/refault counters also increase, but are
host-wide and cannot be exclusively attributed to the probe. These data
support cache turnover as a subsequent-access slowdown mechanism within
this session, without explaining the original v10-to-v13 backend regression.
Stage warming, condition order, two-shard population and CPU allocation limit
transfer. Evaluate contiguous input reads on a representative training working
set before choosing full-atlas scratch staging. No reader modification,
classifier continuation or new run accompanies this review.

## Full-atlas direct-read comparison, CPU only (2026-10-26)

The authorized next probe compares sampled regions and full-bag then the same
retained subset directly from the original12 frozen mounts. GPU is disabled
in metadata and CUDA is hidden from both workers; no model, augmentation,
optimizer or continuation checkpoint is loaded. There is no scratch staging
or forced cache eviction. The existing runner/probe/contract and thin entry
remain the execution path.

The288 fold-training WSI use exactly the training `paired_epoch_order` and
`_sample_indices` functions for the next two epoch indices5/6, seed and60%
retention. Split each order into72-WSI windows, compare both methods on that
same window, and invert method order on alternating windows and passes.
This provides completed paired windows before another whole-epoch scan can
consume the90-minute runtime budget. Both branches synchronize before every
condition and use two readers each. The maximum is2304 combined reads; only
completed paired requests/windows support comparisons if the cutoff intervenes.
An active filesystem call can outlast the deadline. WSI/subset hashes and
selected-value hashes retain direct content-equivalence evidence.

Record average/p95 read+subset time, completed-window concurrent wall and
residency/memory snapshots. Add `/proc/self/status` VmRSS/VmHWM in bytes before
and after reads and after subset selection while full+selected arrays are
alive. VmHWM is a cumulative process high-water mark shared by the two reader
threads; it is not an isolated method-specific or whole-host peak. Method
order/backend warming and finite windows remain confounders, and CPU does
not replicate the T4 model/host-transfer memory allocation. Preserve natural
cache observations and evaluate full-read throughput/memory before changing
the training reader.

Focused verification runs both branch functions concurrently on synthetic
shards:24 reads across eight blocks per branch reproduce independent source
arrays, exact training orders and matching subsets, exclude the held-out fold,
and populate memory fields. Seven existing reader tests, compilation, Ruff,
targeted BasedPyright and kernel build/validation pass. The first72 ordered
WSI cover all12 shards and39.212GB paired full contents. Source
`ceb96369af9dc89bae1b01b5fbc3cd88382a2629` and entry/metadata `59399e4`
are published. Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/22`
around20:45 Bogota on2026-10-26, API RUNNING.
Next useful inspection around22:30 Bogota or when artifacts become available.
No training continuation accompanies this CPU measurement.
Initial logs confirm the exact source,288 training WSI and all12 input files
per branch, with both workers reporting gpu_used=false. Initial stream:
`runs/local/spec0054_io_full_atlas_v22_review/initial_stream.txt`.

V22 inspection around22:30 Bogota confirms platform COMPLETE and worker
exits[0,0]. Both branch results completed=false at the runtime cutoff, with
246 reads each (492 total); in-flight calls finish normally. Epoch index5
window0 completes72 requests per method/branch. Window1 completes72 full
reads and30 sampled reads per branch. Neither full epoch nor epoch6 is
complete. The paired analysis uses only102 WSI per branch with both methods;
the other42 full reads per branch do not enter method comparisons.

All204 branch/epoch/WSI comparisons retain identical subset and selected-value
hashes. Ordered request/subset keys agree between branches, final rows match
JSONL and mincore succeeds throughout. All12 shards are represented. Pooled
matched read+subset means are41.63821s sampled/42.20478s full (full +1.36%);
medians22.72452/24.67909s and p95183.71557/172.22919s. Normal means are
41.12041/40.95470s; SO2 means42.15602/43.45486s. Window0 concurrent wall
is1312.24022/1248.98873s, only4.82% lower with full. In the30 matched WSI
from reversed-order window1, normal means59.05494/67.28858s and SO2
57.95865/66.65760s favor sampled. Full-window1 wall cannot be compared with
the truncated sampled-window1 wall. These data do not establish a repeatable
average gain from selecting full input reads on the larger working set.

Mean pre-read residency on matched rows is.00634 sampled/.00996 full,
substantially lower than the earlier small-panel mixture. The measured
sampled-read mean/median reproduce the scale of v18 training I/O delays;
this session does not reproduce v21's full-read speedup. Largest matched
sampled means occur on shard8 (~71.65s), shard3 (~60.31s) and shard2
(~58.74s), rather than a unique shard1 hotspot. No backend cause is established.
Peak cumulative VmHWM is2.26469/2.25580GiB normal/SO2; largest sampled
RSS snapshots are1.35366/1.34724GiB and full-read snapshots2.23841/2.23380GiB.
RSS/HWM cover both reader threads and retained allocator memory, not isolated
method allocations or a GPU training peak. Minimum observed MemAvailable is
11.156GB host-wide. No OOM or GPU use occurs.

Keep the training reader unchanged pending a measured larger-set mitigation;
scratch staging at full-atlas scale is not measured by this direct-read probe.
Deadline censoring, method order, backend/cache state, only one complete
paired window and CPU allocation limit inference. No classifier continuation,
reader change or new run accompanies the review. Outputs:
`runs/kaggle/spec0054_io_full_atlas_v22/`; reproducible numerical review:
`runs/local/spec0054_io_full_atlas_v22_review/{analyze_v22.py,summary.json}`.

OS cache-management candidates remain unmeasured. The current sampled reader
copies file bytes into owned numpy buffers; its file-page cache can therefore
be advised DONTNEED after consumption without discarding the owned inputs.
Target only the page-aligned bag range, retaining bags needed soon by scheduled
assessments. V20's pre-read eviction experiment does not estimate this
post-consumption policy's throughput. A bounded lookahead using WILLNEED or
readahead could warm upcoming ranges while other work proceeds; the existing
two-WSI data prefetch already overlaps I/O, and extra requests can contend for
the same NFS bandwidth. Measure total elapsed including advice/lookahead cost,
not only the apparent latency of a read issued after prefetch. A proposed narrow
comparison is baseline, post-consumption discard, and discard plus bounded
lookahead, with identical inputs/orders and no GPU. This is a proposal, not
a submitted run. NOREUSE's replacement-policy effect requires Linux6.3+;
record the actual Kaggle kernel before considering it. Where accessible,
per-mount NFS RPC statistics can inform whether transport/server delay dominates.
These per-file hints do not control backend caching. Global drop_caches is
not proposed as a runtime cache policy. Sources:
[fadvise](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html),
[readahead](https://man7.org/linux/man-pages/man2/readahead.2.html),
[kernel VM documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html#drop-caches),
[mountstats](https://man7.org/linux/man-pages/man8/mountstats.8.html).

## CPU cache-policy and native-mmap experiment (2026-10-26)

The existing read-probe compares four input policies: sampled `preadv`
baseline, sampled plus post-copy DONTNEED, discard plus bounded WILLNEED,
and native read-only mmap followed by an owned sampled gather. The shared
training reader and accepted training checkpoints remain unchanged. Use all12
frozen input mounts, the training seed/order/subsets for epoch indices5/6,
72-WSI windows and two readers per branch. Submit only two data futures and
consume FIFO, matching the training prefetch/backpressure pattern. Both branches
run the same policy together. Contract-specified Williams sequences balance
each position and each directed predecessor pair across four complete windows;
partial deadline results cannot assume completed counterbalancing. The maximum
is4608 combined reads within a180-minute runtime budget. Active filesystem
calls may overrun the cutoff.

DONTNEED targets only the page-aligned consumed bag after owned selected
bytes exist. Future hints issue WILLNEED for at most16MiB per WSI, four WSI
ahead; seed the first four hints inside the timed window and issue one new
future hint as each bounded read starts. Persist every hint's target/range
length/timestamps/call duration, including hints whose target was not reached
before cutoff. Request length is not guaranteed actual readahead. No global
cache flush, scratch staging or NFS remount is used.

The mmap arm uses Python's native `mmap(..., ACCESS_READ)` on the bag range
and a NumPy typed view without copying the full bag. Integer-index gathering
creates the owned writable sampled array; release the view before closing
the mapping. C4 acts only on that owned array. All four arms perform identical
C4 and channels-last host preparation using pin_memory=False, with stage wall/
CPU time and raw/prepared value hashes. The prepared digest visits physical
NHWC bytes through a contiguous view, avoiding an extra verification copy.
GPU pinning and H2D are not measured or claimed optimized by this CPU run.
[Python mmap](https://docs.python.org/3.12/library/mmap.html) documents read-only
mapping semantics; file mapping and typed NumPy views are complementary here.

Capture cache residency before/after reading, after discard and after host
preparation, together with process RSS/HWM and the actual kernel release.
Record per-input-mount raw NFS READ counters before/after each window condition,
excluding server addresses. Absence of exposed NFS counters remains an empty
observation. Counters are shared by both branches and potentially shared
bind mounts; do not sum duplicate observations. Compare completed paired
windows using total wall including advice/initial hints, and retain separate
read/host/discard/hint timings and verification overhead. Backend warming,
carryover and CPU allocation remain limitations.

Independent synthetic verification runs both branch functions through48 reads
and16 blocks each: all four policies match independent raw arrays and C4/NHWC
prepared values, preserve frozen-file SHA, training order and held-out-fold
exclusion, and populate memory fields. Contract Williams sequences occupy each
position once per method and cover all12 directed predecessor pairs once.
Seven existing reader tests, compilation, Ruff and targeted BasedPyright pass.
Subagent review found the initial cyclic ordering's predecessor confound; the
contract Williams design corrects it. No mmap lifetime/FIFO issue remained.
Source `3716b489a2ee1bae548aec302ee6386d7c3caba6` and thin entry `adb0c7b`
are published. Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/23`
around22:57 Bogota on2026-10-26; API RUNNING. Initial logs confirm288 train
WSI per branch, all12 inputs, kernel6.18.55+ and GPU/host pinning/H2D disabled.
Runtime cutoff is around01:57:12 Bogota on2026-10-27; next useful inspection
around02:15 or when artifacts become available. Initial stream:
`runs/local/spec0054_io_os_v23_review/initial_stream.txt`.

The original training path opens input descriptors with O_RDONLY and reads
retained bytes into owned NumPy storage. torch.from_numpy shares this buffer;
the explicit host.copy_ then packs it into channels-last pinned storage before
nonblocking H2D on a reader stream. Reinspection of928 consumed WSI per branch
in v18 gives mean read39.19086/39.14295s, host preparation.40559/.34863s
and completed transfer.01505/.01502s normal/SO2. The measured dominant delay
precedes H2D. File-backed mmap changes the host-access mechanism; it does not
establish storage-to-GPU bypass. NVIDIA GPUDirect Storage offers such a path
on compatible deployments, but Kaggle support is unverified. A direct-to-pinned
read could remove one host allocation/copy, with layout/C4 placement and actual
transfer performance requiring a separately authorized GPU measurement. No
such change or measurement is included in this CPU experiment. Evidence:
`runs/local/spec0054_io_os_v23_review/transfer_audit_v18.json`;
[PyTorch shared NumPy storage](https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html),
[pinned/nonblocking transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[NVIDIA GDS](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).

V23 inspection on2026-10-27 around11:38 Bogota confirms COMPLETE and worker
exits[0,0]. Both results completed=false at the180-minute budget, with357
reads each/714 combined. Window0 completes all four policies on72 WSI per
branch; window1 contains only69 discard reads per branch. No full training
epoch or second pass is complete. Comparisons below exclude unmatched window1.
All144 branch/WSI groups preserve selected-value, prepared-value, subset and
C4 hashes across all four methods. Branch request/subset/C4 keys agree;
raw/hint JSONL matches final rows and mincore succeeds. No GPU or OOM occurs.

Pooled matched read means are47.79373s baseline,37.84359s discard,39.95694s
mmap and26.58897s discard+prefetch. Pipeline means (read, hint calls, discard
and CPU host preparation; excluding snapshots/hashes) are48.20771/38.24346/
40.60400/27.07969s. Concurrent window wall including initial hints and all
instrumentation is2400.37805/1968.59100/2094.86510/1498.17998s. Descriptive
wall reductions against baseline are18.0% discard,12.7% mmap and37.6% combined.
Baseline/mmap read CPU means are.40027/.82101s; host preparation.41397/
.64706s. The observed mmap wall improvement does not indicate less CPU work.
All methods retain long tails: p95 reads205.80139/204.45876/207.43684/
200.51904s in the same order. Peak cumulative process HWM is1.94830/1.96875GiB
normal/SO2, covering both readers and instrumentation, not GPU training peaks.

Policy order in the only complete window is baseline, discard, mmap, then
discard+prefetch. Thus the Williams position/predecessor balance was not
completed. Mean pre-read residency is~.000016 baseline, .17975 discard,
.000035 mmap and.31597 combined. All baseline/mmap reads start below1%
resident, but their backend access history still differs. Combined has21/72
WSI per branch above90% residency, consistent with carryover after mmap.
Each hint requests only16MiB (512 patch records), so large preexisting
residencies cannot be uniquely assigned to that advice. DONTNEED actually
reduces measured bag residency to0 in every discard row. These observations
establish cache-state changes, not their causal contribution to throughput.

Each branch issues72 hints to72 unique measured targets, requesting1.20766GB
in total. Hint-call totals are4.48347/4.30010s normal/SO2 and are included in
window wall. Per-input-mount NFS READ counters are empty; no remote-delay
mechanism can be inferred from them. A narrow baseline-versus-discard+prefetch
confirmation with reversed ordering is the next candidate before selecting a
policy; mmap plus discard/prefetch remains an unmeasured combination. No new
run, training-reader change or classifier continuation accompanies this review.
Outputs: `runs/kaggle/spec0054_io_os_v23/`; numerical review and one-shot script:
`runs/local/spec0054_io_os_v23_review/{summary.json,analyze_v23.py}`.

## Combined-policy confirmation with deeper CPU prefetch (2026-10-27)

The authorized next comparison narrows to three packages: baseline sampled
preadv, sampled preadv with post-copy discard and bounded WILLNEED, and native
read-only mmap with MADV_SEQUENTIAL plus that same discard/WILLNEED policy.
Within a WSI the retained patch indices are sorted, so the mapping's sequential
hint describes the local range. It does not label the globally random blob
traversal sequential. No NOREUSE, global sysctl, remount or full-atlas staging
change is included. The shared training reader/checkpoints remain unchanged.

Use the exact first72 WSI of the epoch-index5 training permutation for both
repetitions, with the same60% subsets and C4 codes. This prefix spans all12
shards and39.212GB paired full contents. Execute ABC then CBA. Reversal balances
average position/linear trend; the middle package remains second, so nonlinear
backend warming is not fully counterbalanced. Before every package, request
DONTNEED for only these72 page-aligned bag ranges, record observed residency
without a pass/fail threshold, and synchronize both branches afterward. Keep
reset timestamps/cost and waits distinct from timed input-pipeline wall. The
reset is an experimental control, not a proposed per-epoch training operation.
It does not invalidate remote NFS/backend caches.

Two readers remain active per branch. Baseline retains two data futures;
both candidate packages allow eight submitted/prepared CPU WSI in FIFO order.
Reserve each expected selected-buffer size before submission and release that
reservation only on consumption. Limit reserved prepared storage to3GiB per
branch; every individual frozen selected buffer is smaller than this budget.
Return owned prepared host tensors from the reader so ready futures actually
retain their data. Record maximum pending count, sampled ready count and
reserved bytes; ready counts are sampled lower bounds. This is not a3GiB RSS
cap: source buffers, active C4 work, Torch/index allocations and file-page cache
are additional. RSS/HWM remain separate observations. The candidates test the
complete package including deeper prefetch, rather than isolating each trick.

The epoch plan records all288 training WSI in exact order, blob/byte ranges,
retained byte counts, contiguous-run counts and subset hashes. Actual ranges
come from the frozen physical shard index, not historical cohort patch counts.
Sampling seed/epoch/count reconstruct the selected regions compactly. The plan
excludes full-fold assessment reads and is not a single OS-provided virtual
epoch address space or swap device. Only its bounded prefix is exercised in
this experiment; no full epoch is loaded into RAM. Existing file-backed mmap,
read-ahead hints, owned CPU buffers and explicit discard implement the window.

GPU, pinning and H2D stay disabled. Same C4 and channels-last CPU preparation,
raw/prepared hashes and per-stage timings apply to all packages. Actual policy
wall includes read-ahead calls, discard, CPU preparation and verification;
initial client-cache reset is reported separately. The maximum is864 combined
reads within a240-minute runtime budget, retaining partial outputs when active
I/O outlasts the cutoff. The whole-epoch plan does not change training order
or model updates. [MADV_SEQUENTIAL](https://man7.org/linux/man-pages/man2/madvise.2.html)
is a read-ahead/replacement hint, not a guaranteed acceleration.

Focused synthetic verification performs264 reads across both branches using
depth8 and tighter byte-budget configurations. It checks raw/prepared values
against independent arrays, exact FIFO/subsets/C4, source-file SHA preservation,
physical epoch-plan offsets and both count/byte reservation limits. Seven
existing reader tests, compilation, Ruff, targeted BasedPyright and kernel
build/validation pass. Subagent review found no buffer ownership, reservation
or scheduling issue; nonlinear backend order effects remain a limitation.
Source `f032095bb07b5a2351e8481bf0162b9a90e112b0` and thin entry `faa3e97`
are published. Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/24`
around12:06 Bogota on2026-10-27; API RUNNING. Next useful inspection is
around16:30 Bogota, allowing for the240-minute runtime budget and platform
preparation. No GPU use or classifier continuation. Initial stream:
`runs/local/spec0054_io_combined_v24_review/initial_stream.txt`.
The initial30-second stream contains no events. API RUNNING confirms platform
state, not worker/code-start or any timing measurement; GPU-off metadata is
verified locally. No additional polling accompanies the launch.

## Offline I/O strategy review (2026-10-27)

The exact epoch-index5 read plan, reconstructed with the existing sampler and
actual frozen bag sizes from v18 telemetry, contains288 training WSI and
1,469,274 selected32KiB records per branch. Both branches request96.290GB
paired from160.472GB of full training contents. Historical cohort patch counts
are not physical storage sizes and are excluded from this accounting.

Coalescing adjacent selected runs across small unselected gaps preserves every
selected value while trading extra bytes for fewer application read intervals:

| Largest missing gap read through | Intervals per branch | Paired requested GB |
| --- | ---: | ---: |
| 0 records, current plan | 587,674 | 96.290 |
| 1 record | 234,929 | 119.408 |
| 2 records | 94,074 | 137.870 |
| 3 records | 38,187 | 148.858 |

These are exact offline plan counts, not measured throughput or NFS RPC counts.
The negative full-bag comparison on the larger working set does not establish
the performance of intermediate coalescing. Bytes and run counts both grow
with bag size under fixed60% sampling, so latency-size correlation alone cannot
distinguish bandwidth limits from per-request costs. Coalescing varies those
quantities in opposite directions. Reproducible calculation:
`runs/local/spec0054_io_strategy_review/{analyze_coalescing_epoch5.py,coalescing_epoch5.json}`.

A20KiB local synthetic C example assembles a contiguous virtual address range
from three read-only mappings of two source files, with no payload copy before
access and unchanged source bytes. This establishes address-view correctness,
not acceleration or Kaggle capability. Linux mmap readahead uses the source
VMA's file and file offset, so concatenated addresses do not automatically
prefetch across original-file boundaries
([Linux6.18 filemap.c](https://raw.githubusercontent.com/torvalds/linux/v6.18/mm/filemap.c)).
Mapping each selected run separately would require roughly588,000 regions per
branch for this epoch, with unmeasured mapping overhead and Kaggle limits.
Mapping whole bags avoids that region count but retains subset extraction.
Prototype: `runs/local/spec0054_io_strategy_review/virtual_view_demo.{c,json}`.

The highest potential reduction in repeated remote reads is a verified
canonical scratch copy:160.472GB for train,200.333GB including validation.
Only33.474GB staging has been tested; full capacity/performance remain
unmeasured. Compare staging plus local training/assessment time against direct
reads within one Kaggle session, since scratch is ephemeral. A96.290GB packed
epoch file needs rebuilding when its subset changes; two such files already
exceed the full canonical training footprint. Preserving the original files
and relocating their root could reuse the unchanged training reader; changing
the reader/format requires resolving the recorded resume identity explicitly.

An explicit bounded scheduler can read future requests in source-file/offset
order while consuming the original training FIFO, prioritizing the blocking
WSI and retaining data under a measured RAM budget. Larger lookahead, actual
reader concurrency and per-reader file-descriptor readahead state are distinct
factors. An offline sequential source scan that gathers selected data into a
local epoch file is another hypothesis; include all extra reads and writes.
Lossless chunk compression could reduce source traffic after a verified
repack, but its real FP16 ratio and decoding cost are unknown. If whole bags
must be decoded, compression must exceed approximately1.667 times merely to
match the logical bytes of a raw60% subset, before cache/readahead effects.

FUSE could expose one logical file with kernel readahead, while a userspace
pager via userfaultfd could translate faults and prefetch from the epoch plan.
Both retain underlying source I/O and introduce additional machinery; their
Kaggle permissions are untested. Cached FUSE supports readahead whereas its
direct mode bypasses it
([kernel documentation](https://docs.kernel.org/filesystems/fuse/fuse-io.html)).
io_uring could batch/asynchronously overlap requests within bags; availability
and any gain over bounded threads remain unmeasured. NFS remount controls,
direct I/O and GPU-direct storage require separate capability evidence and
cannot be presumed available. No additional Kaggle job, training modification
or v24 status inspection was performed during this offline review.

## CPU coalescing comparison (2026-10-27)

The existing read-probe contract now compares the original sampled reader with
two gap-coalescing readers, retaining the exact first72 epoch-index5 WSI,
patch subsets, C4 codes, CPU layout preparation and FIFO consumption. All arms
use identical two-reader/two-future limits and selected-buffer reservations.
No forward hints or post-read discard are active in these arms. The existing
per-arm client-cache reset and observed mincore residency apply equally;
remote caching remains uncontrolled. Three Latin-square repetitions balance
each method across execution positions; carryover effects are not fully
counterbalanced. Numerical settings are in `docs/data/spec0054_fold0_run.json`.

The probe-only reader uses a single continuous source interval per preadv and
scatters retained runs directly into the selected output array. Intervening
unselected bytes go to distinct slices of an expendable gap buffer. Groups
split before exceeding SC_IOV_MAX. Row telemetry distinguishes selected bytes
from requested bytes, counts the actual application read intervals and retains
raw/prepared/subset/C4 hashes. This avoids an extra gather of selected data
after over-reading. The original shared ShardBags reader is unchanged.

Focused local checks compare arbitrary FP16 bit patterns, including special
values, with independently indexed source uint16 arrays at nonzero bag offsets.
They observe exact requested lengths and range bounds, force a small iovec
limit, check concurrent reads sharing a descriptor and source immutability.
These establish content/offset correctness, not remote throughput. The entry
allows360 minutes from execution start and no GPU use. The initial preparation
left v24 unchanged; the coalescing comparison now runs in an independent session.
Source `e902d26fece5bd1b7f1df7fd7ab114c573d25aa1` is pinned by the prepared
thin entry. Ten focused tests and396 local synthetic pipeline reads pass,
including both branches, all three methods/repetitions and normal/tight queue
budgets. Raw/prepared hashes, FIFO, requested bytes/intervals, immutable inputs
and reservation/count bounds agree. Compilation, Ruff and targeted
BasedPyright pass. Evidence:
`runs/local/spec0054_io_coalescing_preparation/{check_probe.py,synthetic_check.log,verification.json}`.

Submitted as `maximshtefan/eqvae-spec0054-io-coalescing/1` around13:04 Bogota
on2026-10-27, API RUNNING. The entry is byte-identical to commit `6d64577`;
only kernel id/title differ in the launch metadata, retaining GPU-off and all
original input locators. Launch snapshot/receipt:
`runs/local/spec0054_io_coalescing_launch/`. Possible overlap with v24 is
recorded; inter-session storage interference is unmeasured and external load
may vary within a session. The original v24 job remains unchanged. Next useful
inspection is around19:30 Bogota, allowing for the runtime budget and setup.
Platform RUNNING alone does not establish worker start or measured throughput.

Portable copies of the small supporting artifacts are tracked in
`reports/spec0054_io_20261027/`: training v18 and I/O v21/v22/v23 numerical
summaries, exact epoch coalescing accounting/script, the virtual-mapping
example/result, synthetic coalescing checker/verification and launch
metadata/receipt. They preserve the original bytes and historical observation
times; no new measurement is implied by archiving them. The receipt identifies
source `e902d26`, entry `6d64577` and the separate kernel's exact version. The
entry code is already tracked in the existing kernel directory; only id/title
differ in the archived metadata. Raw outputs and classifier checkpoints remain
recoverable from their exact Kaggle versions. `CURRENT.md` gives the active
continuation snapshot and status/output commands for a fresh checkout.

## Completed CPU policy and coalescing results (2026-10-27)

Both sessions complete all planned reads/blocks and both workers exit0:
v24 has864 reads, coalescing v1 has1296. All raw/prepared/subset/C4 hashes agree
within branch across methods/repetitions; branch request keys and epoch plans
pair exactly, and partial-row JSONL agrees with final JSON. GPU, pinning and
H2D remain disabled. These are input-pipeline CPU measurements, not training
step throughput or evidence from the whole288-WSI training epoch.

The unit below is joint wall for the same72 WSI per branch, running both
branches concurrently; average over each run's repetitions. It includes input
pipeline verification but excludes initial cache reset and synchronization.
Ratios compare methods within the same session only.

| Session / method | Mean joint window wall, s | Saving versus its baseline |
| --- | ---: | ---: |
| v24 sampled baseline, queue2 | 285.842 | — |
| v24 preadv + discard/prefetch, queue8 | 228.301 | 20.13% |
| v24 mmap + discard/prefetch, queue8 | 227.448 | 20.43% |
| Coalescing v1 sampled baseline, queue2 | 267.496 | — |
| Coalescing v1 gap1, queue2 | 179.610 | 32.86% |
| Coalescing v1 gap2, queue2 | 187.754 | 29.81% |

V24 package savings are19.0/21.3% for preadv and19.6/21.2% for mmap across
the two repetitions. Mmap saves only0.37% compared with the preadv package.
Per-WSI read means are4.963/4.787/4.726s, much closer than the window-wall
differences; deeper preparation/scheduling is part of the package and has not
been isolated from the cache policy. Queues reach eight pending/four observed
ready WSI with maximum reserved selected buffers1.963GB; process HWM is
2.27/2.37GiB. Reset maximum observed residency is0.33%; mean residency before
candidate reads is about0.32%, rather than the large v23 carryover. Backend
caching remains unobserved. The original baseline is also much faster than
v23's47.794s mean read, so cross-session improvement cannot establish a reader
effect or resolution of the earlier storage cause.

Gap1 reduces window wall in all three repetitions by36.8/27.2/34.9%; gap2 by
31.1/26.9/31.6%. Gap1 beats gap2 in all three repetitions. Per-WSI mean reads
are4.528/2.533/2.705s. Paired application intervals per window fall287,792 to
114,784/45,762, with requested bytes23.529/29.198/33.721GB. Retained tensors
are unchanged. Peak HWM is1.88/1.80GiB; before-read residency remains low
(maximum1.59%), so a complete cached-WSI population is not supported by the
client snapshots.

NFS READ counters are exposed in both new sessions, unlike v23. All twelve
mount views repeat identical counters; analysis retains one normal-branch
view, without summing aliases or branches. The scope is the shared RPC client,
not an individual bag or branch. Field order and millisecond conversion follow
[Linux6.18 `_print_rpc_iostats`](https://github.com/torvalds/linux/blob/v6.18/net/sunrpc/stats.c).
Coalescing baseline/gap1/gap2 average about280,824/77,983/75,504 READ operations
and24.10/38.89/39.17GB of reply bytes per window. Gap1 therefore reduces
observed RPC operations72.2% while increasing reply traffic61.4%, beyond its
24.1% logical over-read. Reply bytes include protocol overhead and any other
work in that client scope; increased read-ahead is a plausible contributor,
not an isolated measurement. V24 mmap similarly receives about33.66GB versus
24.09GB for the matching preadv package. No observed retransmissions, major
timeouts or READ errors occur in these intervals. Larger requests can improve
this session's wall while increasing traffic; performance under a bandwidth-
limited backend remains uncertain.

V24 timed blocks run12:49:01--13:13:51 Bogota; coalescing runs
13:04:16--13:36:14, so actual overlap is about9min35s. Position balancing does
not remove unknown backend load/carryover. A next narrow comparison could
combine gap1 with the deeper queue/cache policy versus gap1 alone; the two
separate savings are not additive evidence. No training reader was promoted
or classifier resumed. Results and one-shot numerical analysis are tracked in
`reports/spec0054_io_20261027/{v24_combined_summary.json,analyze_v24.py,coalescing_v1_summary.json,analyze_coalescing_v1.py}`.
Raw outputs:
`runs/kaggle/spec0054_io_combined_v24_handoff/` and
`runs/kaggle/spec0054_io_coalescing_v1_handoff/`, recoverable from the exact
Kaggle versions above. Historical launch statuses and scheduled check times
earlier in this spec predate this completed review.

## Gap1 queue/cache combination confirmation (2026-10-27)

The next CPU-only comparison fixes scatter coalescing at one missing record
for every arm. A retains two submitted/prepared WSI; B retains eight with no
forward hints or post-read discard; C uses the same eight-WSI limit plus the
existing bounded WILLNEED and post-copy DONTNEED policy. B/A measures the
queue-depth change, C/B measures the joint cache-policy increment, and C/A
measures the complete combination. This does not separate discard from hints.
The shared training reader and all classifier/checkpoint states remain unchanged.

Replay the same first72 epoch-index5 training WSI, exact60% subsets, C4 codes,
CPU channels-last preparation and FIFO order in ABC/BCA/CAB order. All arms
use two reader threads per branch and a3GiB selected-buffer reservation cap;
active buffers, scatter gap storage, C4 temporaries and OS cache are additional
and RSS/HWM remain separate observations. C hints four WSI ahead, at most16MiB
per future WSI, with initial four hints inside timed wall. The same client-cache
range reset/residency observation and branch barriers precede every arm;
reset/barrier time remains separate. Remote caching, nonlinear load and
carryover remain uncontrolled by position balancing.

Maximum1296 combined reads and360 minutes from entry execution; no GPU,
pinning, H2D or model training. Requested application bytes include hole bytes;
retained bytes, read intervals and raw/prepared/subset/C4 hashes remain recorded.
NFS counters must be deduplicated across identical mount views and branch
snapshots when assessing reply traffic and RPC counts. The earlier33% and20%
gains are not combined estimates; this experiment measures the actual package.
Numerical parameters modify the existing read-probe contract only.

Ten focused reader tests, compilation, Ruff and targeted BasedPyright pass.
The adapted existing synthetic checker verifies396 reads across both branches,
all arms/repetitions and normal/tight reservation budgets. It checks exact
raw/prepared values, requested bytes/intervals, FIFO, unchanged source files,
queue bounds, and66 WILLNEED/264 DONTNEED calls per budget case across both
branches. Only C generates33 hints per branch;99 reset calls plus33 post-read
discard calls per branch account for DONTNEED. Independent subagent design and
diff review found no blocking issue. Archived checker and verification:
`reports/spec0054_io_20261027/{check_gap1_queue_probe.py,gap1_queue_verification.json}`.
Source `1126d6a68bd006f26af4a420acd9185c0ff32d0e` is pinned by the prepared
thin entry. The separate coalescing kernel retains its original CPU-only
metadata and all frozen input locators.

Submitted as `maximshtefan/eqvae-spec0054-io-coalescing/2` around17:08 Bogota
on2026-10-27, API RUNNING. Entry `d35a152` pins the source above; metadata is
byte-identical to the archived coalescing metadata, including GPU-off and all
original12 inputs. No classifier continuation or shared-reader change.
RUNNING establishes platform state; worker start and timing measurements have
not been inspected. First useful check around18:30 Bogota based on the previous
fast-session runtime; a slow session may need the360-minute budget plus setup,
with a later check around23:30. Receipt:
`reports/spec0054_io_20261027/gap1_queue_launch_receipt.json`.

Reinspection around18:36 Bogota confirms API RUNNING. A30-second live stream
returns0 bytes, leaving worker start and numerical/timing progress unconfirmed.
This cannot distinguish platform setup, active execution or delayed log
delivery and does not establish a storage failure. The session is unchanged;
next useful inspection is around23:30 or once logs become available. The
360-minute runtime budget begins at actual entry execution; platform setup and
in-flight reads may extend elapsed time after submission. Snapshot:
`reports/spec0054_io_20261027/gap1_queue_v2_live_1836.json`.

## Bounded I/O followup (2026-10-23)

Reuse the existing read-probe mode in a private Kaggle CPU session, mounting
only the unchanged shard1 dataset and shard2 notebook output as a control.
The numerical contract fixes two near-median fold-training WSI per shard,
two reader threads per branch and two repetitions per method. Both VAE store
processes run concurrently. No classifier, augmentation, H2D or GPU is used.

Compare the existing full contiguous bag read followed by the identical60%
subset selection against the existing sampled-region preadv read. In each
shard the two WSI reverse the order of these methods; repeat each method
before switching. Record bytes, request regions, wall/thread-CPU read time,
subset time, timestamps, physical input paths and hashes of selected values.
Save each completed read immediately. This entails16 reads per branch,32
combined, and leaves all training states untouched. A fixed session cutoff
bounds subsequent reads; a system call already in progress finishes naturally.

First read by this script does not guarantee cold Kaggle/OS/backend caches.
Repeated reads describe small-panel cache behavior, not complete-atlas training
throughput. Identical selected-value hashes establish content parity across
methods; differences in timing remain observations rather than acceptance gates.
The earlier v22 probe already used the shard1 dataset successfully, so its
input type alone is not an established cause of v13's slowdown.

Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/14` on2026-10-23 at10:11
Bogota, source338ff7d and entry0b3ee59. Initial API status RUNNING. Live logs
confirm both CPU workers, GPU unused, four logical CPUs and the expected two
physical input paths. Both full and sampled reads started; no timing result
was yet available in the initial log window. Shared submission-based cutoff
is2026-10-23T10:40:48-05:00; inspect around10:20 or once complete.

Inspection around10:25 Bogota confirms COMPLETE, both workers exit0 and all32
reads complete. All8 branch/WSI groups have identical selected indices and value
hashes across both methods and repetitions. First-script full reads cost
366.19--367.59s on shard1 and274.25--275.08s on shard2; first sampled reads cost
374.41--376.23s and278.25--279.32s. Thread CPU for these reads is0.33--0.61s.
Same-method second reads cost0.0466--0.2282s. Full reads after sampled reads
still cost24.44--25.23s before falling to0.16--0.23s on their second repetition.
These observations implicate storage/cache behavior for both input types and
even contiguous reads. They do not isolate the backend mechanism or explain
the earlier v10/v13 difference. First-method orders use different WSI, so their
initial timings are not a paired cold-cache method comparison. The probe's
small repeated working set does not predict whole-atlas training throughput.
No reader change or further launch accompanied this inspection. Outputs:
`runs/kaggle/spec0054_io_probe_v14/`; local numerical summary:
`runs/local/spec0054_io_probe_v14_review/summary.json`.

The authorized local-copy followup reuses the same probe on shard1 only and
its two fixed training WSI. Per branch, measure the8 input reads first, stream
the full8,340,340,736-byte binary to the contract's Kaggle working directory,
then repeat the8 reads against the copied file using the unchanged reader.
Record full-copy wall/thread-CPU time including SHA256 computation and fsync,
bytes, source/destination paths, destination device and free space. Retain the
streamed digest and the authoritative frozen binary digest as observations.
Both branches copy and probe concurrently, yielding32 reads and two copies.
Persist copy progress and completed records; delete temporary binaries on exit.

After fsync and before local reads, issue POSIX_FADV_DONTNEED for the copied
file. This is advisory rather than proof of a cold cache or physical-disk read.
The input phase warms the selected source regions before the full copy, so
copy timing includes this order effect. The one-hour submission-based cutoff
is checked between reads and16MiB copy chunks; in-flight calls finish naturally.
Copy amortization and local performance on this panel do not establish whether
all12 paired shards fit or sustain the same performance. Classifier training,
frozen inputs and model states remain untouched.

Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/15` on2026-10-23 at10:32
Bogota, source `f4c29f49c4f20f8bb7963e1a5f224d432d642ceb`, entry `dfc0f7d`.
API confirms RUNNING; logs confirm the pinned source, both CPU branches,
four logical CPUs, GPU unused and the expected shard1 paths. Both initial
input methods started; copy timing remains pending. Shared cutoff is
2026-10-23T11:32:01-05:00. Inspect around10:45 Bogota or after completion.

Inspection on2026-10-25 confirms COMPLETE, worker exits[0,0] and all32 reads.
The two8,340,340,736-byte copies have exact authoritative SHA256 matches. All4
branch/WSI groups retain identical selected-value and subset hashes across
input/local methods and repetitions; selected values also match v14. Copy wall
times including hashing and fsync are141.90s normal/133.41s SO2, concurrent,
with141.90s total phase elapsed and45.21/45.40s thread CPU. Input first reads
cost759.27--784.45s, while all local reads cost0.0746--0.5020s and first local
reads per WSI cost0.2237--0.5020s. Source device53 and destination1794 differ.
Input repetitions are also fast; the comparison does not isolate physical
disk performance or remove the advisory-cache limitation. Full copy follows
the initial input reads and therefore is not a cold-start staging estimate.

Observed initial free working space is20,940,505,088 bytes, falling to about
4.26GB after both copies. Shard1 paired binaries occupy16,680,681,472 bytes;
all12 pairs require200,332,607,488 bytes and cannot be staged together in this
measured session. The result supports investigating a shard1 local cache, with
unchanged sample order, but neither total training speed nor a whole-atlas
cache is established. No further execution or training-reader change accompanies
this review. Outputs: `runs/kaggle/spec0054_io_local_copy_v15/`; numerical
summary: `runs/local/spec0054_io_local_copy_v15_review/summary.json`.

On2026-10-25 the user authorized repeating the original direct-input CPU probe
to measure persistence of the slowdown. Reuse the already-published v14 source
`338ff7dd6a7c5a06d4d6fc72e7792a3f33ac23d4` and its original input metadata.
The two WSI per shard1/2, paired subsets, method order, reader count and repeats
remain identical:32 reads and no local copying/cache preparation, GPU or model.
Only the30-minute submission cutoff changes to2026-10-25T14:22:00-05:00.
Record physical paths, hardware, times and value hashes as before. Compare
first accesses and repeated accesses separately against v14 and the input
phase of v15. Session variation does not identify an actual backend incident
and cannot predict whole-atlas training throughput. Inspect around14:05 Bogota
or after completion.

Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/16` on2026-10-25 around
13:53 Bogota. API confirms RUNNING. Initial logs confirm source338ff7d, both
CPU processes, four logical CPUs, GPU unused, the v14 panel and the expected
dataset/notebook input paths (devices54/53). Both initial shard1 methods have
started; completed read timings are pending. This entry requires no new source
publication and uses the old published contract, not the checkout's local-copy
probe configuration. Shared cutoff remains2026-10-25T14:22:00-05:00.

Inspection around14:22 Bogota confirms COMPLETE, worker exits[0,0] and all32
reads. Source/reader/contract hashes match v14, and all32 paired request records
retain identical bag sizes, retained subsets, bytes, region counts and selected
value hashes. First accesses on shard1 cost163.40--167.53s versus366.19--376.23s
in v14 and759.27--784.45s in v15; the mean is55.4% lower than v14. Shard2 costs
269.13--274.99s versus274.25--279.32s in v14, mean1.7% lower. Same-method second
reads cost0.0471--0.2285s; first-access thread CPU is0.33--0.57s. Thus shard1
varies substantially between sessions while multi-minute direct access remains
for both shards. These observations do not identify a Kaggle backend incident
or establish full-atlas training throughput. No new run, local copy or reader
change accompanies the review. Outputs: `runs/kaggle/spec0054_io_direct_repeat_v16/`;
summary: `runs/local/spec0054_io_direct_repeat_v16_review/summary.json`.

First-access delays alone do not establish a steady-state training issue or
justify local staging. The immediate repetitions in v16 are fast, while the
small panel does not measure cache behavior after traversing the working set.
A separate inspection of v13's512 consumed training presentations excludes6
unused drained-prefetch reads and identifies224 WSI revisited in epoch2. Second
training-read medians are32.55s normal/32.42s SO2 versus4.45/4.58s for matching
WSI/subset requests in v10. Shard1 contributes20 revisits with medians203.02/
200.72s versus4.11/3.70s; the other204 take29.86/29.78s versus4.54/4.68s.
Each epoch changes the60% subset and full-bag assessments intervene. This is
evidence of slow subsequent reads in historical v13, not evidence that the new
session retains that behavior. A current throughput measurement must include
revisits after interleaving representative other bags rather than infer them
from first accesses or immediate repeats. No new execution accompanies this
interpretation correction. Numerical review:
`runs/local/spec0054_cls_fold0_c4_v13_review/subsequent_training_reads.json`.

The authorized small followup uses six fixed fold-training WSI across shards1/2.
First read their full bags once to separate initial access from subsequent
sampled requests. Then perform three seeded random permutations of the panel,
one read per WSI per pass, using the existing60% sorted patch sampler with
epoch keys0/1/2. Both branches have identical orders and subsets and two reader
threads. Rotate a pass order if its first WSI equals the preceding pass's last
WSI, avoiding an immediate boundary repeat in the planned request sequence.
Log round, planned request index/previous index/gap, bytes, region count, read
wall/thread-CPU time and selected-index/value hashes. Keep initial and subsequent
timings separate. There are6 initial and18 subsequent reads per branch,48 total.

Reuse the existing probe function and original two input mounts, without local
copying, forced cache eviction, augmentation, H2D or model execution. A30-minute
submission-based cutoff applies before each request; in-flight I/O finishes
naturally. This small-panel check can measure random revisits with intervening
bags but cannot establish behavior after traversing the entire200GB atlas.
Historical probe variants remain pinned to their original published commits.

Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/17` on2026-10-25 around
14:44 Bogota after explicit publication/launch authorization. Source
`03dfbb2bcf6b5bafa75b71698d448b3d70c5bafd`, entry `0bdc98a`; API confirms
RUNNING. Shared cutoff2026-10-25T15:14:01-05:00. Inspect around15:00 Bogota
for subsequent-read timings or after completion. Classifier training is paused.

Inspection on2026-10-25 confirms COMPLETE, both worker exits0 and all48 reads.
The12 initial full reads take1.76--6.65s; the36 random subsequent sampled reads
take0.0532--0.2404s, pooled median0.0975s and mean0.1098s. Pass1/2/3 means are
0.1082/0.1061/0.1054s normal and0.1389/0.0948/0.1052s SO2. Paired orders,
request sizes/regions and subset hashes agree. There are2--9 intervening planned
requests, three distinct subsets per WSI, and all8 overlapping selected-value
checks match v16. The small panel does not reproduce slow subsequent access
in this session and does not require local staging to achieve these timings.
The result is compatible with session-dependent storage/cache behavior; it
does not establish a specific backend incident or whole-atlas training speed.
No classifier continuation or new execution accompanies this review. Outputs:
`runs/kaggle/spec0054_io_random_revisit_v17/`; numerical summary:
`runs/local/spec0054_io_random_revisit_v17_review/summary.json`.

## Accepted classical WSI baseline (2026-10-14)

Extract fixed descriptors from the existing frozen posterior-mu shards, retaining
the historical Otsu population and the same development WSI/folds. No VAE weights,
patch coordinates or MIL continuation states change. The CPU extractor applies
channel-wise spatial block means, flattens each patch in channel/y/x order and
concatenates the per-WSI mean and population standard deviation. Accumulate WSI
moments in FP64; store descriptors and summary tables in FP32. Cache the
intermediate-resolution patch vectors, with physical patch identity and WSI row
offsets, for later bag-of-features analyses. The numerical choices reside in
`docs/data/spec0054_classical_wsi.json`.

Run normal and SO2 extraction independently using the same code and operations.
Outputs are private branch-specific artifacts. Fit standardized multinomial L2
logistic regression on CPU. Choose a shared spatial resolution and regularization
strength by minimum mean unweighted inner-fold CE averaged over both branches.
The inner splits use only outer-training WSI; fit the scaler and class weights
inside each inner split. Refit both heads on all outer-training WSI and report
the held-out fold once, including unweighted CE, macro F1, balanced accuracy,
per-class results, confusion matrices and multiclass Brier score. This is an
exploratory fold0 comparison; the external cohort supplies no selection evidence.

The accepted extension evaluates every frozen outer fold and extends the L2
search toward smaller C. For each outer fold, choose the branch-shared grid/C
using only the inner partitions of that outer training set, then refit on its
complete training set before predicting its holdout. Each development WSI
contributes exactly one out-of-fold prediction per branch. Report per-fold
metrics, their mean/sample standard deviation and pooled out-of-fold metrics;
pooled macro F1 is distinct from mean fold macro F1. The fold0 pilot is retained
separately and remains part of the exploratory history. Reuse the extracted
summary tables; no latent reread or VAE/MIL training is needed.

## Accepted patch-pattern comparison (2026-10-15)

Use the fixed grid8 patch cache to learn MiniBatchKMeans vocabularies separately
for normal and SO2. Sample a capped number of patch rows per WSI using a
WSI-seeded permutation, identically across branches. For every inner training
partition and outer training partition, fit patch StandardScaler and centers
only on sampled patches belonging to that partition. Record training WSI and
sampled row identities. Assign complete bags to each fitted vocabulary and
normalize counts by bag size. Precomputed holdout transformations never update
the scaler or centers. Reuse a partition's histograms across all C candidates.

Compare fixed-grid8 WSI mean/std, a WSI PCA control and cluster proportions.
The PCA control standardizes mean/std on training WSI, fits full-SVD PCA there,
then uses the same standardized L2 logistic head as the other representations.
The PCA output and histogram have equal feature dimensions. This controls head
size but does not isolate every statistical difference between representations.
Keep the frozen outer folds and inner stratification. Select a branch-shared C
separately for each representation using mean inner unweighted CE, then refit
on outer training and report per-fold/pooled OOF metrics for all representations.
Numerical parameters and exact input identities are in the existing classical
contract. Frozen VAE checkpoints, foreground population and coordinates remain.

## Accepted H2T-style conditional means (2026-10-15)

Use the existing1024-dimensional grid8 patch cache without new VAE inference,
latent-shard rereading or PCA. Learn16 shared reference prototypes per training
partition and branch. Reuse deterministic WSI-capped paired patch samples and
the existing MiniBatchKMeans runner; L2-normalize each sampled patch before
fitting, then L2-normalize the fitted centers for nearest-prototype assignment.
This is a H2T-inspired VAE adaptation, not a reproduction of the paper's encoder
or spherical clustering implementation.

For every WSI, assign all cached patch vectors using their unit-normalized
directions, but average their original unnormalized features separately within
each assigned pattern. Concatenate the16 means in prototype order to obtain
16384 WSI features. An absent pattern contributes a zero vector; retain its
patch count as an occupancy diagnostic, not as an additional classifier input.
Accumulate sums in FP64, store feature tables in FP32 and fit heads on FP64
inputs. Zero vectors remain zero under L2 normalization.

Fit vocabularies independently inside all three inner training splits and the
outer training split for each of the five frozen outer folds. Do not fit or
update prototypes using validation patches. Select the common C for the two
branches by the existing paired inner CE criterion, then fit standardized L2
logistic heads on outer training and report held-out metrics and OOF predictions.
Keep earlier mean/std and histogram results as references; do not retune those
arms. Numerical settings and frozen source identities are in the existing
classical contract's `h2t` section. Execute representation construction in two
private CPU jobs, reusing the existing versioned-kernel workflow.

## Accepted cached PANTHER comparison (2026-10-16)

The user authorized PANTHER before label smoothing and requested Kaggle for
both representation construction and classification. Reuse the1024-dimensional
grid8 caches and16 components. Follow the official reference revision
`mahmoodlab/PANTHER@e7ffae402e146363fe1ca3813ffe68d248ec570f`, specifically
`src/scripts/embedding/panther.sh` and the MAP equations in
`src/mil_models/PANTHER/networks.py`: uniform initial mixture, fixed K-means
means, initial diagonal covariance1, tau1 and one MAP-EM update per WSI.
Use raw features without PCA or L2 normalization. The single update uses all
cached patches in a WSI and the same shared training-fitted priors for all WSI.
Priors regularize component masses, first moments and second moments; do not
replace this MAP update with an unregularized GMM fitted to convergence.

Use the CPU KMeans/Lloyd option for prototype construction on the existing
deterministic, paired WSI-capped sample. Retaining128 patches per WSI differs
from the original paper's sampling budget; the extractor remains the frozen
VAE rather than UNI. Explicit numerical settings are in the existing classical
contract's `panther` section. Compute moments in FP64 about a numerical origin
and store final statistics in FP32; this is mathematically the same MAP update
as the FP32 reference. Reference parity is checked on synthetic examples.

Concatenate all16 proportions, then all16x1024 means and all16x1024 diagonal
variances in the official `allcat` order:32784 features per WSI. A second head
uses only the mean slice,16384 features, from the same fitted mixture. Keep
the existing standardized multinomial L2 heads, five outer/three inner folds
and paired inner-CE selection of C, independently for the two representations.
This preserves the earlier classifier comparison rather than adopting the
reference paper's AdamW head training. No label smoothing is added.

Before encoding complete bags, record training/held-out sampled-patch mixture
likelihood, nearest-center and responsibility-weighted residuals, component
mass and assignment entropy for every fitted vocabulary. Then retain per-WSI
component masses, assignment entropy, effective components, variance extrema
and diagonal condition numbers alongside the full-bag statistics. Diagnostics
are observations, not pass/fail gates or parameters selected from validation.
The broader prototype audit below concerns replacing the local-global MIL
aggregator; this authorized classical comparison keeps that model frozen and
uses only the fixed16-component diagnostic scope above.

Run both frozen VAE branches sequentially in one private CPU Kaggle kernel,
then run both paired nested classifier comparisons in that kernel. The two
branches share the CPU allocation; GPU is disabled. Mount the exact classical
version1 outputs. No raw WSI access, latent re-extraction or external-cohort
evaluation is involved.

## Previous regularized spatial-CLS run (2026-10-08)

Continuation input is the private `maximshtefan/eqvae-spec0054-cls-regularized-fold0-resume/3` dataset, containing the v12 latest/best states and cumulative telemetry. Resume latest at normal5228 updates/20,912 presentations/epoch72 cursor176 and SO25182/20,728/epoch71 cursor280. Both have completed the epoch70 assessment. Source remains `f81ca4c574d07d6a2fedddac21eab22eab924500`. The thin entry retains the expired v12 cutoff as execution provenance; replace it with a fresh submission-based deadline, reserving15min of the next session quota, before any new launch.

The authorized followup restarts the existing CLS classifier from its paired
initialization on fold0. The frozen VAE encoders, twelve latent shards,
cohort/folds, class weighting, architecture, batch4 and peakLR0.0005 remain.
The selected horizon is105 epochs:5 warmup +100 cosine, ending at the original
1%-of-peak LR floor0.000005. AEM uses raw attention entropy with constant
coefficient0.05; each training presentation retains a new sorted random60%
patch subset, identically selected for both VAE branches. Validation uses
complete frozen historical Otsu bags and selects the checkpoint by unweighted
mean CE. No representation-rank penalty is added.

Fold0 has288 train/73 validation WSI and30,240 training presentations/7,560
batch4 updates per branch. Both branches run concurrently in separate physical
GPU processes with existing AMP/compilation/prefetch/telemetry. Checkpoints and
buffered tables retain the144-presentation cadence and session guard. This
combined recipe cannot isolate the individual effects of AEM and sampling;
comparison with the prior155-epoch recipe also changes the cosine horizon.
Only fold0 is submitted now. The prior v9 states and learning dynamics stay
preserved as the reference; new recipe states must never resume those weights.
Submitted as `maximshtefan/eqvae-spec0054-mil-fold0/10` on2026-10-08 around
19:16 Bogota, source `f81ca4c574d07d6a2fedddac21eab22eab924500`; initial
API status RUNNING. Shared session cutoff is2026-10-09T07:01:04-05:00.

V10 completed normally with both branches checkpointed at2160 updates/8640
presentations (30 training epochs). The epoch30 full-bag assessment was
interrupted by the session guard and remains pending; the last complete and
best assessment is epoch25. Normal train/validation CE1.25029/1.30101 with
validation macroF1.45263; SO2 CE1.16975/1.16024 with macroF1.50897.
At matched epoch25 the reference has CE/F1 1.29800/.45794 normal and
1.13200/.57203 SO2, so superiority of the new recipe is not observed yet.
All8640 ordered WSI/subset/LR/AEM records match across branches; checkpoint
cadence and successful AMP retries are preserved. Input queue waits account
for about97.2% of recorded step time; mean steps13.38/13.57s exceed the
reference first2160-update means8.00/8.01s. Full artifacts are retained in
runs/kaggle/spec0054_cls_fold0_regularized_v10 and the local review summary.
No continuation or further recipe change accompanied this inspection.

The subsequent authorized continuation was submitted on2026-10-09 around
13:01--13:04 Bogota as `maximshtefan/eqvae-spec0054-mil-fold0/11`, initial
status RUNNING. It resumes both epoch30 states from the verified private
`maximshtefan/eqvae-spec0054-cls-regularized-fold0-resume/1` dataset, including
optimizer/scaler/telemetry and replaying the pending assessment. Source and
recipe remain unchanged. With6h55 quota stated, both GPU processes share
the19:41:02 Bogota cutoff (6h40 from submission preparation,15min margin).
Checkpoints remain every144 presentations/36 updates and at session shutdown.
The105-epoch horizon spans further sessions; no horizon reset occurs.

Inspection on2026-10-12 confirms v11 completed normally with both latest
states at2880 updates/11,520 presentations (40 training epochs). Epoch40
assessment was interrupted by the deadline; last completed assessment is35.
Best validation CE selects normal epoch30 (train/validation CE.99107/1.16403,
validation macroF1.49737) and SO2 epoch35 (1.10371/1.12076, macroF1.46000).
Normal epoch35 worsens to validation CE1.23687/F1.42595; SO2 epoch30 has
CE1.19103/F1.51254. These mixed trends do not establish improved generalization.
Paired inputs and all prior telemetry prefixes are exact; checkpoint cadence
is preserved, with one successful normal AMP backoff and no committed nonfinite
loss/gradient rows. Artifacts and local review are retained under the v11 run
directories.65 epochs remain; continuation must use the new epoch40 states
and replay the pending assessment.

Authorized v12 continuation was submitted on2026-10-12 around22:22 Bogota
from the verified private resume dataset version2 (v11 epoch40 states).
The source f81ca4c574d07d6a2fedddac21eab22eab924500, recipe and reader remain
unchanged; both VAE branches use separate GPUs and preserve cumulative state.
The12-hour budget has a shared cutoff2026-10-13T10:06:50-05:00,11h45 from
submission preparation with15min margin. Checkpoint cadence remains144 WSI
presentations/36 updates; epoch40 assessment is replayed before new updates.

Inspection on2026-10-13 confirms v12 paused normally during training at the
deadline, with checkpoints at normal5228 updates/20,912 presentations
(epoch72/cursor176) and SO25182/20,728 (epoch71/cursor280). Epoch70 assessment
is complete; no pending assessment. Both best by full-bag validation CE are
epoch55: normal train/validation CE.71793/.96126, validation macroF1.60816;
SO2 .67271/.85010, F1.61773. These descriptive fold0 metrics improve over
the prior recipe's best45 CE/F1 1.01946/.57860 and .97660/.57665. At70 normal
CE.71368/1.26642/F1.49796 and SO2 .67487/1.02873/F1.56413 show renewed
validation deterioration after55. The combined recipe has not eliminated
overfitting; its attribution and single-fold limits remain. All prior telemetry
is exact, the common20,728 training presentations remain paired, and the
deadline/checkpoint cadence work. Three AMP backoffs recovered with no committed
nonfinite loss/gradient rows. Mean steps14.09/14.54s returned near v10 levels
without code changes.32.389/33.028 epochs remain to105; saved v12 states must
be used for any subsequent continuation, preserving their within-epoch cursors.

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
The training horizon and full-run artifact budget remain to be frozen before
the classifier campaign begins.

The bounded A0 measurement completed in Kaggle v4 in `1708.38` seconds. Stable
instrumented T0 steps cost `0.139--0.221` seconds, or `10.89--11.65x` the
unobserved step, but added only 12.13 MB of peak allocated device memory. T1
cost `0.54--2.70` seconds for each sampled median/P99/maximum WSI and warmed
T2-lite cost about `0.08--0.24` seconds per sentinel WSI. These absolute costs
support complete T0 on every committed update in the first anchor and
T1/T2-lite only at precommitted boundaries. Cold compilation timings are
retained as runtime observations but are not used to compare the paths.

## Historical local-global Implementation Boundary

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

## Historical local-global first-fold execution

The local first-fold contract is `docs/data/spec0054_fold0_run.json`. It pins
the twelve audited Kaggle producer versions and result hashes. All twelve
inputs have the same FP16 record format and use one reader and training path.
The classifier reads every row of each complete historical
Otsu-selected WSI bag directly from the shard binary. The 330,000-patch A0
physical source is not the full atlas and its `patch_count` column is not a
complete-bag size. The complete atlas has 3,056,833 rows across 361 WSI.

The fold-0 paired anchor has 288 training and 73 validation WSI, seed 1701,
150 epochs and exactly 43,200 training-WSI exposures for each VAE. Effective
batch one is the first run (43,200 committed updates absent AMP backoffs).
The same ordered exposure horizon permits later batch 4 and 8 arms (10,800
and 5,400 updates) without changing model, loss, peak LR or WSI order. The
historical five-epoch 10%-to-peak warmup and 1%-minimum cosine are evaluated
by cumulative WSI exposure; fold-training class weights are recomputed from
its 288 WSI. The model and AdamW parameter groups remain those of Spec 0026/0036.
The classifier uses CUDA FP16 autocast for the train, validation, T1 and
T2-lite forwards, with FP32 classifier head, logits and CE, FP32 trainable
weights and the ordinary GradScaler. AMP backoffs retry
the same effective batch without advancing exposure or order.
No outer-fold metric changes the training path.

T0 includes per-WSI prediction/loss and six-terminal forward summaries plus
actual unscaled-gradient/AdamW-update summaries at every committed update.
T0 also stores the complete bag size and AMP scale before/after the committed
attempt. At T1, the runner snapshots the model and AdamW state, obtains the
sampled unscaled gradient on the disposable copy, and stores both named-layer
activation/activation-gradient summaries and per-parameter weights, gradients,
Adam moments, effective steps, radial components and previous actual updates.
The fixed-boundary prediction rows include cumulative per-WSI learning and
forgetting state, the complete bag's graph degree/occupancy and all 192 FP32
bag-embedding coordinates. With the saved five probabilities and true class,
these coordinates reconstruct the exact unweighted classifier-head gradient
`(p - one_hot(y)) outer embedding` for every WSI and permit embedding-drift,
centroid and head-gradient-similarity analyses. Step wall time and peak device
allocation are recorded with T0. A frozen ten-WSI, fold-train-only gradient
panel samples the lower and upper complete-bag-size quartiles within each of
the five classes at exposure 0 and epochs 1/2/5/20/50/100/150. It measures
unscaled AMP per-WSI gradient norms, pairwise cosine/conflict, `G2`, mean-
gradient energy, gradient variance and a noise-scale proxy on a disposable
model copy. The exact IDs and cadence are in the run contract; these
measurements cannot change weights, optimizer state or WSI order. Fixed full train/validation inference occurs
at exposure 0, halfway through
epoch 1, at epochs 1/2/3/5 and every five epochs from 10 through 150. T1
uses fold-training WSI 28121 at exposure 0, epochs 1/2/5 and every five
epochs thereafter. T2-lite uses WSI 28121 and fold-0 holdout WSI 27739 at
exposure 0, epochs 1/2 and every ten epochs thereafter. These probes operate
on disposable model copies and cannot alter the optimizer or training RNG.
The runner atomically replaces `latest.pt` and its cumulative telemetry
tables every 144 committed WSI exposures (half of fold 0), at fixed evaluation
boundaries and before a 12-hour Kaggle session reaches its 15-minute save
margin. Its state contains the exact order/cursor, scaler, optimizer, exposure
schedule and RNG. A continuation mounts the previous run output and restores
`latest.pt`; no epoch or WSI is skipped, and an AMP backoff is never treated as
a committed update. Sparse immutable boundary checkpoints are saved at the initial and
first-half-epoch points and at precommitted epochs 1/2/3/5/10/20/30/50/75/100/125/150,
so later diagnostics can replay the same model state without changing the
training path. The first complete Kaggle training run was launched as fold-0 kernel v2. This
anchor implements the observational T0/T1/T2-lite subset and a sparse
train-only gradient-conflict/noise panel. Full-gradient sketches of every WSI,
EMA and the counterfactual step-size, SAM, Muon, orthogonal-gradient,
StableMax, spatial-stress and prototype diagnostics remain separate planned
work. The anchor can motivate a later targeted test but cannot establish the
benefit of any unrun method. This fold is
only the first outer-validation partition: across all five folds each of the
361 WSI is held out exactly once, with 288 or 289 training WSI in each fold.
After comparing and freezing the recipe, the final refit uses all 361 WSI for
training, with duration fixed independently of the external 152 WSI.

## Active gated-ABMIL restart

The observed training-only local-global scale/rank/gradient trajectories motivate
restarting the paired fold with direct gated ABMIL. The historical classifiers
and observations remain separate. This is an architectural pilot with a fixed
training recipe; its advantage or stability has not yet been measured.

`src/eqvae/models/gated_abmil.py` keeps the original patch CNN: three
convolutions (16→64, 64→128, 128→192; kernels 5/3/3 and stride 2), each followed
by GroupNorm and GELU, then adaptive spatial average pooling to a patch vector
`h_i` in R^192. Parallel linear maps produce `tanh(V h_i + b_V)` and
`sigmoid(U h_i + b_U)`. Their elementwise product is mapped to one scalar
score `s_i`. With `a_i = softmax_i(s)`, the bag embedding is
`z = sum_i a_i h_i`; a single linear 192→5 head produces logits. The hidden
attention width and optimizer parameters are in the existing first-fold JSON
contract. There are seven convolution/linear layers and three GroupNorm layers,
with 371,782 learned scalar parameters at the frozen width. Local graph blocks,
registers, CLS, final LayerNorm and positional encoding are absent. Coordinates
serve only measurement/atlas identification. No additional regularizer or
alternative optimizer is activated.

CUDA FP16 autocast covers the CNN and the two hidden projections. Attention
score computation, softmax, weighted reduction, head/logits and cross-entropy
use FP32, with FP32 parameters and GradScaler. The same twelve audited FP16
shards supply complete historical Otsu-selected WSI bags through one reader;
no latent or coordinate population changes. Separate concurrent GPU processes
train the normal and continuous-SO(2) branches with identical initialization,
WSI order, folds, loss and exposure schedule. Fold 0 remains 288 train/73 holdout
with effective batch 1 first; batch 4/8 accumulates consecutive full-WSI losses
without changing the 43,200-exposure horizon.

T0 retains per-WSI prediction/loss/margin/entropy, timing/memory/AMP and compact
forward summaries for patch vectors, tanh, sigmoid, gated features, attention
scores/probabilities, bag embedding and logits. Actual unscaled gradients and
AdamW deltas are grouped into patch encoder, gated attention, head and total;
energy shares, update/weight ratios, temporal cosines/sign flips and
counterfactual clip coefficients are unchanged. T1 uses the actual named CNN,
normalization, activation, projection, gate, softmax, multiplication, pooling
and head layers, with all existing parameter/moment/update measurements.
T2-lite replaces the graph/register-specific quantities with normalized
attention entropy, effective patch count, maximum/top-k mass, gate/tanh
saturation, representation spectra and sparse ranked patches/coordinates.
The fixed prediction boundaries, learning/forgetting, full bag embeddings,
head-gradient reconstruction and train-only gradient conflict/noise panel retain
the historical cadence. The observational anchor does not run SAM, Muon,
StableMax, Hessian/Lanczos, EMA or other heavy counterfactual families.

The resumable runner buffers scalar tables in memory, keeps shard descriptors
open, compiles forward plus loss and T0 reductions, and marks only latent axis 0
as potentially dynamic. Backward kernels are generated through Inductor;
GradScaler, optimizer control and detailed telemetry remain outside that
compiled closure. Flushed logs identify reads, transfers, forward/backward,
optimizer attempts, epochs, evaluations and checkpoints. Training saves every
144 exposures, at epoch/boundary points and at the session guard. Saving before
and after a long assessment preserves all committed training if the session
ends there. The scheduler's assessment marker lets continuation rerun the
unfinished boundary from its pre-assessment checkpoint, discarding incomplete
boundary rows and restoring the corresponding learning/forgetting state.

The new `abmil.py` thin entry is configured for a two-full-WSI smoke per GPU,
then a fresh fold. Its source commit must be published and pinned before
submission. Offline build/validation and focused local mathematics/resume tests
are complete; CPU Inductor ran forward/loss/backward for two bag sizes. CUDA
AMP, T4 memory, compile behavior and throughput await the smoke. No new Kaggle
run has been launched.

## Scientific Boundary

- The statistical unit is one WSI. Patches are not independent observations.
- The normal and continuous-`SO(2)` VAE checkpoints remain frozen. Only the
  selected MIL classifier is trained.
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
5. Record the raw numerical differences between telemetry-disabled and enabled
   paths in a focused comparison, without an arbitrary acceptance gate.

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

### C -- selected architectural pilot and later curated screen

The selected next classifier is gated ABMIL (Ilse et al., 2018) on the existing
trainable patch-encoder tokens. Its gated scores are softmax-normalized over
every patch of a complete WSI. This simple variant omits the two local graph
blocks, global registers, global patch summary and CLS block. It must train
from initialization for both frozen VAE branches; existing Spec 0026
checkpoints remain observational records. Reuse the audited FP16 shards,
five-fold WSI assignments and paired WSI-exposure budget. Mean pooling remains
the inexpensive control. A later local-graph-plus-gated-pooling ablation can
test whether spatial context justifies its cost. No new run has been launched.

The direct gated-ABMIL pilot uses the frozen first-fold batch-1 LR/schedule and
WSI-exposure horizon. The training-only local-global diagnostics justify this
restart before completing its batch sweep. Broader architecture comparison
still requires a defensible frozen recipe; optimizer and architecture effects
must be reported separately.

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

A brief train-only throughput probe now compares the existing CNN with two
spatial patch encoders: lossless 2x2 packing, and 4x4 packing followed by a
learned projection. Both proposed encoders use one full self-attention block
and one CLS-query-only block before gated WSI aggregation. The existing thin
Kaggle mechanism loads the pinned probe module; separate normal/SO(2) workers
own separate GPUs and use identical sampled patch indices. Common CE, AMP,
optimizer and T0 measurements isolate encoder execution cost. Cold compilation
and a profiled warmup are excluded from steady-state timing. This short resource
measurement does not resume a fold, select from outer holdout or assess
classification quality; the numerical setup is in
`experiments/spec0054_patch_speed_probe.py`.

The next paired classifier uses a smaller spatial encoder: a nonoverlapping
4x4 stride-4 convolution maps the frozen 16x32x32 means to 64 spatial tokens
of width 64. Learned positions are added once before a shared residual pre-LN
MLP (64->128->64, GELU). A learned CLS then queries itself and the 64 tokens
through native SDPA with two width-32 heads, followed by a residual CLS MLP
and final LN. There is no full token self-attention. Gated ABMIL with hidden
width 128 aggregates the resulting patch vectors into a five-class head.
Shared positions/CLS, LayerNorm and residual paths retain FP32; convolution,
linear projections and spatial SDPA use FP16 autocast. Only WSI patch count
is dynamic in the compiled forward/objective/T0 graph.

Its recipe uses train-only 80% patch retention, sorted seeded subsets paired
by WSI and epoch across VAE branches; diagnostics/inference use complete bags.
The objective is inverse-frequency class-weighted CE minus .01 times raw
ABMIL entropy, with the entropy coefficient cosine-decaying over the paired
50-epoch WSI-exposure horizon. LR peaks at 1e-4 with the existing five-epoch warmup and
cosine schedule. ABMIL score and class-output weights/biases start at zero;
encoder blocks remain nonzero initialized. Effective batch 4 accumulates four
WSI losses before one AdamW step; 1/4/8 remain supported. AMP backoff attempts
retain the same WSI subset and advance neither committed updates nor schedule.
Existing T0/T1/T2-lite are retained, with two spatial-head distributions sampled
on 16 fixed sentinel patches at T2. The authorized short smoke reuses the real
runner for eight committed updates, restores after four, and saves/reloads
T0/T1/T2 and checkpoint state. It does not evaluate an outer holdout or start
the full fold. Numerical configuration remains in the existing fold contract.

The spatial-CLS/gated-ABMIL paired anchor completes 50 epochs: 14,400 training
WSI exposures per VAE for fold 0, or 3,600 committed updates at effective
batch 4. Both LR and AEM finish their decay within this horizon. There is no
adaptive LR reduction or early stopping. `final.pt` at the shared horizon is
the primary paired/OOF reference. An additional exploratory `best.pt` retains
the earliest completed assessment with minimum mean unweighted validation CE,
without AEM. Assessment boundaries and the selection rule are shared across
branches; the selected boundary may differ. Its outer-holdout-selected score
is reported as exploratory. Resume copies `best.pt` and reconstructs prior
validation losses from the existing prediction table. Checkpointing remains
every 144 WSI exposures, and later snapshot/probe epochs outside the new
horizon are removed from the execution contract.

Training prefetch keeps at most two future inputs per VAE process. Two reader
threads read only the sorted retained FP16 records, grouping adjacent indices,
prepare pinned channels-last memory and complete H2D on a persistent CUDA
stream per reader while the main stream trains. Branch files open on the main
thread before concurrent reads and remain open for the session. Results are
consumed in submission order. The speculative queue spans updates and
epochs; assessments drain it and AMP backoffs rebuild it from the uncommitted
cursor. It changes neither paired subsets/order nor committed checkpoint state.
Complete-bag diagnostics remain synchronous. Requested bytes, read-region count,
read/preparation time, H2D completion time and consumer wait are logged; reduced
byte requests are not interpreted as a measured speedup.

The subsequent bounded reader-thread probe mounts only shard 1 and uses four
adjacent median-sized fold-training bags within that shard, with identical
seeded 80% subsets in both VAE branches. A derived cohort CSV retains all
original shard-1 WSI identities, labels and folds for the existing reader.
Following notebook-output mounting failures, the probe uses the private copy
`maximusshtefan/eqvae-spec0054-frozen-latent-shard-1/1`. Its result and index
retain the frozen hashes, and both binary sizes match the producer. The copy
changes the Kaggle input type without regenerating embeddings.
All branch descriptors open before concurrency. After one sampled-read warmup
per bag, the two embedding-store processes synchronously
compare one/two/two/one reader workers, four bags per trial. The probe records
file-read wall/thread-CPU time, complete four-bag elapsed time, process CPU time
and visible CPU layout. It performs no tensor preparation, VAE inference,
classifier steps or H2D. Warm-cache observations do not establish full-atlas cold-storage speed.
The v16 probe completed on two physical CPU cores/four logical threads. Mean
four-bag warm-cache elapsed time was 0.23444/0.21318 s normal and
0.23588/0.22492 s SO2 for one/two readers per process. The paired eight-bag
span fell 7.56%, while summed process CPU time rose 60.91%. There are two
rounds per reader count. Initial warmup reads took 1.19--3.06 s per bag;
concurrent cold reads were not compared. The training path retains one reader
per branch and two future WSI; this bounded result establishes no complete
training speedup.

The authorized first-read follow-up uses eight other median training bags in
shard 1, each read exactly once per store with the same paired, sorted 80%
subset. Four disjoint low/high size pairs run with one/two/two/one readers;
requested counts differ slightly, so compare bytes per elapsed second as well
as wall time. Both pools and binary descriptors persist for the entire probe.
Adjacent retained patches are already grouped into one `preadv` region; no
extra discarded patches are read. There is no warmup or tensor preparation.
This measures first reads by the script, not guaranteed empty Kaggle caches.
The training reader count remains unchanged pending this result.
V17 completed with both workers exiting zero and all paired subsets/order
matching. Mean two-bag elapsed time for one/two readers is 3.14455/2.58905 s
normal and 3.04201/2.60670 s SO2. Two-reader panels request 5.79% fewer bytes;
normalized throughput improves 14.42%/9.94% per store. Across both stores,
throughput rises 225.41 to 244.00 MB/s (8.25%), with 16.38% more summed
process CPU time. This supports a modest first-read concurrency benefit on
the measured host. Only two disjoint trials per count were observed; actual
training gain and controlled cold-cache behavior remain unmeasured.
After v17, two readers per branch were adopted in the local training runner,
retaining the two-future-WSI bound. Focused CPU verification confirms concurrent
reads, ordered sampled tensors, queue drain and exact interrupted/resumed
training agreement for effective batches 1, 4 and 8. The new CUDA path has not
yet been measured in a training run; v17 measured file reads only.

### CNN-128 comparison

The next encoder comparison uses the historical three-convolution patch CNN
with widths 64/128/128, spatial mean pooling and gated ABMIL of hidden width
128. It starts from fresh classifier weights without positional encoding.
The paired batch-4, 80% sampling, LR/AEM schedules and WSI-exposure horizon
are retained. The fold contract increases AdamW weight decay to 0.05 for both
branches; it applies to matrix/convolution weights, excluding biases and
normalization vectors. The completed spatial-CLS anchor used 0.005, so the
comparison also changes decay and cannot isolate an encoder-only effect.
The CNN-128 implementation has 281,222 parameters. T0 retains the eight
ABMIL terminals; T1 captures every convolution, normalization, activation,
pooling and attention/head layer, parameter gradients and optimizer state.
T2 retains WSI-attention participation, gate saturation, patch/attention
representation spectra and top-patch coordinates; spatial CLS-head diagnostics
are removed with that encoder. The classifier begins with uniform WSI attention
and zero class-output weights/biases. Forward/objective/T0 and generated
backward remain compiled with dynamic patch count; optimizer instrumentation
remains outside the compiled graph.
The bounded smoke mounts only the unchanged private shard-1 dataset, derives
its original 36-WSI cohort (26 fold-training bags) and uses train WSI 1080 as
the complete-bag diagnostic sentinel. It runs eight committed updates per VAE,
saves/restores after four and writes T0/T1/T2 plus checkpoint and smoke summary.
At batch 4 it sees 30 WSI, including the two-WSI epoch-tail update; both branches
have the same plan. Smoke mode skips population assessments, including at
epoch boundaries, and does not select a classifier from validation. Its subset
class weights and exposure schedule are operational diagnostics only. The full
fold retains all twelve shards, original cohort, sentinels and fixed horizon.
The v18 CNN128 smoke completed: both workers exit zero, eight committed
updates/30 exposures each and restoration from update4/exposure16. All paired
WSI/sample hashes match. Each branch persisted 30 T0, 7,446 T1 and 426 T2
rows, including every CNN convolution/pooling capture. There were no AMP
backoffs or reviewed T1 nonfinite fractions; training peak allocated memory
was 4.441 GiB per GPU. Execution took 251.460 logged seconds including cold
compilation. These bounded observations verify execution and persistence,
without estimating full-fold learning or throughput.
The authorized fresh full-fold v19 pins the same smoke-verified CNN128 source,
with shard1 mounted as the verified private dataset and shards2--12 as their
original notebook outputs. It uses the full original cohort and paired
14,400-WSI horizon, batch4, decay0.05 and the two-reader prefetch. No CLS or
smoke checkpoint is used to initialize this comparison. Full input mounting
and execution remain pending confirmation after initial RUNNING acceptance.
V19 subsequently mounted all twelve inputs and reached epoch40, 11,520 WSI
exposures/2,880 updates per VAE before the session guard paused during its
assessment. Both processes exited zero and saved loadable pre-assessment
checkpoints; the last complete assessment is epoch35. Paired samples/order
match, AMP backoffs are zero and reviewed T1 nonfinite fractions are zero.
Epoch35 train/validation CE is 1.47034/1.46062 normal and 1.39274/1.38762
SO2, with validation macroF1 .22986/.33089. Matched spatial-CLS epoch35
validation CE was 1.33246/1.24162. This CNN/decay recipe is numerically finite
but has weaker observed fit; encoder and regularization changes are confounded.
The fixed horizon still requires ten epochs/720 updates per branch and a
repeated epoch40 assessment before continuing training.
The authorized v20 continuation mounts the verified private CNN128 resume
dataset version1, restores both epoch40 states and retains the exact source,
recipe and paired horizon. It repeats the pending epoch40 assessment before
the remaining720 updates per branch; initial RUNNING acceptance is recorded.
At fixed train sentinel WSI28121, epoch30 centered patch effective rank is
1.5017/1.2566 with top-four energy96.999%/98.337% for normal/SO2. The matched
CLS sentinel ranks were7.35/5.84. This within-WSI concentration is descriptive
and does not establish an encoder/decay cause of the observed weaker fit.

### Paired spatial-CLS LR range search

CNN128 fold0 completed in v20 at50 epochs/14,400 exposures/3,600 updates per
branch. Final validation CE is1.46037/1.38826 and macroF1.20384/.30817
normal/SO2, versus spatial-CLS final CE1.35750/1.24873 and macroF1.39103/
.46538. Architecture and decay differ, so this comparison does not isolate
the encoder effect.

The next authorized bounded measurement restores the executed88,006-parameter
spatial-CLS encoder and its spatial-head diagnostics. It reuses the existing
runner with `--lr-search`, all twelve frozen FP16 shards and only the288 WSI
of fold0 training. Normal/SO2 run independently and concurrently on GPU0/GPU1
with the same initial model state, WSI order, 80% sorted patch subsets and
effective batch4. AMP, compiled objective/backward, dynamic patch count and
two-reader/two-future-WSI prefetch remain in place.

AdamW retains the most recently agreed decay0.05 (historical CLS used0.005),
betas(.9,.999), epsilon1e-8 and no decay on scalar/vector parameters. AEM
remains fixed at0.01 during the measurement. The existing numerical contract
defines72 preparation updates at LR1e-5, followed by200 updates increasing
geometrically from1e-5 to an exploratory ceiling0.1, inclusive. Progress is
indexed by committed update, so AMP retries retain the same WSI batch and LR.
The maximum budget is272 updates/1,088 training-WSI exposures per branch.

T0 retains CE without/with class weights, positive lambda*entropy magnitude,
objective=weighted CE minus that magnitude, LR, search phase, sample hashes,
unscaled gradient and actual AdamW update measurements. Step logs print the
three loss components. Buffered tables/checkpoints are saved every144 WSI
exposures; initial and pre-sweep model/optimizer snapshots are retained. T1
and T2 run on the fixed train sentinel at update0 and72. The outer holdout,
full-fold assessments and ten-WSI gradient panel are not evaluated in this
measurement. Actual nonfinite loss ends the branch before backward or optimizer
step and records the attempted LR/WSI/losses while retaining committed state;
finite losses have no artificial tolerance gate. An unstable finite portion
remains a measured part of the curve, not a runtime failure.

This trajectory estimates an initial useful LR interval, not a generalization
optimum. Both branches' curves will select one common candidate below observed
deterioration. A separate short fixed-LR confirmation must start from the
original initialization with fresh AdamW state. The main paired fold likewise
starts fresh after confirmation; neither starts from unstable sweep weights.
No automatic LR choice or full fold is launched by this measurement.
The authorized measurement was submitted as Kaggle v21 on2026-10-06,
pinning public sourced593e713a5e5592e08a4be5af882d2a8f8c3eafc. Its first
status is RUNNING; curves and fixed-LR confirmation remain pending.

The next main paired run has a precommitted155-epoch total budget: five
epochs of warmup, then150 epochs of cosine LR decay to1% of the shared peak.
For fold0 train288/effective batch4 this is44,640 WSI exposures and11,160
committed updates, including360 warmup and10,800 cosine updates per branch.
The same normalized exposure budget applies to both VAE branches. Existing
AEM annealing spans the155 total epochs from initial coefficient0.01. The
shared peak LR awaits range-search review and fresh-state confirmation; this
horizon change does not alter the already submitted bounded LR measurement.

V21 completed the bounded search for both branches at272 updates/1,088
exposures and LR0.1 with no AMP backoffs or nonfinite loss. WSI order,
targets, sample hashes and LR match exactly across branches, and retained
initial/preparation/final checkpoints load at updates0/72/272. The brief
cold-start sweep identifies no clear learning optimum: class-weight-normalized
rolling CE stays close to the uniform-prediction baseline ln(5) at low LR,
then deteriorates at high LR. For updates123-152 (LR1.01e-4--3.87e-4), the
sum of weighted CE divided by sum of example weights is1.60894/1.60836
normal/SO2; for updates203-222 (LR.00410--.00988), it is1.69058/1.67767.
Normalization removes the class-composition scale change for uniform
predictions but does not make this trajectory an independent LR comparison.
Attention remains almost uniform during the short sweep, so it establishes
no need for stronger AEM and does not characterize the learned late-training
encoder. A common5e-4 fixed-LR confirmation from original initialization with
fresh AdamW is a provisional proposal below observed deterioration. The
shared peak remains unselected pending that confirmation. Results:
`runs/local/spec0054_cls_lr_search_v21_review/summary.json` and
`lr_curves.png`; no confirmation or main fold was submitted by this review.
Both branches' minimum20-update rolling window is updates160-179, spanning
LR0.0005607--0.001351 and80 WSI presentations. Its weight-normalized CE is
1.58258/1.59779 normal/SO2, a small local improvement over ln(5). The plotted
x coordinate0.001351 is the last LR of that window, not an independently
evaluated optimum. Broad-bin averages can obscure this local valley; both
summaries remain descriptive and do not replace fixed-LR confirmation.
V21 throughput was slower than the prior CLS run despite identical first
1,088 ordered WSI/subset rows. Excluding its first compilation update, mean
step time21.67/21.63s contains21.03/20.97s logged prefetch-input wait;
remaining timed work.641/.660s includes CPU/GPU/logging/optimizer telemetry,
not isolated GPU compute. H2D averages.021/.019s per WSI. Prior CLS first272
updates excluding first10 average8.87/9.20s. The shard reader and model code
are unchanged, while prefetch reader count increased from one to two per
branch. This cross-session comparison locates the extra wait but cannot
attribute it to storage/cache variability or reader contention. A same-session
reader-count comparison is needed before attributing the slowdown to that
change. Evidence:`runs/local/spec0054_cls_lr_search_v21_review/timing_comparison.json`.
The local audit verifies same-descriptor concurrent full/sampled reads preserve
exact row order and the descriptor cursor. The active pipeline preopens
descriptors, owns separate per-read buffers and per-thread CUDA copy streams,
consumes the FIFO on the main thread and joins readers before closing files.
No race was found in this path. New local transfer-start fields separate
`read_seconds`/`read_cpu_seconds` from `host_prepare_seconds`/
`host_prepare_cpu_seconds`, retaining `read_prepare_seconds` and existing
H2D/wait metrics. The v22 probe below measures these stages on Kaggle
without changing the data path.
The authorized brief follow-up reuses the existing single-shard read-probe
mode from sourceca2d3cc60ea3080302ff63a21f71ac3f1ae10feb. Eight distinct
fold-train bags per VAE store form four balanced first-read pairs with
reader counts1/2/2/1. Four repeated-pair trials then compare both counts on
the same two previously read panels. Both store processes run concurrently,
one per physical GPU, with paired sorted80% subsets, preopened descriptors,
persistent pools and per-reader copy streams. It separately measures read,
pinned channels-last host preparation and completed H2D; no model, training,
backward or compilation occurs. Total16 requests per store,32 combined,
mounting only shard1. First-read observations do not prove empty OS/storage
caches, and repeated-read results are not whole-cohort cold throughput.
Submitted as Kaggle `maximusshtefan/eqvae-spec0054-mil-fold0/22` on
2026-10-07 around15:11 Bogota; subsequent inspection confirms COMPLETE,
both workers exit0 and eight trials per branch. Paired WSI/subset hashes match.
For identical repeated panels, total paired wall (two WSI per branch,
read/preparation/H2D) is .308/.305s with one reader and .267/.240s with two:
13.3%/21.1% reductions,17.2% combined,20.8% more bytes/s. Process CPU cost
increases42.4% for those same panels. First-read different panels give
203.8/297.1MB/s with one/two readers, a descriptive46% throughput difference
confounded by panel and initial allocation costs. First-read raw reads span
.935--2.193s per WSI versus .057--.255s pinned host preparation and
.010--.026s H2D. First-read wall greatly exceeds read thread CPU, while
repeated read/preparation wall is close to thread CPU. V21's large input
delay is not reproduced; its exact cause remains unresolved. Retain two
readers per branch; this probe shows no reason to revert them, but measures
neither whole-cohort cache behavior nor training throughput. Aggregate:
`runs/local/spec0054_reader_timing_v22_review/summary.json`.

### Fresh spatial-CLS LR confirmation

The authorized follow-up compares shared peaks0.0005 and0.001 for eight
fold0 training epochs each. Each independent VAE process executes both
candidates sequentially on its assigned physical GPU; each candidate resets
the same seed/model initialization and creates fresh AdamW/scaler state.
The existing runner's `--lr-confirmation` reuses all twelve frozen shards,
batch4, paired80% sorted subsets, two readers, AMP, compiled objective/backward
and dynamic patch dimension. The contract retains155 total epochs for AEM's
cosine schedule; confirmation stops after8 training epochs and holds LR at
the peak after the5-epoch exposure warmup. There is no LR decay during the
three fixed-peak epochs. Per candidate/branch budget:2,304 training-WSI
presentations and576 committed updates (360 warmup,216 fixed-peak).

Existing T0 records separate class-weighted/unweighted CE, AEM magnitude and
total objective, gradients, AdamW updates, AMP state and step time; reader
events separate read, pinned preparation, H2D and input wait timings.
Train-only T1/T2-lite run at initialization and epochs1/2/5/8; the existing
ten-train-WSI gradient panel runs at initialization and epochs1/2/5. No outer
holdout or full-fold evaluation is used. Tables and resumable latest state
flush every144 presentations/36 updates, with initial/final checkpoints in
separate `lr_0.0005`/`lr_0.001` output directories. Pending diagnostic boundaries
replay on resume; a completed candidate performs no additional update.
The last three epochs' matched training CE and gradient/update behavior
compare early fit and stability, not generalization or a proven LR optimum.
No automatic peak selection or155-epoch main fold is part of this run.
Submitted on2026-10-07 around16:05 Bogota as
`maximusshtefan/eqvae-spec0054-mil-fold0/23`, source
`7b56af79766fe6f4f9fde87c1d195ce973bb687c`; first status RUNNING.
Initial status establishes acceptance, not mounting/training or comparative
results. Next useful inspection after16:50 Bogota.
Subsequent inspection confirms v23 COMPLETE with both worker exits0 and all
four candidates at epoch8/576 updates/2,304 exposures. Initial weights and
ordered WSI/targets/80% subset hashes match across all four; latest/final
weights match per candidate. Each saves2,304 T0,28,846 T1 and1,165 T2 rows
and diagnostics at epochs0/1/2/5/8. No AMP backoff, recorded nonfinite loss or
gradient, or recorded T1 nonfinite fraction occurs. Training allocation peaks
at3.958--3.996GiB per T4; both physical GPU mappings are verified.

Epochs6--8 mean class-weighted CE is1.49301/1.49821 normal/SO2 with5e-4
versus1.54778/1.56189 with1e-3. Unweighted CE is1.52996/1.50064 versus
1.59516/1.58296. Mean AEM magnitude is.08551/.08480 versus.08555/.08504,
and loss-component accounting residual is exactly0. At epoch8 weighted CE
is1.46077/1.45475 versus1.53686/1.56270. Median global actual-update/weight
ratios are.001641/.001502 versus.002933/.003143 over the last216 updates.
These matched early-fit results favor the common5e-4 candidate for the next
fresh155-epoch recipe; outer-fold generalization remains unmeasured here.
Mean update time excluding the first is5.34/5.32s versus6.37/6.36s; candidate
order/storage state confound any causal LR timing interpretation. V21's
sustained22-second input delay is not reproduced. Review evidence:
`runs/local/spec0054_cls_lr_confirmation_v23_review/summary.json` and
`classification_ce.png`. No main training run is submitted by this inspection.

### Selected spatial-CLS fold0 recipe

Following confirmation, the authorized fresh fold0 run fixes common peak
LR0.0005,155 total epochs (5 warmup +150 cosine), effective batch4,
weight decay0.05, AEM initial0.01 annealed over155 epochs, paired80% sorted
patch sampling and the existing two-reader pipeline. The full288 train/73
validation cohort and all twelve frozen shards are retained. Each branch's
complete horizon is44,640 WSI presentations/11,160 committed updates. Both
VAE branches run concurrently in separate physicalGPU0/GPU1 processes.
The existing full-fold validation, checkpoint selection, T0/T1/T2 and gradient
panel cadence resumes; tables/latest state flush every144 presentations.

The first session has a remaining stated GPU budget of6h30. A shared absolute
deadline, supplied only by the thin entry via `--session-deadline-unix`, stops
both branches at6h15 after submission, reserving15 minutes for checkpoint and
process shutdown. Counting from submission includes queue/input preparation
conservatively. The guard is checked before training updates, AMP retries and
evaluation/diagnostic bags. It saves committed progress or retains the saved
pre-assessment state for replay. The session deadline is excluded from the
scientific checkpoint identity, allowing later continuation with a new runtime
deadline and the same frozen recipe/source. It does not shorten the155-epoch
scientific horizon. The current submission authorizes fold0 only; subsequent
folds use the same recipe on distinct validation partitions, followed by a
separate final refit on all361 WSI with no validation-dependent stopping.
Submitted as `maximusshtefan/eqvae-spec0054-mil-fold0/24` on2026-10-07
around19:56--19:57 Bogota, source
`0f1868f35a7bffaea0b66189b2dbd8ee47659348`. The shared runtime cutoff is
2026-10-08T02:11:27-05:00, excluding that deadline from checkpoint identity.
This is a quota-limited first session of the155-epoch fold, not a shortened
scientific horizon or a continuation of the eight-epoch confirmation weights.
Inspection on2026-10-08 confirms v24 COMPLETE with two normal worker exits
after the planned runtime pause around02:11 Bogota. Latest checkpoints load
at1712/1707 updates and6848/6828 exposures normal/SO2, both after23 full
epochs plus224/204 WSI of epoch24. The first6828 ordered WSI/target/subset/LR
rows match. Latest-save exposure gaps are at most144, including final saves.
The155-epoch horizon remains incomplete.

The last complete train/validation full-bag assessment is epoch20: normal
CE1.29440/1.40451 and validation macroF1.31023; SO2 CE1.24604/1.23381 and
validation macroF1.46110. Normal's best validation-CE state remains epoch10
(CE1.38685, macroF1.35434); SO2's is epoch20. Normal's train/validation gap
increases after10, while SO2 has a transient epoch15 regression before its
epoch20 improvement. These are exploratory fold0 dynamics, not evidence of
five-fold performance. One SO2 AMP gradient overflow at update1440 lowers
scale65536 to32768 and the batch is retried successfully; no other backoff or
recorded nonfinite committed losses/gradients occurs. Per-GPU training peaks
are4.1124/4.1129GiB; mean noninitial update time8.256/8.264s and last100-update
means10.52/10.73s. No recipe or scientific horizon is changed by this review.
Evidence: `runs/local/spec0054_cls_fold0_v24_review/summary.json` and
`train_validation_ce.png`; full outputs/checkpoints are under
`runs/kaggle/spec0054_cls_fold0_v24/`. No continuation is submitted yet.

The subsequently authorized exact continuation ran as
`maximshtefan/eqvae-spec0054-mil-fold0/9` from the shared v24 outputs with
the same source/recipe and separate physical T4GPU0/GPU1 processes. It
completed normally at the2026-10-08T15:20:52-05:00 runtime cutoff, retaining
normal/SO2 latest states at4800/4831 updates and19,200/19,324 presentations
(66.67/67.10 equivalent epochs);155 epochs remain unfinished. The cumulative
first19,200 paired T0 order/subset/LR/update rows match, and prior T0 history
is preserved. Latest-save gaps remain at most144 presentations. Four/three
AMP backoffs retry successfully; committed losses/gradients remain finite.
Both best checkpoints by validation CE are epoch45: normal CE1.01946,
macroF1.57860; SO2 CE.97660, macroF1.57665. At the last complete assessment,
epoch65, train/validation CE is.44138/1.49621 normal and.47721/1.16734 SO2,
with validation macroF1.43680/.52680. These exploratory fold0 observations
show improved learning and a pronounced later generalization gap, without
changing the fixed recipe or five-fold protocol. Outputs and loadable latest/
best checkpoints are under `runs/kaggle/spec0054_cls_fold0_original_v9/`;
review evidence is in `runs/local/spec0054_cls_fold0_original_v9_review/`.

Late T2 measurements add a mechanistic observation to that review. On the
fixed train sentinel28121 (13,178 full-bag patches), attention effective
support drops from69.48%/79.93% at epoch20 to1.26%/1.60% at60, normal/SO2;
maximum patch mass at60 is6.65%/6.86%. On validation sentinel27739
(11,855 patches), support drops87.55%/90.30% to8.86%/14.56%. Thus late
concentration is observed on both sentinels, without establishing its
prevalence or causal role across the cohort. Effective fractions use the
prediction table's full bag size; patch_tokens.count counts scalar elements.
The AEM coefficient remains.0062533 at65, versus initial.01. The user's
proposed constant AEM.01 and70% retention are recorded as an unimplemented
alternative requiring a separate experiment. Literature context is saved
in the review summary: [AEM's final paper](https://arxiv.org/html/2406.15303v3)
uses cosine weight annealing to reduce parameter sensitivity, while
[random-sampling MIL](https://arxiv.org/abs/2403.05351) finds dataset-dependent
sampling optima. Neither source establishes an ovarian-subtyping optimum
of70% or superiority of constant AEM for this frozen-latent setup.

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
The instrumentation comparison covers the diagnostic-disabled and enabled
paths and records logits, loss, optimizer/scaler state, raw update differences
and skipped-attempt sequences. These are descriptive measurements; A0 does not
turn them into a pass/fail decision.

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

Before replacing the local-global MIL aggregator with PANTHER, perform the
broader offline T2 prototype audit on
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
