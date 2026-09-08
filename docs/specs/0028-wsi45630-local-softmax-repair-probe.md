# Spec 0028: WSI45630 Local-Softmax Repair Probe

Status: executed / closed
Implementation readiness: complete; evidence consumed by Spec 0026
Owner/workstream: Spec 0026 local-attention backend selection
Last updated: 2026-09-01

## Purpose

Complete the timing comparison that Spec 0027 version 1 correctly skipped.
Preserve its independently assembled oracle, real 32,595-patch WSI45630 graph,
softmax-plus-static-null algebra, candidates, chunk sizes and timing protocol.
Change only the pre-timing gradient decision so ordinary FP16 reduction-order
differences do not reject the vectorized explicit implementation itself.

## Prior Evidence And Scope

Version 1 artifact SHA-256 is
`299dcbfbf03b2301d03cd7fedd109c7833039c5a4a307f00fd56c55d593e7753`.
It authenticated 32,595 nodes and 793,081 edges on Torch `2.13.0+cu130` / Tesla
T4. Explicit, automatic SDPA and forced-efficient SDPA failed elementwise
`allclose` only for K/V projection-weight gradients while maximum relative L2
was at most `1.047e-3`; Flash rejected the additive mask. No timing row exists.

This repair remains a generated-token kernel probe. It reads no latent payload,
does not instantiate the CNN or full MIL model, does not learn, and cannot
establish architecture quality or full-model capacity.

## Locked Inputs And Algebra

Reuse only private dataset
`maximusshtefan/eqvae-wsi45630-capacity-inputs`, version 1, with the exact
contract/pointer hashes and canonical radius-2 graph from Spec 0027. Use
generated FP16 `X[32595,192]`, six width-32 heads, one learned static null key,
one learned null bias, fixed zero null value, learned radial bias and softmax
over valid patch neighbours plus that one null slot. Preserve Spec 0027's
initialization, seed, two warmups, five measured forward/backward iterations,
and chunk sizes `2048`, `8192`, `32595`.

Candidates remain explicit gathered FP32 attention, automatic SDPA, forced
efficient SDPA and forced Flash SDPA. Unsupported forced backends are recorded,
not treated as algebra failures. A candidate may not allocate `N x N`.

## Revised Correctness Contract

Use the same deliberately nonzero radial/null parameters and deterministic
summed upstream gradient as version 1. Every oracle gradient norm must exceed
`1e-6`; all compared values must be finite.

First validate vectorized explicit attention against the independent per-query
oracle on 257 queries:

- output must pass elementwise `rtol=5e-3, atol=5e-3`;
- each input/parameter gradient must have relative L2 error at most `2e-3` and
  cosine similarity at least `0.999`.

Then validate each supported SDPA backend directly against that accepted
vectorized explicit result and independently against the oracle using the same
output and gradient criteria. Record, but do not gate gradients on, the old
elementwise `allclose`, violation count/fraction, maximum/mean absolute error,
norms, relative L2 and cosine similarity.

The `2e-3` threshold is precommitted before version 2 timing. It is approximately
twice FP16 unit roundoff and still below 0.2% whole-gradient discrepancy; version
1 calibrates that this admits the observed explicit numerical floor rather than
silently relaxing a backend-specific failure. Output remains strictly
elementwise checked. No threshold may change after launch.

Only correctness-passing candidates receive the 12 full-graph row slots. Select
the finite successful row with lowest mean forward/backward time. Record all
failures, timing samples and peak allocated/reserved VRAM.

## Outputs

- Requested metadata ID:
  `maximusshtefan/eqvae-wsi45630-local-attention-probe`; Kaggle returned the
  title-derived canonical ID
  `maximusshtefan/eqvae-wsi45630-local-attention-repair-probe`, accepted version
  2. Evidence retrieval is bound to that canonical ID at `/2`.
- Output: `spec0028_local_attention_repair_probe.json` plus exact versioned log.
- Evidence root: `runs/kaggle/wsi45630_local_attention_repair_probe_v2/`.
- Source: `kaggle/kernels/wsi45630_local_attention_probe/run.py`.

The artifact must bind runtime identity, input/graph hashes, all correctness
metrics, 12 rows and the selected row or an explicit terminal failure.

## Remote Authorization

The user's fresh 2026-09-01 authorization approved exactly one repair push plus
required status/output reads for private version 2. It authorized no training or
further repair. The executed push required:

```text
KAGGLE_PUSH_CONFIRMED=1
KAGGLE_FULL_DATASET_CONFIRMED=1
KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED=1
```

It must require the immutable version-1 push receipt/artifact hashes, reject all
CLI overrides/wait, parse Kaggle's accepted-version message, require version 2,
then write a distinct non-overridable consumed receipt. Retrieval must use that
exact version and stage JSON plus a nonempty log before one final install.

Guard token: `spec0028_local_softmax_repair_probe_authorized`.

The independently reviewed, immutable version-2 upload bytes are:

- `run.py` SHA-256:
  `8217f9538dd57009f93d0343418a8d3e18d160176d9e3ef0471147ae69c55e12`;
- `kernel-metadata.json` SHA-256:
  `69e3d9abd665a3f438809ef2832fc505431abd46f73f9e00050b6eb40615e8a6`.

The path/ID-keyed guard must enforce both hashes, create an exclusive consumed
attempt claim before the network call, and upload a verified snapshot. Ambiguous
CLI failure or an unexpected accepted version consumes the authority and permits
no retry.

## Acceptance Criteria

1. Focused CPU tests independently exercise irregular masks, radial codes,
   nonzero null parameters, normwise/cosine metrics and elementwise diagnostics.
2. The version-2 guard behavior is tested with a fake Kaggle CLI, including
   exact version 2, one-use consumption and staged exact-version retrieval.
3. Full repository quality, repository/workspace preflight, `git diff --check`
   and two independent clean-context reviews pass before push.
4. Every timed row passed the locked revised gate; unsupported rows remain
   explicit and cannot win.
5. Retrieved JSON/log hashes and accepted version are recorded in `CURRENT.md`.
6. Spec 0026 changes only from accepted evidence; a failed version 2 is not
   silently converted into a backend selection.

## Execution Evidence

Version 2 was accepted at `2026-09-02T01:09:30Z`; push-receipt SHA-256 is
`a9b2f9280f538d9d8930675188a1934566b8cf991361145655ed060b9c6686bb`.
Kaggle's accepted response warned that the title did not resolve to the sealed
metadata ID and returned the canonical private slug
`maximusshtefan/eqvae-wsi45630-local-attention-repair-probe`. Exact version `/2`
completed. Its artifact records the authenticated 32,595-node, 793,081-edge graph
and 12 rows. Explicit, automatic SDPA and forced-efficient SDPA passed; forced
Flash was unsupported. The lowest-mean row was explicit chunk 8192 at
`129.252548 ms`, 1,669,328,384 peak allocated bytes and 2,141,192,192 peak
reserved bytes.

Artifact SHA-256 is
`b1082db4f991b2124b9e828a6cf22605ed36ecda3558b0ac6c71793c4cab9a9e`;
normalized log SHA-256 is
`fdc78705a574a835e8a272b6aa4e046de511a05afd7dd29a026ea571c933de64`;
raw downloaded log SHA-256 is
`530b7dd256761d30a05fde107c24904aa47728da402beb187c16154b6506b1cf`;
retrieval-receipt SHA-256 is
`9213d006e359206fb6718de6b31c662630abf0fcf5a56af6045596219c81287f`.

This selects the explicit backend/chunk among tested softmax implementations. It
does not compare local sigmoid with softmax or validate the stacked/full MIL
model. The one-shot authority is consumed; no retry, version 3 or repair is
authorized.

## Blockers

None. Spec 0026 retains the later full-model capacity requirement.

## Related Files

- `docs/specs/0027-wsi45630-local-softmax-kernel-probe.md`
- `docs/specs/0026-local-global-sigmoid-mil.md`
- `scripts/kaggle_kernel.sh`
- `CURRENT.md`
