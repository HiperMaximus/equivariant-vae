# Decision 0011: Kaggle Code Delivery — Embedded Payload Now, pip-install When Public

Date: 2026-07-16

> Narrowed by decision 0012 (2026-07-17): the run kernels now set
> `enable_internet = "true"` solely to `pip install --upgrade` the torch stack at
> runtime. The embedded-code and empty-`*_sources` faces of the invariant below are
> unchanged; only the internet-off face is relaxed, and only for torch.

Decision: keep delivering our code to Kaggle as a self-contained embedded payload
(a single generated `run.py` with the whole `src/eqvae` tree base64-zipped inside
it), with `enable_internet = "false"` and empty `*_sources`, for as long as the
GitHub repo is private. When/if the repo is made public (pending the thesis
professor's approval), switch delivery to
`pip install "git+https://github.com/HiperMaximus/equivariant-vae.git@<commit-sha>"`
over the network, and retire the embed builder.

## Why the current design is deliberate, not accidental

`enable_internet = "false"`, empty `dataset_sources`/`kernel_sources`, and the
embedded payload are three faces of ONE invariant: hermeticity. With no network and
no attached sources, a run can only use what is baked into the payload, so it cannot
silently depend on something undeclared and it stays reproducible offline. Spec 0003
"Known Risks" states the rationale: "Enabling internet in Kaggle can hide undeclared
dependency and code-source assumptions." Internet-off is not leftover caution — it is
the enforcement of this invariant. (A flip to `enable_internet = "true"` was applied
and then reverted on 2026-07-16 once this rationale surfaced; see decision 0010. A
NARROWER flip — internet on for a runtime torch upgrade only, with code still
embedded and `*_sources` still empty — was later adopted for the run kernels; see
decision 0012.)

## Why pip-install is the future upgrade (and dataset_sources is not)

Now that `eqvae` is a real installable package (the `[build-system]` added
2026-07-16; see the Spec 0001 package/import policy), `pip install git+...@<sha>`
becomes available on Kaggle with internet on — pinned to a commit SHA, which is a
strong, content-addressed provenance. It is blocked today only because the repo is
PRIVATE (a private-repo install needs a GitHub token in Kaggle Secrets — a credential
on a third-party platform) and internet is off.

`dataset_sources` was evaluated as an alternative to the embed and REJECTED as a
lateral move, not an upgrade:

- Two artifacts to keep version-synced (the kernel AND a code-dataset) instead of
  one; more drift surface.
- Provenance becomes "which dataset version was attached", looser than a hash + git
  SHA baked into the one file.
- Same pre-run upload cost as building the embed — no win there.
- It also requires relaxing the empty-source-lists half of the hermeticity guard.
- Its only advantage is "more idiomatic", but the embed already ships the real
  multi-file `src/eqvae` tree, so we never suffered single-file flattening.

So: stay on embed while private; when public, go straight to pip-install; skip
`dataset_sources`.

## Kaggle torch version

"Kaggle rides near-latest torch" is a working assumption (originally the user's),
corroborated at the mechanism level — the Kaggle GPU image supports CUDA `sm_120`
(Blackwell), which only a recent torch/CUDA build targets — but NOT pinned to an exact
version, which drifts with Kaggle's environment image. The environment is not
API-pinnable: there is no `kernel-metadata.json` field for it, and API script-kernel
pushes always use Kaggle's latest environment. The reproducibility-correct fix is to
MEASURE it — record `torch.__version__` / `torch.version.cuda` in the Kaggle run
telemetry (today the `runtime` block records `device`/`cuda_available` but not the
torch version). Tracked as an open follow-up.
