# Kaggle CLI Workflow

Status: draft workflow scaffold
Last updated: 2026-06-10

Kaggle is a remote execution surface, not a Git remote. This repo remains the
source of truth for experiment code, specs, configs, and paper-facing claims.

## Current State

Historical Kaggle notebooks live in:

```text
kaggle/train_runs
kaggle/dataset_generation
kaggle/generate_dataset_Classification_With_Masks
```

They are JSON notebooks kept as historical evidence and behavior-inventory input.
Do not edit them into the new baseline.

The first CLI-managed script-kernel scaffold lives in:

```text
kaggle/kernels/non_eq_vae_debug
```

It is not push-ready yet. It intentionally exits until the real spec 0001
launcher replaces the placeholder.

Spec 0001 runtime benchmarking requires two accelerator modes:

- `single_visible_t4`: run on one visible GPU with `world_size = 1`;
- `dual_t4_ddp`: run on two T4 GPUs with `world_size = 2`.

The preferred implementation is one dual-T4 benchmark kernel where
`single_visible_t4` restricts visibility to the first GPU and `dual_t4_ddp`
launches two ranks with `torchrun --standalone --nproc_per_node=2` or an
equivalent self-spawn launcher.

Verified Kaggle accelerator metadata:

- on 2026-06-11, `kaggle kernels pull maximusshtefan/non-eq-vae -m` downloaded
  metadata for the existing notebook that the Kaggle UI showed as GPU T4 x2;
- the pulled `kernel-metadata.json` has `"machine_shape": "NvidiaTeslaT4"`,
  `"enable_gpu": true`, and `"enable_tpu": false`;
- Kaggle CLI 2.2.1 accepts `--accelerator ACC` and passes that string directly
  as `request.machine_shape`, so benchmark metadata/tooling should use
  `NvidiaTeslaT4` rather than inventing a separate `T4x2` string.

Because the metadata field does not encode the count of visible T4 devices, the
benchmark must verify the actual allocation at runtime. `dual_t4_ddp` rows must
record `cuda_device_count == 2`, two T4 device names, `world_size == 2`, and
`nproc_per_node == 2`; otherwise the row fails with
`failure_kind = "wrong_accelerator"`. `single_visible_t4` rows may use the same
Kaggle machine shape but must mask visibility to one GPU and record
`visible_device_count == 1`.

The behavior inventory now lives at:

```text
docs/behavior_inventory_kaggle.md
```

On 2026-06-06, `./scripts/kaggle_kernel.sh check` passed on this laptop with
Kaggle CLI 2.2.1. Authentication is still a user-local secret and should be
rechecked before remote reads or writes.

On 2026-06-11, the read-only API preflight command:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
```

confirmed:

- OAuth access-token generation works without printing the token;
- `kernels list`, `kernels status`, and `kernels logs` work for
  `maximusshtefan/non-eq-vae`;
- `datasets files` works for
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`;
- `kaggle quota -v` fails with Kaggle's authentication-required message even
  though OAuth token generation works;
- `kaggle kernels files maximusshtefan/non-eq-vae -v` also fails with the same
  authentication-required message.

Therefore the benchmark workflow must not rely on the CLI quota endpoint as the
only gate. Before a remote benchmark push, run the API preflight, check GPU
quota/availability in the Kaggle web UI if the quota endpoint still warns, and
let the benchmark itself fail rows with `failure_kind = "wrong_accelerator"` if
Kaggle does not allocate two visible T4 devices for `dual_t4_ddp`.

The user visually confirmed the Kaggle web UI quota on 2026-06-11: phone
verification is complete, identity verification is not complete, and Kaggle GPU
quota shows `00:07 / 30 hrs` used. Identity verification is not currently a
benchmark blocker as long as the UI continues to expose GPU quota and notebook
GPU selection.

## Local Commands

Validate the local scaffold:

```bash
./scripts/kaggle_kernel.sh validate
```

After spec 0001 implementation creates repo code/configs, build the self-contained
kernel payload before any remote push:

```bash
./scripts/kaggle_kernel.sh build
```

The generated `kaggle/kernels/*/payload/` directory is ignored and must be
rebuilt from source before remote pushes. Payload metadata includes both
`pyproject.toml` and `uv.lock`; spec 0001 kernels must not resolve or install
dependencies on Kaggle unless a later spec explicitly changes that rule.

Check whether the Kaggle CLI is installed and whether local metadata is valid:

```bash
./scripts/kaggle_kernel.sh check
```

After explicit user permission for remote reads, run the read-only API
preflight before remote benchmark pushes:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
```

Push a script kernel only after explicit user permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
```

Check remote status after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
```

Download outputs into ignored local run artifacts after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
```

Pulling from Kaggle can overwrite local files and requires explicit permission:

```bash
KAGGLE_PULL_CONFIRMED=1 ./scripts/kaggle_kernel.sh pull
```

## Credentials

Kaggle credentials are local secrets. Do not store, print, or commit them.

The official Kaggle API supports local CLI authentication and the standard local
token file. Agents must ask before running network commands or remote writes.

## Dataset Sources

Attach Kaggle datasets through `kernel-metadata.json`, not by hard-coding UI
display names in the script.

Use exact dataset slugs, for example:

```json
"dataset_sources": ["owner/dataset-slug"]
```

The current scaffold uses the confirmed pre-shuffled patch dataset:

```json
"dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
```

Other confirmed historical slugs are recorded in
`docs/behavior_inventory_kaggle.md`. Do not attach
`maximusshtefan/non-eq-vae-output` to spec 0001 or any new normal VAE baseline.
A future historical-reproduction spec would need to opt into that source
explicitly.

The pre-shuffled patch dataset is the confirmed train/validation patch source.
It contains `ubc_train_shuffled.*` and `ubc_ocean_valid.*`, but no held-out test
shard. Final evaluation needs a separate sealed test dataset/source from the
UBC-OCEAN WSIs with supplemental masks. Those masks are non-exhaustive and should
not be interpreted as full-WSI negative/positive coverage.

The push wrapper refuses remote writes while `dataset_sources` is empty, while
the placeholder guard remains, while the bundled payload is missing, while the
dataset slug differs from `maximusshtefan/patches-pre-shuffled-ubc-ocean`, or
while spec 0001 and the spec index are not marked `locked / implementation-ready`.

For spec 0001 benchmark kernels, the wrapper or metadata validation must require
`machine_shape == "NvidiaTeslaT4"` and the single-visible versus dual-DDP launch
mode recorded above.

## GitHub Linking

Kaggle's web UI can show a notebook as linked from GitHub, but that is not the
workflow here. For agentic work, the repo should generate or own the script
kernel folder, and the Kaggle API should upload that folder.

If someone edits a kernel in the Kaggle UI, pull it locally, inspect the diff,
and reconcile it into the repo. Do not let UI edits become the source of truth.

## Official References

- Kaggle API README: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
- Kaggle kernel commands: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md
- Kaggle kernel metadata: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
