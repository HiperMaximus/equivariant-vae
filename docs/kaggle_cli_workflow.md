# Kaggle CLI Workflow

Kaggle experiments are CLI-managed script kernels. Frozen checkpoints and data
are mounted from their published Kaggle datasets. Each kernel uses a small
direct `code_file` that loads the public repository and records the exact Git
commit in its output. `build` only compiles and validates that file; the
repository is never embedded as a ZIP/base64 payload.

## Commands

```bash
./scripts/kaggle_kernel.sh build <kernel-dir>
./scripts/kaggle_kernel.sh validate <kernel-dir>
./scripts/kaggle_kernel.sh check <kernel-dir>
./scripts/kaggle_kernel.sh push <kernel-dir>
./scripts/kaggle_kernel.sh status owner/slug/version
./scripts/kaggle_kernel.sh logs owner/slug/version
./scripts/kaggle_kernel.sh output owner/slug/version <new-output-dir>
```

`push` uploads only `kernel-metadata.json` and its direct `code_file`. It requires
the metadata owner to match the authenticated account. Record the exact
owner/slug/version reported by Kaggle in `CURRENT.md`; input locators remain
unchanged in the metadata and may belong to other owners.

For a parameter rerun, edit the existing contract, commit and push it to the
public repository, then push the same thin kernel. Do not duplicate model code,
weights, contracts, or experiment modules inside the Kaggle entrypoint.

Keep checks that affect numerical correctness: valid Python/metadata, finite
tensors, and the experiment's declared mathematical tolerances. Kaggle's input
version and the output's Git commit identify the external artifacts. Use a new
output directory when downloading a new version so earlier evidence is not
overwritten.
