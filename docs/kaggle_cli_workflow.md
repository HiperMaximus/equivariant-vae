# Kaggle CLI Workflow

Kaggle experiments are CLI-managed script kernels. Frozen checkpoints and data
are mounted from their published Kaggle datasets. New kernels use a small direct
`code_file` that loads the public repository and calls experiment code from the
exact Git commit recorded in the output.

Do not embed the repository as ZIP/base64 or add a new branch to
`build_kaggle_embedded_kernel.py`. That builder remains only for reproducing old
kernels that already have a `run_template.py`. For a direct kernel, `build`
simply compiles and validates its declared `code_file`.

## Commands

```bash
./scripts/kaggle_kernel.sh build <kernel-dir>
./scripts/kaggle_kernel.sh validate <kernel-dir>
./scripts/kaggle_kernel.sh check <kernel-dir>
./scripts/kaggle_kernel.sh push <kernel-dir>
./scripts/kaggle_kernel.sh status-launch <launch-receipt.json>
./scripts/kaggle_kernel.sh logs-launch <launch-receipt.json>
./scripts/kaggle_kernel.sh output-launch <launch-receipt.json> <new-output-dir>
```

`push` submits the existing package and records the exact Kaggle
owner/slug/version under `runs/local/kaggle_launches/`. Preserve that locator:
the active account is not necessarily the owner of every input.

For a parameter rerun, edit the existing contract, commit and push it to the
public repository, then push the same thin kernel. Do not duplicate model code,
weights, contracts, or experiment modules inside the Kaggle entrypoint.

Keep checks that affect numerical correctness: valid Python/metadata, finite
tensors, and the experiment's declared mathematical tolerances. Kaggle's input
version and the output's Git commit identify the external artifacts. Use a new
output directory when downloading a new version so earlier evidence is not
overwritten.
