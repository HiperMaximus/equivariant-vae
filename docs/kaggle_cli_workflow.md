# Kaggle CLI Workflow

Kaggle experiments are CLI-managed script kernels. The repository runner and
machine-readable contract are the editable sources; generated `run.py` files
are build artifacts.

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

For a parameter rerun, edit the existing contract or runner, rebuild, validate,
and push the same kernel. Do not add extra launcher or orchestration machinery.

Keep checks that affect numerical correctness: valid Python/metadata, exact
input and checkpoint identity, finite tensors, and the experiment's declared
mathematical tolerances. Use a new output directory when downloading a new
version so earlier evidence is not overwritten.
