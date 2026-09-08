# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0916, PLR2004, T201, TRY003
"""Build the exact Spec 0034 full compiled fixed-25 MIL probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

from eqvae.models.local_global_mil import EXPECTED_PARAMETER_COUNT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT: Final = Path.cwd()
DEFAULT_ROOT: Final = Path(
    "kaggle/kernels/wsi45630_full_compile_probe/package",
)
SPEC_PATH: Final = Path("docs/specs/0034-full-compiled-fixed25-mil-probe.md")
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/wsi45630_full_compile_probe/run_template.py",
)
MODEL_PATH: Final = Path("src/eqvae/models/local_global_mil.py")
CANDIDATE_PATH: Final = Path("src/eqvae/models/local_attention_candidates.py")
INPUT_ROOT: Final = Path("runs/local/wsi45630_capacity")
INPUT_RECEIPT_PATH: Final = INPUT_ROOT / "input_receipt.json"
INPUT_CONTRACT_PATH: Final = INPUT_ROOT / "bundle/wsi45630_capacity_input.json"
INPUT_DATASET_REFERENCE: Final = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
INPUT_DATASET_VERSION: Final = 1
INPUT_CONTRACT_SHA256: Final = (
    "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660"
)
POINTER_SHA256: Final = (
    "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
)
KERNEL_ID: Final = "maximusshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe"
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
)
PACKAGE_FILES: Final = {"kernel-metadata.json", "run.py"}
WSI_ID: Final = 45_630
PATCH_COUNT: Final = 32_595
DIAGNOSIS_INDEX: Final = 1


def build() -> dict[str, object]:
    """Create a new immutable local launch package."""
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        message = f"Refusing to overwrite {output}"
        raise FileExistsError(message)
    _validate_input_authority(ROOT)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        message = f"Stale staging directory exists: {staging}"
        raise FileExistsError(message)
    kernel = staging / "kernel"
    try:
        kernel.mkdir(parents=True)
        _write_json(kernel / "kernel-metadata.json", _kernel_metadata())
        (kernel / "run.py").write_bytes(_render(ROOT, contract=_contract(ROOT)))
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate()


def validate() -> dict[str, object]:
    """Reject input, source, metadata, contract, or rendered-code drift."""
    _validate_input_authority(ROOT)
    kernel = ROOT / DEFAULT_ROOT / "kernel"
    entries = list(kernel.iterdir())
    if {path.name for path in entries} != PACKAGE_FILES or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise ValueError("Spec 0034 kernel package allow-list differs")
    contract = _contract(ROOT)
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata():
        raise ValueError("Spec 0034 kernel metadata differs")
    expected = _render(ROOT, contract=contract)
    run_path = kernel / "run.py"
    if run_path.read_bytes() != expected:
        raise ValueError("Rendered Spec 0034 launcher differs")
    if run_path.stat().st_size >= 1_000_000:
        raise ValueError("Spec 0034 launcher exceeds Kaggle's 1 MB limit")
    return contract


def _validate_input_authority(root: Path) -> None:
    receipt = _read_object(root / INPUT_RECEIPT_PATH)
    contract_path = root / INPUT_CONTRACT_PATH
    contract = _read_object(contract_path)
    if (
        receipt.get("status") != "verified"
        or receipt.get("visibility") != "private"
        or receipt.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or receipt.get("dataset_version") != INPUT_DATASET_VERSION
        or receipt.get("input_contract_sha256") != INPUT_CONTRACT_SHA256
        or _sha256(contract_path) != INPUT_CONTRACT_SHA256
    ):
        raise ValueError("WSI45630 private input authority differs")
    files = receipt.get("files")
    if not isinstance(files, dict) or files != contract.get("files"):
        raise ValueError("WSI45630 input receipt file provenance differs")
    pointer = cast("dict[str, object]", files).get("probe/pointers.csv")
    if not isinstance(pointer, dict):
        raise TypeError("WSI45630 pointer receipt differs")
    pointer_record = cast("dict[str, object]", pointer)
    if pointer_record.get("sha256") != POINTER_SHA256:
        raise ValueError("WSI45630 pointer receipt differs")
    if (
        contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("diagnosis_index") != DIAGNOSIS_INDEX
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise ValueError("WSI45630 input contract scope differs")


def _contract(root: Path) -> dict[str, object]:
    return {
        "schema_version": "spec0034.full_compiled_fixed25_mil.v1",
        "authorization": "spec0034_pinned_torch_retry_v7_authorized",
        "scope": "capacity_optimization_only_not_learning_or_evaluation",
        "spec_sha256": _sha256(root / SPEC_PATH),
        "model": {
            "source": MODEL_PATH.as_posix(),
            "sha256": _sha256(root / MODEL_PATH),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "candidate": {
            "source": CANDIDATE_PATH.as_posix(),
            "sha256": _sha256(root / CANDIDATE_PATH),
            "backend": "whole_bag_fixed25_inductor",
        },
        "input_dataset": {
            "reference": INPUT_DATASET_REFERENCE,
            "version": INPUT_DATASET_VERSION,
            "contract_sha256": INPUT_CONTRACT_SHA256,
            "pointer_sha256": POINTER_SHA256,
        },
        "kernel_sources": [
            {"reference": source, "version": 1} for source in KERNEL_SOURCES
        ],
        "wsi_id": WSI_ID,
        "patch_count": PATCH_COUNT,
        "diagnosis_index": DIAGNOSIS_INDEX,
        "initialization_seed": 3401,
        "runtime_dependency": {
            "torch": "2.14.0",
            "cuda_wheel": "cu130",
            "index_url": "https://download.pytorch.org/whl/cu130",
            "install_scope": "torch_only_no_domain_libraries",
        },
        "precision": {
            "input": "float16_channels_last",
            "autocast": "float16",
            "normalization": "standard_pytorch_amp_policy",
            "classifier_loss": "float32",
            "grad_scaler": {
                "api": "torch.amp.GradScaler",
                "init_scale": 32768.0,
                "growth_interval": 1_000_000,
                "overflow_policy": "skip_update_and_continue",
            },
        },
        "compile": {
            "backend": "inductor",
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": True,
            "recompile_limit": 3,
            "dynamic_axes": "N_only_permissive_cached_specialization",
            "optimizer": "grad_scaler_and_native_fused_adamw_eager",
        },
        "correctness": {
            "criterion": "compiler_training_effect_lte_max_amp_or_repeat_effect",
            "steps": 5,
            "precision_control": "same_fixed25_eager_fp32",
            "repeat_control": "same_fixed25_eager_amp",
            "decision_units": "per_parameter_optimizer_state_and_behavior_per_step",
            "amp_skip_gate": "eager_replay_compiled_histories_must_match",
        },
        "optimizer": {
            "name": "AdamW",
            "fused": True,
            "capturable": True,
            "learning_rate": 2e-4,
            "matrix_weight_decay": 1e-4,
        },
        "execution": {
            "branch_devices": {"normal_vae": 0, "so2_vae": 1},
            "compile_warmups": 2,
            "dynamic_reuse_bag_size": 64,
            "measured_steps": 5,
            "measured_committed_steps_required": 5,
            "branch_failure_policy": "record_and_continue_other_branch",
            "benchmark_workload": "single_wsi_repeated_not_epoch_requeue",
            "checkpointing": False,
            "cudagraphs": False,
            "fallback": None,
        },
        "output": "spec0034_full_compiled_fixed25_mil_probe.json",
    }


def _kernel_metadata() -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae WSI45630 full compiled fixed25 MIL probe",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [INPUT_DATASET_REFERENCE],
        "competition_sources": [],
        "kernel_sources": list(KERNEL_SOURCES),
        "model_sources": [],
    }


def _render(root: Path, *, contract: Mapping[str, object]) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    model = _embedded_model_source(root / MODEL_PATH)
    candidate = _embedded_candidate_source(root / CANDIDATE_PATH)
    values = {
        "contract_json": contract_json,
        "contract_sha256": hashlib.sha256(contract_json.encode()).hexdigest(),
        "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
    }
    for name in values:
        if template.count(f"${name}") != 1:
            message = f"Spec 0034 template must contain one ${name}"
            raise ValueError(message)
    rendered = Template(template).substitute(values)
    sentinels = {
        "# __EMBEDDED_MODEL__\n": "if False:\n" + textwrap.indent(model, "    "),
        "# __EMBEDDED_CANDIDATE__\n": "if False:\n"
        + textwrap.indent(candidate, "    "),
    }
    for sentinel, body in sentinels.items():
        if rendered.count(sentinel) != 1:
            raise ValueError(f"Spec 0034 template sentinel differs: {sentinel!r}")
        rendered = rendered.replace(sentinel, body, 1)
    payload = rendered.encode()
    compile(payload, str(root / TEMPLATE_PATH), "exec")
    return payload


def _embedded_model_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    if source.count(future) != 1:
        raise ValueError("Canonical MIL model future import differs")
    embedded = source.replace(future, "", 1)
    compile(embedded, str(path), "exec")
    return embedded


def _embedded_candidate_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    source = source.replace("from __future__ import annotations\n", "", 1)
    runtime_import = """from eqvae.models.local_global_mil import (
    ATTENTION_HEADS,
    HEAD_WIDTH,
    LOCAL_MAX_DEGREE,
    RADIAL_CODEBOOK,
    TOKEN_WIDTH,
    LocalAttentionGraph,
    SparseLocalSoftmaxAttention,
)
"""
    if source.count(runtime_import) != 1:
        raise ValueError("Candidate runtime import block differs")
    embedded = source.replace(runtime_import, "", 1)
    compile(embedded, str(path), "exec")
    return embedded


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the immutable Spec 0034 package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("build", "validate"),
        nargs="?",
        default="build",
    )
    args = parser.parse_args(argv)
    result = validate() if cast("str", args.action) == "validate" else build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
