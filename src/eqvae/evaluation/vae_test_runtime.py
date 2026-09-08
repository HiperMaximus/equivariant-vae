# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: COM812, DOC201, DOC501, E501, EM101, EM102, FBT003, PLC0206, PLR0913, PLR0914, PLR0916, PLW0717, TRY003, TRY300, TRY301
"""Label-free dual-T4 runtime for the Spec 0045 reconstruction evaluation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import mmap
import shutil
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, cast

import torch
from torch import Tensor, nn

from eqvae.data.latent_shards import (
    LATENT_CHANNELS,
    LATENT_HEIGHT,
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LATENT_VALUES,
    LATENT_WIDTH,
    LatentRowIdentity,
    WorkManifest,
    WorkManifestRow,
    parse_latent_shard_header,
)
from eqvae.data.wsi_batches import iter_wsi_patch_batches
from eqvae.evaluation.vae_test import (
    REDACTED_LOCATION_HEADER,
    REMOTE_METRIC_HEADER,
    TEST_ROW_COUNT,
    TEST_WSI_COUNT,
    TestLocation,
    load_test_locations,
    location_identity_sha256,
    reconstruction_metrics,
    sha256_file,
    state_dict_sha256,
)
from eqvae.models.registry import build_model

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _Decoder(nn.Module):
    """Typed seam shared by both frozen decoder implementations."""

    def decode(self, latent: Tensor) -> Tensor:
        """Decode one latent tensor batch."""
        raise NotImplementedError


INPUT_ROOT: Final = Path("/kaggle/input")
WORKING_ROOT: Final = Path("/kaggle/working")
CONTRACT_NAME: Final = "spec0045_vae_test_input.json"
OUTPUT_DIRNAME: Final = "vae_test_reconstruction"
LATENT_KERNEL_SOURCES: Final = tuple(
    f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}" for run in range(1, 6)
)
MODEL_KINDS: Final = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}
MODEL_PARAMETER_COUNTS: Final = {"normal_vae": 3_958_435, "so2_vae": 1_180_035}
METRIC_KEYS: Final = REMOTE_METRIC_HEADER[len(REDACTED_LOCATION_HEADER) :]
FORBIDDEN_LABEL_TERMS: Final = (
    "diagnosis",
    "tissue",
    "label",
    "truth",
    "target",
    "oracle",
)
GPU_COUNT: Final = 2
IMAGE_BATCH_NDIM: Final = 4
RGB_CHANNELS: Final = 3
EXPECTED_TORCH_VERSION: Final = "2.14.0+cu130"


class _LatentReader:
    """Minimal fixed-record mmap reader for one authenticated Spec 0021 binary."""

    def __init__(self, path: Path, *, expected_bytes: int) -> None:
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Latent binary size differs: {path.name}")
        with path.open("rb") as handle:
            header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
        expected_size = (
            LATENT_SHARD_HEADER_SIZE + header.tensor_count * LATENT_RECORD_BYTES
        )
        if expected_size != expected_bytes:
            raise ValueError(f"Latent header count differs: {path.name}")
        self.path = path
        self.count = header.tensor_count
        self._file: BinaryIO | None = None
        self._mapping: mmap.mmap | None = None

    def __getitem__(self, index: int) -> Tensor:
        """Return one read-only FP32 latent view at the exact file index."""
        if not 0 <= index < self.count:
            raise IndexError(f"Latent file index {index} outside {self.path.name}")
        mapping = self._ensure_mapping()
        offset = LATENT_SHARD_HEADER_SIZE + index * LATENT_RECORD_BYTES
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            tensor = torch.frombuffer(
                mapping,
                dtype=torch.float32,
                count=LATENT_VALUES,
                offset=offset,
            )
        return tensor.reshape(LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)

    def close(self) -> None:
        """Close mmap state after every derived GPU tensor has been released."""
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _ensure_mapping(self) -> mmap.mmap:
        if self._mapping is None:
            self._file = self.path.open("rb")
            self._mapping = mmap.mmap(
                self._file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
            if hasattr(self._mapping, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                self._mapping.madvise(mmap.MADV_RANDOM)
        return self._mapping


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact mounted-input evaluation and publish status last."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-input-contract-sha256", required=True)
    parser.add_argument("--expected-dataset-reference", required=True)
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument("--working-root", default=str(WORKING_ROOT))
    args = parser.parse_args(argv)
    run_evaluation(
        expected_input_contract_sha256=cast(
            "str",
            args.expected_input_contract_sha256,
        ),
        expected_dataset_reference=cast("str", args.expected_dataset_reference),
        input_root=Path(cast("str", args.input_root)),
        working_root=Path(cast("str", args.working_root)),
    )
    return 0


def run_evaluation(
    *,
    expected_input_contract_sha256: str,
    expected_dataset_reference: str,
    input_root: Path,
    working_root: Path,
) -> dict[str, object]:
    """Evaluate every paired location and atomically publish the accepted package."""
    output = working_root / OUTPUT_DIRNAME
    staging = working_root / f".{OUTPUT_DIRNAME}.building"
    if output.exists() or staging.exists():
        raise FileExistsError("Spec 0045 output or staging path already exists")
    started = time.perf_counter()
    readers: dict[str, dict[int, _LatentReader]] = {}
    try:
        _require_exact_runtime()
        bundle_root, contract = _resolve_input_bundle(
            input_root=input_root,
            expected_sha256=expected_input_contract_sha256,
            expected_dataset_reference=expected_dataset_reference,
        )
        locations = load_test_locations(
            bundle_root / "vae_test_locations.csv",
            expected_sha256=cast(
                "str",
                cast(
                    "Mapping[str, object]",
                    cast("Mapping[str, object]", contract["population"])[
                        "redacted_location"
                    ],
                )["sha256"],
            ),
        )
        latent_roots = _resolve_latent_roots(
            input_root=input_root,
            contract=contract,
        )
        readers = _open_latent_readers(latent_roots=latent_roots, contract=contract)
        wsi_dir = _resolve_wsi_dir(input_root)
        devices = _require_exact_dual_t4()
        models = {
            branch: _load_model(
                bundle_root=bundle_root,
                contract=contract,
                branch=branch,
                device=devices[index],
            )
            for index, branch in enumerate(MODEL_KINDS)
        }
        staging.mkdir(parents=True)
        metric_path = staging / "per_patch_metrics.csv.gz"
        wsi_evidence, metric_digest, inf_counts = _evaluate_rows(
            locations=locations,
            readers=readers,
            models=models,
            devices=devices,
            wsi_dir=wsi_dir,
            expected_wsi_evidence=cast("list[object]", contract["wsi_evidence"]),
            batch_size=cast(
                "int",
                cast("Mapping[str, object]", contract["execution"])["batch_size"],
            ),
            metric_path=metric_path,
        )
        elapsed = time.perf_counter() - started
        runtime = {
            "schema_version": "spec0045.runtime.v1",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(index) for index in range(2)],
            "precision": "FP32",
            "decoder_input": "stored_posterior_mu",
            "optimizer_updates": 0,
            "patch_count": len(locations),
            "elapsed_seconds": elapsed,
            "patches_per_second": len(locations) / elapsed,
        }
        _write_json(staging / "runtime.json", runtime)
        _write_json(staging / "wsi_evidence.json", wsi_evidence)
        run_contract = {
            "schema_version": "spec0045.remote_run_contract.v1",
            "input_contract_sha256": expected_input_contract_sha256,
            "dataset_reference": expected_dataset_reference,
            "kernel_sources": list(LATENT_KERNEL_SOURCES),
            "location_identity_sha256": location_identity_sha256(locations),
            "metric_row_identity_sha256": metric_digest,
            "metric_inf_counts": inf_counts,
            "source_contract": contract,
        }
        _write_json(staging / "run_contract.json", run_contract)
        return _publish_completed_output(
            staging=staging,
            output=output,
            row_count=len(locations),
            wsi_count=len(wsi_evidence),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for branch_readers in readers.values():
            for reader in branch_readers.values():
                reader.close()


def _publish_completed_output(
    *,
    staging: Path,
    output: Path,
    row_count: int,
    wsi_count: int,
) -> dict[str, object]:
    """Publish the immutable payload first and write the pass status last."""
    output_files = {
        path.name: _file_record(path) for path in staging.iterdir() if path.is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": "spec0045.remote_manifest.v1",
        "status": "complete",
        "row_count": row_count,
        "wsi_count": wsi_count,
        "output_files": output_files,
    }
    _write_json(staging / "manifest.json", manifest)
    manifest_sha256 = sha256_file(staging / "manifest.json")
    _write_json(
        staging / "status.json",
        {
            "schema_version": "spec0045.remote_status.v1",
            "status": "pass",
            "manifest_sha256": manifest_sha256,
            "optimizer_updates": 0,
        },
    )
    staging.replace(output)
    return manifest


def _resolve_input_bundle(
    *,
    input_root: Path,
    expected_sha256: str,
    expected_dataset_reference: str,
) -> tuple[Path, dict[str, object]]:
    matches = [
        path
        for path in input_root.rglob(CONTRACT_NAME)
        if path.is_file()
        and not path.is_symlink()
        and sha256_file(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError("Expected one exact Spec 0045 input contract")
    contract_path = matches[0]
    root = contract_path.parent
    contract = _read_object(contract_path)
    if (
        contract.get("schema_version") != "spec0045.vae_test_input.v1"
        or contract.get("scope") != "frozen_vae_full_test_reconstruction_label_free"
        or contract.get("status") != "complete"
        or contract.get("dataset_reference") != expected_dataset_reference
        or contract.get("kernel_sources") != list(LATENT_KERNEL_SOURCES)
        or contract.get("producer_versions") != dict.fromkeys(LATENT_KERNEL_SOURCES, 1)
        or contract.get("evaluator_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("Spec 0045 mounted contract identity differs")
    files = cast("Mapping[str, object]", contract.get("files"))
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*files, CONTRACT_NAME}:
        raise ValueError("Spec 0045 mounted input allow-list differs")
    for relative, raw_record in files.items():
        if any(term in relative.lower() for term in FORBIDDEN_LABEL_TERMS):
            raise ValueError(f"Mounted input exposes forbidden path {relative}")
        record = cast("Mapping[str, object]", raw_record)
        path = root / relative
        if path.stat().st_size != record.get("bytes") or sha256_file(
            path
        ) != record.get("sha256"):
            raise ValueError(f"Mounted input byte differs: {relative}")
    header = (
        (root / "vae_test_locations.csv")
        .open(
            encoding="utf-8",
        )
        .readline()
    )
    if any(term in header.lower() for term in FORBIDDEN_LABEL_TERMS):
        raise ValueError("Mounted location CSV exposes forbidden fields")
    return root, contract


def _resolve_latent_roots(
    *,
    input_root: Path,
    contract: Mapping[str, object],
) -> dict[int, Path]:
    sources = cast("Mapping[str, object]", contract["latent_sources"])
    roots: dict[int, Path] = {}
    for run in range(1, 6):
        source = cast("Mapping[str, object]", sources[str(run)])
        pair = cast("Mapping[str, object]", source["pair_audit"])
        name = cast("str", pair["name"])
        matches = [
            path
            for path in input_root.rglob(name)
            if path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == pair["bytes"]
            and sha256_file(path) == pair["sha256"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one exact latent pair source for run {run}")
        root = matches[0].parent
        pair_payload = _read_object(matches[0])
        if (
            pair_payload.get("status") != "complete"
            or pair_payload.get("run_number") != run
        ):
            raise ValueError(f"Latent pair audit {run} is not complete")
        roots[run] = root
    return roots


def _open_latent_readers(
    *,
    latent_roots: Mapping[int, Path],
    contract: Mapping[str, object],
) -> dict[str, dict[int, _LatentReader]]:
    sources = cast("Mapping[str, object]", contract["latent_sources"])
    output: dict[str, dict[int, _LatentReader]] = {
        "normal_vae": {},
        "so2_vae": {},
    }
    try:
        for run, root in latent_roots.items():
            source = cast("Mapping[str, object]", sources[str(run)])
            for branch in output:
                artifact = cast("Mapping[str, object]", source[branch])
                binary = cast("Mapping[str, object]", artifact["binary"])
                sidecar = cast("Mapping[str, object]", artifact["sidecar"])
                sidecar_path = root / cast("str", sidecar["name"])
                binary_path = root / cast("str", binary["name"])
                if (
                    sidecar_path.stat().st_size != sidecar["bytes"]
                    or sha256_file(sidecar_path) != sidecar["sha256"]
                ):
                    raise ValueError(f"Latent sidecar differs for {branch} run {run}")
                sidecar_payload = _read_object(sidecar_path)
                if (
                    sidecar_payload.get("status") != "complete"
                    or sidecar_payload.get("model_name") != branch
                    or sidecar_payload.get("file_size") != binary["bytes"]
                    or binary_path.stat().st_size != binary["bytes"]
                    or sha256_file(binary_path) != binary["sha256"]
                ):
                    raise ValueError(f"Latent sidecar contract differs for {branch}")
                output[branch][run] = _LatentReader(
                    binary_path,
                    expected_bytes=cast("int", binary["bytes"]),
                )
        return output
    except BaseException:
        for branch_readers in output.values():
            for reader in branch_readers.values():
                reader.close()
        raise


def _load_model(
    *,
    bundle_root: Path,
    contract: Mapping[str, object],
    branch: str,
    device: torch.device,
) -> _Decoder:
    weights = cast("Mapping[str, object]", contract["weights"])
    record = cast("Mapping[str, object]", weights[branch])
    path = bundle_root / f"{branch}_state.pt"
    if (
        path.stat().st_size != record["state_file_bytes"]
        or sha256_file(path) != record["state_file_sha256"]
    ):
        raise ValueError(f"Derived state file differs for {branch}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in raw.items()
    ):
        raise TypeError(f"Derived state payload differs for {branch}")
    state = cast("dict[str, Tensor]", raw)
    if state_dict_sha256(state) != record["state_dict_sha256"]:
        raise ValueError(f"Derived state identity differs for {branch}")
    model = build_model(MODEL_KINDS[branch])
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != cast(
            "int",
            record["parameter_count"],
        )
        or cast("int", record["parameter_count"]) != MODEL_PARAMETER_COUNTS[branch]
    ):
        raise ValueError(f"Frozen parameter count differs for {branch}")
    model.load_state_dict(state, strict=True)
    prepared = model.to(device).eval().requires_grad_(False)
    return cast("_Decoder", cast("object", prepared))


def _evaluate_rows(
    *,
    locations: Sequence[TestLocation],
    readers: Mapping[str, Mapping[int, _LatentReader]],
    models: Mapping[str, _Decoder],
    devices: tuple[torch.device, torch.device],
    wsi_dir: Path,
    expected_wsi_evidence: Sequence[object],
    batch_size: int,
    metric_path: Path,
) -> tuple[list[dict[str, object]], str, dict[str, int]]:
    manifest = _as_work_manifest(locations)
    expected_evidence = {
        cast("int", cast("Mapping[str, object]", item)["wsi_id"]): cast(
            "Mapping[str, object]",
            item,
        )
        for item in expected_wsi_evidence
    }
    evidence: list[dict[str, object]] = []
    digest = hashlib.sha256(b"eqvae_spec0045_metric_row_v1")
    inf_counts = {"normal_psnr_img": 0, "so2_psnr_img": 0}
    normal_stream = torch.cuda.Stream(device=devices[0])
    so2_stream = torch.cuda.Stream(device=devices[1])
    with _gzip_text_writer(metric_path) as text_handle:
        writer = csv.writer(text_handle, lineterminator="\n")
        writer.writerow(REMOTE_METRIC_HEADER)
        for batch in iter_wsi_patch_batches(
            manifest=manifest,
            wsi_dir=wsi_dir,
            batch_size=batch_size,
        ):
            batch_locations = locations[
                batch.row_start : batch.row_start + len(batch.identities)
            ]
            normal_mu = torch.stack([
                readers["normal_vae"][location.run_number][location.file_index]
                for location in batch_locations
            ])
            so2_mu = torch.stack([
                readers["so2_vae"][location.run_number][location.file_index]
                for location in batch_locations
            ])
            with torch.inference_mode():
                with torch.cuda.device(devices[0]), torch.cuda.stream(normal_stream):
                    target_normal = _normalize_uint8_batch(
                        batch.images_uint8.to(devices[0], non_blocking=True),
                    )
                    normal_reconstruction = models["normal_vae"].decode(
                        normal_mu.to(devices[0], non_blocking=True),
                    )
                    normal_metrics = reconstruction_metrics(
                        reconstruction=normal_reconstruction,
                        target_normalized=target_normal,
                    )
                with torch.cuda.device(devices[1]), torch.cuda.stream(so2_stream):
                    target_so2 = _normalize_uint8_batch(
                        batch.images_uint8.to(devices[1], non_blocking=True),
                    )
                    so2_reconstruction = models["so2_vae"].decode(
                        so2_mu.to(devices[1], non_blocking=True),
                    )
                    so2_metrics = reconstruction_metrics(
                        reconstruction=so2_reconstruction,
                        target_normalized=target_so2,
                    )
            normal_stream.synchronize()
            so2_stream.synchronize()
            vectors = {
                "normal_mae_norm": normal_metrics["mae_norm"].cpu(),
                "normal_mse_norm": normal_metrics["mse_norm"].cpu(),
                "normal_psnr_img": normal_metrics["psnr_img"].cpu(),
                "normal_ssim_img": normal_metrics["ssim_img"].cpu(),
                "so2_mae_norm": so2_metrics["mae_norm"].cpu(),
                "so2_mse_norm": so2_metrics["mse_norm"].cpu(),
                "so2_psnr_img": so2_metrics["psnr_img"].cpu(),
                "so2_ssim_img": so2_metrics["ssim_img"].cpu(),
            }
            for offset, location in enumerate(batch_locations):
                metric_values = tuple(
                    float(vectors[name][offset].item()) for name in METRIC_KEYS
                )
                for name, value in zip(METRIC_KEYS, metric_values, strict=True):
                    if name in inf_counts and value == float("inf"):
                        inf_counts[name] += 1
                row = (*location.csv_values(), *metric_values)
                writer.writerow(_format_value(value) for value in row)
                for value in row:
                    digest.update(str(_format_value(value)).encode())
                    digest.update(b"\0")
                digest.update(b"\n")
            if batch.final_wsi_evidence is not None:
                observed = asdict(batch.final_wsi_evidence)
                expected = expected_evidence.get(batch.final_wsi_evidence.wsi_id)
                if expected is None or any(
                    observed[name] != expected[name]
                    for name in ("wsi_id", "png_bytes", "png_sha256")
                ):
                    raise ValueError(
                        f"Official WSI bytes differ for {batch.final_wsi_evidence.wsi_id}",
                    )
                evidence.append(observed)
    if len(evidence) != TEST_WSI_COUNT or len(locations) != TEST_ROW_COUNT:
        raise ValueError("Spec 0045 completed population differs")
    return evidence, digest.hexdigest(), inf_counts


def _as_work_manifest(locations: Sequence[TestLocation]) -> WorkManifest:
    rows = tuple(
        WorkManifestRow(
            identity=LatentRowIdentity(
                atlas_row_index=location.atlas_row_index,
                wsi_id=location.wsi_id,
                x=location.x,
                y=location.y,
            ),
            split="test",
            diagnosis_label="",
            diagnosis_index=0,
            tissue_label="",
            cancer_selected=True,
            tissue_selected=False,
        )
        for location in locations
    )
    ranges: list[tuple[int, int, int]] = []
    active_wsi = rows[0].identity.wsi_id
    start = 0
    for index, row in enumerate(rows[1:], start=1):
        if row.identity.wsi_id != active_wsi:
            ranges.append((active_wsi, start, index))
            active_wsi = row.identity.wsi_id
            start = index
    ranges.append((active_wsi, start, len(rows)))
    return WorkManifest(Path("vae_test_locations.csv"), "", 0, rows, tuple(ranges))


def _resolve_wsi_dir(input_root: Path) -> Path:
    candidates = (
        input_root / "UBC-OCEAN/train_images",
        input_root / "competitions/UBC-OCEAN/train_images",
    )
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise ValueError("Expected one official UBC-OCEAN train_images mount")
    return matches[0]


def _require_exact_dual_t4() -> tuple[torch.device, torch.device]:
    """Reject every Kaggle accelerator topology outside the frozen dual-T4 plan."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != GPU_COUNT:
        raise RuntimeError("Spec 0045 requires exactly two CUDA devices")
    names = tuple(torch.cuda.get_device_name(index) for index in range(GPU_COUNT))
    if names != ("Tesla T4", "Tesla T4"):
        raise RuntimeError(f"Spec 0045 requires dual Tesla T4, observed {names}")
    return torch.device("cuda:0"), torch.device("cuda:1")


def _require_exact_runtime() -> None:
    """Require the previously exercised PyTorch/CUDA build before test access."""
    if torch.__version__ != EXPECTED_TORCH_VERSION or torch.version.cuda != "13.0":
        raise RuntimeError(
            f"Spec 0045 requires Torch {EXPECTED_TORCH_VERSION}; observed "
            f"{torch.__version__} CUDA {torch.version.cuda}",
        )


def _normalize_uint8_batch(images: Tensor) -> Tensor:
    """Apply the frozen uint8/127.5-1 target transform without extra dependencies."""
    if (
        images.dtype != torch.uint8
        or images.ndim != IMAGE_BATCH_NDIM
        or images.shape[1] != RGB_CHANNELS
    ):
        raise ValueError("Spec 0045 target batch must be BCHW uint8 RGB")
    return images.to(dtype=torch.float32).div_(127.5).sub_(1.0)


def _gzip_text_writer(path: Path) -> io.TextIOWrapper:
    binary = path.open("xb")
    compressed = gzip.GzipFile(fileobj=binary, mode="wb", mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


if __name__ == "__main__":
    raise SystemExit(main())
