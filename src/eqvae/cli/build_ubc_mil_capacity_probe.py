# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, PLR0914, PLR0915, PLW0717, TRY003
"""Build the one-off unchunked largest-WSI MIL capacity probe."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import textwrap
import zipfile
from collections import Counter
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
    WSI_BAG_HEADER,
    WSI_INSTANCE_HEADER,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

KERNEL_ID: Final = "maximusshtefan/eqvae-ubc-ocean-largest-wsi-mil-capacity-probe"
CONFIG_SCHEMA: Final = "spec0023.mil_capacity_probe_config.v1"
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/ubc_ocean_mil_capacity_probe/run_template.py",
)
SPEC_PATH: Final = Path("docs/specs/0023-matched-supervised-latent-evaluation.md")
DEFAULT_MANIFEST_ROOT: Final = Path("runs/local/ubc_ocean_supervised_manifests")
DEFAULT_OUTPUT: Final = Path("runs/local/ubc_ocean_mil_capacity_probe")
EXPECTED_WSI_ID: Final = 45630
EXPECTED_INSTANCE_COUNT: Final = 8149
EXPECTED_PART_COUNTS: Final = {4: 3339, 11: 4810}
INITIALIZATION_SEED: Final = 1701
WARMUP_STEPS: Final = 1
MEASURED_STEPS: Final = 1
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
)
UPLOAD_FILES: Final = frozenset({
    "kernel-metadata.json",
    "run.py",
    "spec0023_mil_capacity_probe_config.json",
})


def build_probe(
    *,
    repo_root: Path,
    manifest_root: Path,
    output_root: Path,
    enforce_real: bool = True,
    _skip_validation: bool = False,
) -> dict[str, object]:
    """Build one exact local upload directory without contacting Kaggle."""
    catalog_path = manifest_root / "physical_parts.csv"
    instance_path = manifest_root / "wsi/wsi_cancer_train_instances.csv"
    bag_path = manifest_root / "wsi/wsi_cancer_train_bags.csv"
    catalog_rows = _read_csv(catalog_path, CATALOG_HEADER)
    instance_rows = _read_csv(instance_path, WSI_INSTANCE_HEADER)
    bag_rows = _read_csv(bag_path, WSI_BAG_HEADER)
    if not bag_rows:
        raise ValueError("Training WSI bag manifest is empty")

    largest = max(bag_rows, key=lambda row: int(row["instance_count"]))
    start = int(largest["instance_start"])
    count = int(largest["instance_count"])
    selected = instance_rows[start : start + count]
    if len(selected) != count or any(
        row["wsi_id"] != largest["wsi_id"] or row["split"] != "train"
        for row in selected
    ):
        raise ValueError("Largest training WSI range differs from its instances")

    part_counts = Counter(int(row["part"]) for row in selected)
    parts = set(part_counts)
    selected_catalog = [row for row in catalog_rows if int(row["part"]) in parts]
    if len(selected_catalog) != 2 * len(parts):
        raise ValueError("Largest-bag parts do not have one aligned model pair")
    sources = tuple(dict.fromkeys(row["kaggle_source"] for row in selected_catalog))
    if enforce_real and (
        int(largest["wsi_id"]) != EXPECTED_WSI_ID
        or count != EXPECTED_INSTANCE_COUNT
        or dict(part_counts) != EXPECTED_PART_COUNTS
        or sources != KERNEL_SOURCES
    ):
        raise ValueError("Real largest-WSI identity or physical topology changed")

    probe_instances: list[dict[str, str]] = []
    for probe_index, row in enumerate(selected):
        copied = dict(row)
        copied["instance_row"] = str(probe_index)
        probe_instances.append(copied)
    probe_bag = dict(largest)
    probe_bag.update({"bag_row": "0", "instance_start": "0"})

    class_counts = Counter(int(row["diagnosis_index"]) for row in bag_rows)
    if set(class_counts) != set(range(5)):
        raise ValueError("Training WSI bags must contain all five diagnoses")
    training_wsi_count = len(bag_rows)
    class_weights = {
        str(index): training_wsi_count / (5 * class_counts[index]) for index in range(5)
    }
    assets = {
        "probe/physical_parts.csv": _csv_bytes(CATALOG_HEADER, selected_catalog),
        "probe/wsi_instances.csv": _csv_bytes(WSI_INSTANCE_HEADER, probe_instances),
        "probe/wsi_bags.csv": _csv_bytes(WSI_BAG_HEADER, [probe_bag]),
    }
    source_records = [
        {
            "kaggle_source": source,
            "binaries": sorted(
                (
                    {
                        "name": row["binary_name"],
                        "bytes": int(row["binary_bytes"]),
                        "sha256": row["binary_sha256"],
                    }
                    for row in selected_catalog
                    if row["kaggle_source"] == source
                ),
                key=lambda record: cast("str", record["name"]),
            ),
            "binary_bytes": sum(
                int(row["binary_bytes"])
                for row in selected_catalog
                if row["kaggle_source"] == source
            ),
        }
        for source in sources
    ]
    config: dict[str, object] = {
        "schema_version": CONFIG_SCHEMA,
        "spec_sha256": _sha256(repo_root / SPEC_PATH),
        "full_manifest_sha256": {
            "physical_parts.csv": _sha256(catalog_path),
            "wsi_cancer_train_instances.csv": _sha256(instance_path),
            "wsi_cancer_train_bags.csv": _sha256(bag_path),
        },
        "embedded_assets_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in assets.items()
        },
        "wsi_id": int(largest["wsi_id"]),
        "split": "train",
        "diagnosis_index": int(largest["diagnosis_index"]),
        "instance_count": count,
        "part_counts": {str(part): part_counts[part] for part in sorted(part_counts)},
        "training_wsi_count": training_wsi_count,
        "diagnosis_class_counts": {
            str(index): class_counts[index] for index in range(5)
        },
        "diagnosis_class_weights": class_weights,
        "kernel_sources": list(sources),
        "source_records": source_records,
        "mounted_binary_bytes": sum(
            cast("int", record["binary_bytes"]) for record in source_records
        ),
        "model_devices": {"normal_vae": 0, "so2_vae": 1},
        "initialization_seed": INITIALIZATION_SEED,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        "precision": "FP16-autocast-with-GradScaler",
        "grad_scaler_init_scale": 32_768,
        "grad_scaler_growth_interval": 1_000_000,
        "checkpoint_chunk_size": None,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "session_limit_seconds": 28_800,
        "required_deadline_reserve_seconds": 3_600,
        "saved_output_limit_bytes": 20_000_000_000,
        "projected_output_bytes": 100_000,
        "output_allowlist": ["spec0023_mil_capacity_probe.json"],
        "test_release_status": "not_authorized_not_mounted",
        "binary_integrity_basis": (
            "completed_audit_catalog_sha256_plus_mounted_size_and_header_count;"
            "no_full_binary_rehash"
        ),
    }
    config_bytes = _canonical_json(config)
    payload = _source_zip(repo_root, assets)
    if output_root.exists():
        message = f"Refusing to overwrite {output_root}"
        raise FileExistsError(message)
    output_root.mkdir(parents=True)
    try:
        (output_root / "spec0023_mil_capacity_probe_config.json").write_bytes(
            config_bytes,
        )
        metadata = {
            "id": KERNEL_ID,
            "title": "eqvae UBC-OCEAN largest-WSI MIL capacity probe",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": list(sources),
            "model_sources": [],
        }
        (output_root / "kernel-metadata.json").write_text(
            f"{json.dumps(metadata, indent=2)}\n",
            encoding="utf-8",
        )
        (output_root / "run.py").write_bytes(
            _render_wrapper(repo_root, payload, config_bytes),
        )
        if not _skip_validation:
            validate_probe(
                repo_root=repo_root,
                manifest_root=manifest_root,
                output_root=output_root,
                enforce_real=enforce_real,
            )
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return config


def validate_probe(
    *,
    repo_root: Path,
    manifest_root: Path,
    output_root: Path,
    enforce_real: bool = True,
) -> None:
    """Reject a stale or broadened one-off probe upload directory."""
    observed = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed != set(UPLOAD_FILES):
        raise ValueError("MIL capacity-probe upload allow-list differs")
    expected_root = output_root.parent / f".{output_root.name}.validation"
    if expected_root.exists():
        shutil.rmtree(expected_root)
    try:
        expected = build_probe(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=expected_root,
            enforce_real=enforce_real,
            _skip_validation=True,
        )
        config = _read_object(output_root / "spec0023_mil_capacity_probe_config.json")
        if config != expected:
            raise ValueError("MIL capacity-probe config differs")
        for name in UPLOAD_FILES:
            if (output_root / name).read_bytes() != (expected_root / name).read_bytes():
                message = f"MIL capacity-probe {name} differs"
                raise ValueError(message)
    finally:
        shutil.rmtree(expected_root, ignore_errors=True)


def _render_wrapper(repo_root: Path, payload: bytes, config_bytes: bytes) -> bytes:
    substitutions = {
        "embedded_payload_b64": "\n".join(
            textwrap.wrap(base64.b64encode(payload).decode("ascii"), 76),
        ),
        "embedded_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "embedded_config_b64": base64.b64encode(config_bytes).decode("ascii"),
        "embedded_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    template = Template((repo_root / TEMPLATE_PATH).read_text(encoding="utf-8"))
    return template.substitute(substitutions).encode()


def _source_zip(repo_root: Path, assets: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    source_root = repo_root / "src/eqvae"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        entries = {
            (
                Path("src/eqvae") / path.relative_to(source_root)
            ).as_posix(): path.read_bytes()
            for path in sorted(source_root.rglob("*"))
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        entries.update(assets)
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def _read_csv(path: Path, header: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(header):
            message = f"Unexpected CSV header in {path}"
            raise ValueError(message)
        return [dict(row) for row in reader]


def _csv_bytes(
    header: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("build", "validate"),
        nargs="?",
        default="build",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the exact real one-off Kaggle probe directory."""
    args = _parser().parse_args(argv)
    repo_root = cast("Path", args.repo_root).resolve()
    manifest_root = cast("Path", args.manifest_root)
    output_root = cast("Path", args.output_root)
    if cast("str", args.mode) == "build":
        build_probe(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=output_root,
        )
    else:
        validate_probe(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=output_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
