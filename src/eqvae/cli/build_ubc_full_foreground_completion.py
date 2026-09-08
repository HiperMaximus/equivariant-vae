# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, FURB118, PLR0914, PLR0916, PLR2004, PLW0717, SLF001, T201, TRY003
"""Prepare eight missing-only paired extraction jobs and verify compact outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

from eqvae.cli import build_ubc_wsi45630_completion as prior
from eqvae.cli.build_ubc_cancer_topup import GLOBAL_AUDIT_PATH
from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.cli.generate_ubc_eval_manifests import SPLIT_PATH, SPLIT_SHA256
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_UNION_SHA256,
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

OUTPUT_ROOT: Final = Path("runs/local/full_foreground_completion")
SPEC_PATH: Final = Path("docs/specs/0024-full-foreground-latent-completion.md")
TEMPLATE_PATH: Final = Path("kaggle/kernels/full_foreground_completion/run_template.py")
DATASET_REFERENCE: Final = "maximusshtefan/eqvae-full-foreground-inputs"
CONTRACT_NAME: Final = "full_foreground_input.json"
WORKER_CONTRACT_NAME: Final = "spec0022_topup_inference_contract.json"
MANIFEST_NAME: Final = "cancer_topup_manifest.csv"
PAIR_NAME: Final = "spec0022_cancer_topup_pair_audit.json"
RUN_COUNT: Final = 8
TARGET_ROWS: Final = 1_750_221
MISSING_ROWS: Final = 1_077_164
EXPECTED_RUN_ROWS: Final = (
    140_268,
    137_761,
    135_378,
    131_266,
    132_584,
    136_829,
    135_184,
    127_894,
)
COMPLETION_ROOT: Final = Path("runs/local/wsi45630_completion/bundle/inference")
COMPLETION_EVIDENCE: Final = Path("runs/kaggle/wsi45630_completion_v1")
COMPLETION_MANIFEST_SHA256: Final = (
    "df4d5281c585bd39826a6d528eba2b8b1ddd15444e173c0c59cf3b6750202adc"
)
COMPLETION_CONTRACT_SHA256: Final = (
    "7b060e303a3ddc0559e7d616eaed20a983ff5287492a1d30280494f474a9481a"
)
COMPLETION_PAIR_SHA256: Final = (
    "267a8b8128e1a637352223ee3e5939edd9c47b29e7e8617fdbedcb1aae1ea20b"
)
TOPUP_LOG_SHA256: Final = (
    "88432e7897ca8675a66a3de147797ae2ff8acc68d31e8adf0038cc2361e10ccc"
)
COMPLETION_LOG_SHA256: Final = (
    "967b6955baa432d909532543fb8b5ade2972f2b5b33b43a68fa8d2e09074af57"
)

type Identity = tuple[int, int, int, int]


@dataclass(frozen=True)
class Prepared:
    """Offline coordinate plan; never contains a latent tensor or an image."""

    parts: tuple[tuple[Identity, ...], ...]
    plan: dict[str, object]
    worker_base: dict[str, object]


def load_coordinates(
    path: Path,
    *,
    expected_header: Sequence[str] | None = None,
    splits: Mapping[int, str] | None = None,
) -> tuple[Identity, ...]:
    """Read coordinates in numeric order with frozen WSI split inheritance."""
    rows: list[Identity] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if expected_header is not None and tuple(reader.fieldnames or ()) != tuple(
            expected_header,
        ):
            raise ValueError(f"Coordinate header differs: {path}")
        for raw in reader:
            row = tuple(int(raw[key]) for key in prior.MANIFEST_HEADER)
            identity = cast("Identity", row)
            if splits is not None and raw["split"] != splits.get(identity[1]):
                raise ValueError("Coordinate split differs from the frozen WSI split")
            rows.append(identity)
    _require_order(rows)
    return tuple(rows)


def _key(row: Identity) -> tuple[int, int, int]:
    return row[1], row[3], row[2]


def _require_order(rows: Sequence[Identity]) -> None:
    previous: Identity | None = None
    for row in rows:
        if min(row) < 0 or row[2] % 256 or row[3] % 256:
            raise ValueError("Coordinates must be nonnegative full-grid origins")
        if previous is not None and (
            _key(row) <= _key(previous) or row[0] <= previous[0]
        ):
            raise ValueError("Coordinates must have unique numeric wsi_id,y,x order")
        previous = row


def derive_missing(
    candidates: Sequence[Identity],
    sources: Mapping[str, Sequence[Identity]],
) -> tuple[tuple[Identity, ...], dict[str, int]]:
    """Subtract the entire disjoint physical inventory, not current MIL selections."""
    _require_order(candidates)
    by_atlas: dict[int, Identity] = {}
    by_coordinate: dict[tuple[int, int, int], tuple[Identity, str]] = {}
    for source, rows in sources.items():
        _require_order(rows)
        for row in rows:
            if row[0] in by_atlas or _key(row) in by_coordinate:
                raise ValueError(
                    "Stored sources overlap in atlas or coordinate identity",
                )
            by_atlas[row[0]] = row
            by_coordinate[_key(row)] = row, source
    reused = dict.fromkeys(sources, 0)
    missing: list[Identity] = []
    for row in candidates:
        existing = by_coordinate.get(_key(row))
        atlas_match = by_atlas.get(row[0])
        if (existing is not None and existing[0] != row) or (
            atlas_match is not None and atlas_match != row
        ):
            raise ValueError("Stored coordinate/atlas identity drift")
        if existing is None:
            missing.append(row)
        else:
            reused[existing[1]] += 1
    if sum(reused.values()) + len(missing) != len(candidates):
        raise ValueError("Missing subtraction does not exactly cover the target")
    return tuple(missing), reused


def partition_missing(
    rows: Sequence[Identity],
    *,
    count: int = RUN_COUNT,
) -> tuple[tuple[Identity, ...], ...]:
    """Apply the existing nearest-remaining-mean whole-WSI allocation rule."""
    _require_order(rows)
    sizes = [
        (wsi, sum(1 for _ in group)) for wsi, group in groupby(rows, lambda r: r[1])
    ]
    if not 1 <= count <= len(sizes):
        raise ValueError("Cannot create the requested nonempty whole-WSI parts")
    parts: list[tuple[Identity, ...]] = []
    cursor = offset = 0
    remaining_rows = len(rows)
    for number in range(count):
        remaining_parts = count - number
        target = remaining_rows / remaining_parts
        max_end = len(sizes) - remaining_parts + 1
        selected = 0
        while cursor < max_end:
            next_count = sizes[cursor][1]
            if selected and abs(selected - target) <= abs(
                selected + next_count - target,
            ):
                break
            selected += next_count
            cursor += 1
        parts.append(tuple(rows[offset : offset + selected]))
        offset += selected
        remaining_rows -= selected
    if offset != len(rows):
        raise ValueError("Work parts do not exactly cover missing coordinates")
    return tuple(parts)


def project_job(
    rows: int,
    wsi_count: int,
    largest_wsi: int,
    timings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Budget row/slide cost plus an explicit largest-WSI admission margin."""
    row_rate = max(
        cast("float", item["elapsed_seconds"]) / cast("int", item["row_count"])
        for item in timings
    )
    multi_wsi_rate = max(
        cast("float", item["elapsed_seconds"]) / cast("int", item["wsi_count"])
        for item in timings
        if cast("int", item["wsi_count"]) > 1
    )
    single = math.ceil(
        max(
            cast("float", item["elapsed_seconds"])
            for item in timings
            if item["wsi_count"] == 1
        ),
    )
    bound = max(math.ceil(largest_wsi * row_rate), math.ceil(multi_wsi_rate), single)
    row_projection = math.ceil(rows * row_rate)
    wsi_projection = math.ceil(wsi_count * multi_wsi_rate)
    inference = max(row_projection, wsi_projection) + bound
    binary_bytes = 2 * (LATENT_SHARD_HEADER_SIZE + rows * LATENT_RECORD_BYTES)
    if binary_bytes + 10_000_000 > 20_000_000_000 or inference > 25_200:
        raise ValueError("Whole-WSI job exceeds saved-output or session allowance")
    return {
        "method": "max_observed_seconds_per_row_or_wsi_v1",
        "planning_extension": "max_row_or_multi_wsi_plus_largest_wsi_margin",
        "source_runs": list(timings),
        "supplement_rows": rows,
        "supplement_wsi_count": wsi_count,
        "largest_missing_wsi_rows": largest_wsi,
        "row_projection_seconds": row_projection,
        "wsi_projection_seconds": wsi_projection,
        "worst_observed_seconds_per_wsi": bound,
        "projected_inference_seconds": inference,
        "validation_reserve_seconds": 3_600,
        "projected_total_seconds": inference + 3_600,
        "session_limit_seconds": 28_800,
        "fits": True,
    }


def prepare(repo_root: Path) -> Prepared:
    """Authenticate prior metadata and derive the fixed eight-run plan offline."""
    inputs = {
        name: prior._require_hash(path, expected)
        for name, path, expected in prior._sources(repo_root)
    }
    extra_sources = (
        ("frozen_split", SPLIT_PATH, SPLIT_SHA256),
        (
            "completed_wsi45630_manifest",
            COMPLETION_ROOT / MANIFEST_NAME,
            COMPLETION_MANIFEST_SHA256,
        ),
        (
            "completed_wsi45630_contract",
            COMPLETION_ROOT / WORKER_CONTRACT_NAME,
            COMPLETION_CONTRACT_SHA256,
        ),
        (
            "completed_wsi45630_pair",
            COMPLETION_EVIDENCE / "dataset" / PAIR_NAME,
            COMPLETION_PAIR_SHA256,
        ),
    )
    for name, path, digest in extra_sources:
        inputs[name] = prior._require_hash(repo_root / path, digest)
    prior._validate_prior_authorities(repo_root)
    old = prior._read_object(repo_root / prior.OLD_CONTRACT_PATH)
    inputs["base_global_audit"] = prior._require_hash(
        repo_root / GLOBAL_AUDIT_PATH,
        cast("str", old["base_global_audit_sha256"]),
    )
    with (repo_root / SPLIT_PATH).open(encoding="utf-8", newline="") as handle:
        splits = {int(row["wsi_id"]): row["split"] for row in csv.DictReader(handle)}
    candidates = load_coordinates(repo_root / prior.CANDIDATE_PATH, splits=splits)
    sources = {
        "base": load_coordinates(repo_root / prior.UNION_PATH, splits=splits),
        "part11": load_coordinates(
            repo_root / prior.OLD_MANIFEST_PATH,
            expected_header=prior.MANIFEST_HEADER,
        ),
        "wsi45630": load_coordinates(
            repo_root / COMPLETION_ROOT / MANIFEST_NAME,
            expected_header=prior.MANIFEST_HEADER,
        ),
    }
    missing, reused = derive_missing(candidates, sources)
    if (
        len(candidates) != TARGET_ROWS
        or len(missing) != MISSING_ROWS
        or reused != {"base": 576_375, "part11": 74_033, "wsi45630": 22_649}
        or len({row[1] for row in candidates}) != 152
        or len({row[1] for row in missing}) != 145
        or any(row[1] == prior.WSI_ID for row in missing)
    ):
        raise ValueError("Full-foreground inventory differs from Spec 0024")
    parts = partition_missing(missing)
    if tuple(map(len, parts)) != EXPECTED_RUN_ROWS:
        raise ValueError("Eight-run whole-WSI partition differs from the locked plan")
    timings, pngs = _prior_evidence(repo_root, old)
    runs: list[dict[str, object]] = []
    for number, rows in enumerate(parts, 1):
        counts = Counter(row[1] for row in rows)
        projection = project_job(len(rows), len(counts), max(counts.values()), timings)
        runs.append({
            "run_number": number,
            "producer": f"maximusshtefan/eqvae-full-foreground-{number:02d}",
            "wsi_ids": list(counts),
            "row_count": len(rows),
            "expected_binary_output_bytes": 2 * (64 + len(rows) * LATENT_RECORD_BYTES),
            "deadline_projection": projection,
        })
    return Prepared(
        parts,
        {
            "schema_version": "spec0024.full_foreground_plan.v1",
            "status": "prepared_not_extracted",
            "inputs": inputs,
            "target_rows": len(candidates),
            "missing_rows": len(missing),
            "reused_foreground_rows": reused,
            "retained_nonforeground_base_rows": len(sources["base"]) - reused["base"],
            "runs": runs,
            "source_pngs": pngs,
            "logical_integration": "blocked_until_all_remote_pairs_verify",
        },
        old,
    )


def _prior_evidence(
    repo_root: Path,
    old: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    deadline = cast("Mapping[str, object]", old["deadline_projection"])
    sealed = cast("list[dict[str, object]]", deadline["source_runs"])
    records: list[dict[str, object]] = []
    pngs: dict[str, dict[str, object]] = {}
    evidence_paths: list[tuple[str, Path, Path, str, str]] = []
    for item in sealed:
        run = cast("int", item["run_number"])
        root = repo_root / f"runs/kaggle/ubc_ocean_latents/run_{run:02d}_metadata"
        evidence_paths.append((
            f"base_{run:02d}",
            root / f"dataset/spec0021_pair_audit_run_{run:02d}_of_05.json",
            root / f"eqvae-ubc-ocean-latent-run-{run:02d}.log",
            cast("str", item["pair_audit_sha256"]),
            cast("str", item["run_log_sha256"]),
        ))
    evidence_paths.extend((
        (
            "part11",
            repo_root / prior.PAIR_AUDIT_PATH,
            repo_root
            / prior.PAIR_AUDIT_PATH.parent.parent
            / "eqvae-ubc-ocean-cancer-latent-top-up.log",
            prior.PAIR_AUDIT_SHA256,
            TOPUP_LOG_SHA256,
        ),
        (
            "wsi45630",
            repo_root / COMPLETION_EVIDENCE / "dataset" / PAIR_NAME,
            repo_root / COMPLETION_EVIDENCE / "eqvae-wsi45630-completion.log",
            COMPLETION_PAIR_SHA256,
            COMPLETION_LOG_SHA256,
        ),
    ))
    for source, pair_path, log_path, pair_hash, log_hash in evidence_paths:
        prior._require_hash(pair_path, pair_hash)
        prior._require_hash(log_path, log_hash)
        pair = prior._read_object(pair_path)
        if pair.get("status") != "complete":
            raise ValueError("Prior producer is not complete")
        evidence = cast("list[dict[str, object]]", pair["completed_wsi_evidence"])
        log = cast("list[dict[str, object]]", json.loads(log_path.read_text()))
        records.append({
            "source": source,
            "row_count": pair["row_count"],
            "wsi_count": len(evidence),
            "elapsed_seconds": log[-1]["time"],
            "pair_audit_sha256": pair_hash,
            "run_log_sha256": log_hash,
        })
        for item in evidence:
            wsi = str(item["wsi_id"])
            identity = {key: item[key] for key in ("png_bytes", "png_sha256")}
            if wsi in pngs and pngs[wsi] != identity:
                raise ValueError("Prior producers disagree on original PNG identity")
            pngs[wsi] = identity
    return records, pngs


def _metadata() -> dict[str, object]:
    return {
        "id": DATASET_REFERENCE,
        "title": "eqvae full foreground inputs",
        "licenses": [{"name": "other"}],
    }


def _worker_contract(
    prepared: Prepared,
    run: Mapping[str, object],
    manifest: Path,
    spec_sha256: str,
) -> dict[str, object]:
    contract = dict(prepared.worker_base)
    contract.update({
        "spec_sha256": spec_sha256,
        "completion_run_number": run["run_number"],
        "producer": run["producer"],
        "supplement_manifest": {
            "path": MANIFEST_NAME,
            "sha256": prior._sha256(manifest),
            "row_count": run["row_count"],
            "header": list(prior.MANIFEST_HEADER),
        },
        "expected_binary_output_bytes": run["expected_binary_output_bytes"],
        "deadline_projection": run["deadline_projection"],
    })
    return contract


def build(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Stage one shared immutable input and eight thin kernels offline."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    prepared = prepare(repo_root)
    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging path: {staging}")
    bundle = staging / "bundle"
    spec_hash = prior._sha256(repo_root / SPEC_PATH)
    runs = cast("list[dict[str, object]]", prepared.plan["runs"])
    try:
        (bundle / "checkpoints").mkdir(parents=True)
        for model, source in (
            ("normal_vae", prior.NORMAL_CHECKPOINT_PATH),
            ("so2_vae", prior.SO2_CHECKPOINT_PATH),
        ):
            shutil.copy2(
                repo_root / source,
                bundle / f"checkpoints/{model}_step_060000.pt",
            )
        prior._copy_source(repo_root, bundle / "src")
        for run, rows in zip(runs, prepared.parts, strict=True):
            number = cast("int", run["run_number"])
            root = bundle / f"runs/run_{number:02d}"
            root.mkdir(parents=True)
            prior._write_manifest(root / MANIFEST_NAME, rows)
            prior._write_canonical_json(
                root / WORKER_CONTRACT_NAME,
                _worker_contract(
                    prepared,
                    run,
                    root / MANIFEST_NAME,
                    spec_hash,
                ),
            )
        contract: dict[str, object] = {
            "schema_version": "spec0024.full_foreground_input.v1",
            "dataset_reference": DATASET_REFERENCE,
            "spec_sha256": spec_hash,
            "files": prior._artifact_records(bundle, exclude=set()),
            "runs": runs,
        }
        prior._write_canonical_json(bundle / CONTRACT_NAME, contract)
        prior._write_canonical_json(bundle / prior.METADATA_NAME, _metadata())
        prior._write_canonical_json(staging / "plan.json", prepared.plan)
        stage_upload_envelope(bundle_root=bundle, destination=staging / "upload")
        _render_kernels(repo_root, staging, contract)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate(repo_root=repo_root, output_root=output_root)


def _kernel_config(root: Path, run: Mapping[str, object]) -> dict[str, object]:
    number = cast("int", run["run_number"])
    worker = root / f"bundle/runs/run_{number:02d}" / WORKER_CONTRACT_NAME
    return {
        "run_number": number,
        "producer": run["producer"],
        "row_count": run["row_count"],
        "wsi_ids": run["wsi_ids"],
        "input_contract_sha256": prior._sha256(root / "bundle" / CONTRACT_NAME),
        "worker_contract_sha256": prior._sha256(worker),
        "manifest_sha256": prior._sha256(worker.with_name(MANIFEST_NAME)),
    }


def _kernel_metadata(run: Mapping[str, object]) -> dict[str, object]:
    number = cast("int", run["run_number"])
    return {
        "id": run["producer"],
        "title": f"eqvae full foreground {number:02d}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_REFERENCE],
        "competition_sources": ["UBC-OCEAN"],
        "kernel_sources": [],
        "model_sources": [],
    }


def _kernel_bytes(repo_root: Path, config: Mapping[str, object]) -> bytes:
    source = (
        Template((repo_root / TEMPLATE_PATH).read_text())
        .substitute(
            config=json.dumps(dict(config), sort_keys=True, separators=(",", ":")),
        )
        .encode()
    )
    if len(source) >= 1_000_000:
        raise ValueError("Kaggle script must be smaller than 1 MB")
    compile(source, "run.py", "exec")
    return source


def _render_kernels(
    repo_root: Path,
    root: Path,
    contract: Mapping[str, object],
) -> None:
    for run in cast("list[dict[str, object]]", contract["runs"]):
        number = cast("int", run["run_number"])
        kernel = root / f"kernels/run_{number:02d}"
        kernel.mkdir(parents=True)
        config = _kernel_config(root, run)
        (kernel / "run.py").write_bytes(_kernel_bytes(repo_root, config))
        prior._write_canonical_json(kernel / "config.json", config)
        prior._write_canonical_json(
            kernel / "kernel-metadata.json",
            _kernel_metadata(run),
        )


def validate(
    *,
    repo_root: Path,
    output_root: Path,
    require_receipt: bool = False,
) -> dict[str, object]:
    """Recompute coverage and reject stale inputs, kernels, or receipts."""
    prepared = prepare(repo_root)
    bundle = output_root / "bundle"
    contract = prior._read_object(bundle / CONTRACT_NAME)
    if (
        contract.get("schema_version") != "spec0024.full_foreground_input.v1"
        or contract.get("dataset_reference") != DATASET_REFERENCE
        or contract.get("spec_sha256") != prior._sha256(repo_root / SPEC_PATH)
        or contract.get("runs") != prepared.plan["runs"]
        or prior._read_object(output_root / "plan.json") != prepared.plan
        or prior._read_object(bundle / prior.METADATA_NAME) != _metadata()
    ):
        raise ValueError("Full-foreground plan or input contract differs")
    files = prior._records(contract["files"])
    if files != prior._artifact_records(
        bundle,
        exclude={CONTRACT_NAME, prior.METADATA_NAME},
    ):
        raise ValueError("Input bundle bytes or allow-list differ")
    expected_names = {f"src/{name}" for name in prior._source_files(repo_root)}
    expected_names.update(
        f"checkpoints/{model}_step_060000.pt" for model in EXPECTED_CHECKPOINT_SHA256
    )
    runs = cast("list[dict[str, object]]", contract["runs"])
    for run, expected_rows in zip(runs, prepared.parts, strict=True):
        number = cast("int", run["run_number"])
        root = bundle / f"runs/run_{number:02d}"
        expected_names.update(
            f"runs/run_{number:02d}/{name}"
            for name in (MANIFEST_NAME, WORKER_CONTRACT_NAME)
        )
        rows = load_coordinates(
            root / MANIFEST_NAME,
            expected_header=prior.MANIFEST_HEADER,
        )
        if rows != expected_rows or prior._read_object(
            root / WORKER_CONTRACT_NAME,
        ) != _worker_contract(
            prepared,
            run,
            root / MANIFEST_NAME,
            cast("str", contract["spec_sha256"]),
        ):
            raise ValueError("Per-run coordinates or worker binding differ")
        config = _kernel_config(output_root, run)
        kernel = output_root / f"kernels/run_{number:02d}"
        if (
            (kernel / "run.py").read_bytes() != _kernel_bytes(repo_root, config)
            or prior._read_object(kernel / "config.json") != config
            or prior._read_object(kernel / "kernel-metadata.json")
            != _kernel_metadata(run)
            or {p.name for p in kernel.iterdir()}
            != {"run.py", "config.json", "kernel-metadata.json"}
        ):
            raise ValueError("Generated kernel/config/producer binding differs")
    if set(files) != expected_names:
        raise ValueError(
            "Shared input contains an unexpected file or missing dependency",
        )
    for name, path in prior._source_files(repo_root).items():
        if files[f"src/{name}"]["sha256"] != prior._sha256(path):
            raise ValueError(f"Source snapshot differs: {name}")
    for model, digest in EXPECTED_CHECKPOINT_SHA256.items():
        if files[f"checkpoints/{model}_step_060000.pt"]["sha256"] != digest:
            raise ValueError("Shared frozen checkpoint differs")
    _validate_upload(output_root / "upload", bundle)
    if require_receipt:
        receipt = prior._read_object(output_root / "input_receipt.json")
        if receipt != _receipt(bundle, contract):
            raise ValueError("Private immutable version-1 receipt differs")
    return {
        "status": "locally_prepared",
        "run_count": len(runs),
        "new_rows_per_model": MISSING_ROWS,
        "input_contract_sha256": prior._sha256(bundle / CONTRACT_NAME),
        "bundle_bytes": sum(cast("int", record["bytes"]) for record in files.values()),
        "max_wrapper_bytes": max(
            (output_root / f"kernels/run_{i:02d}/run.py").stat().st_size
            for i in range(1, RUN_COUNT + 1)
        ),
        "remote_receipt_verified": require_receipt,
    }


def _validate_upload(upload: Path, bundle: Path) -> None:
    import zipfile  # noqa: PLC0415

    if {p.name for p in upload.iterdir()} != {"bundle.zip", prior.METADATA_NAME}:
        raise ValueError("Shared upload envelope differs")
    if prior._read_object(upload / prior.METADATA_NAME) != _metadata():
        raise ValueError("Upload dataset identity differs")
    files = prior._artifact_records(bundle, exclude={prior.METADATA_NAME})
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        if len(archive.namelist()) != len(files) or set(archive.namelist()) != set(
            files,
        ):
            raise ValueError("Upload archive allow-list differs")
        for name in files:
            if archive.read(name) != (bundle / name).read_bytes():
                raise ValueError(f"Upload archive bytes differ: {name}")


def _receipt(bundle: Path, contract: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "verified",
        "visibility": "private",
        "dataset_version": 1,
        "dataset_reference": DATASET_REFERENCE,
        "files": contract["files"],
        "input_contract_sha256": prior._sha256(bundle / CONTRACT_NAME),
    }


def verify_download(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Seal a private-v1 receipt only after the guarded remote download byte-matches."""
    validate(repo_root=repo_root, output_root=output_root)
    remote = output_root / "remote_v1"
    status = prior._read_object(remote / "status.json")
    info = cast(
        "Mapping[str, object]",
        prior._read_object(remote / "metadata/dataset-metadata.json")["info"],
    )
    if (
        status != {"status": "ready", "current_version_number": 1}
        or info.get("ownerUser") != "maximusshtefan"
        or info.get("datasetSlug") != DATASET_REFERENCE.split("/")[1]
        or info.get("isPrivate") is not True
    ):
        raise ValueError("Expected ready private input dataset version 1")
    bundle = output_root / "bundle"
    if prior._artifact_records(
        remote / "download",
        exclude=set(),
    ) != prior._artifact_records(bundle, exclude={prior.METADATA_NAME}):
        raise ValueError("Downloaded immutable input bytes differ")
    receipt = _receipt(bundle, prior._read_object(bundle / CONTRACT_NAME))
    target = output_root / "input_receipt.json"
    if target.exists():
        raise FileExistsError("Immutable input receipt already exists")
    prior._write_canonical_json(target, receipt)
    return {
        "status": "verified",
        "input_contract_sha256": receipt["input_contract_sha256"],
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def verify_pair_metadata(
    *,
    metadata_root: Path,
    manifest: Path,
    contract_path: Path,
    expected_pngs: Mapping[str, Mapping[str, object]],
    producer: str,
) -> dict[str, object]:
    """Accept one source-qualified pair from compact producer evidence."""
    rows = load_coordinates(manifest, expected_header=prior.MANIFEST_HEADER)
    contract = prior._read_object(contract_path)
    pair = prior._read_object(metadata_root / PAIR_NAME)
    manifest_hash = prior._sha256(manifest)
    wsi_ids = list(dict.fromkeys(row[1] for row in rows))
    if (
        not rows
        or contract.get("producer") != producer
        or (
            pair.get("schema_version") != "spec0022.cancer_topup_pair_audit.v1"
            or pair.get("status") != "complete"
            or pair.get("row_count") != len(rows)
            or pair.get("input_contract_sha256") != prior._sha256(contract_path)
            or pair.get("supplement_manifest_sha256") != manifest_hash
            or pair.get("base_union_sha256") != EXPECTED_UNION_SHA256
            or any(
                pair.get(f"{model.replace('_vae', '')}_checkpoint_sha256") != digest
                for model, digest in EXPECTED_CHECKPOINT_SHA256.items()
            )
        )
    ):
        raise ValueError("Completed pair does not bind the planned producer/contract")
    artifacts = cast("Mapping[str, Mapping[str, object]]", pair["artifacts"])
    if set(artifacts) != set(EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("Completion must contain exactly both model artifacts")
    expected_files = {PAIR_NAME}
    for model, checkpoint in EXPECTED_CHECKPOINT_SHA256.items():
        record = artifacts[model]
        name = f"{model}_mu_cancer_topup.json"
        expected_files.add(name)
        path = metadata_root / name
        sidecar = prior._read_object(path)
        expected_source = {
            "logical_basename": MANIFEST_NAME,
            "sha256": manifest_hash,
            "run_number": 1,
            "row_count": len(rows),
            "first_identity": dict(zip(prior.MANIFEST_HEADER, rows[0], strict=True)),
            "last_identity": dict(zip(prior.MANIFEST_HEADER, rows[-1], strict=True)),
        }
        binary_bytes = LATENT_SHARD_HEADER_SIZE + len(rows) * LATENT_RECORD_BYTES
        crc = sidecar.get("payload_crc32")
        if (
            record.get("bin_name") != f"{model}_mu_cancer_topup.bin"
            or record.get("bin_bytes") != binary_bytes
            or not _is_sha256(record.get("bin_sha256"))
            or record.get("sidecar_name") != name
            or record.get("sidecar_bytes") != path.stat().st_size
            or record.get("sidecar_sha256") != prior._sha256(path)
            or sidecar.get("schema_version") != "spec0020.latent_shard.v1"
            or sidecar.get("status") != "complete"
            or sidecar.get("model_name") != model
            or sidecar.get("checkpoint_sha256") != checkpoint
            or sidecar.get("pinned_union_sha256") != EXPECTED_UNION_SHA256
            or sidecar.get("source_manifest") != expected_source
            or sidecar.get("file_size") != binary_bytes
            or sidecar.get("payload_bytes") != len(rows) * LATENT_RECORD_BYTES
            or not _is_sha256(sidecar.get("payload_sha256"))
            or isinstance(crc, bool)
            or not isinstance(crc, int)
            or not 0 <= crc < 2**32
            or sidecar.get("completed_wsi_ids") != wsi_ids
            or sidecar.get("completed_wsi_count") != len(wsi_ids)
            or sidecar.get("tensor")
            != {
                "count": len(rows),
                "dtype": "float32_le",
                "shape": [16, 32, 32],
                "layout": "CHW",
                "record_bytes": LATENT_RECORD_BYTES,
            }
        ):
            raise ValueError(f"Source-qualified {model} sidecar identity differs")
    if {p.name for p in metadata_root.iterdir()} != expected_files:
        raise ValueError(
            "Compact completion evidence must contain only three JSON files",
        )
    evidence = cast("list[dict[str, object]]", pair["completed_wsi_evidence"])
    if [item["wsi_id"] for item in evidence] != wsi_ids:
        raise ValueError("Pair WSI completion order differs")
    for item in evidence:
        expected = expected_pngs[str(item["wsi_id"])]
        if any(
            item.get(key) != expected[key] for key in ("png_bytes", "png_sha256")
        ) or not _is_sha256(item.get("transcript_sha256")):
            raise ValueError(
                "Original PNG identity differs across old and new producers",
            )
    return {
        "producer": producer,
        "pair_audit_sha256": prior._sha256(metadata_root / PAIR_NAME),
        "manifest_sha256": manifest_hash,
        "row_count": len(rows),
        "artifacts": artifacts,
    }


def verify_completion(
    *,
    repo_root: Path,
    output_root: Path,
    metadata_root: Path,
) -> dict[str, object]:
    """Write the aggregate audit only after all eight exact/disjoint pairs verify."""
    validate(repo_root=repo_root, output_root=output_root, require_receipt=True)
    prepared = prepare(repo_root)
    runs = cast("list[dict[str, object]]", prepared.plan["runs"])
    pngs = cast("Mapping[str, Mapping[str, object]]", prepared.plan["source_pngs"])
    pairs: list[dict[str, object]] = []
    for run in runs:
        name = f"run_{cast('int', run['run_number']):02d}"
        inference = output_root / "bundle/runs" / name
        pairs.append(
            verify_pair_metadata(
                metadata_root=metadata_root / name / "dataset",
                manifest=inference / MANIFEST_NAME,
                contract_path=inference / WORKER_CONTRACT_NAME,
                expected_pngs=pngs,
                producer=cast("str", run["producer"]),
            ),
        )
    audit: dict[str, object] = {
        "schema_version": "spec0024.full_foreground_completion.v1",
        "status": "all_pairs_verified_logical_integration_pending",
        "plan_sha256": prior._sha256(output_root / "plan.json"),
        "input_contract_sha256": prior._sha256(output_root / "bundle" / CONTRACT_NAME),
        "target_rows": TARGET_ROWS,
        "new_rows_per_model": MISSING_ROWS,
        "reused_foreground_rows": prepared.plan["reused_foreground_rows"],
        "exact_disjoint_target_coverage": True,
        "original_png_identity_match": True,
        "pairs": pairs,
        "logical_views_activated": False,
    }
    destination = output_root / "completion_audit.json"
    if destination.exists():
        raise FileExistsError("Completion audit already exists")
    prior._write_canonical_json(destination, audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare fixed packages or verify downloaded compact evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("build", "validate", "verify-download", "verify-completion"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--require-receipt", action="store_true")
    args = parser.parse_args(argv)
    repo_root = cast("Path", args.repo_root).resolve()
    output_root = cast("Path", args.output_root)
    action = cast("str", args.action)
    if action == "build":
        result = build(repo_root=repo_root, output_root=output_root)
    elif action == "verify-download":
        result = verify_download(repo_root=repo_root, output_root=output_root)
    elif action == "verify-completion":
        metadata_root = cast("Path | None", args.metadata_root)
        if metadata_root is None:
            parser.error("verify-completion requires --metadata-root")
        result = verify_completion(
            repo_root=repo_root,
            output_root=output_root,
            metadata_root=metadata_root,
        )
    else:
        result = validate(
            repo_root=repo_root,
            output_root=output_root,
            require_receipt=cast("bool", args.require_receipt),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
