# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, PERF401, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLW0717, T201, TRY003, TRY300, TRY301
"""Build the frozen Spec 0022 cancer bags and label-free top-up input."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from eqvae.cli.generate_ubc_eval_manifests import (
    CANCER_HEADER,
    SPLIT_PATH,
    SPLIT_SHA256,
    sha256_file,
    validate_pinned_hash,
)
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_UNION_SHA256,
    EXPECTED_WORK_MANIFEST_SHA256,
    UNION_HEADER,
    LatentArtifact,
    LatentRowIdentity,
    ModelName,
    WorkManifestRow,
    load_work_manifest,
    validate_latent_artifact,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SPEC_PATH: Final = Path("docs/specs/0022-additive-cancer-coverage-topup.md")
CANDIDATE_PATH: Final = Path(
    "runs/local/ubc_ocean_eval_manifests/cancer_ae_patch_manifest.csv",
)
CONSUMPTION_ROOT: Final = Path("runs/local/ubc_ocean_eval_consumption")
UNION_PATH: Final = CONSUMPTION_ROOT / "union_patch_manifest.csv"
WORK_ROOT: Final = CONSUMPTION_ROOT / "work_shards"
GLOBAL_AUDIT_PATH: Final = Path(
    "runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/"
    "spec0021_latent_store_global_audit.json",
)
BASE_EVIDENCE_ROOT: Final = Path("runs/kaggle/ubc_ocean_latents")
NORMAL_CHECKPOINT_PATH: Final = Path(
    "runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt",
)
SO2_CHECKPOINT_PATH: Final = Path(
    "runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/"
    "checkpoints/step_060000.pt",
)
OUTPUT_ROOT: Final = Path("runs/local/ubc_ocean_cancer_topup")
CANDIDATE_SHA256: Final = (
    "710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c"
)
OLD_CANCER_SHA256: Final = {
    "train": "03e8e0d7aefb8d30d049c8521c44ebc88ca3709ea26a65765b772df2b3af65fb",
    "validation": ("3d96c739320ab1facbdf2bb93b18fc8d2c951a7f30a3d61f5ce3587e4c402ce5"),
    "test": "1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc",
}
SPLITS: Final = ("train", "validation", "test")
SELECTION_SEED: Final = 20_260_827
PURPOSE_RETAIN: Final = 0
PURPOSE_ADD: Final = 1
EXPECTED_TARGET_COUNTS: Final = {
    "train": 308_359,
    "validation": 66_706,
    "test": 66_194,
}
EXPECTED_TARGET_TOTAL: Final = 441_259
SAVED_OUTPUT_LIMIT_BYTES: Final = 20_000_000_000
METADATA_RESERVE_BYTES: Final = 10_000_000
SESSION_LIMIT_SECONDS: Final = 28_800
VALIDATION_RESERVE_SECONDS: Final = 3_600
LATENT_HEADER_BYTES: Final = 64
LATENT_RECORD_BYTES: Final = 65_536
SHA256_HEX_LENGTH: Final = 64
SUPPLEMENT_HEADER: Final = ("atlas_row_index", "wsi_id", "x", "y")
INSTANCE_HEADER: Final = (
    "instance_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "diagnosis_label",
    "diagnosis_index",
    "split",
    "store_source",
    "shard_number",
    "file_index",
)
BAG_HEADER: Final = (
    "bag_row",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "split",
    "instance_start",
    "instance_count",
)
ACCESS_TRANSCRIPT_HEADER: Final = (
    "run_name",
    "model_name",
    "logical_dataset_sha256",
    "instance_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "split",
    "store_source",
    "shard_number",
    "file_index",
)
TOPUP_OUTPUT_ALLOWLIST: Final = frozenset({
    "normal_vae_mu_cancer_topup.bin",
    "normal_vae_mu_cancer_topup.json",
    "so2_vae_mu_cancer_topup.bin",
    "so2_vae_mu_cancer_topup.json",
    "spec0022_cancer_topup_pair_audit.json",
})


@dataclass(frozen=True)
class CandidateRow:
    """One mask-independent candidate with frozen WSI metadata."""

    identity: LatentRowIdentity
    diagnosis_label: str
    diagnosis_index: int
    split: str

    @property
    def order_key(self) -> tuple[int, int, int]:
        """Authoritative WSI/y/x publication order."""
        return (self.identity.wsi_id, self.identity.y, self.identity.x)


@dataclass(frozen=True)
class SplitRow:
    """One canonical WSI assignment."""

    diagnosis_label: str
    diagnosis_index: int
    split: str


@dataclass(frozen=True)
class PhysicalLocation:
    """One exact record in either a base shard or the top-up shard."""

    source: str
    shard_number: int
    file_index: int


def target_bag_size(available: int) -> int:
    """Derive the locked quarter-coverage bag size without float rounding."""
    if available < 1:
        raise ValueError("Every WSI must have at least one cancer candidate")
    return min(available, max(1_000, (available + 3) // 4))


def select_wsi_target(
    candidates: Sequence[CandidateRow],
    existing: Sequence[CandidateRow],
    *,
    seed: int = SELECTION_SEED,
) -> tuple[CandidateRow, ...]:
    """Freeze one WSI target before physical-union membership is consulted."""
    if not candidates:
        raise ValueError("Cannot select an empty WSI candidate group")
    wsi_id = candidates[0].identity.wsi_id
    if any(row.identity.wsi_id != wsi_id for row in (*candidates, *existing)):
        raise ValueError("One target selection cannot mix WSIs")
    candidate_by_identity = {row.identity: row for row in candidates}
    if len(candidate_by_identity) != len(candidates):
        raise ValueError(f"Duplicate candidate identity in WSI {wsi_id}")
    existing_ids = {row.identity for row in existing}
    if (
        len(existing_ids) != len(existing)
        or not existing_ids <= candidate_by_identity.keys()
    ):
        raise ValueError(
            f"Existing cancer rows are not a unique candidate subset for WSI {wsi_id}",
        )
    target_count = target_bag_size(len(candidates))
    if len(existing) == target_count:
        return tuple(existing)
    if len(existing) > target_count:
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, wsi_id, PURPOSE_RETAIN])),
        )
        indices = cast(
            "np.ndarray[tuple[int], np.dtype[np.int64]]",
            rng.choice(len(existing), size=target_count, replace=False),
        )
        selected_ids = {existing[int(index)].identity for index in indices}
    else:
        remaining = [row for row in candidates if row.identity not in existing_ids]
        deficit = target_count - len(existing)
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, wsi_id, PURPOSE_ADD])),
        )
        indices = cast(
            "np.ndarray[tuple[int], np.dtype[np.int64]]",
            rng.choice(len(remaining), size=deficit, replace=False),
        )
        selected_ids = existing_ids | {
            remaining[int(index)].identity for index in indices
        }
    return tuple(row for row in candidates if row.identity in selected_ids)


def materialize_cancer_topup(
    *,
    candidate_path: Path = CANDIDATE_PATH,
    old_cancer_paths: Mapping[str, Path] | None = None,
    split_path: Path = SPLIT_PATH,
    union_path: Path = UNION_PATH,
    work_paths: Mapping[int, Path] | None = None,
    global_audit_path: Path = GLOBAL_AUDIT_PATH,
    normal_checkpoint_path: Path = NORMAL_CHECKPOINT_PATH,
    so2_checkpoint_path: Path = SO2_CHECKPOINT_PATH,
    output_root: Path = OUTPUT_ROOT,
    expected_candidate_sha256: str | None = CANDIDATE_SHA256,
    expected_old_sha256: Mapping[str, str] | None = OLD_CANCER_SHA256,
    expected_split_sha256: str | None = SPLIT_SHA256,
    expected_union_sha256: str | None = EXPECTED_UNION_SHA256,
    expected_work_sha256: Mapping[int, str] | None = EXPECTED_WORK_MANIFEST_SHA256,
    expected_checkpoint_sha256: Mapping[ModelName, str] = EXPECTED_CHECKPOINT_SHA256,
    pair_audit_paths: Mapping[int, Path] | None = None,
    run_log_paths: Mapping[int, Path] | None = None,
    enforce_real_counts: bool = True,
) -> dict[str, object]:
    """Publish an immutable plan only after the completed base audit validates."""
    old_paths = dict(
        old_cancer_paths
        or {split: CONSUMPTION_ROOT / f"cancer_{split}.csv" for split in SPLITS},
    )
    resolved_work_paths = dict(
        work_paths
        or {run: WORK_ROOT / f"run_{run:02d}_of_05.csv" for run in range(1, 6)},
    )
    resolved_pair_audits = dict(
        pair_audit_paths
        or {
            run: BASE_EVIDENCE_ROOT / f"run_{run:02d}_metadata/dataset/"
            f"spec0021_pair_audit_run_{run:02d}_of_05.json"
            for run in range(1, 6)
        },
    )
    resolved_run_logs = dict(
        run_log_paths
        or {
            run: BASE_EVIDENCE_ROOT / f"run_{run:02d}_metadata/"
            f"eqvae-ubc-ocean-latent-run-{run:02d}.log"
            for run in range(1, 6)
        },
    )
    _require_exact_keys(old_paths, SPLITS, "old cancer paths")
    _require_exact_keys(resolved_work_paths, tuple(range(1, 6)), "work paths")
    _require_exact_keys(resolved_pair_audits, tuple(range(1, 6)), "pair audit paths")
    _require_exact_keys(resolved_run_logs, tuple(range(1, 6)), "run log paths")
    input_hashes = {
        "candidate": sha256_file(candidate_path),
        "split": sha256_file(split_path),
        "union": sha256_file(union_path),
        "base_global_audit": sha256_file(global_audit_path),
        **{f"old_cancer_{split}": sha256_file(old_paths[split]) for split in SPLITS},
        **{
            f"work_{run:02d}": sha256_file(resolved_work_paths[run])
            for run in range(1, 6)
        },
        **{
            f"pair_audit_{run:02d}": sha256_file(resolved_pair_audits[run])
            for run in range(1, 6)
        },
        **{
            f"run_log_{run:02d}": sha256_file(resolved_run_logs[run])
            for run in range(1, 6)
        },
    }
    _validate_optional_hash(candidate_path, expected_candidate_sha256, "candidate")
    _validate_optional_hash(split_path, expected_split_sha256, "split")
    _validate_optional_hash(union_path, expected_union_sha256, "union")
    if expected_old_sha256 is not None:
        for split in SPLITS:
            _validate_optional_hash(
                old_paths[split],
                expected_old_sha256[split],
                f"old cancer {split}",
            )
    if expected_work_sha256 is not None:
        for run in range(1, 6):
            _validate_optional_hash(
                resolved_work_paths[run],
                expected_work_sha256[run],
                f"work manifest {run}",
            )
    _validate_base_global_audit(
        global_audit_path,
        expected_union_sha256=input_hashes["union"],
    )
    split_rows = _load_split(split_path)
    candidates, candidates_by_wsi = _load_candidates(candidate_path, split_rows)
    existing_by_wsi = _load_existing(old_paths, candidates, split_rows)

    targets: list[CandidateRow] = []
    target_by_wsi: dict[int, tuple[CandidateRow, ...]] = {}
    coverage: list[dict[str, object]] = []
    undercovered = Counter[str]()
    omitted = Counter[str]()
    nominal_additions = Counter[str]()
    for wsi_id, group in candidates_by_wsi.items():
        old = existing_by_wsi[wsi_id]
        selected = select_wsi_target(group, old)
        target_by_wsi[wsi_id] = selected
        targets.extend(selected)
        split = split_rows[wsi_id].split
        if len(old) < len(selected):
            undercovered[split] += 1
            nominal_additions[split] += len(selected) - len(old)
        omitted[split] += len(
            {row.identity for row in old} - {row.identity for row in selected},
        )
        coverage.append(_coverage_record(group, old, selected))
    if tuple(row.order_key for row in targets) != tuple(
        sorted(row.order_key for row in targets),
    ):
        raise AssertionError("Frozen target lost authoritative order")

    base_locations = _load_base_locations(
        union_path=union_path,
        work_paths=resolved_work_paths,
        expected_work_sha256=(
            {run: input_hashes[f"work_{run:02d}"] for run in range(1, 6)}
            if expected_work_sha256 is None
            else expected_work_sha256
        ),
    )
    target_ids = {row.identity for row in targets}
    supplement_rows = [row for row in targets if row.identity not in base_locations]
    supplement_locations = {
        row.identity: PhysicalLocation("topup", 1, index)
        for index, row in enumerate(supplement_rows)
    }
    if target_ids & set(supplement_locations) & set(base_locations):
        raise AssertionError("A target identity resolved to both base and top-up")
    locations = {
        identity: PhysicalLocation("base", run, index)
        for identity, (run, index, _row) in base_locations.items()
        if identity in target_ids
    }
    locations.update(supplement_locations)
    if set(locations) != target_ids:
        raise ValueError("Every target must resolve exactly once to base or top-up")

    split_counts = Counter(row.split for row in targets)
    if enforce_real_counts:
        observed = {split: split_counts[split] for split in SPLITS}
        if observed != EXPECTED_TARGET_COUNTS or len(targets) != EXPECTED_TARGET_TOTAL:
            raise ValueError(
                f"Spec 0022 target totals disagree: {observed}, total={len(targets)}",
            )

    normal_hash = _validate_checkpoint(
        normal_checkpoint_path,
        expected_checkpoint_sha256["normal_vae"],
        "normal_vae",
    )
    so2_hash = _validate_checkpoint(
        so2_checkpoint_path,
        expected_checkpoint_sha256["so2_vae"],
        "so2_vae",
    )
    expected_binary_bytes = 2 * (
        LATENT_HEADER_BYTES + len(supplement_rows) * LATENT_RECORD_BYTES
    )
    if expected_binary_bytes + METADATA_RESERVE_BYTES > SAVED_OUTPUT_LIMIT_BYTES:
        raise ValueError(
            "Top-up binaries plus metadata reserve exceed Kaggle output cap",
        )
    deadline_projection = _project_deadline(
        pair_audit_paths=resolved_pair_audits,
        run_log_paths=resolved_run_logs,
        supplement_rows=len(supplement_rows),
        supplement_wsi_count=len({row.identity.wsi_id for row in supplement_rows}),
    )

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing top-up plan: {output_root}",
        )
    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale top-up staging directory exists: {staging}")
    inference = staging / "inference_bundle"
    logical = staging / "logical"
    inference.mkdir(parents=True)
    logical.mkdir(parents=True)
    try:
        supplement_path = inference / "cancer_topup_manifest.csv"
        _write_supplement(supplement_path, supplement_rows)
        normal_copy = inference / "normal_vae_step_060000.pt"
        so2_copy = inference / "so2_vae_step_060000.pt"
        shutil.copyfile(normal_checkpoint_path, normal_copy)
        shutil.copyfile(so2_checkpoint_path, so2_copy)
        if sha256_file(normal_copy) != normal_hash or sha256_file(so2_copy) != so2_hash:
            raise ValueError("Copied top-up checkpoint bytes changed")
        logical_files = _write_logical_files(
            logical,
            targets,
            target_by_wsi,
            locations,
            split_rows,
        )
        _write_json(
            logical / "access_transcript_schema.json",
            {
                "schema_version": "spec0022.access_transcript_schema.v1",
                "header": list(ACCESS_TRANSCRIPT_HEADER),
                "test_release_policy": "train_validation_transcript_must_exclude_test",
            },
        )
        spec_hash = sha256_file(SPEC_PATH)
        supplement_hash = sha256_file(supplement_path)
        inference_contract = {
            "schema_version": "spec0022.cancer_topup_inference_input.v1",
            "status": "complete",
            "spec_sha256": spec_hash,
            "base_global_audit_sha256": input_hashes["base_global_audit"],
            "candidate_manifest_sha256": input_hashes["candidate"],
            "base_union_sha256": input_hashes["union"],
            "supplement_manifest": {
                "path": supplement_path.relative_to(inference).as_posix(),
                "sha256": supplement_hash,
                "row_count": len(supplement_rows),
                "header": list(SUPPLEMENT_HEADER),
            },
            "checkpoints": {
                "normal_vae": {
                    "path": normal_copy.relative_to(inference).as_posix(),
                    "sha256": normal_hash,
                },
                "so2_vae": {
                    "path": so2_copy.relative_to(inference).as_posix(),
                    "sha256": so2_hash,
                },
            },
            "tensor": {"dtype": "float32_le", "shape": [16, 32, 32], "layout": "CHW"},
            "expected_binary_output_bytes": expected_binary_bytes,
            "saved_output_limit_bytes": SAVED_OUTPUT_LIMIT_BYTES,
            "metadata_reserve_bytes": METADATA_RESERVE_BYTES,
            "deadline_projection": deadline_projection,
        }
        _write_canonical_json(
            inference / "spec0022_topup_inference_contract.json",
            inference_contract,
        )
        _write_json(
            inference / "dataset-metadata.json",
            {
                "id": "maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs",
                "title": "eqvae UBC-OCEAN cancer top-up inputs",
                "licenses": [{"name": "other"}],
            },
        )
        inference_files = _artifact_records(inference)
        forbidden_headers = {
            "diagnosis_label",
            "diagnosis_index",
            "split",
            "tissue_label",
            "mask",
        }
        if forbidden_headers & set(SUPPLEMENT_HEADER):
            raise AssertionError(
                "Inference manifest contains forbidden semantic fields",
            )
        logical_contract = {
            "schema_version": "spec0022.cancer_logical_dataset.v1",
            "status": "planned",
            "spec_sha256": spec_hash,
            "selection_seed": SELECTION_SEED,
            "candidate_manifest_sha256": input_hashes["candidate"],
            "base_union_sha256": input_hashes["union"],
            "base_global_audit_sha256": input_hashes["base_global_audit"],
            "supplement_manifest_sha256": supplement_hash,
            "files": logical_files,
            "target_counts": {split: split_counts[split] for split in SPLITS},
            "target_total": len(targets),
            "supplement_rows": len(supplement_rows),
        }
        _write_json(
            logical / "spec0022_logical_dataset_contract.json",
            logical_contract,
        )
        plan_audit: dict[str, object] = {
            "schema_version": "spec0022.cancer_topup_plan_audit.v1",
            "status": "complete",
            "inputs": input_hashes,
            "selection": {
                "seed": SELECTION_SEED,
                "target_counts": {split: split_counts[split] for split in SPLITS},
                "target_total": len(targets),
                "nominal_additions": {
                    split: nominal_additions[split] for split in SPLITS
                },
                "undercovered_wsi": {split: undercovered[split] for split in SPLITS},
                "old_rows_omitted": {split: omitted[split] for split in SPLITS},
                "supplement_rows": len(supplement_rows),
                "base_reused_rows": len(targets) - len(supplement_rows),
            },
            "coverage_by_wsi": coverage,
            "inference_bundle_files": inference_files,
            "logical_files": logical_files,
            "acceptance": {
                "target_frozen_before_union_resolution": True,
                "supplement_disjoint_from_base": not bool(
                    set(supplement_locations) & set(base_locations),
                ),
                "model_independent_logical_files": True,
                "inference_manifest_label_free": True,
                "base_global_audit_complete": True,
                "saved_output_fits": True,
                "deadline_projection_fits": deadline_projection["fits"],
            },
        }
        _write_json(staging / "spec0022_cancer_topup_plan_audit.json", plan_audit)
        _fsync_directory(staging)
        staging.replace(output_root)
        _fsync_directory(output_root.parent)
        return plan_audit
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def finalize_logical_audit(
    *,
    plan_root: Path = OUTPUT_ROOT,
    topup_pair_audit_path: Path,
) -> dict[str, object]:
    """Bind the completed aligned pair and publish the final logical audit last."""
    destination = plan_root / "spec0022_logical_dataset_audit.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    plan = _read_object(plan_root / "spec0022_cancer_topup_plan_audit.json")
    contract_path = (
        plan_root / "inference_bundle/spec0022_topup_inference_contract.json"
    )
    contract = _read_object(contract_path)
    pair = _read_object(topup_pair_audit_path)
    if (
        plan.get("schema_version") != "spec0022.cancer_topup_plan_audit.v1"
        or plan.get("status") != "complete"
        or contract.get("schema_version") != "spec0022.cancer_topup_inference_input.v1"
        or contract.get("status") != "complete"
    ):
        raise ValueError("Spec 0022 plan and inference contracts must be complete")
    supplement = cast("Mapping[str, object]", contract["supplement_manifest"])
    checkpoints = cast("Mapping[str, Mapping[str, object]]", contract["checkpoints"])
    if (
        pair.get("schema_version") != "spec0022.cancer_topup_pair_audit.v1"
        or pair.get("status") != "complete"
        or pair.get("supplement_manifest_sha256") != supplement["sha256"]
        or pair.get("row_count") != supplement["row_count"]
        or pair.get("normal_checkpoint_sha256") != checkpoints["normal_vae"]["sha256"]
        or pair.get("so2_checkpoint_sha256") != checkpoints["so2_vae"]["sha256"]
    ):
        raise ValueError(
            "Top-up pair audit does not match the frozen inference contract",
        )
    artifacts_value = pair.get("artifacts")
    if not isinstance(artifacts_value, dict) or set(
        cast("dict[str, object]", artifacts_value),
    ) != {"normal_vae", "so2_vae"}:
        raise ValueError("Top-up pair audit must bind exactly two model artifacts")
    pair_root = topup_pair_audit_path.parent
    observed_files = {path.name for path in pair_root.iterdir() if path.is_file()}
    if observed_files != set(TOPUP_OUTPUT_ALLOWLIST) or any(
        not path.is_file() for path in pair_root.iterdir()
    ):
        raise ValueError("Top-up output directory differs from its exact allow-list")
    supplement_path = plan_root / "inference_bundle" / cast("str", supplement["path"])
    validated: dict[str, LatentArtifact] = {}
    for model_name, bin_name in (
        ("normal_vae", "normal_vae_mu_cancer_topup.bin"),
        ("so2_vae", "so2_vae_mu_cancer_topup.bin"),
    ):
        typed_model = cast("ModelName", model_name)
        checkpoint_sha256 = cast("str", checkpoints[model_name]["sha256"])
        artifact = validate_latent_artifact(
            bin_path=pair_root / bin_name,
            manifest_path=supplement_path,
            run_number=1,
            model_name=typed_model,
            checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed_model],
            expected_manifest_sha256=cast("str", supplement["sha256"]),
            expected_union_sha256=cast("str", contract["base_union_sha256"]),
            validate_payload=True,
            coordinate_only_manifest=True,
        )
        record = cast("dict[str, object]", artifacts_value)[model_name]
        _validate_pair_artifact_record(artifact, record)
        validated[model_name] = artifact
    if validated["normal_vae"].manifest.rows != validated["so2_vae"].manifest.rows:
        raise ValueError(
            "Top-up model artifacts do not expose identical row identities",
        )
    validated_logical_files = _validate_logical_dataset(plan_root, plan)
    payload: dict[str, object] = {
        "schema_version": "spec0022.cancer_logical_dataset_audit.v1",
        "status": "complete",
        "plan_audit_sha256": sha256_file(
            plan_root / "spec0022_cancer_topup_plan_audit.json",
        ),
        "inference_contract_sha256": sha256_file(contract_path),
        "topup_pair_audit": {
            "path": topup_pair_audit_path.name,
            "sha256": sha256_file(topup_pair_audit_path),
        },
        "base_global_audit_sha256": cast("Mapping[str, str]", plan["inputs"])[
            "base_global_audit"
        ],
        "logical_contract_sha256": sha256_file(
            plan_root / "logical/spec0022_logical_dataset_contract.json",
        ),
        "validated_logical_files": validated_logical_files,
        "acceptance": {
            "pair_complete_and_aligned": True,
            "same_logical_files_for_both_models": True,
        },
    }
    _write_json_exclusive(destination, payload)
    return payload


def finalize_logical_audit_from_metadata(
    *,
    plan_root: Path = OUTPUT_ROOT,
    topup_metadata_root: Path,
) -> dict[str, object]:
    """Seal the logical dataset from the remote writer's compact final metadata."""
    destination = plan_root / "spec0022_logical_dataset_audit.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    plan_path = plan_root / "spec0022_cancer_topup_plan_audit.json"
    contract_path = (
        plan_root / "inference_bundle/spec0022_topup_inference_contract.json"
    )
    plan = _read_object(plan_path)
    contract = _read_object(contract_path)
    pair_path = topup_metadata_root / "spec0022_cancer_topup_pair_audit.json"
    pair = _read_object(pair_path)
    if (
        plan.get("schema_version") != "spec0022.cancer_topup_plan_audit.v1"
        or plan.get("status") != "complete"
        or contract.get("schema_version") != "spec0022.cancer_topup_inference_input.v1"
        or contract.get("status") != "complete"
    ):
        raise ValueError("Spec 0022 plan and inference contracts must be complete")
    supplement = cast("Mapping[str, object]", contract["supplement_manifest"])
    checkpoints = cast("Mapping[str, Mapping[str, object]]", contract["checkpoints"])
    if (
        pair.get("schema_version") != "spec0022.cancer_topup_pair_audit.v1"
        or pair.get("status") != "complete"
        or pair.get("supplement_manifest_sha256") != supplement["sha256"]
        or pair.get("row_count") != supplement["row_count"]
        or pair.get("base_union_sha256") != contract["base_union_sha256"]
        or pair.get("input_contract_sha256") != sha256_file(contract_path)
        or pair.get("normal_checkpoint_sha256") != checkpoints["normal_vae"]["sha256"]
        or pair.get("so2_checkpoint_sha256") != checkpoints["so2_vae"]["sha256"]
    ):
        raise ValueError(
            "Top-up pair audit does not match the frozen inference contract",
        )
    expected_files = {
        "normal_vae_mu_cancer_topup.json",
        "so2_vae_mu_cancer_topup.json",
        "spec0022_cancer_topup_pair_audit.json",
    }
    observed_files = {
        path.name for path in topup_metadata_root.iterdir() if path.is_file()
    }
    if observed_files != expected_files or any(
        not path.is_file() for path in topup_metadata_root.iterdir()
    ):
        raise ValueError("Top-up compact metadata directory differs")
    artifacts_value = pair.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise TypeError("Top-up pair artifacts must be an object")
    artifacts = cast("Mapping[str, object]", artifacts_value)
    if set(artifacts) != {
        "normal_vae",
        "so2_vae",
    }:
        raise ValueError("Top-up pair audit must bind exactly two model artifacts")
    supplement_path = (
        plan_root
        / "inference_bundle"
        / cast(
            "str",
            supplement["path"],
        )
    )
    supplement_rows = _load_supplement_identities(supplement_path)
    expected_wsi_ids = sorted({row.wsi_id for row in supplement_rows})
    sidecar_wsi_ids: list[list[int]] = []
    for model_name in ("normal_vae", "so2_vae"):
        record_value = artifacts[model_name]
        if not isinstance(record_value, dict):
            raise TypeError("Top-up pair artifact record must be an object")
        record = cast("dict[str, object]", record_value)
        sidecar_name = f"{model_name}_mu_cancer_topup.json"
        binary_name = f"{model_name}_mu_cancer_topup.bin"
        sidecar_path = topup_metadata_root / sidecar_name
        sidecar = _read_object(sidecar_path)
        checkpoint = cast("str", checkpoints[model_name]["sha256"])
        binary_bytes = LATENT_HEADER_BYTES + len(supplement_rows) * LATENT_RECORD_BYTES
        completed = sidecar.get("completed_wsi_ids")
        if (
            record.get("bin_name") != binary_name
            or record.get("bin_bytes") != binary_bytes
            or not _is_sha256(record.get("bin_sha256"))
            or record.get("sidecar_name") != sidecar_name
            or record.get("sidecar_bytes") != sidecar_path.stat().st_size
            or record.get("sidecar_sha256") != sha256_file(sidecar_path)
            or sidecar.get("schema_version") != "spec0020.latent_shard.v1"
            or sidecar.get("status") != "complete"
            or sidecar.get("model_name") != model_name
            or sidecar.get("checkpoint_sha256") != checkpoint
            or sidecar.get("pinned_union_sha256") != contract["base_union_sha256"]
            or sidecar.get("file_size") != binary_bytes
            or sidecar.get("payload_bytes") != binary_bytes - LATENT_HEADER_BYTES
            or not _is_sha256(sidecar.get("payload_sha256"))
            or not isinstance(completed, list)
            or completed != expected_wsi_ids
            or sidecar.get("completed_wsi_count") != len(expected_wsi_ids)
        ):
            raise ValueError(f"Top-up {model_name} compact metadata differs")
        source_value = sidecar.get("source_manifest")
        tensor_value = sidecar.get("tensor")
        if not isinstance(source_value, dict) or not isinstance(tensor_value, dict):
            raise TypeError(f"Top-up {model_name} sidecar contract must be objects")
        source = cast("Mapping[str, object]", source_value)
        tensor = cast("Mapping[str, object]", tensor_value)
        if (
            source.get("sha256") != supplement["sha256"]
            or source.get("row_count") != supplement["row_count"]
            or tensor.get("count") != supplement["row_count"]
            or tensor.get("dtype") != "float32_le"
            or tensor.get("layout") != "CHW"
            or tensor.get("shape") != [16, 32, 32]
            or tensor.get("record_bytes") != LATENT_RECORD_BYTES
        ):
            raise ValueError(f"Top-up {model_name} sidecar contract differs")
        sidecar_wsi_ids.append(cast("list[int]", completed))
    evidence_value = pair.get("completed_wsi_evidence")
    if not isinstance(evidence_value, list):
        raise TypeError("Top-up compact WSI completion evidence must be an array")
    evidence = cast("list[object]", evidence_value)
    if (
        sidecar_wsi_ids[0] != sidecar_wsi_ids[1]
        or len(evidence) != len(expected_wsi_ids)
        or sorted(
            _object_int(cast("Mapping[str, object]", row), "wsi_id")
            for row in evidence
            if isinstance(row, dict)
        )
        != expected_wsi_ids
    ):
        raise ValueError("Top-up compact WSI completion evidence differs")
    validated_logical_files = _validate_logical_dataset(plan_root, plan)
    payload: dict[str, object] = {
        "schema_version": "spec0022.cancer_logical_dataset_audit.v1",
        "status": "complete",
        "plan_audit_sha256": sha256_file(plan_path),
        "inference_contract_sha256": sha256_file(contract_path),
        "topup_pair_audit": {
            "path": pair_path.name,
            "sha256": sha256_file(pair_path),
        },
        "base_global_audit_sha256": cast("Mapping[str, str]", plan["inputs"])[
            "base_global_audit"
        ],
        "logical_contract_sha256": sha256_file(
            plan_root / "logical/spec0022_logical_dataset_contract.json",
        ),
        "validated_logical_files": validated_logical_files,
        "acceptance": {
            "remote_pair_audit_complete_and_aligned": True,
            "same_logical_files_for_both_models": True,
            "latent_binaries_remain_remote": True,
        },
    }
    _write_json_exclusive(destination, payload)
    return payload


def _validate_pair_artifact_record(
    artifact: LatentArtifact,
    record_value: object,
) -> None:
    if artifact.file_sha256 is None or not isinstance(record_value, dict):
        raise ValueError("Top-up pair artifact record is incomplete")
    record = cast("dict[str, object]", record_value)
    expected: dict[str, object] = {
        "bin_name": artifact.bin_path.name,
        "bin_bytes": artifact.bin_path.stat().st_size,
        "bin_sha256": artifact.file_sha256,
        "sidecar_name": artifact.sidecar_path.name,
        "sidecar_bytes": artifact.sidecar_path.stat().st_size,
        "sidecar_sha256": sha256_file(artifact.sidecar_path),
    }
    if record != expected:
        raise ValueError("Top-up pair artifact record disagrees with validated files")


def _validate_logical_dataset(
    plan_root: Path,
    plan: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    logical_root = plan_root / "logical"
    expected_names = {
        f"wsi_cancer_{split}_{kind}.csv"
        for split in SPLITS
        for kind in ("instances", "bags")
    }
    records_value = plan.get("logical_files")
    if not isinstance(records_value, dict):
        raise TypeError("Spec 0022 plan logical_files must be an object")
    records = cast("dict[str, object]", records_value)
    if set(records) != expected_names:
        raise ValueError("Spec 0022 plan must bind exactly six logical CSVs")
    contract = _read_object(
        logical_root / "spec0022_logical_dataset_contract.json",
    )
    if (
        contract.get("schema_version") != "spec0022.cancer_logical_dataset.v1"
        or contract.get("status") != "planned"
        or contract.get("files") != records
    ):
        raise ValueError("Logical contract and plan file records disagree")
    coverage_value = plan.get("coverage_by_wsi")
    if not isinstance(coverage_value, list):
        raise TypeError("Spec 0022 coverage_by_wsi must be an array")
    coverage: dict[int, dict[str, object]] = {}
    for value in cast("list[object]", coverage_value):
        if not isinstance(value, dict):
            raise TypeError("Spec 0022 WSI coverage record must be an object")
        record = cast("dict[str, object]", value)
        wsi_id = _object_int(record, "wsi_id")
        if wsi_id in coverage:
            raise ValueError(f"Duplicate Spec 0022 coverage WSI {wsi_id}")
        coverage[wsi_id] = record
    supplement = _load_supplement_identities(
        plan_root / "inference_bundle/cancer_topup_manifest.csv",
    )
    supplement_index = {identity: index for index, identity in enumerate(supplement)}
    topup_seen: set[LatentRowIdentity] = set()
    all_seen: set[LatentRowIdentity] = set()
    validated: dict[str, dict[str, object]] = {}
    split_counts: dict[str, int] = {}
    bag_wsis: set[int] = set()
    for split in SPLITS:
        instance_name = f"wsi_cancer_{split}_instances.csv"
        bag_name = f"wsi_cancer_{split}_bags.csv"
        instance_path = logical_root / instance_name
        bag_path = logical_root / bag_name
        summaries, instance_count = _validate_instance_file(
            instance_path,
            split=split,
            all_seen=all_seen,
            topup_seen=topup_seen,
            supplement_index=supplement_index,
        )
        _validate_bag_file(
            bag_path,
            split=split,
            summaries=summaries,
            coverage=coverage,
            bag_wsis=bag_wsis,
        )
        split_counts[split] = instance_count
        for name, path, count in (
            (instance_name, instance_path, instance_count),
            (bag_name, bag_path, len(summaries)),
        ):
            observed: dict[str, object] = {
                "path": path.relative_to(logical_root).as_posix(),
                "row_count": count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if records[name] != observed:
                raise ValueError(f"Logical file record changed for {name}")
            validated[name] = observed
    if topup_seen != set(supplement) or bag_wsis != set(coverage):
        raise ValueError("Logical bags do not consume the frozen WSI/supplement sets")
    selection_value = plan.get("selection")
    if not isinstance(selection_value, dict):
        raise TypeError("Spec 0022 selection audit must be an object")
    selection = cast("dict[str, object]", selection_value)
    target_counts = selection.get("target_counts")
    if target_counts != split_counts or sum(split_counts.values()) != selection.get(
        "target_total",
    ):
        raise ValueError("Logical instance totals disagree with the frozen selection")
    return validated


def _load_supplement_identities(path: Path) -> tuple[LatentRowIdentity, ...]:
    result: list[LatentRowIdentity] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUPPLEMENT_HEADER:
            raise ValueError("Unexpected supplement manifest header")
        for raw in reader:
            result.append(_csv_identity(raw, path))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate supplement identity")
    return tuple(result)


def _validate_instance_file(
    path: Path,
    *,
    split: str,
    all_seen: set[LatentRowIdentity],
    topup_seen: set[LatentRowIdentity],
    supplement_index: Mapping[LatentRowIdentity, int],
) -> tuple[list[tuple[int, str, int, int, int]], int]:
    summaries: list[tuple[int, str, int, int, int]] = []
    previous_key: tuple[int, int, int] | None = None
    active: tuple[int, str, int, int] | None = None
    count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INSTANCE_HEADER:
            raise ValueError(f"Unexpected logical instance header in {path.name}")
        for index, raw in enumerate(reader):
            if _csv_int(raw, "instance_row", path) != index or raw["split"] != split:
                raise ValueError(f"Logical instance row/split failure in {path.name}")
            identity = _csv_identity(raw, path)
            key = (identity.wsi_id, identity.y, identity.x)
            if previous_key is not None and key <= previous_key:
                raise ValueError(f"Logical instance order failure in {path.name}")
            previous_key = key
            if identity in all_seen:
                raise ValueError(f"Duplicate logical target identity {identity}")
            all_seen.add(identity)
            diagnosis_label = raw["diagnosis_label"]
            diagnosis_index = _csv_int(raw, "diagnosis_index", path)
            source = raw["store_source"]
            shard = _csv_int(raw, "shard_number", path)
            file_index = _csv_int(raw, "file_index", path)
            if source == "topup":
                if shard != 1 or supplement_index.get(identity) != file_index:
                    raise ValueError(f"Invalid top-up location in {path.name}")
                topup_seen.add(identity)
            elif source == "base":
                if shard not in range(1, 6) or file_index < 0:
                    raise ValueError(f"Invalid base location in {path.name}")
            else:
                raise ValueError(f"Invalid store source in {path.name}")
            if active is None:
                active = (identity.wsi_id, diagnosis_label, diagnosis_index, index)
            elif identity.wsi_id != active[0]:
                summaries.append((*active, index - active[3]))
                active = (identity.wsi_id, diagnosis_label, diagnosis_index, index)
            elif (diagnosis_label, diagnosis_index) != (active[1], active[2]):
                raise ValueError(f"Within-WSI label mismatch in {path.name}")
            count = index + 1
    if active is not None:
        summaries.append((*active, count - active[3]))
    return summaries, count


def _validate_bag_file(
    path: Path,
    *,
    split: str,
    summaries: Sequence[tuple[int, str, int, int, int]],
    coverage: Mapping[int, Mapping[str, object]],
    bag_wsis: set[int],
) -> None:
    cursor = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BAG_HEADER:
            raise ValueError(f"Unexpected logical bag header in {path.name}")
        rows = list(reader)
    if len(rows) != len(summaries):
        raise ValueError(f"Bag/instance WSI count mismatch in {path.name}")
    for index, (raw, summary) in enumerate(zip(rows, summaries, strict=True)):
        wsi_id, label, diagnosis_index, start, count = summary
        record = coverage.get(wsi_id)
        if (
            _csv_int(raw, "bag_row", path) != index
            or raw["split"] != split
            or _csv_int(raw, "wsi_id", path) != wsi_id
            or raw["diagnosis_label"] != label
            or _csv_int(raw, "diagnosis_index", path) != diagnosis_index
            or _csv_int(raw, "instance_start", path) != cursor
            or start != cursor
            or _csv_int(raw, "instance_count", path) != count
            or record is None
            or record.get("split") != split
            or _object_int(record, "target_count") != count
            or target_bag_size(_object_int(record, "candidate_count")) != count
            or wsi_id in bag_wsis
        ):
            raise ValueError(f"Logical bag range/metadata failure in {path.name}")
        bag_wsis.add(wsi_id)
        cursor += count


def _csv_identity(raw: Mapping[str, str], path: Path) -> LatentRowIdentity:
    return LatentRowIdentity(
        _csv_int(raw, "atlas_row_index", path),
        _csv_int(raw, "wsi_id", path),
        _csv_int(raw, "x", path),
        _csv_int(raw, "y", path),
    )


def _csv_int(raw: Mapping[str, str], key: str, path: Path) -> int:
    try:
        return int(raw[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid {key} in {path.name}") from error


def _object_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_split(path: Path) -> dict[int, SplitRow]:
    expected = (
        "wsi_id",
        "diagnosis_label",
        "diagnosis_index",
        "is_updated_image_id",
        "split",
    )
    result: dict[int, SplitRow] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected split header: {reader.fieldnames!r}")
        for raw in reader:
            wsi_id = int(raw["wsi_id"])
            split = raw["split"]
            if split not in SPLITS or wsi_id in result:
                raise ValueError(f"Invalid or duplicate split WSI {wsi_id}")
            result[wsi_id] = SplitRow(
                raw["diagnosis_label"],
                int(raw["diagnosis_index"]),
                split,
            )
    if list(result) != sorted(result):
        raise ValueError("Split rows must be ordered by WSI ID")
    return result


def _load_candidates(
    path: Path,
    split_rows: Mapping[int, SplitRow],
) -> tuple[dict[LatentRowIdentity, CandidateRow], dict[int, tuple[CandidateRow, ...]]]:
    identities: dict[LatentRowIdentity, CandidateRow] = {}
    grouped: dict[int, list[CandidateRow]] = defaultdict(list)
    previous: tuple[int, int, int] | None = None
    previous_atlas = -1
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANCER_HEADER:
            raise ValueError(
                f"Unexpected cancer candidate header: {reader.fieldnames!r}",
            )
        for index, raw in enumerate(reader):
            identity = LatentRowIdentity(
                int(raw["atlas_row_index"]),
                int(raw["wsi_id"]),
                int(raw["x"]),
                int(raw["y"]),
            )
            key = (identity.wsi_id, identity.y, identity.x)
            if (
                previous is not None and key <= previous
            ) or identity.atlas_row_index <= previous_atlas:
                raise ValueError(f"Cancer candidate order failure at row {index}")
            previous, previous_atlas = key, identity.atlas_row_index
            split = split_rows.get(identity.wsi_id)
            if split is None or (
                raw["diagnosis_label"],
                int(raw["diagnosis_index"]),
                raw["split"],
            ) != (split.diagnosis_label, split.diagnosis_index, split.split):
                raise ValueError(
                    f"Candidate metadata disagrees for WSI {identity.wsi_id}",
                )
            row = CandidateRow(
                identity,
                split.diagnosis_label,
                split.diagnosis_index,
                split.split,
            )
            if identity in identities:
                raise ValueError(f"Duplicate cancer candidate identity {identity}")
            identities[identity] = row
            grouped[identity.wsi_id].append(row)
    if set(grouped) != set(split_rows):
        raise ValueError("Cancer candidates do not cover the frozen WSI split exactly")
    return identities, {wsi: tuple(rows) for wsi, rows in grouped.items()}


def _load_existing(
    paths: Mapping[str, Path],
    candidates: Mapping[LatentRowIdentity, CandidateRow],
    split_rows: Mapping[int, SplitRow],
) -> dict[int, tuple[CandidateRow, ...]]:
    grouped: dict[int, list[CandidateRow]] = defaultdict(list)
    seen: set[LatentRowIdentity] = set()
    for expected_split in SPLITS:
        previous: tuple[int, int, int] | None = None
        with paths[expected_split].open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CANCER_HEADER:
                raise ValueError(
                    f"Unexpected old cancer header in {paths[expected_split]}",
                )
            for index, raw in enumerate(reader):
                identity = LatentRowIdentity(
                    int(raw["atlas_row_index"]),
                    int(raw["wsi_id"]),
                    int(raw["x"]),
                    int(raw["y"]),
                )
                key = (identity.wsi_id, identity.y, identity.x)
                if previous is not None and key <= previous:
                    raise ValueError(
                        f"Old cancer order failure in {expected_split} at row {index}",
                    )
                previous = key
                candidate = candidates.get(identity)
                split = split_rows.get(identity.wsi_id)
                if (
                    candidate is None
                    or split is None
                    or split.split != expected_split
                    or raw["split"] != expected_split
                ):
                    message = (
                        "Old cancer row is outside its frozen candidate split: "
                        f"{identity}"
                    )
                    raise ValueError(
                        message,
                    )
                if identity in seen:
                    raise ValueError(f"Duplicate old cancer identity {identity}")
                seen.add(identity)
                grouped[identity.wsi_id].append(candidate)
    if set(grouped) != set(split_rows):
        raise ValueError("Old cancer selections do not cover all frozen WSIs")
    return {wsi: tuple(rows) for wsi, rows in grouped.items()}


def _load_base_locations(
    *,
    union_path: Path,
    work_paths: Mapping[int, Path],
    expected_work_sha256: Mapping[int, str],
) -> dict[LatentRowIdentity, tuple[int, int, WorkManifestRow]]:
    union_rows: list[LatentRowIdentity] = []
    with union_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != UNION_HEADER:
            raise ValueError("Unexpected base union header")
        for raw in reader:
            union_rows.append(
                LatentRowIdentity(
                    int(raw["atlas_row_index"]),
                    int(raw["wsi_id"]),
                    int(raw["x"]),
                    int(raw["y"]),
                ),
            )
    result: dict[LatentRowIdentity, tuple[int, int, WorkManifestRow]] = {}
    concatenated: list[LatentRowIdentity] = []
    for run in range(1, 6):
        work = load_work_manifest(
            work_paths[run],
            run_number=run,
            expected_sha256=expected_work_sha256[run],
        )
        for file_index, row in enumerate(work.rows):
            if row.identity in result:
                raise ValueError(f"Duplicate physical base identity {row.identity}")
            result[row.identity] = (run, file_index, row)
            concatenated.append(row.identity)
    if concatenated != union_rows:
        raise ValueError(
            "Five base work manifests do not concatenate to the complete union",
        )
    return result


def _write_supplement(path: Path, rows: Sequence[CandidateRow]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUPPLEMENT_HEADER)
        for row in rows:
            identity = row.identity
            writer.writerow((
                identity.atlas_row_index,
                identity.wsi_id,
                identity.x,
                identity.y,
            ))
        handle.flush()
        os.fsync(handle.fileno())


def _write_logical_files(
    root: Path,
    targets: Sequence[CandidateRow],
    target_by_wsi: Mapping[int, Sequence[CandidateRow]],
    locations: Mapping[LatentRowIdentity, PhysicalLocation],
    split_rows: Mapping[int, SplitRow],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        instances = [row for row in targets if row.split == split]
        instance_path = root / f"wsi_cancer_{split}_instances.csv"
        bag_path = root / f"wsi_cancer_{split}_bags.csv"
        with instance_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(INSTANCE_HEADER)
            for instance_row, row in enumerate(instances):
                location = locations[row.identity]
                identity = row.identity
                writer.writerow((
                    instance_row,
                    identity.atlas_row_index,
                    identity.wsi_id,
                    identity.x,
                    identity.y,
                    row.diagnosis_label,
                    row.diagnosis_index,
                    row.split,
                    location.source,
                    location.shard_number,
                    location.file_index,
                ))
            handle.flush()
            os.fsync(handle.fileno())
        start = 0
        bag_count = 0
        with bag_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(BAG_HEADER)
            for wsi_id, metadata in split_rows.items():
                if metadata.split != split:
                    continue
                count = len(target_by_wsi[wsi_id])
                writer.writerow((
                    bag_count,
                    wsi_id,
                    metadata.diagnosis_label,
                    metadata.diagnosis_index,
                    split,
                    start,
                    count,
                ))
                start += count
                bag_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if start != len(instances):
            raise AssertionError(f"Bag ranges do not exhaust {split} instances")
        for path, count in ((instance_path, len(instances)), (bag_path, bag_count)):
            records[path.name] = {
                "path": path.relative_to(root).as_posix(),
                "row_count": count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def _coverage_record(
    candidates: Sequence[CandidateRow],
    old: Sequence[CandidateRow],
    target: Sequence[CandidateRow],
) -> dict[str, object]:
    xs = [row.identity.x for row in candidates]
    ys = [row.identity.y for row in candidates]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    dx, dy = max(1, xmax - xmin + 1), max(1, ymax - ymin + 1)

    def cells(rows: Sequence[CandidateRow]) -> set[tuple[int, int]]:
        return {
            (
                min(9, ((row.identity.x - xmin) * 10) // dx),
                min(9, ((row.identity.y - ymin) * 10) // dy),
            )
            for row in rows
        }

    occupied = cells(candidates)
    old_cells, target_cells = cells(old), cells(target)
    first = candidates[0]
    return {
        "wsi_id": first.identity.wsi_id,
        "split": first.split,
        "candidate_count": len(candidates),
        "old_count": len(old),
        "target_count": len(target),
        "occupied_cells": len(occupied),
        "old_occupied_cells": len(old_cells),
        "target_occupied_cells": len(target_cells),
        "old_coverage": len(old_cells) / len(occupied),
        "target_coverage": len(target_cells) / len(occupied),
    }


def _validate_base_global_audit(path: Path, *, expected_union_sha256: str) -> None:
    payload = _read_object(path)
    validation_value = payload.get("validation")
    validation = (
        cast("dict[str, object]", validation_value)
        if isinstance(validation_value, dict)
        else None
    )
    if (
        payload.get("schema_version") != "spec0021.latent_store_global_audit.v1"
        or payload.get("status") != "complete"
        or validation is None
        or validation.get("union_sha256") != expected_union_sha256
    ):
        raise ValueError("Completed Spec 0021 global audit is missing or mismatched")
    artifacts_value = payload.get("artifacts")
    if not isinstance(artifacts_value, dict) or set(
        cast("dict[str, object]", artifacts_value),
    ) != {
        "normal_vae",
        "so2_vae",
    }:
        raise ValueError("Spec 0021 global audit does not bind both base model stores")


def _project_deadline(
    *,
    pair_audit_paths: Mapping[int, Path],
    run_log_paths: Mapping[int, Path],
    supplement_rows: int,
    supplement_wsi_count: int,
) -> dict[str, object]:
    """Conservatively project the one-off job from all completed base runs."""
    if supplement_rows < 1 or supplement_wsi_count < 1:
        raise ValueError("The one-off top-up must contain rows from at least one WSI")
    evidence: list[dict[str, object]] = []
    seconds_per_row: list[float] = []
    seconds_per_wsi: list[float] = []
    for run in range(1, 6):
        audit = _read_object(pair_audit_paths[run])
        completed_value = audit.get("completed_wsi_evidence")
        row_count = audit.get("row_count")
        if (
            audit.get("schema_version") != "spec0021.latent_pair_audit.v1"
            or audit.get("status") != "complete"
            or audit.get("run_number") != run
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
            or not isinstance(completed_value, list)
            or not completed_value
        ):
            raise ValueError(f"Spec 0021 run {run:02d} evidence is incomplete")
        completed = cast("list[object]", completed_value)
        log_value = cast(
            "object",
            json.loads(run_log_paths[run].read_text(encoding="utf-8")),
        )
        if not isinstance(log_value, list) or not log_value:
            raise ValueError(f"Spec 0021 run {run:02d} log is malformed")
        log = cast("list[object]", log_value)
        final_record = log[-1]
        if not isinstance(final_record, dict):
            raise TypeError(f"Spec 0021 run {run:02d} log is malformed")
        final_time = cast("dict[str, object]", final_record).get("time")
        if (
            isinstance(final_time, bool)
            or not isinstance(final_time, (int, float))
            or final_time <= 0
        ):
            raise ValueError(f"Spec 0021 run {run:02d} log lacks a final elapsed time")
        elapsed = float(final_time)
        wsi_count = len(completed)
        seconds_per_row.append(elapsed / row_count)
        seconds_per_wsi.append(elapsed / wsi_count)
        evidence.append({
            "run_number": run,
            "row_count": row_count,
            "wsi_count": wsi_count,
            "elapsed_seconds": elapsed,
            "pair_audit_sha256": sha256_file(pair_audit_paths[run]),
            "run_log_sha256": sha256_file(run_log_paths[run]),
        })
    row_projection = supplement_rows * max(seconds_per_row)
    wsi_projection = supplement_wsi_count * max(seconds_per_wsi)
    worst_wsi_seconds = math.ceil(max(seconds_per_wsi))
    inference_seconds = math.ceil(max(row_projection, wsi_projection))
    projected_total = inference_seconds + VALIDATION_RESERVE_SECONDS
    if projected_total > SESSION_LIMIT_SECONDS:
        raise ValueError(
            "Top-up deadline projection plus validation reserve exceeds one "
            "Kaggle session",
        )
    return {
        "method": "max_observed_seconds_per_row_or_wsi_v1",
        "source_runs": evidence,
        "supplement_rows": supplement_rows,
        "supplement_wsi_count": supplement_wsi_count,
        "row_projection_seconds": math.ceil(row_projection),
        "wsi_projection_seconds": math.ceil(wsi_projection),
        "worst_observed_seconds_per_wsi": worst_wsi_seconds,
        "projected_inference_seconds": inference_seconds,
        "validation_reserve_seconds": VALIDATION_RESERVE_SECONDS,
        "projected_total_seconds": projected_total,
        "session_limit_seconds": SESSION_LIMIT_SECONDS,
        "fits": True,
    }


def _validate_checkpoint(path: Path, expected: str, model: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{model} checkpoint SHA-256 mismatch: {observed}")
    return observed


def _validate_optional_hash(path: Path, expected: str | None, name: str) -> None:
    if expected is not None:
        validate_pinned_hash(path, expected, name)


def _require_exact_keys[K](
    mapping: Mapping[K, object],
    keys: Sequence[K],
    name: str,
) -> None:
    if set(mapping) != set(keys):
        raise ValueError(f"{name} must contain exactly {list(keys)!r}")


def _artifact_records(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_object(path: Path) -> dict[str, object]:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", raw)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    _fsync_file(path)


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n",
        encoding="utf-8",
    )
    _fsync_file(path)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the one frozen top-up plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize-from-metadata", type=Path)
    args = parser.parse_args(argv)
    metadata_root = cast("Path | None", args.finalize_from_metadata)
    if metadata_root is not None:
        result = finalize_logical_audit_from_metadata(
            topup_metadata_root=metadata_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if cast("bool", args.validate_only):
        if not GLOBAL_AUDIT_PATH.is_file():
            raise FileNotFoundError(
                "Spec 0021 CPU finalizer/global audit is still required before "
                "sealing Spec 0022",
            )
        if not OUTPUT_ROOT.is_dir():
            raise FileNotFoundError("The sealed Spec 0022 top-up plan does not exist")
        plan = _read_object(OUTPUT_ROOT / "spec0022_cancer_topup_plan_audit.json")
        if plan.get("status") != "complete":
            raise ValueError("Spec 0022 plan audit is not complete")
        print(json.dumps(plan["selection"], indent=2, sort_keys=True))
        return 0
    materialize_cancer_topup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
