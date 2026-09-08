# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0916, PLR2004, SLF001, T201, TRY003, TRY301
"""Join the fifteen completed latent pairs into reference-only full MIL views."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import shutil
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.cli import build_ubc_full_foreground_completion as completion
from eqvae.cli import build_ubc_supervised_manifests as legacy
from eqvae.cli import build_ubc_wsi45630_completion as prior
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_WORK_MANIFEST_SHA256,
    ModelName,
)

if TYPE_CHECKING:
    from _csv import Writer
    from collections.abc import Iterator, Mapping, Sequence

OUTPUT_ROOT: Final = Path("runs/local/ubc_ocean_full_foreground_manifests")
COMPLETION_SHA256: Final = (
    "5f21633d42cce71955ba3f6f0cf9c2dbd5a6d4b5b19b8339e745775d1fe76aae"
)
COVERAGE_PATH: Final = Path("runs/local/full_foreground_atlas_coverage_check.json")
COVERAGE_SHA256: Final = (
    "6eb4a0de6c012a56b1eaf5c42c553eec2ece28293257c0d0b172ba1c4d8c7139"
)
TEST_INSTANCE_HEADER: Final = tuple(
    name for name in legacy.WSI_INSTANCE_HEADER if not name.startswith("diagnosis_")
)
TEST_BAG_HEADER: Final = tuple(
    name for name in legacy.WSI_BAG_HEADER if not name.startswith("diagnosis_")
)
type Identity = tuple[int, int, int, int]
type LocatedRow = tuple[Identity, int, int]


@dataclass(frozen=True)
class SourcePart:
    """One authenticated physical manifest; row positions are binary indices."""

    part: int
    manifest: Path
    manifest_sha256: str
    count: int
    metadata_root: Path


def _pinned_object(path: Path, digest: str) -> dict[str, object]:
    prior._require_hash(path, digest)
    return legacy._read_object(path)


def _append_pair(
    rows: list[tuple[object, ...]],
    pair: Mapping[str, object],
    part: int,
    producer: str,
) -> None:
    artifacts = legacy._mapping(pair, "artifacts")
    count = legacy._integer(pair, "row_count")
    for model in legacy.MODELS:
        record = legacy._mapping(artifacts, model)
        rows.append((
            part,
            model,
            producer,
            record["bin_name"],
            count,
            record["bin_bytes"],
            record["bin_sha256"],
            record["sidecar_name"],
            record["sidecar_bytes"],
            record["sidecar_sha256"],
        ))


def _validate_source(source: SourcePart, rows: Sequence[tuple[object, ...]]) -> None:
    prior._require_hash(source.manifest, source.manifest_sha256)
    pair_rows = [
        dict(zip(legacy.CATALOG_HEADER, r, strict=True))
        for r in rows
        if r[0] == source.part
    ]
    if len(pair_rows) != 2 or {r["model_name"] for r in pair_rows} != set(
        legacy.MODELS,
    ):
        raise ValueError("Source must resolve exactly one aligned model pair")
    if len({r["kaggle_source"] for r in pair_rows}) != 1:
        raise ValueError("Paired source producers differ")
    for record in pair_rows:
        sidecar_path = source.metadata_root / str(record["sidecar_name"])
        sidecar = _pinned_object(sidecar_path, str(record["sidecar_sha256"]))
        source_metadata = legacy._mapping(sidecar, "source_manifest")
        if (
            sidecar_path.stat().st_size != record["sidecar_bytes"]
            or sidecar.get("model_name") != record["model_name"]
            or sidecar.get("checkpoint_sha256")
            != EXPECTED_CHECKPOINT_SHA256[cast("ModelName", record["model_name"])]
            or sidecar.get("status") != "complete"
            or source_metadata.get("sha256") != source.manifest_sha256
            or source_metadata.get("logical_basename") != source.manifest.name
            or source_metadata.get("row_count") != source.count
            or record["row_count"] != source.count
            or record["binary_bytes"] != 64 + 65_536 * source.count
            or sidecar.get("file_size") != record["binary_bytes"]
            or legacy._mapping(sidecar, "tensor")
            != {
                "count": source.count,
                "dtype": "float32_le",
                "layout": "CHW",
                "shape": [16, 32, 32],
                "record_bytes": 65_536,
            }
        ):
            raise ValueError("Source-qualified manifest/sidecar pairing differs")


def _load_sources(
    repo: Path,
    audit: Mapping[str, object],
    plan: Mapping[str, object],
) -> tuple[list[SourcePart], list[tuple[object, ...]]]:
    pins = legacy._mapping(plan, "inputs")
    base = _pinned_object(repo / legacy.BASE_AUDIT, str(pins["base_global_audit"]))
    topup = _pinned_object(
        repo / prior.PAIR_AUDIT_PATH,
        str(pins["completed_pair_audit"]),
    )
    rows, counts = legacy._catalog_rows(base, topup)
    sources = [
        SourcePart(
            part,
            repo / legacy.BASE_WORK_ROOT / f"run_{part:02d}_of_05.csv",
            EXPECTED_WORK_MANIFEST_SHA256[part],
            counts[part],
            repo / f"runs/kaggle/ubc_ocean_latents/run_{part:02d}_metadata/dataset",
        )
        for part in range(1, 6)
    ]
    sources.append(
        SourcePart(
            11,
            repo / prior.OLD_MANIFEST_PATH,
            str(pins["completed_part11_manifest"]),
            counts[11],
            repo / prior.PAIR_AUDIT_PATH.parent,
        ),
    )
    largest = _pinned_object(
        repo / completion.COMPLETION_EVIDENCE / "dataset" / completion.PAIR_NAME,
        str(pins["completed_wsi45630_pair"]),
    )
    _append_pair(rows, largest, 12, "maximusshtefan/eqvae-wsi45630-completion")
    sources.append(
        SourcePart(
            12,
            repo / completion.COMPLETION_ROOT / completion.MANIFEST_NAME,
            str(pins["completed_wsi45630_manifest"]),
            legacy._integer(largest, "row_count"),
            repo / completion.COMPLETION_EVIDENCE / "dataset",
        ),
    )
    pairs = cast("list[dict[str, object]]", audit["pairs"])
    if len(pairs) != 8:
        raise ValueError("Eight authenticated completion pairs are required")
    for number, pair in enumerate(pairs, 1):
        producer = f"maximusshtefan/eqvae-full-foreground-{number:02d}"
        if pair["producer"] != producer:
            raise ValueError("Completion producer order differs")
        _append_pair(rows, pair, number + 12, producer)
        sources.append(
            SourcePart(
                number + 12,
                repo
                / completion.OUTPUT_ROOT
                / f"bundle/runs/run_{number:02d}"
                / completion.MANIFEST_NAME,
                str(pair["manifest_sha256"]),
                legacy._integer(pair, "row_count"),
                repo
                / f"runs/kaggle/full_foreground_completion/run_{number:02d}/dataset",
            ),
        )
    for source in sources:
        _validate_source(source, rows)
    return sources, rows


def _coordinates(path: Path) -> Iterator[tuple[Identity, str | None]]:
    previous: Identity | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            identity = cast(
                "Identity",
                tuple(int(raw[k]) for k in prior.MANIFEST_HEADER),
            )
            if previous is not None and (
                _key(identity) <= _key(previous) or identity[0] <= previous[0]
            ):
                raise ValueError(
                    "Coordinates must have unique numeric wsi_id,y,x order",
                )
            previous = identity
            yield identity, raw.get("split")


def _key(identity: Identity) -> tuple[int, int, int]:  # noqa: FURB118
    return identity[1], identity[3], identity[2]


def _physical_rows(
    source: SourcePart,
    splits: Mapping[int, str],
) -> Iterator[LocatedRow]:
    count = 0
    for index, (identity, split) in enumerate(_coordinates(source.manifest)):
        if identity[1] not in splits or (
            split is not None and split != splits[identity[1]]
        ):
            raise ValueError("Physical manifest differs from the frozen WSI split")
        yield identity, source.part, index
        count += 1
    if count != source.count:
        raise ValueError("Physical manifest row count differs from its paired catalog")


def resolve_foreground(
    sources: Sequence[SourcePart],
    candidate: Path,
    splits: Mapping[int, str],
    excluded: Counter[int],
) -> Iterator[LocatedRow]:
    """Merge without renumbering physical gaps, dropping tails or reading labels.

    Yields:
        Exact target identity, physical part and original binary record index.

    """
    targets = iter(_coordinates(candidate))
    target = next(targets, None)
    previous: Identity | None = None
    merged = heapq.merge(
        *(_physical_rows(s, splits) for s in sources),
        key=lambda r: _key(r[0]),
    )
    for identity, part, index in merged:
        if previous is not None and (
            _key(identity) <= _key(previous) or identity[0] <= previous[0]
        ):
            raise ValueError("Stored source overlap or atlas identity drift")
        previous = identity
        if target is not None and _key(identity) > _key(target[0]):
            raise ValueError("Missing foreground coordinate")
        if target is not None and _key(identity) == _key(target[0]):
            if identity != target[0]:
                raise ValueError("Stored/target atlas identity drift")
            if target[1] != splits.get(identity[1]):
                raise ValueError("Candidate differs from the frozen WSI split")
            yield identity, part, index
            target = next(targets, None)
        elif part in range(1, 6):
            excluded[part] += 1
        else:
            raise ValueError(
                "Supplement contains an unplanned nonforeground coordinate",
            )
    if target is not None:
        raise ValueError("Missing foreground coordinate at end of inventory")


def write_views(
    *,
    output: Path,
    sources: Sequence[SourcePart],
    candidate: Path,
    split_path: Path,
    expected_per_wsi: Mapping[int, int],
) -> dict[str, object]:
    """Write development bags plus separate label-free reserved test mappings."""
    splits = legacy._load_split(split_path)
    labels: dict[int, tuple[str, int]] = {}
    with split_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["split"] in {"train", "validation"}:
                labels[int(raw["wsi_id"])] = (
                    raw["diagnosis_label"],
                    int(raw["diagnosis_index"]),
                )
    excluded: Counter[int] = Counter()
    counts: Counter[str] = Counter()
    per_wsi: Counter[int] = Counter()
    bag_rows: dict[str, list[tuple[object, ...]]] = {s: [] for s in legacy.SPLITS}
    with ExitStack() as stack:
        writers: dict[str, Writer] = {}
        for split in legacy.SPLITS:
            folder = output / ("sealed_test" if split == "test" else "development")
            folder.mkdir(exist_ok=True)
            handle = stack.enter_context(
                (folder / f"wsi_cancer_{split}_instances.csv").open(
                    "x",
                    encoding="utf-8",
                    newline="",
                ),
            )
            writers[split] = csv.writer(handle, lineterminator="\n")
            writers[split].writerow(
                TEST_INSTANCE_HEADER if split == "test" else legacy.WSI_INSTANCE_HEADER,
            )
        for identity, part, index in resolve_foreground(
            sources,
            candidate,
            splits,
            excluded,
        ):
            wsi = identity[1]
            split = splits[wsi]
            label = () if split == "test" else labels[wsi]
            if wsi not in per_wsi:
                bag_rows[split].append((
                    len(bag_rows[split]),
                    wsi,
                    *label,
                    split,
                    counts[split],
                ))
            writers[split].writerow((
                counts[split],
                *identity,
                *label,
                split,
                part,
                index,
            ))
            counts[split] += 1
            per_wsi[wsi] += 1
    if dict(per_wsi) != dict(expected_per_wsi) or set(per_wsi) != set(splits):
        raise ValueError(
            "Per-WSI full-foreground coverage differs from independent audit",
        )
    for split, bags in bag_rows.items():
        folder = output / ("sealed_test" if split == "test" else "development")
        legacy._write_csv(
            folder / f"wsi_cancer_{split}_bags.csv",
            TEST_BAG_HEADER if split == "test" else legacy.WSI_BAG_HEADER,
            ((*row, per_wsi[cast("int", row[1])]) for row in bags),
        )
    return {
        "split_rows": dict(counts),
        "split_wsis": {s: len(b) for s, b in bag_rows.items()},
        "per_wsi": {
            str(w): {"split": splits[w], "rows": n} for w, n in per_wsi.items()
        },
        "excluded_base_rows": sum(excluded.values()),
    }


def seal_development(output: Path, summary: Mapping[str, object]) -> dict[str, object]:
    """Pin exactly the development files without exposing test identities or labels."""
    dev = output / "development"
    files = {
        p.name: legacy._file_record(p, sum(1 for _ in p.open()) - 1)
        for p in sorted(dev.glob("*.csv"))
    }
    per_wsi = legacy._mapping(summary, "per_wsi")
    contract: dict[str, object] = {
        "schema_version": "spec0025.full_foreground_development.v1",
        "completion_audit_sha256": COMPLETION_SHA256,
        "files": files,
        "bags": {
            s: {
                w: legacy._mapping(per_wsi, w)["rows"]
                for w in per_wsi
                if legacy._mapping(per_wsi, w)["split"] == s
            }
            for s in ("train", "validation")
        },
        "test_release": "not_authorized",
    }
    legacy._write_json(dev / "dataset.json", contract)
    return contract


def build(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Build only local metadata from pinned completed evidence, never binary data."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    completed_root = repo_root / completion.OUTPUT_ROOT
    audit = _pinned_object(completed_root / "completion_audit.json", COMPLETION_SHA256)
    coverage = _pinned_object(repo_root / COVERAGE_PATH, COVERAGE_SHA256)
    plan = _pinned_object(completed_root / "plan.json", str(audit["plan_sha256"]))
    sources, catalog = _load_sources(repo_root, audit, plan)
    pins = legacy._mapping(plan, "inputs")
    candidate, split = repo_root / prior.CANDIDATE_PATH, repo_root / legacy.SPLIT_PATH
    prior._require_hash(candidate, str(pins["candidate_manifest"]))
    prior._require_hash(split, str(pins["frozen_split"]))
    expected = {row[0]: row[2] for row in cast("list[list[int]]", coverage["per_wsi"])}
    staging = output_root.with_name(f".{output_root.name}.building")
    staging.mkdir(parents=True, exist_ok=False)
    try:  # noqa: PLW0717
        summary = write_views(
            output=staging,
            sources=sources,
            candidate=candidate,
            split_path=split,
            expected_per_wsi=expected,
        )
        if (
            summary["split_rows"]
            != {"train": 1_224_875, "validation": 264_178, "test": 261_168}
            or summary["split_wsis"] != {"train": 106, "validation": 23, "test": 23}
            or summary["excluded_base_rows"] != 23_023
        ):
            raise ValueError("Full-foreground integration totals differ")
        legacy._write_csv(
            staging / "development/physical_parts.csv",
            legacy.CATALOG_HEADER,
            catalog,
        )
        seal_development(staging, summary)
        result: dict[str, object] = {
            "schema_version": "spec0025.full_foreground_integration.v1",
            "status": "complete",
            "completion_audit_sha256": COMPLETION_SHA256,
            "coverage_audit_sha256": COVERAGE_SHA256,
            "plan_sha256": audit["plan_sha256"],
            "candidate_sha256": pins["candidate_manifest"],
            "split_sha256": pins["frozen_split"],
            "coverage": summary,
            "sources": [
                {
                    "part": s.part,
                    "manifest": str(s.manifest.relative_to(repo_root)),
                    "manifest_sha256": s.manifest_sha256,
                    "rows": s.count,
                    "producer": next(r[2] for r in catalog if r[0] == s.part),
                    "producer_version": 1,
                }
                for s in sources
            ],
            "files": {
                str(p.relative_to(staging)): {
                    "bytes": p.stat().st_size,
                    "sha256": legacy._sha256(p),
                }
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
            "test_release": "not_authorized",
            "latent_payloads_read": False,
            "existing_tissue_and_quarter_views_modified": False,
        }
        legacy._write_json(staging / "integration_audit.json", result)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Create the fixed reference dataset locally without any remote action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = build(
        repo_root=cast("Path", args.repo_root),
        output_root=cast("Path", args.output_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
