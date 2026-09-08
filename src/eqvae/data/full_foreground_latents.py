# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: DOC201, DOC501, EM101, EM102, TRY003
"""Development-only full-foreground bags over the existing physical reader."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final, Self, cast

from eqvae.data.supervised_latents import (
    SupervisedLatentStore,
    WSIBagDataset,
    _load_wsi_bags,
    _load_wsi_instances,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from eqvae.data.latent_shards import ModelName
    from eqvae.data.supervised_latents import LearningSplit

DEVELOPMENT_FILES: Final = frozenset({
    "physical_parts.csv",
    "wsi_cancer_train_instances.csv",
    "wsi_cancer_train_bags.csv",
    "wsi_cancer_validation_instances.csv",
    "wsi_cancer_validation_bags.csv",
})


class FullForegroundBagDataset(WSIBagDataset):
    """Keep sealed test mappings outside the supported learning entry point."""

    def __init__(
        self,
        *,
        development_root: Path,
        expected_contract_sha256: str,
        split: LearningSplit,
        model_name: ModelName,
        source_roots: Mapping[str, Path],
    ) -> None:
        """Authenticate membership before constructing any mounted physical store."""
        if split not in {"train", "validation"}:
            raise ValueError("Only train/validation are available; test is sealed")
        contract_path = development_root / "dataset.json"
        if _sha256(contract_path) != expected_contract_sha256:
            raise ValueError("Development contract differs from its external hash pin")
        contract = cast("dict[str, object]", json.loads(contract_path.read_text()))
        files = cast("dict[str, dict[str, object]]", contract["files"])
        if (
            contract.get("schema_version") != "spec0025.full_foreground_development.v1"
            or contract.get("test_release") != "not_authorized"
            or frozenset(files) != DEVELOPMENT_FILES
            or frozenset(p.name for p in development_root.iterdir())
            != DEVELOPMENT_FILES | {"dataset.json"}
        ):
            raise ValueError("Development-only file allow-list differs")
        for name, record in files.items():
            path = development_root / name
            if (
                path.stat().st_size != record["bytes"]
                or _sha256(path) != record["sha256"]
            ):
                raise ValueError(f"Development file identity differs: {name}")
        self.instances = _load_wsi_instances(
            development_root / f"wsi_cancer_{split}_instances.csv",
        )
        self.bags = _load_wsi_bags(
            development_root / f"wsi_cancer_{split}_bags.csv",
            self.instances,
        )
        expected_bags = cast("dict[str, dict[str, int]]", contract["bags"])
        if (
            set(expected_bags) != {"train", "validation"}
            or set(expected_bags["train"]) & set(expected_bags["validation"])
            or any(row.split != split for row in self.instances)
            or not self.bags
            or {str(bag.wsi_id): bag.instance_count for bag in self.bags}
            != expected_bags[split]
        ):
            raise ValueError(
                "Full-foreground bags differ from the requested frozen split",
            )
        self.store = SupervisedLatentStore(
            catalog_path=development_root / "physical_parts.csv",
            model_name=model_name,
            source_roots=source_roots,
        )

    def __enter__(self) -> Self:
        """Return the guarded dataset with complete-bag read behavior unchanged."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the reused store's process-local binary mappings."""
        self.store.close()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
