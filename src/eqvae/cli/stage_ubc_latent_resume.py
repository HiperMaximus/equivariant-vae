# Copyright 2026 HiperMaximus
"""Stage one immutable Spec 0021 run-specific resume dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.inference.input_bundle import stage_resume_bundle

if TYPE_CHECKING:
    from collections.abc import Sequence

RESUME_RECEIPT_SCHEMA = "spec0021.resume_dataset_receipt.v1"


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and stage one run's exact accepted recovery window.

    Returns:
        Zero after the staged bundle passes its complete validation.

    """
    args = _parser().parse_args(argv)
    run_number = cast("int", args.run_number)
    artifacts_dir = Path(cast("str", args.artifacts_dir)).resolve()
    if (artifacts_dir / "dataset").is_dir():
        artifacts_dir /= "dataset"
    input_contract = Path(cast("str", args.input_contract)).resolve()
    run_config = Path(cast("str", args.run_config)).resolve()
    prior_receipt_path = Path(cast("str", args.prior_receipt)).resolve()
    dataset_slug = f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run_number:02d}-resume"
    dataset_version = _next_version(prior_receipt_path, dataset_slug, run_number)
    destination_root = Path(cast("str", args.output_root)).resolve()
    destination = (
        destination_root / f"run_{run_number:02d}" / f"version_{dataset_version:04d}"
    )
    authority = stage_resume_bundle(
        destination,
        artifacts_dir=artifacts_dir,
        work_manifest_path=(
            input_contract.parent
            / "manifests"
            / "work_shards"
            / f"run_{run_number:02d}_of_05.csv"
        ),
        dataset_slug=dataset_slug,
        dataset_version=dataset_version,
        run_number=run_number,
        input_bundle_sha256=_sha256(input_contract),
        run_config_sha256=_sha256(run_config),
    )
    sys.stdout.write(
        f"{
            json.dumps(
                {
                    'dataset_reference': authority.dataset_slug,
                    'dataset_version': authority.dataset_version,
                    'provenance_sha256': authority.provenance_sha256,
                    'run_config_sha256': authority.run_config_sha256,
                    'staged_directory': str(destination),
                },
                sort_keys=True,
            )
        }\n",
    )
    return 0


def _next_version(receipt_path: Path, dataset_slug: str, run_number: int) -> int:
    if not receipt_path.is_file():
        return 1
    payload = cast("object", json.loads(receipt_path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = "Prior resume receipt must contain a JSON object"
        raise TypeError(message)
    payload = cast("dict[str, object]", payload)
    if (
        payload.get("schema_version") != RESUME_RECEIPT_SCHEMA
        or payload.get("dataset_reference") != dataset_slug
        or payload.get("run_number") != run_number
    ):
        message = "Prior resume receipt identity mismatch"
        raise ValueError(message)
    version = payload.get("dataset_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        message = "Prior resume receipt version must be positive"
        raise ValueError(message)
    return version + 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_number", type=int, choices=range(1, 6))
    parser.add_argument("artifacts_dir")
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--prior-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
