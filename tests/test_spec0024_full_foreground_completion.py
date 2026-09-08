# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
"""Focused invariants for the eight missing-only paired extraction packages."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from eqvae.cli import build_ubc_full_foreground_completion as builder
from eqvae.cli import build_ubc_wsi45630_completion as prior
from eqvae.cli import generate_ubc_cancer_topup as worker_module

if TYPE_CHECKING:
    from collections.abc import Callable


def test_full_foreground_subtracts_all_stored_rows_and_rejects_identity_drift() -> None:
    """Reuse stored-but-unselected foreground while keeping nonforeground out.

    This DERIVED set difference prevents duplicate encoding without allowing
    physical storage membership to change the independently frozen target.
    """
    rows = tuple((i, 10, i * 256, 0) for i in range(5))
    sources = {
        "base": (rows[0], rows[2], (9, 10, 2304, 0)),
        "part11": (rows[1],),
        "completed": (rows[3],),
    }
    missing, reused = builder.derive_missing(rows, sources)
    assert missing == (rows[4],)
    assert reused == {"base": 2, "part11": 1, "completed": 1}
    with pytest.raises(ValueError, match="overlap"):
        builder.derive_missing(rows, {**sources, "duplicate": (rows[2],)})
    with pytest.raises(ValueError, match="identity drift"):
        builder.derive_missing(rows, {"wrong_atlas": ((99, 10, 0, 0),)})
    with pytest.raises(ValueError, match="identity drift"):
        builder.derive_missing(rows, {"wrong_coordinate": ((0, 10, 256, 0),)})


def test_full_foreground_parts_preserve_whole_wsi_numeric_order_and_offsets(
    tmp_path: Path,
) -> None:
    """Whole-WSI partitions concatenate exactly and physical indices restart per part.

    This DERIVED relationship catches WSI splits, lexicographic IDs, dropped
    rows and offset drift before any paired encoder or mmap can use them.
    """
    rows: list[builder.Identity] = []
    for wsi, count in ((1, 4), (2, 1), (10, 7), (30, 2)):
        for i in range(count):
            rows.append((len(rows), wsi, i * 256, 0))
    parts = builder.partition_missing(rows, count=2)
    assert tuple(map(len, parts)) == (5, 9)
    assert tuple(row for part in parts for row in part) == tuple(rows)
    assert {row[1] for row in parts[0]}.isdisjoint(row[1] for row in parts[1])
    for number, part in enumerate(parts):
        path = tmp_path / f"{number}.csv"
        prior._write_manifest(path, part)  # noqa: SLF001
        observed = builder.load_coordinates(path, expected_header=prior.MANIFEST_HEADER)
        assert tuple(enumerate(observed)) == tuple(enumerate(part))
    path = tmp_path / "split.csv"
    path.write_text(
        "atlas_row_index,wsi_id,x,y,split\n0,10,0,0,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split"):
        builder.load_coordinates(path, splits={10: "train"})
    with pytest.raises(ValueError, match="numeric"):
        builder.partition_missing((rows[5], rows[0]), count=2)


def test_full_foreground_deadline_uses_largest_wsi_and_saved_output_cap() -> None:
    """Budget actual largest missing WSI plus reserve, not an average-slide bound.

    The elapsed values are MEASURED completed-run evidence; the resulting
    relationships enforce the locked output/session policy, not a benchmark.
    """
    timings: list[dict[str, object]] = [
        {"row_count": 74_033, "wsi_count": 65, "elapsed_seconds": 11119.359830337},
        {"row_count": 118_513, "wsi_count": 29, "elapsed_seconds": 10209.547744842},
        {"row_count": 22_649, "wsi_count": 1, "elapsed_seconds": 2408.269866317},
    ]
    result = builder.project_job(135_184, 16, 20_924, timings)
    expected_bound, expected_inference, expected_total = 3143, 23_447, 27_047
    assert result["worst_observed_seconds_per_wsi"] == expected_bound
    assert result["projected_inference_seconds"] == expected_inference
    assert result["projected_total_seconds"] == expected_total
    with pytest.raises(ValueError, match="saved-output or session"):
        builder.project_job(152_512, 16, 20_924, timings)
    with pytest.raises(ValueError, match="saved-output or session"):
        builder.project_job(135_184, 75, 20_924, timings)


def test_full_foreground_thin_wrapper_isolates_one_run_and_shares_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the selected four label-free files reach the unchanged paired worker.

    This DELIBERATE isolation catches cross-run manifest/contract substitution,
    extra labelled inputs, or checkpoint duplication inside the inference root.
    """
    bundle = tmp_path / "bundle"
    for number in (1, 2):
        root = bundle / f"runs/run_{number:02d}"
        root.mkdir(parents=True)
        prior._write_manifest(root / builder.MANIFEST_NAME, ((number, number, 0, 0),))  # noqa: SLF001
        prior._write_canonical_json(  # noqa: SLF001
            root / builder.WORKER_CONTRACT_NAME,
            {
                "schema_version": "spec0022.cancer_topup_inference_input.v1",
                "status": "complete",
                "completion_run_number": number,
                "producer": f"fixture/part-{number:02d}",
            },
        )
    (bundle / "checkpoints").mkdir()
    for model in ("normal_vae", "so2_vae"):
        (bundle / f"checkpoints/{model}_step_060000.pt").write_bytes(b"fixture")
    contract = bundle / builder.CONTRACT_NAME
    prior._write_canonical_json(  # noqa: SLF001
        contract,
        {
            "dataset_reference": builder.DATASET_REFERENCE,
            "files": prior._artifact_records(bundle, exclude=set()),  # noqa: SLF001
        },
    )
    selected = bundle / "runs/run_02"
    config: dict[str, object] = {
        "run_number": 2,
        "producer": "fixture/part-02",
        "row_count": 1,
        "input_contract_sha256": prior._sha256(contract),  # noqa: SLF001
        "manifest_sha256": prior._sha256(selected / builder.MANIFEST_NAME),  # noqa: SLF001
        "worker_contract_sha256": prior._sha256(  # noqa: SLF001
            selected / builder.WORKER_CONTRACT_NAME,
        ),
    }
    code = builder._kernel_bytes(Path(__file__).parents[1], config)  # noqa: SLF001
    upload_limit = 1_000_000
    assert len(code) < upload_limit
    path = tmp_path / "wrapper.py"
    path.write_bytes(code)
    spec = importlib.util.spec_from_file_location("completion_fixture", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKING_ROOT", tmp_path)
    monkeypatch.setattr(module, "INPUT_ROOT", bundle)
    resolve = cast("Callable[[], Path]", module._resolve_bundle)  # noqa: SLF001
    assert resolve() == bundle
    monkeypatch.setattr(module, "CONFIG", {**config, "run_number": 1})
    with pytest.raises(RuntimeError, match="Per-run manifest"):
        resolve()
    monkeypatch.setattr(module, "CONFIG", config)
    labelled = bundle / "test_labels.csv"
    labelled.write_text("wsi_id,label\n2,hidden\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="allow-list"):
        resolve()
    labelled.unlink()
    calls: list[Path] = []

    def worker(**kwargs: Path) -> dict[str, object]:
        root = kwargs["input_root"]
        worker_module._validate_input_tree(root, root / builder.WORKER_CONTRACT_NAME)  # noqa: SLF001
        assert builder.load_coordinates(root / builder.MANIFEST_NAME) == ((2, 2, 0, 0),)
        assert all(path.is_symlink() for path in root.iterdir())
        assert (root / "normal_vae_step_060000.pt").resolve() == (
            bundle / "checkpoints/normal_vae_step_060000.pt"
        ).resolve()
        calls.append(root)
        return {"status": "complete", "row_count": 1}

    monkeypatch.setattr("eqvae.cli.generate_ubc_cancer_topup.run_topup", worker)
    execute = cast("Callable[[Path, Path], None]", module._execute)  # noqa: SLF001
    execute(bundle, tmp_path / "images")
    assert len(calls) == 1
    assert not calls[0].exists()
    assert (bundle / "checkpoints/normal_vae_step_060000.pt").read_bytes() == b"fixture"


def test_full_foreground_completion_rejects_foreign_sidecars_and_changed_png(
    tmp_path: Path,
) -> None:
    """Producer contract, sidecar identities, and original PNG bytes must agree.

    This DELIBERATE provenance guard rejects same-filename different-source
    outputs and changed images without opening any latent binary.
    """
    manifest, contract, metadata, pngs = _pair_fixture(tmp_path)

    def verify() -> dict[str, object]:
        return builder.verify_pair_metadata(
            metadata_root=metadata,
            manifest=manifest,
            contract_path=contract,
            expected_pngs=pngs,
            producer="fixture/run-01",
        )

    result = verify()
    assert result["row_count"] == len(builder.load_coordinates(manifest))
    pair_path = metadata / builder.PAIR_NAME
    pair = prior._read_object(pair_path)  # noqa: SLF001
    original = pair_path.read_bytes()
    pair["input_contract_sha256"] = "f" * 64
    prior._write_canonical_json(pair_path, pair)  # noqa: SLF001
    with pytest.raises(ValueError, match="producer/contract"):
        verify()
    pair_path.write_bytes(original)
    sidecar = metadata / "normal_vae_mu_cancer_topup.json"
    saved = sidecar.read_bytes()
    payload = prior._read_object(sidecar)  # noqa: SLF001
    payload["model_name"] = "so2_vae"
    prior._write_canonical_json(sidecar, payload)  # noqa: SLF001
    with pytest.raises(ValueError, match="sidecar"):
        verify()
    sidecar.write_bytes(saved)
    pngs["10"]["png_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="PNG"):
        verify()


def _pair_fixture(root: Path) -> tuple[Path, Path, Path, dict[str, dict[str, object]]]:
    manifest = root / builder.MANIFEST_NAME
    rows = ((0, 10, 0, 0), (1, 10, 256, 0))
    prior._write_manifest(manifest, rows)  # noqa: SLF001
    contract = root / builder.WORKER_CONTRACT_NAME
    prior._write_canonical_json(contract, {"producer": "fixture/run-01"})  # noqa: SLF001
    metadata = root / "dataset"
    metadata.mkdir()
    pngs: dict[str, dict[str, object]] = {
        "10": {"png_bytes": 1234, "png_sha256": "b" * 64},
    }
    artifacts: dict[str, object] = {}
    for model, checkpoint in builder.EXPECTED_CHECKPOINT_SHA256.items():
        name = f"{model}_mu_cancer_topup.json"
        sidecar: dict[str, object] = {
            "schema_version": "spec0020.latent_shard.v1",
            "status": "complete",
            "model_name": model,
            "checkpoint_sha256": checkpoint,
            "pinned_union_sha256": builder.EXPECTED_UNION_SHA256,
            "source_manifest": {
                "logical_basename": builder.MANIFEST_NAME,
                "sha256": prior._sha256(manifest),  # noqa: SLF001
                "run_number": 1,
                "row_count": 2,
                "first_identity": dict(
                    zip(prior.MANIFEST_HEADER, rows[0], strict=True),
                ),
                "last_identity": dict(
                    zip(prior.MANIFEST_HEADER, rows[-1], strict=True),
                ),
            },
            "tensor": {
                "count": 2,
                "dtype": "float32_le",
                "shape": [16, 32, 32],
                "layout": "CHW",
                "record_bytes": 65_536,
            },
            "file_size": 64 + 2 * 65_536,
            "payload_bytes": 2 * 65_536,
            "payload_crc32": 17,
            "payload_sha256": "c" * 64,
            "completed_wsi_ids": [10],
            "completed_wsi_count": 1,
        }
        prior._write_canonical_json(metadata / name, sidecar)  # noqa: SLF001
        artifacts[model] = {
            "bin_name": f"{model}_mu_cancer_topup.bin",
            "bin_bytes": 64 + 2 * 65_536,
            "bin_sha256": "d" * 64,
            "sidecar_name": name,
            "sidecar_bytes": (metadata / name).stat().st_size,
            "sidecar_sha256": prior._sha256(metadata / name),  # noqa: SLF001
        }
    pair: dict[str, object] = {
        "schema_version": "spec0022.cancer_topup_pair_audit.v1",
        "status": "complete",
        "row_count": 2,
        "input_contract_sha256": prior._sha256(contract),  # noqa: SLF001
        "supplement_manifest_sha256": prior._sha256(manifest),  # noqa: SLF001
        "base_union_sha256": builder.EXPECTED_UNION_SHA256,
        "normal_checkpoint_sha256": builder.EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        "so2_checkpoint_sha256": builder.EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        "artifacts": artifacts,
        "completed_wsi_evidence": [
            {"wsi_id": 10, **pngs["10"], "transcript_sha256": "e" * 64},
        ],
    }
    prior._write_canonical_json(metadata / builder.PAIR_NAME, pair)  # noqa: SLF001
    return manifest, contract, metadata, pngs
