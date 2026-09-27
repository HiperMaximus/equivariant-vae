"""Small physical-row check for the once-only FP16 shard consumer."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spec0054_shard_bags import ShardBags


def test_complete_wsi_reads_preserve_index_order_and_paired_offsets(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.csv"
    with cohort.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("wsi_id", "patch_count", "diagnosis_index", "fold"))
        writer.writeheader()
        writer.writerows([
            {"wsi_id": 4, "patch_count": 1, "diagnosis_index": 2, "fold": 0},
            {"wsi_id": 8, "patch_count": 1, "diagnosis_index": 1, "fold": 1},
            {"wsi_id": 12, "patch_count": 2, "diagnosis_index": 3, "fold": 0},
        ])
    contract = {"shards": [{"shard": 1, "rows": 3}, {"shard": 2, "rows": 2}]}
    expected_results = {}
    atlas_index = 0
    for shard, entries in (
        (1, [(4, 0, 0, 2, 0), (4, 256, 0, 2, 0), (8, 0, 256, 1, 1)]),
        (2, [(12, 0, 0, 3, 0), (12, 256, 0, 3, 0)]),
    ):
        suffix = f"shard_{shard:02d}_of_12"
        directory = tmp_path / f"source_{shard}"
        directory.mkdir()
        index = directory / f"index_{suffix}.csv"
        fields = ("file_index", "atlas_row_index", "wsi_id", "x", "y", "diagnosis_index", "fold")
        with index.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for file_index, (wsi_id, x, y, diagnosis, fold) in enumerate(entries):
                writer.writerow({"file_index": file_index, "atlas_row_index": atlas_index,
                                 "wsi_id": wsi_id, "x": x, "y": y,
                                 "diagnosis_index": diagnosis, "fold": fold})
                atlas_index += 1
        files = {"index": {"sha256": hashlib.sha256(index.read_bytes()).hexdigest()}}
        for branch, offset in (("normal_vae", 0), ("so2_vae", 100)):
            values = np.stack([np.full((16, 32, 32), offset + atlas_index - len(entries) + row,
                                       dtype="<f2") for row in range(len(entries))])
            binary = directory / f"{branch}_mu_{suffix}.fp16.bin"
            binary.write_bytes(values.tobytes())
            files[branch] = {"name": binary.name, "bytes": binary.stat().st_size}
        result = directory / f"result_{suffix}.json"
        result.write_text(json.dumps({"rows_written": len(entries), "files": files}))
        expected_results[shard] = hashlib.sha256(result.read_bytes()).hexdigest()
    bags = ShardBags(tmp_path, contract, cohort, expected_results)
    assert sorted(bags.locations) == [4, 8, 12]
    assert bags.locations[4].coordinates == ((0, 0), (256, 0))
    assert bags.locations[8].first_record == 2
    assert bags.locations[12].first_record == 0
    assert bags.read("normal_vae", 4).dtype == np.dtype("float16")
    assert bags.read("normal_vae", 4)[:, 0, 0, 0].tolist() == [0, 1]
    assert bags.read("so2_vae", 8)[:, 0, 0, 0].tolist() == [102]
    assert bags.read("normal_vae", 12)[:, 0, 0, 0].tolist() == [3, 4]


def test_exposure_schedule_and_table_resume(tmp_path: Path) -> None:
    import torch

    from experiments.spec0054_mil_fold0 import ExposureSchedule, _restore_rows
    from eqvae.training.mil_telemetry_io import write_telemetry_table

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.0002)
    config = {"epochs": 150, "warmup_epochs": 5, "peak_lr": 0.0002}
    schedule = ExposureSchedule(optimizer, 288, config)
    assert schedule.set_next(1) == pytest.approx(0.00002)
    assert schedule.set_next(1440) == pytest.approx(0.0002)
    assert schedule.set_next(43200) == pytest.approx(0.000002)
    schedule.last_exposure = 1440
    restored = ExposureSchedule(optimizer, 288, config)
    restored.load_state_dict(schedule.state_dict())
    assert restored.last_exposure == 1440
    table = tmp_path / "t0.npz"
    write_telemetry_table(table, [
        {"update": 1, "wsi_id": 4, "loss": 0.25},
        {"update": 2, "wsi_id": 8, "loss": 0.5},
    ])
    assert _restore_rows(table, 1) == [{"loss": 0.25, "update": 1, "wsi_id": 4}]


def test_fixed_boundary_head_gradient_matches_autograd() -> None:
    import torch
    from torch.nn import functional

    from experiments.spec0054_mil_fold0 import _head_gradient_norms

    head = torch.nn.Linear(4, 3)
    embedding = torch.tensor([0.2, -0.4, 0.8, 0.1])
    logits = head(embedding)
    functional.cross_entropy(logits.unsqueeze(0), torch.tensor([2])).backward()
    bias_norm, weight_norm = _head_gradient_norms(logits, 2, embedding)
    assert bias_norm == pytest.approx(float(head.bias.grad.norm()))
    assert weight_norm == pytest.approx(float(head.weight.grad.norm()))


def test_gradient_noise_panel_matches_direct_two_wsi_case() -> None:
    import torch

    from experiments.spec0054_mil_fold0 import _gradient_noise_summary

    records, norms, cosines = _gradient_noise_summary(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert norms.tolist() == [1.0, 1.0]
    assert cosines[0, 1] == 0
    assert records["mean_individual_squared_norm"] == 1
    assert records["mean_gradient_squared_norm"] == 0.5
    assert records["gradient_variance"] == 0.5
    assert records["gradient_noise_scale"] == 1
    opposite, _, _ = _gradient_noise_summary(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    assert opposite["noise_scale_defined"] == 0
    assert np.isnan(opposite["gradient_noise_scale"])
    assert opposite["pairwise_conflict_fraction"] == 1
