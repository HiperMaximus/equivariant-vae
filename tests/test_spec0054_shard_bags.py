"""Small physical-row check for the once-only FP16 shard consumer."""

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from experiments.spec0054_shard_bags import BagLocation, RECORD_BYTES, ShardBags


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
        files: dict[str, dict[str, str | int]] = {"index": {"sha256": hashlib.sha256(index.read_bytes()).hexdigest()}}
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
    descriptor = bags.descriptors[("normal_vae", 1)]
    assert bags.read("normal_vae", 4)[:, 0, 0, 0].tolist() == [0, 1]
    assert bags.descriptors[("normal_vae", 1)] == descriptor
    assert bags.read("so2_vae", 8)[:, 0, 0, 0].tolist() == [102]
    assert bags.read("normal_vae", 12)[:, 0, 0, 0].tolist() == [3, 4]
    bags.close()
    assert bags.descriptors == {}


def test_sampled_read_requests_only_retained_records_in_order(tmp_path, monkeypatch):
    reader = ShardBags.__new__(ShardBags)
    reader.locations = {8: BagLocation(1, 2, 10, 0, 1, ())}
    reader.files = {branch: {1: tmp_path / f"{branch}.bin"}
                    for branch in ("normal_vae", "so2_vae")}
    reader.descriptors = {}
    generator = np.random.default_rng(8)
    for binary in reader.files.values():
        generator.standard_normal((12, 16, 32, 32)).astype("<f2").tofile(binary[1])
    indices = np.array([0, 1, 3, 4, 5, 6, 8, 9])  # 80%, in three contiguous regions.
    calls = []
    preadv = os.preadv
    def observed_read(fd, buffers, offset):
        calls.append((offset, sum(len(buffer) for buffer in buffers)))
        return preadv(fd, buffers, offset)
    monkeypatch.setattr(os, "preadv", observed_read)
    try:
        for branch in reader.files:
            full = reader.read(branch, 8)
            sampled = reader.read(branch, 8, indices)
            np.testing.assert_array_equal(sampled, full[indices])
            assert calls[-3:] == [(2 * RECORD_BYTES, 2 * RECORD_BYTES),
                                  (5 * RECORD_BYTES, 4 * RECORD_BYTES),
                                  (10 * RECORD_BYTES, 2 * RECORD_BYTES)]
            assert sum(size for _, size in calls[-3:]) == 8 * RECORD_BYTES
        np.testing.assert_array_equal(reader.read("normal_vae", 8, np.array([9])),
                                      reader.read("normal_vae", 8)[[9]])
    finally:
        reader.close()


def test_two_readers_share_descriptor_without_changing_offsets(tmp_path: Path) -> None:
    reader = ShardBags.__new__(ShardBags)
    reader.locations = {8: BagLocation(1, 2, 6, 0, 1, ()),
                        12: BagLocation(1, 8, 6, 0, 1, ())}
    binary = tmp_path / "latents.bin"
    values = np.random.default_rng(8).standard_normal((14, 16, 32, 32)).astype("<f2")
    values.tofile(binary)
    reader.files = {"normal_vae": {1: binary}}
    descriptor = os.open(binary, os.O_RDONLY)
    reader.descriptors = {("normal_vae", 1): descriptor}
    barrier = Barrier(2)

    def read(request):
        wsi_id, indices = request
        barrier.wait()
        return reader.read("normal_vae", wsi_id, indices)

    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            indices = np.array([0, 1, 3, 5])
            for requests in (((8, indices), (12, indices)), ((8, None), (12, indices))):
                outputs = list(threads.map(read, requests))
                for (wsi_id, selected), output in zip(requests, outputs, strict=True):
                    location = reader.locations[wsi_id]
                    expected = values[location.first_record:location.first_record + location.count]
                    np.testing.assert_array_equal(output, expected if selected is None else expected[selected])
                assert not np.shares_memory(outputs[0], outputs[1])
                assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
                assert reader.descriptors[("normal_vae", 1)] == descriptor
    finally:
        reader.close()


@pytest.mark.parametrize("maximum_gap,expected_records,expected_intervals", [(0, 7, 4), (1, 8, 3), (2, 10, 2)])
def test_probe_coalescing_scatter_preserves_bits_and_physical_ranges(
    tmp_path, monkeypatch, maximum_gap, expected_records, expected_intervals,
):
    from experiments.spec0054_mil_fold0 import _read_coalesced

    # Raw uint16 patterns also cover signed zero, infinities and NaN payloads.
    bits = np.random.default_rng(19).integers(0, 65536, (19, 16, 32, 32), dtype=np.uint16)
    binary = tmp_path / "latents.bin"
    bits.tofile(binary)
    original = hashlib.sha256(binary.read_bytes()).hexdigest()
    descriptor = os.open(binary, os.O_RDONLY)
    indices = np.array([0, 1, 3, 4, 7, 8, 12])
    preadv = os.preadv
    calls = []

    def observed(fd, buffers, offset):
        calls.append((offset, sum(len(buffer) for buffer in buffers), len(buffers)))
        return preadv(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", observed)
    try:
        actual, requested, regions = _read_coalesced(descriptor, 2, indices, maximum_gap)
        np.testing.assert_array_equal(actual.view(np.uint16), bits[indices + 2])
        assert requested == expected_records * RECORD_BYTES
        assert regions == len(calls) == expected_intervals
        assert calls[0][0] == 2 * RECORD_BYTES
        assert all(2 * RECORD_BYTES <= offset and offset + size <= 15 * RECORD_BYTES
                   for offset, size, _ in calls)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
        # A small kernel iovec limit must split groups without changing output.
        monkeypatch.setattr(os, "sysconf", lambda name: 3)
        calls.clear()
        limited, requested_limited, regions_limited = _read_coalesced(descriptor, 2, indices, maximum_gap)
        np.testing.assert_array_equal(limited.view(np.uint16), bits[indices + 2])
        assert all(vector_count <= 3 for _, _, vector_count in calls)
        assert requested_limited == sum(size for _, size, _ in calls)
        assert regions_limited == len(calls)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda offset: _read_coalesced(
                descriptor, offset, indices, maximum_gap)[0], (0, 4)))
        for offset, output in zip((0, 4), outputs, strict=True):
            np.testing.assert_array_equal(output.view(np.uint16), bits[indices + offset])
        assert not np.shares_memory(outputs[0], outputs[1])
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == original
    finally:
        os.close(descriptor)


def test_exposure_schedule_and_table_resume(tmp_path: Path) -> None:
    import torch

    from experiments.spec0054_mil_fold0 import ExposureSchedule, _restore_rows
    from eqvae.training.mil_telemetry_io import write_telemetry_table

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.0002)
    config = json.loads(Path("docs/data/spec0054_fold0_run.json").read_text())
    schedule = ExposureSchedule(optimizer, 288, config)
    assert schedule.set_next(1) == pytest.approx(0.00005)
    assert schedule.set_next(1440) == pytest.approx(0.0005)
    assert schedule.set_next(55 * 288) == pytest.approx(0.0002525)
    assert schedule.set_next(105 * 288) == pytest.approx(0.000005)
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


def test_lr_confirmation_holds_peak_and_constant_aem() -> None:
    import torch

    from experiments.spec0054_mil_fold0 import ExposureSchedule, _aem_weight

    config = {"epochs": 105, "warmup_epochs": 5, "aem_weight": 0.05,
              "run_mode": "lr_confirmation"}
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])
    for peak in (0.0005, 0.001):
        candidate = {**config, "peak_lr": peak}
        schedule = ExposureSchedule(optimizer, 288, candidate)
        assert schedule.set_next(1) == pytest.approx(0.1 * peak)
        assert schedule.set_next(5 * 288) == pytest.approx(peak)
        restored = ExposureSchedule(optimizer, 288, candidate)
        restored.load_state_dict(schedule.state_dict())
        for exposure in (5 * 288 + 4, 6 * 288, 8 * 288):
            assert restored.set_next(exposure) == pytest.approx(peak)
        for exposure in (0, 8 * 288, 65 * 288, 105 * 288):
            assert _aem_weight(exposure, 288, candidate) == pytest.approx(0.05)


def test_fixed_boundary_head_gradient_matches_autograd() -> None:
    import torch
    from torch.nn import functional

    from experiments.spec0054_mil_fold0 import _head_gradient_norms

    head = torch.nn.Linear(4, 3)
    embedding = torch.tensor([0.2, -0.4, 0.8, 0.1])
    logits = head(embedding)
    functional.cross_entropy(logits.unsqueeze(0), torch.tensor([2])).backward()
    bias_norm, weight_norm = _head_gradient_norms(logits, 2, embedding)
    assert head.bias.grad is not None and head.weight.grad is not None
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


@pytest.mark.parametrize("budget_records", [48, 11])
def test_training_prefetch_fifo_bytes_policy_and_rng(tmp_path, monkeypatch, budget_records):
    from threading import Condition, Event

    import torch

    from experiments import spec0054_mil_fold0 as runner
    from experiments.spec0054_mil_fold0 import (
        _prefetch_training_inputs, _sample_c4, _sample_indices,
    )

    reader = ShardBags.__new__(ShardBags)
    counts = [9, 7, 8, 6, 9, 8, 7, 6, 9, 8, 7]
    reader.locations = {}
    first = 2
    for wsi, count in enumerate(counts):
        reader.locations[wsi] = BagLocation(1, first, count, 0, 1, ())
        first += count
    values = np.random.default_rng(91).standard_normal((first, 16, 32, 32)).astype("<f2")
    binary = tmp_path / "latents.bin"
    values.tofile(binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    reader.files = {"normal_vae": {1: binary}}
    reader.descriptors = {}
    config = {"seed": 19, "c4_augmentation": True,
              "training_io": {"reader_workers": 2, "future_wsi": 8,
                              "max_ready_bytes_per_branch": budget_records * RECORD_BYTES,
                              "coalesced_gap_records": 1, "lookahead_wsi": 4,
                              "prefetch_bytes_per_wsi": 5 * RECORD_BYTES}}
    requests = [(3, wsi, _sample_indices(count, 19, 3, wsi, 0.6))
                for wsi, count in enumerate(counts)]
    observations = []
    hints = []
    changed = Condition()
    second_ready = Event()
    read_coalesced = runner._read_coalesced

    def observed_read(fd, first_record, indices, maximum_gap):
        if first_record == reader.locations[0].first_record:
            assert second_ready.wait(timeout=10)
        return read_coalesced(fd, first_record, indices, maximum_gap)

    monkeypatch.setattr(runner, "_read_coalesced", observed_read)

    def log(event, **details):
        with changed:
            observations.append((event, details))
            if event == "bag_load_done" and details["wsi_id"] == 1:
                second_ready.set()
            changed.notify_all()

    def advise(fd, offset, length, advice):
        hints.append((fd, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", advise)
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    actual = []
    try:
        with _prefetch_training_inputs(reader, "normal_vae", torch.device("cpu"),
                                       config, iter(requests), log) as prepared:
            submitted = [row for event, row in observations if event == "bag_prefetch_submit"]
            initial_count = len(submitted)
            assert initial_count == (8 if budget_records == 48 else 2)
            # Completed futures retain all their reservations until consumed.
            with changed:
                assert changed.wait_for(lambda: sum(event == "bag_load_done"
                                                     for event, _ in observations) == initial_count,
                                        timeout=10)
            assert sum(event == "bag_prefetch_submit" for event, _ in observations) == initial_count
            assert next(row["wsi_id"] for event, row in observations if event == "bag_load_done") == 1
            for wsi, indices, host in prepared:
                location = reader.locations[wsi]
                expected = values[location.first_record + indices].copy()
                expected = np.stack([np.rot90(patch, int(turns), axes=(-2, -1))
                                     for patch, turns in zip(expected, _sample_c4(len(indices), 19, 3, wsi), strict=True)])
                np.testing.assert_array_equal(host.numpy().view(np.uint16), expected.view(np.uint16))
                assert host.is_contiguous(memory_format=torch.channels_last)
                actual.append(wsi)
        assert actual == list(range(len(counts)))
        reserved = pending = 0
        for event, row in observations:
            if event == "bag_prefetch_submit":
                reserved += row["selected_bytes"]
                pending += 1
            elif event == "bag_prefetch_consume":
                reserved -= len(requests[row["wsi_id"]][2]) * RECORD_BYTES
                pending -= 1
            else:
                continue
            assert reserved == row["reserved_selected_bytes"]
            assert pending == row["pending_wsi"]
            assert 0 <= reserved <= budget_records * RECORD_BYTES
            assert 0 <= pending <= 8
        assert reserved == pending == 0
        transfers = [row for event, row in observations if event == "bag_transfer_start"]
        for row in transfers:
            indices = requests[row["wsi_id"]][2]
            assert row["selected_bytes"] == len(indices) * RECORD_BYTES
            assert row["requested_bytes"] >= row["selected_bytes"]
            assert row["read_regions"] >= 1
        descriptor = reader.descriptors["normal_vae", 1]
        willneed = [(offset, length) for fd, offset, length, advice in hints
                    if advice == os.POSIX_FADV_WILLNEED and fd == descriptor]
        assert sorted(willneed) == sorted((location.first_record * RECORD_BYTES,
                                          min(location.count, 5) * RECORD_BYTES)
                                         for location in reader.locations.values())
        discarded = [(offset, length) for fd, offset, length, advice in hints
                     if advice == os.POSIX_FADV_DONTNEED and fd == descriptor]
        assert sorted(discarded) == sorted((location.first_record * RECORD_BYTES,
                                           location.count * RECORD_BYTES)
                                          for location in reader.locations.values())
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == digest
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        assert torch.equal(torch_rng, torch.get_rng_state())
    finally:
        reader.close()


def test_training_prefetch_keeps_pinned_source_through_h2d(tmp_path, monkeypatch):
    import weakref
    from contextlib import nullcontext
    from threading import Event

    import torch

    from experiments import spec0054_mil_fold0 as runner
    from experiments.spec0054_mil_fold0 import _prefetch_training_inputs

    reader = ShardBags.__new__(ShardBags)
    reader.locations = {8: BagLocation(1, 0, 2, 0, 1, ())}
    binary = tmp_path / "latents.bin"
    values = np.arange(2 * 16 * 32 * 32, dtype=np.uint16).reshape(2, 16, 32, 32)
    values.tofile(binary)
    reader.files = {"normal_vae": {1: binary}}
    reader.descriptors = {}
    config = {"seed": 7, "c4_augmentation": False,
              "training_io": {"reader_workers": 2, "future_wsi": 8,
                              "max_ready_bytes_per_branch": 2 * RECORD_BYTES,
                              "coalesced_gap_records": 1, "lookahead_wsi": 4,
                              "prefetch_bytes_per_wsi": RECORD_BYTES}}
    synchronizing, release_dma = Event(), Event()
    calls, hosts = [], []
    copy_stream, main_stream = object(), object()
    read_coalesced = runner._read_coalesced

    def owned_read(*args):
        output = read_coalesced(*args)
        assert output[0].flags.owndata
        calls.append("owned_read")
        return output

    monkeypatch.setattr(runner, "_read_coalesced", owned_read)

    class GPUResult:
        def record_stream(self, stream):
            assert stream is main_stream
            assert "synchronized" in calls
            calls.append("record_stream")

    class Host:
        def copy_(self, source):
            np.testing.assert_array_equal(source.numpy().view(np.uint16), values)
            calls.append("host_copy")

        def to(self, device, non_blocking):
            assert device.type == "cuda" and non_blocking
            calls.append("async_h2d")
            return GPUResult()

    def empty_like(source, memory_format, pin_memory):
        assert memory_format is torch.channels_last and pin_memory
        host = Host()
        hosts.append(weakref.ref(host))
        return host

    class Ready:
        def record(self, stream):
            assert stream is copy_stream
            calls.append("event_record")

        def synchronize(self):
            synchronizing.set()
            assert release_dma.wait(timeout=10)
            assert hosts[0]() is not None
            calls.append("synchronized")

    def advise(fd, offset, length, advice):
        if advice == os.POSIX_FADV_DONTNEED:
            assert "owned_read" in calls and "host_copy" not in calls
            calls.append("discard_after_copy")

    monkeypatch.setattr(torch, "empty_like", empty_like)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: copy_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Ready)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: main_stream)
    monkeypatch.setattr(os, "posix_fadvise", advise)
    try:
        with _prefetch_training_inputs(reader, "normal_vae", torch.device("cuda:0"), config,
                                       iter([(0, 8, np.array([0, 1]))]),
                                       lambda *args, **kwargs: None) as prepared:
            assert synchronizing.wait(timeout=10)
            assert hosts[0]() is not None and "synchronized" not in calls
            release_dma.set()
            assert next(prepared)[0] == 8
            assert calls == ["owned_read", "discard_after_copy", "host_copy", "async_h2d", "event_record",
                             "synchronized", "record_stream"]
            assert hosts[0]() is None
    finally:
        release_dma.set()
        reader.close()
