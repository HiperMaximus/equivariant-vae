"""Synthetic identity/order verification of the CPU source benchmark."""
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.spec0054_shard1_mount_check import run_benchmark

repo = Path(__file__).resolve().parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='spec0054_source_check_') as temp:
    root = Path(temp)
    fake = root / 'repo'
    docs = fake / 'docs/data'
    docs.mkdir(parents=True)
    inputs = root / 'inputs'
    dataset = inputs / 'datasets/maximusshtefan/eqvae-spec0054-frozen-latent-shard-1/spec0054_fp16_latents'
    notebook = inputs / 'notebooks/maximshtefan/eqvae-spec0054-fp16-latent-shard-1/spec0054_fp16_latents'
    dataset.mkdir(parents=True)
    ids = [4, 281, 286, 431, 970, 1080, 1660, 1666, 1774, 1943, 3055]
    counts = [7, 9, 11, 6, 8, 12, 10, 14, 13, 16, 15]
    cohort = [{'wsi_id': w, 'diagnosis_index': i % 5, 'fold': 0 if i == 0 else i % 4 + 1}
              for i, w in enumerate(ids)]
    cohort += [{'wsi_id': 6000, 'diagnosis_index': 2, 'fold': 1}]
    with (docs / 'spec0054_cohort_folds.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cohort[0]))
        writer.writeheader()
        writer.writerows(cohort)
    entries = []
    offsets = {}
    for row, count in zip(cohort, counts, strict=False):
        offsets[row['wsi_id']] = len(entries)
        for coordinate in range(count):
            entries.append({'file_index': len(entries), 'atlas_row_index': len(entries),
                            'wsi_id': row['wsi_id'], 'x': coordinate * 256, 'y': 0,
                            'diagnosis_index': row['diagnosis_index'], 'fold': row['fold']})
    index = dataset / 'index_shard_01_of_12.csv'
    with index.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    files = {'index': {'name': index.name, 'bytes': index.stat().st_size, 'sha256': sha(index)}}
    values = {}
    for branch, seed in [('normal_vae', 24), ('so2_vae', 55)]:
        values[branch] = np.random.default_rng(seed).standard_normal((len(entries), 16, 32, 32)).astype('<f2')
        binary = dataset / f'{branch}_mu_shard_01_of_12.fp16.bin'
        values[branch].tofile(binary)
        files[branch] = {'name': binary.name, 'bytes': binary.stat().st_size}
    result = dataset / 'result_shard_01_of_12.json'
    result.write_text(json.dumps({'rows_written': len(entries), 'files': files}))
    notebook.parent.mkdir(parents=True)
    shutil.copytree(dataset, notebook)
    config = json.loads((repo / 'docs/data/spec0054_fold0_run.json').read_text())
    config['sources'] = [{'shard': 1, 'result_sha256': sha(result)}]
    (docs / 'spec0054_fold0_run.json').write_text(json.dumps(config))
    extraction = {'shards': [{'shard': 1, 'first_wsi': ids[0], 'last_wsi': ids[-1], 'rows': len(entries)}]}
    (docs / 'spec0054_fp16_latent_extraction.json').write_text(json.dumps(extraction))
    original = {str(p): sha(p) for directory in (dataset, notebook) for p in directory.iterdir()}
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    real_check_output = subprocess.check_output
    def source(*args, **kwargs):
        return 'synthetic-source\n' if args[0][0] == 'git' else real_check_output(*args, **kwargs)
    with patch('experiments.spec0054_shard1_mount_check.subprocess.check_output', source):
        measured = run_benchmark(fake, inputs, root / 'benchmark.json', time.time() + 120)
    assert measured['completed'] and len(measured['arms']) == 4
    assert measured['source_order'] == ('dataset', 'kernel_output', 'kernel_output', 'dataset')
    train = ids[1:]
    all_train = train + [6000]
    count_by_wsi = dict(zip(ids, counts, strict=True))
    expected_orders = {None: train}
    for epoch in (5, 6, 7):
        positions = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch])).permutation(len(all_train))
        expected_orders[epoch] = [all_train[i] for i in positions if all_train[i] in train]
    for branch, data in measured['branches'].items():
        assert len(data['reads']) == 160 and len(data['windows']) == 16
        for arm, source_name in enumerate(measured['source_order']):
            for epoch in (None, 5, 6, 7):
                rows = [row for row in data['reads'] if row['arm'] == arm and row['epoch'] == epoch]
                assert [row['wsi_id'] for row in rows] == expected_orders[epoch]
                for row in rows:
                    wsi = row['wsi_id']
                    raw = values[branch][offsets[wsi]:offsets[wsi] + count_by_wsi[wsi]]
                    assert row['source'] == source_name
                    if epoch is None:
                        assert row['raw_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
                    else:
                        generator = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, wsi]))
                        indices = np.sort(generator.choice(len(raw), size=math.ceil(.6 * len(raw)), replace=False))
                        rotations = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, wsi, 0xC4])).integers(0, 4, len(indices), dtype=np.uint8)
                        expected = np.stack([np.rot90(patch, int(turn), axes=(-2, -1)) for patch, turn in zip(raw[indices], rotations, strict=True)])
                        assert row['sample_indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
                        assert row['c4_rotations_sha256'] == hashlib.sha256(rotations.tobytes()).hexdigest()
                        assert row['prepared_physical_sha256'] == hashlib.sha256(expected.transpose(0, 2, 3, 1).copy().tobytes()).hexdigest()
    logged = [json.loads(line) for line in (root / 'shard1_source_benchmark_events.jsonl').read_text().splitlines()]
    assert sum(row['event'] == 'read_verified' for row in logged) == 320
    assert max(row['pending_wsi'] for row in logged if row['event'] == 'bag_prefetch_submit') == 8
    assert all(row['reserved_selected_bytes'] <= config['training_io']['max_ready_bytes_per_branch'] for row in logged if row['event'] == 'bag_prefetch_submit')
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(sha(Path(path)) == digest for path, digest in original.items())
    with patch('experiments.spec0054_shard1_mount_check.subprocess.check_output', source):
        expired = run_benchmark(fake, inputs, root / 'expired.json', time.time() - 1)
    assert expired['deadline_reached'] and not expired['completed'] and not expired['arms']
    assert all(not data['reads'] for data in expired['branches'].values())
summary = {'synthetic_verified_reads': 320, 'branches': 2, 'arms': 4, 'training_wsi': 10,
           'full_and_sampled_hashes_match_independent_formula': True, 'source_order': ['dataset', 'kernel_output', 'kernel_output', 'dataset'],
           'global_epoch_order_filter_matches': True, 'rng_unchanged': True, 'files_immutable': True,
           'pending_peak': 8, 'byte_reservations_bounded': True, 'deadline_zero_reads': True}
Path(__file__).with_name('source_benchmark_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
