"""Offline exact byte/call accounting for frozen epoch-5 subset read plans."""
import json
from pathlib import Path
import numpy as np
from eqvae.training.mil_telemetry_io import load_telemetry_table
from experiments.spec0054_mil_fold0 import _sample_indices

source = Path('runs/kaggle/spec0054_cls_fold0_c4_v18/spec0054_cls_fold0/normal_vae/t0.npz')
output = Path('runs/local/spec0054_io_strategy_review/coalescing_epoch5.json')
table = load_telemetry_table(source)
sizes = {int(wsi): int(count) for wsi, count in zip(table['wsi_id'], table['bag_size'])}
record_bytes, seed, epoch, retention = 32768, 1701, 5, 0.6
thresholds = (0, 1, 2, 3, 7, 31)
totals = {gap: {'contiguous_reads_per_branch': 0, 'records_read_per_branch': 0} for gap in thresholds}
selected_records = 0
for wsi_id, count in sorted(sizes.items()):
    indices = _sample_indices(count, seed, epoch, wsi_id, retention)
    gaps = np.diff(indices) - 1
    selected_records += len(indices)
    for gap in thresholds:
        totals[gap]['contiguous_reads_per_branch'] += 1 + int(np.count_nonzero(gaps > gap))
        totals[gap]['records_read_per_branch'] += len(indices) + int(gaps[(gaps > 0) & (gaps <= gap)].sum())

result = {
    'scope': 'offline exact read-plan accounting; not throughput measurement',
    'source_table': str(source), 'size_source': 'actual frozen bag_size from cumulative t0, one count per WSI',
    'branch_sizes_paired': True, 'wsi_count': len(sizes), 'seed': seed, 'epoch_index': epoch,
    'patch_retention': retention, 'record_bytes': record_bytes,
    'full_records_per_branch': sum(sizes.values()), 'selected_records_per_branch': selected_records,
    'full_training_paired_bytes': 2 * sum(sizes.values()) * record_bytes,
    'selected_training_paired_bytes': 2 * selected_records * record_bytes,
    'maximum_selected_bag_bytes': max(int(np.ceil(retention * count)) * record_bytes for count in sizes.values()),
    'coalescing': [],
    'interpretation': 'Merge every gap of at most threshold missing records; retain identical selected values after over-read. Read intervals do not extend outside first/last selected rows.',
    'limitations': ['No new NFS reads, timing, scratch capacity, or RPC count is measured.',
                    'Contiguous read count means application read intervals, not necessarily NFS RPC count.',
                    'Bytes are paired normal/SO2; call counts are per branch.',
                    'Historical cohort patch_count is not used.']}
for gap in thresholds:
    total = totals[gap]
    result['coalescing'].append({
        'maximum_missing_gap_records': gap, **total,
        'paired_bytes_read': 2 * total['records_read_per_branch'] * record_bytes,
        'byte_ratio_vs_selected': total['records_read_per_branch'] / selected_records,
        'read_count_ratio_vs_baseline': total['contiguous_reads_per_branch'] / totals[0]['contiguous_reads_per_branch']})
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
