"""One-shot numerical review of the completed Spec0054 combined-policy probe."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path('runs/kaggle/spec0054_io_coalescing_v1_handoff')
branches = ('normal_vae', 'so2_vae')
data = {b: json.loads((root / 'spec0054_io_probe' / b / 'read_probe.json').read_text()) for b in branches}
methods = data[branches[0]]['probe']['methods']

def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'maximum': float(a.max())}

def request_key(row):
    return (row['block_index'], row['method'], row['wsi_id'], row['epoch'],
            row['sample_indices_sha256'], row['c4_rotations_sha256'])

log = (root / 'eqvae-spec0054-io-coalescing.log').read_text()
match = re.search(r'\\"event\\": \\"workers_exit\\", \\"statuses\\": \[(\d+), (\d+)\]', log)
review = {'run': 'maximshtefan/eqvae-spec0054-io-coalescing/1', 'platform_status': 'COMPLETE',
          'source_commit': data[branches[0]]['source_commit'], 'worker_exit_codes': [int(x) for x in match.groups()] if match else None,
          'completed': all(d['completed'] for d in data.values()),
          'total_reads': sum(len(d['rows']) for d in data.values()), 'branches': {}, 'methods': {},
          'paired_request_keys': sorted(map(request_key, data[branches[0]]['rows'])) == sorted(map(request_key, data[branches[1]]['rows'])),
          'paired_epoch_plans': data[branches[0]]['epoch_read_plan'] == data[branches[1]]['epoch_read_plan'],
          'limitations': ['All arms have identical policy, reader count and future depth; coalescing gap size differs.',
                          'ABC/BCA/CAB balances execution position, not all carryover or nonlinear backend load.',
                          'mincore observes client residency, not backend cache.',
                          'This 72-WSI CPU window is not an entire training epoch or a GPU throughput measurement.',
                          'Combined-policy kernel v24 overlaps part of this run; inter-session interference is unmeasured.']}
for branch, d in data.items():
    hashes = {}
    for row in d['rows']:
        hashes.setdefault(row['wsi_id'], set()).add(tuple(row[k] for k in ('selected_values_sha256', 'prepared_values_sha256', 'sample_indices_sha256', 'c4_rotations_sha256')))
    row_file = root / 'spec0054_io_probe' / branch / 'read_probe_rows.jsonl'
    row_lines = [json.loads(line) for line in row_file.read_text().splitlines()]
    first, last = min(b['started_unix'] for b in d['blocks']), max(b['ended_unix'] for b in d['blocks'])
    review['branches'][branch] = {'completed': d['completed'], 'row_count': len(d['rows']), 'block_count': len(d['blocks']),
        'all_blocks_complete': all(b['completed'] for b in d['blocks']),
        'all_wsi_hashes_identical_across_methods_and_repetitions': all(len(v) == 1 for v in hashes.values()),
        'rows_jsonl_equal_to_final_json': sorted(row_lines,key=request_key) == sorted(d['rows'],key=request_key),
        'gpu_used': d['hardware']['gpu_used'], 'host_pinning': d['hardware']['host_pinning'],
        'h2d_measured': d['hardware']['h2d_measured'],
        'first_timed_block_utc': datetime.fromtimestamp(first, timezone.utc).isoformat(),
        'last_timed_block_utc': datetime.fromtimestamp(last, timezone.utc).isoformat(),
        'process_hwm_bytes': max(r['after_prepare']['process_memory_bytes']['VmHWM'] for r in d['rows']),
        'result_sha256': hashlib.sha256((root / 'spec0054_io_probe' / branch / 'read_probe.json').read_bytes()).hexdigest()}
for method in methods:
    rows = [r for d in data.values() for r in d['rows'] if r['method'] == method]
    blocks = [b for d in data.values() for b in d['blocks'] if b['method'] == method]
    walls = []
    for repeat in range(data[branches[0]]['probe']['passes']):
        pair = [b for b in blocks if b['block_index'] == repeat]
        walls.append(max(b['ended_unix'] for b in pair) - min(b['started_unix'] for b in pair))
    review['methods'][method] = {
        'read_seconds': stats([r['read_seconds'] for r in rows]),
        'pipeline_seconds': stats([r['pipeline_seconds'] for r in rows]),
        'read_cpu_seconds': stats([r['read_cpu_seconds'] for r in rows]),
        'host_prepare_seconds': stats([r['host_prepare_seconds'] for r in rows]),
        'selected_bytes_per_repeat': sum(r.get('sampled_bytes',r['requested_bytes']) for r in rows) / data[branches[0]]['probe']['passes'],
        'requested_bytes_per_repeat': sum(r['requested_bytes'] for r in rows) / data[branches[0]]['probe']['passes'],
        'application_intervals_per_repeat': (sum(r['read_regions'] for r in rows) / data[branches[0]]['probe']['passes']) if all(r['read_regions'] is not None for r in rows) else None,
        'joint_window_wall_seconds_by_repeat': walls, 'sum_joint_window_wall_seconds': sum(walls),
        'mean_joint_window_wall_seconds': float(np.mean(walls)),
        'residency_before_read': stats([r['before_read']['resident_fraction'] for r in rows]),
        'reset_residency_maximum': max(r['resident_fraction'] for b in blocks for r in b['residency_after_cache_reset']),
        'max_pending_wsi': max(b['max_pending_wsi'] for b in blocks),
        'max_ready_wsi_observed': max(b['max_ready_wsi_observed'] for b in blocks),
        'max_reserved_host_bytes': max(b['max_reserved_host_bytes'] for b in blocks),
        'cache_reset_seconds_sum_both_branches': sum(b['cache_reset_seconds'] for b in blocks),
        'nfs_read_counters_exposed': any(b['nfs_read_before'] or b['nfs_read_after'] for b in blocks)}
baseline = review['methods']['sampled_regions']
for result in review['methods'].values():
    result['joint_wall_ratio_to_baseline'] = result['sum_joint_window_wall_seconds'] / baseline['sum_joint_window_wall_seconds']
    result['joint_wall_ratio_by_repeat'] = [a/b for a,b in zip(result['joint_window_wall_seconds_by_repeat'],baseline['joint_window_wall_seconds_by_repeat'])]
# READ RPC fields follow Linux net/sunrpc/stats.c, _print_rpc_iostats.
# The twelve mount views repeat the same counters. Retain one branch/view;
# never add the aliases or the two branch snapshots together.
review['nfs_counter_scope'] = 'normal branch snapshot; one identical mount view; shared RPC-client scope, not per WSI/branch; reply bytes include protocol overhead'
review['nfs_counter_field_source'] = 'https://github.com/torvalds/linux/blob/v6.18/net/sunrpc/stats.c'
for method in methods:
    nfs = []
    for block in data[branches[0]]['blocks']:
        if block['method'] != method:
            continue
        before, after = block['nfs_read_before'], block['nfs_read_after']
        before_unique = {tuple(v) for v in before.values()}
        after_unique = {tuple(v) for v in after.values()}
        item = {'repeat':block['block_index'], 'mount_view_count':len(before),
                'identical_counter_views':len(before_unique)==len(after_unique)==1}
        if item['identical_counter_views']:
            delta = [a-b for a,b in zip(next(iter(after_unique)),next(iter(before_unique)))]
            operations, transmissions, timeouts, sent, received, queue, rtt, execution, errors = delta
            item.update(operations=operations, transmissions=transmissions, major_timeouts=timeouts,
                        bytes_sent=sent, reply_bytes_received=received, errors=errors,
                        mean_queue_ms=queue/operations, mean_rtt_ms=rtt/operations,
                        mean_execution_ms=execution/operations)
        nfs.append(item)
    review['methods'][method]['shared_nfs_read_by_repeat'] = nfs

Path('runs/local/spec0054_io_coalescing_v1_review/summary.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review,indent=2))
