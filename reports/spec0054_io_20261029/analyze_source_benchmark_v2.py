"""Read-only paired review of dataset versus notebook-output shard1 reads."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'runs/kaggle/spec0054_shard1_source_v2'
r = json.loads((ROOT / 'shard1_source_benchmark.json').read_text())
events = [json.loads(line) for line in (ROOT / 'shard1_source_benchmark_events.jsonl').read_text().splitlines()]
rows = [row for branch in r['branches'].values() for row in branch['reads']]


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), **({key: float(value) for key, value in {
        'mean': a.mean(), 'median': np.median(a), 'p95': np.quantile(a, .95),
        'max': a.max(), 'sum': a.sum()}.items()} if len(a) else {})}


def saving(a, b):
    return 100 * (1 - b / a)


groups = defaultdict(list)
for row in rows:
    groups[row['branch'], row['phase'], row['epoch'], row['wsi_id']].append(row)
parity = []
for key, group in groups.items():
    fields = ('raw_sha256',) if key[1] == 'full' else (
        'sample_indices_sha256', 'c4_rotations_sha256', 'prepared_physical_sha256',
        'selected_bytes', 'requested_bytes', 'read_regions')
    parity.append({'key': list(key), 'n': len(group),
                   'four_source_visit_pairs': len({(row['source'], row['source_visit']) for row in group}) == 4,
                   'equal': all(len({row[field] for row in group}) == 1 for field in fields)})
config = json.loads((REPO / 'docs/data/spec0054_fold0_run.json').read_text())
with (REPO / 'docs/data/spec0054_cohort_folds.csv').open() as handle:
    all_train = sorted(int(row['wsi_id']) for row in csv.DictReader(handle) if int(row['fold']) != config['fold'])
orders = {}
for branch, data in r['branches'].items():
    for arm in range(4):
        for epoch in (None, 5, 6, 7):
            selected = [row for row in data['reads'] if row['arm'] == arm and row['epoch'] == epoch]
            if epoch is None:
                expected = r['training_wsi_ids']
            else:
                positions = np.random.default_rng(np.random.SeedSequence([r['seed'], epoch])).permutation(len(all_train))
                expected = [all_train[i] for i in positions if all_train[i] in r['training_wsi_ids']]
            orders[f'{branch}/{arm}/{epoch}'] = [row['wsi_id'] for row in selected] == expected
sources = {}
for source in ('dataset', 'kernel_output'):
    source_rows = [row for row in rows if row['source'] == source]
    arms = [arm for arm in r['arms'] if arm['source'] == source]
    windows = [window for data in r['branches'].values() for window in data['windows'] if window['source'] == source]
    sources[source] = {'joint_arm_wall_seconds_including_hashing': stats([arm['wall_seconds_including_hashing'] for arm in arms]),
                       'whole_arm_process_cpu_seconds': stats([arm['process_cpu_seconds_including_hashing'] for arm in arms]),
                       'phases': {}, 'by_branch': {}}
    for phase in ('full', 'sampled'):
        selected = [row for row in source_rows if row['phase'] == phase]
        sources[source]['phases'][phase] = {
            'read_seconds': stats([row['read_seconds'] for row in selected]),
            'consumer_hash_seconds': stats([row['hash_seconds'] for row in selected]),
            'branch_window_wall_seconds_including_hashing': stats([w['wall_seconds_including_hashing'] for w in windows if w['phase'] == phase]),
            'hotspot3055_read_seconds': stats([row['read_seconds'] for row in selected if row['wsi_id'] == 3055]),
            'by_source_visit': {str(visit): stats([row['read_seconds'] for row in selected if row['source_visit'] == visit]) for visit in (0, 1)},
            'by_epoch': {str(epoch): stats([row['read_seconds'] for row in selected if row['epoch'] == epoch]) for epoch in ((None,) if phase == 'full' else (5, 6, 7))},
        }
    for branch in r['branches']:
        sources[source]['by_branch'][branch] = {
            phase: stats([row['read_seconds'] for row in source_rows if row['branch'] == branch and row['phase'] == phase])
            for phase in ('full', 'sampled')}
paired_arms = []
for pair in (0, 1):
    arms = {arm['source']: arm for arm in r['arms'][pair * 2:pair * 2 + 2]}
    a, b = arms['dataset'], arms['kernel_output']
    paired_arms.append({'pair': pair, 'dataset_seconds': a['wall_seconds_including_hashing'],
                        'output_seconds': b['wall_seconds_including_hashing'],
                        'saving_percent': saving(a['wall_seconds_including_hashing'], b['wall_seconds_including_hashing'])})
fields = ('branch', 'source', 'source_visit', 'phase', 'epoch', 'wsi_id')
row_map = {tuple(row[field] for field in fields): row for row in rows}
logged = [event for event in events if event['event'] == 'read_verified']
jsonl_matches = len(logged) == len(rows) and all(
    {key: value for key, value in event.items() if key not in ('event', 'session_seconds')}
    == row_map[tuple(event[field] for field in fields)] for event in logged)
summary = {
    'kernel': 'maximshtefan/eqvae-spec0054-shard1-input-comparison/2', 'api_status': 'COMPLETE',
    'source_commit': r['source_commit'], 'completed': r['completed'], 'deadline_reached': r['deadline_reached'],
    'gpu_used': r['gpu_used'], 'session_seconds_including_hashing': r['session_seconds_including_hashing'],
    'verified_reads': len(rows), 'complete_arms': sum(arm['completed'] for arm in r['arms']),
    'complete_windows': sum(w['completed'] for d in r['branches'].values() for w in d['windows']),
    'identity_groups': len(parity), 'all_groups_four_equal_observations': all(p['n'] == 4 and p['four_source_visit_pairs'] and p['equal'] for p in parity),
    'all_orders_equal_expected': all(orders.values()), 'jsonl_matches_final_json': jsonl_matches,
    'sources': sources, 'paired_arms': paired_arms,
    'whole_arm_mean_saving_percent': saving(sources['dataset']['joint_arm_wall_seconds_including_hashing']['mean'], sources['kernel_output']['joint_arm_wall_seconds_including_hashing']['mean']),
    'read_mean_saving_percent': {phase: saving(sources['dataset']['phases'][phase]['read_seconds']['mean'], sources['kernel_output']['phases'][phase]['read_seconds']['mean']) for phase in ('full', 'sampled')},
    'pending_wsi_peak': max(e['pending_wsi'] for e in events if e['event'] == 'bag_prefetch_submit'),
    'reserved_selected_bytes_peak': max(e['reserved_selected_bytes'] for e in events if e['event'] == 'bag_prefetch_submit'),
    'min_available_host_bytes': min(e['meminfo_kib']['MemAvailable'] * 1024 for e in events if 'meminfo_kib' in e),
    'peak_process_rss_bytes': max(e['process_kib']['VmHWM'] * 1024 for e in events if 'process_kib' in e),
    'file_identities': r['files'],
    'limitations': ['Single CPU session with natural cache and ABBA order, no isolated cold-cache claim.',
                    'Paired panel is12.04GiB per source, full atlas ~200GB was not cycled.',
                    'Read timers exclude hashes but whole-arm/window times include verification.',
                    'No GPU, H2D or model training, and the minute-scale slowdown did not recur.'],
}
Path(__file__).with_name('source_benchmark_v2_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('verified_reads', 'complete_arms', 'complete_windows', 'all_groups_four_equal_observations', 'all_orders_equal_expected', 'jsonl_matches_final_json', 'whole_arm_mean_saving_percent', 'read_mean_saving_percent', 'paired_arms')}, indent=2))
for source, data in sources.items():
    print(source, 'arm', data['joint_arm_wall_seconds_including_hashing'])
    for phase, p in data['phases'].items():
        print(phase, p['read_seconds'], 'hotspot', p['hotspot3055_read_seconds'])
