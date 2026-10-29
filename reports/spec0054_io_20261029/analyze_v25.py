"""Read-only review of the completed v25 continuation and historical timings."""
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from eqvae.training.mil_dynamics_checkpoint import load_dynamics_checkpoint
from eqvae.training.mil_telemetry_io import load_telemetry_table

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports/spec0054_io_20261029'
RUNS = ROOT / 'runs/kaggle'
V25 = RUNS / 'spec0054_cls_fold0_c4_v25/spec0054_cls_fold0'
V18 = RUNS / 'spec0054_cls_fold0_c4_v18/spec0054_cls_fold0'
V10 = RUNS / 'spec0054_cls_fold0_regularized_v10/spec0054_cls_fold0'


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), **({k: float(v) for k, v in {
        'mean': a.mean(), 'median': np.median(a), 'p95': np.quantile(a, .95),
        'p99': np.quantile(a, .99), 'max': a.max(), 'sum': a.sum(),
    }.items()} if len(a) else {})}


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assessments(table, last):
    out = []
    for exp in sorted(set(table['exposures'])):
        if exp > last:
            continue
        row = {'exposures': int(exp), 'epoch': float(exp / 288)}
        for split in ('train', 'validation'):
            mask = (table['exposures'] == exp) & (table['split'] == split)
            y, p = table['target'][mask].astype(int), table['prediction'][mask].astype(int)
            cm = np.zeros((5, 5), dtype=int)
            np.add.at(cm, (y, p), 1)
            den = cm.sum(0) + cm.sum(1)
            f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(5), where=den > 0)
            row[split] = {'n': int(mask.sum()), 'ce': float(table['unweighted_loss'][mask].mean()),
                          'macro_f1': float(f1.mean()), 'accuracy': float((y == p).mean())}
        out.append(row)
    return out


log = (RUNS / 'spec0054_cls_fold0_c4_v25/eqvae-spec0054-mil-fold0.log').read_text()
logrows = json.loads(log[log.index('[{"stream_name"'):])
timed = []
for row in logrows:
    try:
        event = json.loads(row['data'])
    except (ValueError, TypeError):
        continue
    if isinstance(event, dict) and 'event' in event:
        timed.append({**event, 'kernel_seconds': row['time']})

result = {'kernel': 'maximshtefan/eqvae-spec0054-mil-fold0/25', 'status': 'COMPLETE',
          'worker_exit_codes': next(e['statuses'] for e in timed if e['event'] == 'workers_exit'),
          'timing_limitation': 'Historical session comparisons use different epochs and storage sessions; they are descriptive, not a randomized GPU reader comparison.',
          'branches': {}}
tables = {}
for branch in ('normal_vae', 'so2_vae'):
    directory = V25 / branch
    cp = load_dynamics_checkpoint(directory / 'latest.pt')
    best = load_dynamics_checkpoint(directory / 'best.pt')
    t = tables[branch] = load_telemetry_table(directory / 't0.npz')
    prior = load_telemetry_table(V18 / branch / 't0.npz')
    es = events(directory / 'events.jsonl')
    ts = [e for e in timed if e.get('branch') == branch]
    steps = [e for e in es if e['event'] == 'step_done']
    transfers = defaultdict(deque)
    for e in es:
        if e['event'] == 'bag_transfer_start' and e.get('prefetch'):
            transfers[e['wsi_id'], e['sample_indices_sha256'], e['c4_rotations_sha256']].append(e)
    consumed = [transfers[int(wsi), str(subset), str(c4)].popleft()
                for wsi, subset, c4 in zip(t['wsi_id'][1440:], t['sample_indices_sha256'][1440:],
                                          t['c4_rotations_sha256'][1440:], strict=True)]
    phase_reads = defaultdict(list)
    for index, read in enumerate(consumed):
        phase_reads[str((1440 + index) // 288 + 1)].append(read['read_seconds'])
    shard = {e['wsi_id']: e['shard'] for e in es if e['event'] == 'bag_read_start'}
    completed = assessments(load_telemetry_table(directory / 'predictions.npz'),
                            cp['scheduler']['last_assessment_exposure'])
    phases = []
    for start in [e for e in ts if e['event'] in ('evaluation_start', 'diagnostics_start')]:
        kind = start['event'].removesuffix('_start')
        end = next(e for e in ts if e['event'] == kind + '_done' and e['exposures'] == start['exposures'])
        phases.append({'phase': kind, 'exposures': start['exposures'],
                       'seconds': end['kernel_seconds'] - start['kernel_seconds']})
    oldsteps = [e for e in events(V10 / branch / 'events.jsonl')
                if e['event'] == 'step_done' and 360 < e['update'] <= cp['committed_update']]
    prefix = {k: bool(np.array_equal(t[k][:1440], v, equal_nan=True)
                      if v.dtype.kind in 'fc' else np.array_equal(t[k][:1440], v))
              for k, v in prior.items()}
    submission = [e for e in es if e['event'] == 'bag_prefetch_submit']
    by_epoch = defaultdict(list)
    for e in steps:
        by_epoch[str((e['exposures'] - 1) // 288 + 1)].append(e['step_seconds'])
    waits = [e['wait_seconds'] for e in es if e['event'] == 'bag_prefetch_consume']
    state = {k: cp[k] for k in ('committed_update', 'exposure_count', 'epoch', 'within_epoch_cursor', 'scheduler')}
    result['branches'][branch] = {
        'state': state, 'new_updates': cp['committed_update'] - 360,
        'new_presentations': cp['exposure_count'] - 1440,
        'worker_start': next(e for e in es if e['event'] == 'worker_start'),
        'resume': next(e for e in es if e['event'] == 'resume_done'),
        'v18_telemetry_prefix_equal': all(prefix.values()),
        'v18_prefix_differing_fields': [k for k, equal in prefix.items() if not equal],
        'best_epoch': best['exposure_count'] / 288,
        'best_assessment': next(a for a in completed if a['exposures'] == best['exposure_count']),
        'complete_assessments': completed, 'step_seconds': stats([e['step_seconds'] for e in steps]),
        'step_seconds_by_epoch': {k: stats(v) for k, v in by_epoch.items()},
        'matched_update_v10_step_seconds': stats([e['step_seconds'] for e in oldsteps]),
        'consumed_read_seconds': stats([e['read_seconds'] for e in consumed]),
        'consumed_read_seconds_by_epoch': {k: stats(v) for k, v in phase_reads.items()},
        'consumed_read_seconds_by_shard': {str(sh): stats([e['read_seconds'] for e in consumed if shard[e['wsi_id']] == sh]) for sh in range(1, 13)},
        'queue_wait_seconds': stats(waits),
        'queue_wait_fraction_of_step_seconds': sum(waits) / sum(e['step_seconds'] for e in steps),
        'prefetched_unused_reads': sum(len(v) for v in transfers.values()),
        'max_pending_wsi': max(e['pending_wsi'] for e in submission),
        'max_reserved_selected_bytes': max(e['reserved_selected_bytes'] for e in submission),
        'max_step_peak_allocated_bytes': max(e['peak_allocated_bytes'] for e in steps),
        'max_transfer_seconds': max(e.get('transfer_seconds', 0) for e in es),
        'amp_skipped_attempts': sum(e['amp_skipped_attempts'] for e in steps),
        'nonfinite_loss_rows': int(sum(np.count_nonzero(~np.isfinite(t[k])) for k in ('unweighted_loss', 'objective_loss'))),
        'nonfinite_telemetry_max': {k: float(np.max(t[k])) for k in t if 'nonfinite' in k},
        'phase_times': phases,
        'event_counts': dict(Counter(e['event'] for e in es)),
    }
fields = ('wsi_id', 'target', 'sample_indices_sha256', 'c4_rotations_sha256', 'update', 'exposures', 'lr', 'aem_coefficient')
n = min(len(tables[b]['update']) for b in tables)
result['common_prefix_rows'] = n
result['paired_common_prefix_equal'] = {k: bool(np.array_equal(tables['normal_vae'][k][:n], tables['so2_vae'][k][:n])) for k in fields}
result['historical_v18'] = {b: {k: v for k, v in br.items() if k in ('step_seconds', 'read_seconds', 'queue_wait_fraction')}
                           for b, br in json.loads((ROOT / 'runs/local/spec0054_cls_fold0_c4_v18_review/summary.json').read_text())['branches'].items()}
(OUTPUT / 'v25_summary.json').write_text(json.dumps(result, indent=2) + '\n')
for branch, br in result['branches'].items():
    print(branch, 'state', br['state'], 'best', br['best_assessment']['validation'])
    print('steps', br['step_seconds'], 'reads', br['consumed_read_seconds'], 'wait_fraction', br['queue_wait_fraction_of_step_seconds'])
    print('epochs', br['step_seconds_by_epoch'])
print('paired', result['common_prefix_rows'], result['paired_common_prefix_equal'])
