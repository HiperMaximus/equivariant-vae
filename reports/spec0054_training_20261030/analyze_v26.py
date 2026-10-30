"""Read-only review of the completed v26 continuation and historical timings."""
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from eqvae.training.mil_dynamics_checkpoint import load_dynamics_checkpoint
from eqvae.training.mil_telemetry_io import load_telemetry_table

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports/spec0054_training_20261030'
RUNS = ROOT / 'runs/kaggle'
V25 = RUNS / 'spec0054_cls_fold0_c4_v25/spec0054_cls_fold0'
V26 = RUNS / 'spec0054_cls_fold0_c4_v26/spec0054_cls_fold0'


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


review_log = json.loads((ROOT / 'runs/local/spec0054_mil_v26_review_events.json').read_text())
result = {'kernel': 'maximshtefan/eqvae-spec0054-mil-fold0/26', 'api_status': 'COMPLETE',
          'worker_exit_codes': next(e['statuses'] for e in review_log if e.get('event') == 'workers_exit'),
          'remote_identity': json.loads((OUTPUT / 'v26_remote_identity.json').read_text()),
          'log_scope': 'Remote stdout stream contains only its final tail; branch events.jsonl and saved tables are authoritative for full-session counts.',
          'timing_limitation': 'Different epochs and storage sessions. Only shard1 input route changed; improvement in other unchanged shards prevents assigning the whole speedup to this route.',
          'branches': {}}
tables = {}
config = json.loads((ROOT / 'docs/data/spec0054_fold0_run.json').read_text())
for branch in ('normal_vae', 'so2_vae'):
    path = V26 / branch
    cp = load_dynamics_checkpoint(path / 'latest.pt')
    best = load_dynamics_checkpoint(path / 'best.pt')
    prior_cp = load_dynamics_checkpoint(V25 / branch / 'latest.pt')
    prior = load_telemetry_table(V25 / branch / 't0.npz')
    t = tables[branch] = load_telemetry_table(path / 't0.npz')
    n = len(prior['update'])
    es = events(path / 'events.jsonl')
    steps = [e for e in es if e['event'] == 'step_done']
    prepared = defaultdict(deque)
    for e in es:
        if e['event'] == 'bag_transfer_start' and e.get('prefetch'):
            prepared[e['wsi_id'], e['sample_indices_sha256'], e['c4_rotations_sha256']].append(e)
    committed_keys = list(zip(t['wsi_id'][n:].tolist(), t['sample_indices_sha256'][n:].tolist(), t['c4_rotations_sha256'][n:].tolist(), strict=True))
    # Restored AMP retries can prepare the same input again. Each committed
    # epoch/WSI hash is unique; its last preparation belongs to the retry
    # eventually committed, while earlier attempt/drain records stay separate.
    consumed = [prepared[int(w), str(s), str(c)].pop()
                for w, s, c in zip(t['wsi_id'][n:], t['sample_indices_sha256'][n:], t['c4_rotations_sha256'][n:], strict=True)]
    shards = {e['wsi_id']: e['shard'] for e in es if e['event'] == 'bag_read_start'}
    prefix = {key: bool(np.array_equal(t[key][:n], value, equal_nan=True)
                        if value.dtype.kind in 'fc' else np.array_equal(t[key][:n], value))
              for key, value in prior.items()}
    complete = assessments(load_telemetry_table(path / 'predictions.npz'), cp['scheduler']['last_assessment_exposure'])
    by_epoch = defaultdict(list)
    for e in steps:
        by_epoch[str((e['exposures'] - 1) // 288 + 1)].append(e['step_seconds'])
    peak_submit = [e for e in es if e['event'] == 'bag_prefetch_submit']
    waits = [e['wait_seconds'] for e in es if e['event'] == 'bag_prefetch_consume']
    boundaries = {0, 144} | {epoch * 288 for epoch in config['evaluation_early_epochs']} | set(range(config['evaluation_every_epochs'] * 288, (config['epochs'] + 1) * 288, config['evaluation_every_epochs'] * 288))
    result['branches'][branch] = {
        'state': {key: cp[key] for key in ('committed_update', 'exposure_count', 'epoch', 'within_epoch_cursor', 'scheduler')},
        'session_updates': cp['committed_update'] - prior_cp['committed_update'],
        'session_presentations': cp['exposure_count'] - prior_cp['exposure_count'],
        'worker_start': next(e for e in es if e['event'] == 'worker_start'),
        'resume_done': next(e for e in es if e['event'] == 'resume_done'),
        'identity_equal_v25': json.loads((path / 'identity.json').read_text()) == json.loads((V25 / branch / 'identity.json').read_text()),
        'v25_telemetry_prefix_equal': all(prefix.values()),
        'v25_prefix_differing_fields': [key for key, equal in prefix.items() if not equal],
        'best_epoch': best['exposure_count'] / 288,
        'best_assessment': next(a for a in complete if a['exposures'] == best['exposure_count']),
        'latest_complete_assessment': complete[-1], 'complete_assessments': complete,
        'pending_boundary_assessment': cp['exposure_count'] in boundaries and cp['scheduler']['last_assessment_exposure'] < cp['exposure_count'],
        'session_steps_observed': len(steps), 'step_seconds': stats([e['step_seconds'] for e in steps]),
        'step_seconds_by_epoch': {key: stats(value) for key, value in by_epoch.items()},
        'consumed_read_seconds': stats([e['read_seconds'] for e in consumed]),
        'reads_by_shard': {str(sh): stats([e['read_seconds'] for e in consumed if shards[e['wsi_id']] == sh]) for sh in range(1, 13)},
        'reads_non_shard1': stats([e['read_seconds'] for e in consumed if shards[e['wsi_id']] != 1]),
        'queue_wait_seconds': stats(waits),
        'queue_wait_fraction_of_step_seconds': sum(waits) / sum(e['step_seconds'] for e in steps),
        'max_gpu_allocated_bytes': max(e['peak_allocated_bytes'] for e in steps),
        'max_pending_wsi': max(e['pending_wsi'] for e in peak_submit),
        'max_reserved_selected_bytes': max(e['reserved_selected_bytes'] for e in peak_submit),
        'prepared_records_not_selected_in_successful_committed_pass_including_retries': sum(len(value) for value in prepared.values()),
        'committed_input_keys_unique': len(set(committed_keys)) == len(committed_keys),
        'amp_skipped_attempts': sum(e['amp_skipped_attempts'] for e in steps),
        'amp_event_counts': dict(Counter(e['event'] for e in es if 'amp' in e['event'])),
        'nonfinite_committed_losses': int(sum(np.count_nonzero(~np.isfinite(t[key])) for key in ('unweighted_loss', 'objective_loss'))),
        'max_nonfinite_telemetry': {key: float(np.max(t[key])) for key in t if 'nonfinite' in key},
        'complete_assessment_count': len(complete),
        'assessment_event_count_session': sum(e['event'] == 'assessment_done' for e in es),
        'event_counts': dict(Counter(e['event'] for e in es)),
    }
    del es, prepared, consumed, prior
fields = ('wsi_id', 'target', 'sample_indices_sha256', 'c4_rotations_sha256', 'update', 'exposures', 'lr', 'aem_coefficient')
n = min(len(t['update']) for t in tables.values())
result['common_prefix_rows'] = n
result['paired_common_prefix_equal'] = {key: bool(np.array_equal(tables['normal_vae'][key][:n], tables['so2_vae'][key][:n])) for key in fields}
result['v25_reference'] = {branch: {key: r[key] for key in ('step_seconds', 'consumed_read_seconds', 'queue_wait_fraction_of_step_seconds', 'consumed_read_seconds_by_shard')}
                         for branch, r in json.loads((ROOT / 'reports/spec0054_io_20261029/v25_summary.json').read_text())['branches'].items()}
(OUTPUT / 'v26_summary.json').write_text(json.dumps(result, indent=2) + '\n')
for branch, r in result['branches'].items():
    print(branch, 'state', r['state'], 'best_epoch', r['best_epoch'], 'best', r['best_assessment']['validation'], 'latest', r['latest_complete_assessment']['validation'])
    print('step', r['step_seconds'], 'read', r['consumed_read_seconds'], 'shard1', r['reads_by_shard']['1'], 'other_shards', r['reads_non_shard1'])
    print('wait_fraction', r['queue_wait_fraction_of_step_seconds'], 'amp_skips', r['amp_skipped_attempts'], 'nonfinite_losses', r['nonfinite_committed_losses'], 'prefix', r['v25_telemetry_prefix_equal'])
print('paired', result['common_prefix_rows'], result['paired_common_prefix_equal'])
