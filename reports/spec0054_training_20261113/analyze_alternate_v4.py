"""Read saved alternate-account gradient-penalty session; no inference or tuning."""
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from eqvae.training.mil_dynamics_checkpoint import load_dynamics_checkpoint

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
RUN = ROOT / 'runs/kaggle/spec0054_cls_fold0_gp_alternate_v4/spec0054_cls_fold0_gp'
CONTROL = json.loads((ROOT / 'reports/spec0054_training_20261102/v28_summary.json').read_text())


def table(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key not in ('format', 'columns')}


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), **{key: float(value) for key, value in zip(
        ('mean', 'median', 'p95', 'maximum', 'sum'),
        (values.mean(), np.median(values), np.quantile(values, .95), values.max(), values.sum()))}}


def assessments(predictions, last):
    results = []
    for exposure in np.unique(predictions['exposures']):
        if exposure > last:
            continue
        result = {'exposures': int(exposure), 'epoch': float(exposure / 288)}
        for split in ('train', 'validation'):
            mask = (predictions['exposures'] == exposure) & (predictions['split'] == split)
            y, pred = predictions['target'][mask].astype(int), predictions['prediction'][mask].astype(int)
            cm = np.zeros((5, 5), dtype=int)
            np.add.at(cm, (y, pred), 1)
            den = cm.sum(0) + cm.sum(1)
            f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(5), where=den > 0)
            result[split] = {'n': int(mask.sum()), 'ce': float(predictions['unweighted_loss'][mask].mean()),
                             'macro_f1': float(f1.mean()), 'accuracy': float((pred == y).mean())}
        results.append(result)
    return results


report = {'kernel': 'maximusshtefan/eqvae-spec0054-gradient-penalty-fold0/4', 'api_status': 'COMPLETE',
          'worker_exit_codes': [0, 0],
          'scope': 'Sixth session of the exploratory fold0 gradient-penalty candidate, not completion of105 epochs.',
          'log_scope': 'Downloaded branch event files contain full history; remote stdout is a final tail.',
          'timing_limitation': 'Different storage sessions prevent causal throughput comparison with control or smoke.',
          'branches': {}}
PRIOR = ROOT / 'runs/kaggle/spec0054_cls_fold0_gp_alternate_v3/spec0054_cls_fold0_gp'
PRIOR_SUMMARY = json.loads((ROOT / 'reports/spec0054_training_20261112/alternate_v3_summary.json').read_text())
tables = {}
for branch in ('normal_vae', 'so2_vae'):
    directory = RUN / branch
    cp, best = (load_dynamics_checkpoint(directory / name) for name in ('latest.pt', 'best.pt'))
    data = tables[branch] = table(directory / 't0.npz')
    predictions = table(directory / 'predictions.npz')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    resume = next(event for event in events if event['event'] == 'resume_done')
    prior_data = table(PRIOR / branch / 't0.npz')
    session_data = {key: values[data['update'] > resume['update']] for key, values in data.items()}
    steps = [event for event in events if event['event'] == 'step_done']
    prepared = defaultdict(deque)
    for event in events:
        if event['event'] == 'bag_transfer_start' and event.get('prefetch'):
            prepared[event['wsi_id'], event['sample_indices_sha256'], event['c4_rotations_sha256']].append(event)
    # The last preparation is the successful attempt when AMP replays an input.
    reads = [prepared[int(wsi), str(indices), str(c4)].pop() for wsi, indices, c4 in zip(
        session_data['wsi_id'], session_data['sample_indices_sha256'], session_data['c4_rotations_sha256'], strict=True)]
    shard_by_wsi = {event['wsi_id']: event['shard'] for event in events if event['event'] == 'bag_read_start'}
    waits = [event['wait_seconds'] for event in events if event['event'] == 'bag_prefetch_consume']
    complete = assessments(predictions, cp['scheduler']['last_assessment_exposure'])
    latest = complete[-1]
    selected = next(row for row in complete if row['exposures'] == best['exposure_count'])
    control = next(row for row in CONTROL['branches'][branch]['complete_assessments']
                   if row['exposures'] == latest['exposures'])
    largest = data['sampled_bag_size'].max()
    mask = data['sampled_bag_size'] == largest
    attention = table(directory / 't2.npz')
    attention_mask = (attention['capture_name'] == 'attention') & (attention['record_kind'] == 'summary') & (attention['exposures'] <= latest['exposures'])
    attention_last_exposure = int(attention['exposures'][attention_mask].max())
    attention_mask &= attention['exposures'] == attention_last_exposure
    attention_keys = {'normalized_entropy_mean', 'effective_instance_count_mean', 'maximum_mass_mean', 'top_100_mass_mean'}
    identity = json.loads((directory / 'identity.json').read_text())
    report['branches'][branch] = {
        'identity': identity,
        'source_matches_expected': identity['source_commit'] == 'e0d1b81433e5b6787b2329f9ec31afc9bf854a6b',
        'worker_start': next(event for event in events if event['event'] == 'worker_start'),
        'training_recipe': next(event for event in events if event['event'] == 'training_recipe'),
        'resume_done': resume,
        'resumed_pending_assessment_completed_before_training': (next(index for index, event in enumerate(events) if event['event'] == 'assessment_done' and event['exposures'] == resume['exposures']) < next(index for index, event in enumerate(events) if event['event'] == 'step_done')) if PRIOR_SUMMARY['branches'][branch]['pending_boundary_assessment'] else None,
        'identity_equals_alternate_v3': identity == json.loads((PRIOR / branch / 'identity.json').read_text()),
        'restored_t0_prefix_equal': {key: bool(np.array_equal(data[key][:len(value)], value, equal_nan=True) if value.dtype.kind in 'fc' else np.array_equal(data[key][:len(value)], value)) for key, value in prior_data.items()},
        'session_added_updates': cp['committed_update'] - resume['update'],
        'session_added_presentations': cp['exposure_count'] - resume['exposures'],
        'resume_event_count': sum(event['event'] == 'resume_done' for event in events),
        'state': {key: cp[key] for key in ('committed_update', 'exposure_count', 'epoch', 'within_epoch_cursor', 'scheduler')},
        'epoch_fraction_completed': cp['exposure_count'] / 288,
        'checkpoint_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ('latest.pt', 'best.pt')},
        'final_checkpoint_exists': (directory / 'final.pt').exists(),
        'session_pause': next(event for event in events if event['event'] in ('session_pause', 'session_pause_during_assessment')),
        'pending_boundary_assessment': any(event['event'] == 'session_pause_during_assessment' for event in events),
        'complete_assessments': complete, 'best_assessment': selected,
        'latest_complete_assessment': latest, 'control_at_same_latest_boundary': control,
        'step_seconds': stats([event['step_seconds'] for event in steps]),
        'consumed_training_read_seconds': stats([event['read_seconds'] for event in reads]),
        'read_seconds_by_shard': {str(shard): stats([event['read_seconds'] for event in reads if shard_by_wsi[event['wsi_id']] == shard])
                                  for shard in range(1, 13)},
        'queue_wait_fraction_of_summed_step_wall': sum(waits) / sum(event['step_seconds'] for event in steps),
        'nonwait_step_wall_seconds': sum(event['step_seconds'] for event in steps) - sum(waits),
        'max_allocated_bytes': max(event['peak_allocated_bytes'] for event in steps),
        'largest_sampled_bag': {'sampled_patches': int(largest), 'wsi_ids': np.unique(data['wsi_id'][mask]).tolist(),
                                'presentations': int(mask.sum()),
                                'max_containing_step_allocated_bytes': int(data['peak_allocated_bytes'][mask].max())},
        'amp_backoffs': [{key: event[key] for key in ('update', 'attempt', 'scale_before', 'scale_after')} for event in events if event['event'] == 'amp_backoff'],
        'latest_full_bag_sentinel_attention_epoch': attention_last_exposure / 288,
        'latest_full_bag_sentinel_attention': [{key: attention[key][index].item() for key in ('wsi_id', 'metric_name', 'value')} for index in np.flatnonzero(attention_mask) if str(attention['metric_name'][index]) in attention_keys],
        'amp_skipped_attempts': sum(event['amp_skipped_attempts'] for event in steps),
        'nonfinite_numeric_training_values': {key: int((~np.isfinite(value)).sum()) for key, value in data.items() if value.dtype.kind in 'fc'},
        'gradient_penalty': stats(data['gradient_penalty']),
        'gradient_penalty_to_weighted_smoothed_ce': stats(data['gradient_penalty'] / data['weighted_smoothed_ce']),
        'objective_decomposition_max_abs_error': float(np.max(np.abs(data['objective_loss'] - (
            data['weighted_smoothed_ce'] - data['aem_penalty'] + data['gradient_penalty'])))),
        'unused_or_retried_preparations': sum(len(value) for value in prepared.values()),
        'event_counts': dict(Counter(event['event'] for event in events)),
    }

n = min(len(data['update']) for data in tables.values())
report['paired_common_prefix_rows'] = n
report['paired_common_prefix_equal'] = {key: bool(np.array_equal(tables['normal_vae'][key][:n], tables['so2_vae'][key][:n]))
    for key in ('wsi_id', 'target', 'update', 'exposures', 'sample_indices_sha256', 'c4_rotations_sha256',
                'class_weight', 'lr', 'aem_coefficient', 'gradient_penalty_weight')}
common = min(value['latest_complete_assessment']['exposures'] for value in report['branches'].values())
report['control_at_common_boundary'] = {branch: next(row for row in CONTROL['branches'][branch]['complete_assessments'] if row['exposures'] == common) for branch in tables}
log_entries = json.loads((RUN.parent / 'eqvae-spec0054-gradient-penalty-fold0.log').read_text())
report['worker_exit_codes'] = next(json.loads(entry['data'])['statuses'] for entry in log_entries if '"event": "workers_exit"' in entry['data'])
report['common_complete_assessment'] = {branch: next(row for row in value['complete_assessments'] if row['exposures'] == common) for branch, value in report['branches'].items()}
(OUT / 'alternate_v4_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
for branch, result in report['branches'].items():
    print(branch, 'progress', result['epoch_fraction_completed'], 'latest', result['latest_complete_assessment'])
    print('memoryGiB', result['max_allocated_bytes'] / 2**30, 'largest', result['largest_sampled_bag'])
    print('read', result['consumed_training_read_seconds'], 'queue_wait_fraction', result['queue_wait_fraction_of_summed_step_wall'])
