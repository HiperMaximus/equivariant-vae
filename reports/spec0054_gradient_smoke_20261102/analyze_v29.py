"""One-shot description of the exact v29 train-only calibration output."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eqvae.training.mil_dynamics_checkpoint import load_dynamics_checkpoint

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/kaggle/spec0054_gradient_smoke_v29/spec0054_cls_fold0_gp_smoke'
SOURCE = '6c1ad07688095453cd709bd7e6b988b292b2ee7b'


def table(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key not in ('format', 'columns')}


def stats(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return {'n': len(values), 'nonfinite': int((~finite).sum()),
            **({key: float(value) for key, value in zip(
                ('minimum', 'p10', 'median', 'p90', 'maximum'), np.quantile(values, [0, .1, .5, .9, 1]))}
               if len(values) and finite.all() else {'quantiles': None})}


def ratio(numerator, denominator):
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=valid)


with (ROOT / 'docs/data/spec0054_cohort_folds.csv').open() as source:
    cohort = {int(row['wsi_id']): row for row in csv.DictReader(source)}
report = {'kernel': 'maximshtefan/eqvae-spec0054-mil-fold0/29', 'expected_source': SOURCE,
          'scope': 'Eight early updates per branch, shard1 fold-train only; first update excluded from coefficient estimation.',
          'limitations': ['Candidate is conditional on these smoke states; no generalization estimate.',
                         'A 5% Euclidean gradient norm ratio is not a 5% AdamW parameter-update effect.',
                         'Timing includes two extra calibration derivatives, not uninstrumented production throughput.',
                         'Shard1 observations do not establish maximum full-atlas memory.'], 'branches': {}}
tables, gradient_ratios, loss_ratios, valid_rows = {}, [], [], []
for branch in ('normal_vae', 'so2_vae'):
    directory, initial = RUN / branch, RUN / 'smoke_initial' / branch
    identity = json.loads((directory / 'identity.json').read_text())
    smoke = json.loads((directory / 'smoke.json').read_text())
    counts = Counter(int(row['diagnosis_index']) for row in cohort.values() if int(row['fold']) != identity['config']['fold'])
    expected_count = sum(counts.values())
    expected_weights = [expected_count / (5 * counts[k]) for k in range(5)]
    data = tables[branch] = table(directory / 't0.npz')
    prior = table(initial / 't0.npz')
    prefix = {key: bool(np.array_equal(data[key][:len(value)], value, equal_nan=True)
                       if value.dtype.kind in 'fc' else np.array_equal(data[key][:len(value)], value))
              for key, value in prior.items()}
    loss_ratio = ratio(data['gradient_penalty'], data['weighted_smoothed_ce'])
    gradient_ratio = ratio(data['gradient_penalty_gradient_l2'], data['base_objective_gradient_l2'])
    post = data['update'] > 1
    valid = (np.isfinite(loss_ratio) & np.isfinite(gradient_ratio)
             & np.isfinite(data['classification_gradient_norm_squared'])
             & np.isfinite(data['gradient_penalty_base_cosine']))
    gradient_ratios.append(gradient_ratio[post]); loss_ratios.append(loss_ratio[post]); valid_rows.append(valid[post])
    _, steps = np.unique(data['update'], return_index=True)
    phases = {}
    for phase, path in (('initial', initial), ('resumed', directory)):
        events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
        step_events = [event for event in events if event['event'] == 'step_done']
        cp = load_dynamics_checkpoint(path / 'latest.pt')
        phases[phase] = {'worker_start': next(event for event in events if event['event'] == 'worker_start'),
                         'resume_events': [event for event in events if event['event'] == 'resume_done'],
                         'checkpoint': {key: cp[key] for key in ('committed_update', 'exposure_count', 'epoch', 'within_epoch_cursor')},
                         'max_step_allocated_bytes': max(event['peak_allocated_bytes'] for event in step_events),
                         'amp_backoffs': sum(event['event'] == 'amp_backoff' for event in events),
                         'evaluation_event_count': sum(event['event'].startswith('evaluation') for event in events),
                         'prediction_file_exists': (path / 'predictions.npz').exists()}
    groups = {}
    for name, selected in (('first_update', ~post), ('post_first_update', post)):
        groups[name] = {key: stats(value[selected]) for key, value in (
            ('raw_gradient_norm_squared', data['classification_gradient_norm_squared']),
            ('penalty', data['gradient_penalty']), ('penalty_to_weighted_smoothed_ce', loss_ratio),
            ('added_gradient_to_base_gradient', gradient_ratio),
            ('gradient_cosine', data['gradient_penalty_base_cosine']))}
        groups[name]['undefined_calibration_rows'] = int((~valid[selected]).sum())
    report['branches'][branch] = {
        'identity': identity, 'source_matches_expected': identity['source_commit'] == SOURCE, 'smoke': smoke,
        'full_fold_weights_match': smoke['class_weights'] == expected_weights,
        'canonical_schedule_count_matches': smoke['schedule_train_count'] == expected_count,
        'restored_t0_prefix_equal': all(prefix.values()), 'prefix_differences': [key for key, equal in prefix.items() if not equal],
        'phases': phases, 'groups': groups, 'step_seconds': stats(data['step_seconds'][steps]),
        'max_allocated_bytes': int(data['peak_allocated_bytes'].max()),
        'amp_skipped_attempts': int(data['amp_skipped_attempts'][steps].sum()),
        'full_bag_patches': stats(data['bag_size']), 'sampled_bag_patches': stats(data['sampled_bag_size']),
        'unique_training_wsi_count': len(set(map(int, data['wsi_id']))),
        'nontraining_presentations': sum(int(cohort[int(wsi)]['fold']) == identity['config']['fold'] for wsi in data['wsi_id']),
        'nonfinite_observations': {key: int((~np.isfinite(data[key])).sum()) for key in data if data[key].dtype.kind in 'fc'},
    }

paired_fields = ('wsi_id', 'target', 'update', 'exposures', 'sample_indices_sha256', 'c4_rotations_sha256',
                 'class_weight', 'lr', 'aem_coefficient', 'gradient_penalty_weight')
report['paired_history_equal'] = {key: bool(np.array_equal(tables['normal_vae'][key], tables['so2_vae'][key])) for key in paired_fields}
recipe = report['branches']['normal_vae']['identity']['config']['gradient_penalty_smoke']
gradient, losses = np.concatenate(gradient_ratios), [np.median(values) for values in loss_ratios]
report['selection_recipe'] = recipe
report['pooled_post_first_gradient_ratio'] = stats(gradient)
if np.concatenate(valid_rows).all() and np.median(gradient) > 0 and max(losses) > 0:
    by_gradient = recipe['weight'] * recipe['target_median_gradient_ratio'] / float(np.median(gradient))
    by_loss = recipe['weight'] * recipe['maximum_median_loss_ratio'] / float(max(losses))
    report['shared_initial_candidate'] = {'lambda': min(by_gradient, by_loss), 'gradient_based_lambda': by_gradient,
                                         'loss_cap_lambda': by_loss, 'reason': 'Smaller of shared gradient-scale estimate and worst-branch median loss cap.'}
else:
    report['shared_initial_candidate'] = None
    report['candidate_undefined_reason'] = 'Nonfinite/undefined post-first calibration observation or nonpositive ratio median; no rows silently removed.'
Path(__file__).with_name('v29_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report['shared_initial_candidate']))
