"""Descriptive train/validation comparison using complete exploratory fold0 assessments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'reports/spec0054_training_20261030/v26_summary.json'
r = json.loads(source.read_text())
report = {'source_kernel': 'maximshtefan/eqvae-spec0054-mil-fold0/26',
          'last_complete_assessment_epoch': 80, 'validation_wsi': 73, 'training_wsi': 288,
          'v27_status': 'RUNNING; single current log read has no worker output',
          'comparison': 'Ordinary unweighted CE and macro-F1 on full unaugmented bags with dropout disabled in both splits.',
          'interpretation': 'A train/validation gap and worse CE after the selected epoch55 are indications of overfitting; metric fluctuations and F1 improvement prevent claiming monotonic generalization deterioration.',
          'branches': {}}
fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True)
for column, (branch, title) in enumerate([('normal_vae', 'Normal'), ('so2_vae', 'SO(2)')]):
    b = r['branches'][branch]
    assessments = b['complete_assessments']
    selected = b['best_assessment']
    last = b['latest_complete_assessment']
    report['branches'][branch] = {
        'selected_epoch': b['best_epoch'], 'selected': selected, 'last_complete': last,
        'ce_gap_last': last['validation']['ce'] - last['train']['ce'],
        'f1_gap_last': last['train']['macro_f1'] - last['validation']['macro_f1'],
        'train_ce_change_selected_to_last': last['train']['ce'] - selected['train']['ce'],
        'validation_ce_change_selected_to_last': last['validation']['ce'] - selected['validation']['ce'],
        'validation_f1_change_selected_to_last': last['validation']['macro_f1'] - selected['validation']['macro_f1'],
    }
    x = [a['epoch'] for a in assessments]
    for row, (metric, ylabel) in enumerate([('ce', 'CE sin ponderar ↓'), ('macro_f1', 'Macro-F1 ↑')]):
        ax = axes[row, column]
        for split, label, color in [('train', 'Entrenamiento', '#2463a7'), ('validation', 'Validación', '#cc672c')]:
            ax.plot(x, [a[split][metric] for a in assessments], marker='o', ms=3, lw=1.8,
                    color=color, label=label)
        ax.axvline(b['best_epoch'], color='#6d7278', ls='--', lw=1.1,
                   label='Checkpoint por CE (época 55)' if row == 0 else None)
        ax.grid(alpha=.2)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 81)
        ax.set_ylim((.65, 1.7) if row == 0 else (0, .8))
        if row == 0:
            ax.set_title(title)
        else:
            ax.set_xlabel('Época')
    axes[0, column].legend(loc='lower left', fontsize=8)
fig.suptitle('ABMIL · fold 0 · entrenamiento y validación', fontsize=14)
fig.text(.5, .013, 'Evaluaciones completas: 288 WSI de entrenamiento y 73 de validación. Última evaluación: época 80.',
         ha='center', fontsize=9, color='#454545')
fig.tight_layout(rect=(0, .035, 1, .96))
out = Path(__file__).parent
fig.savefig(out / 'train_validation_v26.png', dpi=170)
plt.close(fig)
(out / 'overfit_review.json').write_text(json.dumps(report, indent=2) + '\n')
for branch, b in report['branches'].items():
    print(branch, {k: v for k, v in b.items() if k not in ('selected', 'last_complete')})
