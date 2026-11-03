"""Plot complete fold0 assessment histories through the105-epoch horizon."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).parent
report = json.loads((out / 'v28_summary.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True)
for col, (branch, title) in enumerate((('normal_vae', 'Normal'), ('so2_vae', 'SO(2)'))):
    b = report['branches'][branch]
    assessments = b['complete_assessments']
    x = [a['epoch'] for a in assessments]
    for row, (metric, label) in enumerate((('ce', 'CE sin ponderar ↓'), ('macro_f1', 'Macro-F1 ↑'))):
        ax = axes[row, col]
        for split, name, color in (('train', 'Entrenamiento', '#2463a7'), ('validation', 'Validación', '#cc672c')):
            ax.plot(x, [a[split][metric] for a in assessments], color=color,
                    marker='o', ms=3, lw=1.8, label=name)
        ax.axvline(b['best_epoch'], color='#6d7278', ls='--', lw=1.1,
                   label=f"Checkpoint por CE (época {b['best_epoch']:g})" if row == 0 else None)
        ax.set_ylabel(label)
        ax.set_xlim(0, 106)
        ax.set_ylim((.65, 1.7) if row == 0 else (0, .8))
        ax.grid(alpha=.2)
        if row == 0:
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_xlabel('Época')
fig.suptitle('ABMIL · fold 0 · trayectoria completa hasta época 105', fontsize=14)
fig.text(.5, .013, 'Bolsas completas sin aumento: 288 WSI de entrenamiento y 73 de validación. Dropout desactivado.',
         ha='center', fontsize=9, color='#454545')
fig.tight_layout(rect=(0, .035, 1, .96))
fig.savefig(out / 'train_validation_v28.png', dpi=170)
plt.close(fig)
