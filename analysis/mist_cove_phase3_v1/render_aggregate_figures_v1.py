"""Render aggregate-only source comparison figures after verification.

No EEG, model checkpoints, individual predictions or source descriptors read.
Error bars describe optimization seeds, not population uncertainty.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    report = json.loads(Path(source).read_text())
    if report['status'] != 'VERIFIED_ALL_18_COMPLETE_SOURCE_ONLY':
        raise ValueError('Verified complete source experiment required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    methods = ['q1', 'availability', 'background', 'rounded_q1', 'raw_context', 'evidence']
    values = report['aggregate']
    means = [values[m]['mean_subject_macro_f1']['mean'] for m in methods]
    deviations = [values[m]['mean_subject_macro_f1']['sample_sd_across_optimization_seeds'] for m in methods]
    fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
    positions = np.arange(6)
    ax.bar(positions, means, yerr=deviations, capsize=4, color=['#245b82']*4+['#65995d']*2)
    for seed, marker in [(123, 'o'), (456, 's'), (789, '^')]:
        scores = [next(r['metrics']['mean_subject_macro_f1'] for r in report['rows'] if r['seed'] == seed and r['variant'] == m) for m in methods]
        ax.scatter(positions, scores, marker=marker, s=25, facecolor='white', edgecolor='black', label=f'Seed {seed}', zorder=3)
    ax.set(xticks=positions, xticklabels=methods, ylim=(0, 1), ylabel='Participant-average fixed-five Macro-F1',
           title='Frozen source VAL: four participants, three optimization seeds')
    ax.legend(loc='upper left', ncols=3, fontsize=8)
    ax.text(.5, -.16, 'Bars: seed mean; error bars: seed SD. No reserved or external evaluation.', ha='center', transform=ax.transAxes, fontsize=9)
    fig.savefig(output/'source_comparison.png', dpi=160); plt.close(fig)
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_f1 = np.asarray([values[m]['per_stage_f1_mean'] for m in methods])
    fig, ax = plt.subplots(figsize=(8, 4.7), layout='constrained')
    image = ax.imshow(stage_f1, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    for i in range(6):
        for j in range(5):
            ax.text(j, i, f'{stage_f1[i,j]:.3f}', ha='center', va='center', color='white' if stage_f1[i,j] > .6 else 'black')
    ax.set(xticks=range(5), xticklabels=stages, yticks=range(6), yticklabels=methods,
           title='Pooled source-VAL stage F1, averaged across optimization seeds')
    fig.colorbar(image, ax=ax, label='F1'); fig.savefig(output/'stage_f1.png', dpi=160); plt.close(fig)
    with (output/'source_metrics.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'seed', 'participant_macro_f1', 'pooled_macro_f1', 'accuracy', 'kappa', 'balanced_accuracy', 'nll', 'brier', 'ece15', 'best_epoch', 'realized_epochs', 'optimizer_updates', 'parameters', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes', 'epoch_wall_seconds_excluding_checkpoint_writes']+[f'{stage}_F1' for stage in stages]+[f'{stage}_recall' for stage in stages])
        for row in report['rows']:
            met = row['metrics']
            writer.writerow([row['variant'], row['seed'], *[met[k] for k in ['mean_subject_macro_f1', 'macro_f1', 'accuracy', 'kappa', 'balanced_accuracy', 'nll', 'brier_score', 'ece_15_equal_width_bins']], row['best_epoch'], row['actual_epochs'], row['optimizer_steps'], row['parameters'], row['peak_gpu_allocated_bytes'], row['peak_gpu_reserved_bytes'], row['epoch_wall_seconds_excluding_checkpoint_writes'], *met['per_stage_f1'], *met['per_stage_recall']])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
