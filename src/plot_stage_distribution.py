"""
plot_stage_distribution.py
==========================
Combined fibrosis-stage histogram for UMM and MAINZ, reconstructed from the four
`class_prevalence_<task>.csv` files that export_cohort_figures writes.

    F0/1 = fibrosis class 0
    F3/4 = two_stage class 1
    F4   = cirrhosis class 1
    F3   = F3/4 - F4
    F2   = n - F0/1 - F3/4

F0 and F1 stay merged: categorize_micro maps them to the same class in all four
tasks. Apply the ANALYTIC_GRADES patch in preprocess() if you need them apart.

Cross-check: three_stage class 1 (F2/3) must equal F2 + F3. The script asserts it.

Run:  python plot_stage_distribution.py [input_dir] [output_dir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UMM_COL, MAINZ_COL = '#4878A8', '#D65F5F'
STAGES = ['F0/1', 'F2', 'F3', 'F4']
UMM_SPLITS = ('train', 'val', 'test')


def _pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 0, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def load_counts(input_dir):
    """Return {'UMM': {stage: count}, 'MAINZ': {...}} plus the split sizes."""
    frames = {}
    for task in ('fibrosis', 'two_stage', 'cirrhosis', 'three_stage'):
        p = Path(input_dir) / f'class_prevalence_{task}.csv'
        if not p.exists():
            raise SystemExit(f'missing {p}')
        frames[task] = pd.read_csv(p)

    def take(task, cls, cohort):
        df = frames[task]
        sel = df.split.isin(UMM_SPLITS) if cohort == 'UMM' else (df.split == 'prospective')
        return int(df.loc[sel & (df['class'] == cls), 'count'].sum())

    counts, sizes = {}, {}
    for cohort in ('UMM', 'MAINZ'):
        f01 = take('fibrosis', 0, cohort)
        f34 = take('two_stage', 1, cohort)
        f4 = take('cirrhosis', 1, cohort)
        n = f01 + take('fibrosis', 1, cohort)
        counts[cohort] = {'F0/1': f01, 'F2': n - f01 - f34, 'F3': f34 - f4, 'F4': f4}
        sizes[cohort] = n

        f23 = take('three_stage', 1, cohort)
        derived = counts[cohort]['F2'] + counts[cohort]['F3']
        if f23 != derived:
            raise SystemExit(f'{cohort}: three_stage F2/3 = {f23} but F2+F3 = {derived} '
                             f'-- the four task runs are not from the same cohort')

    umm = frames['fibrosis']
    per_split = {s: int(umm.loc[umm.split == s, 'n'].iloc[0]) for s in UMM_SPLITS}
    return counts, sizes, per_split


def plot(counts, sizes, output_dir):
    _pub_style()
    x = np.arange(len(STAGES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(140 / 25.4, 62 / 25.4))

    for off, cohort, col in ((-w / 2, 'UMM', UMM_COL), (w / 2, 'MAINZ', MAINZ_COL)):
        n = sizes[cohort]
        pct = [100.0 * counts[cohort][s] / n for s in STAGES]
        bars = ax.bar(x + off, pct, w, color=col, edgecolor='white', linewidth=0.7,
                      label=f'{cohort} (n = {n})')
        for b, s in zip(bars, STAGES):
            c = counts[cohort][s]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9,
                    f'{c}\n{100.0 * c / n:.1f}%', ha='center', va='bottom',
                    fontsize=6.2, linespacing=1.15, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_xlabel('Histological fibrosis stage')
    ax.set_ylabel('Share of cohort (%)')
    ax.set_ylim(0, max(100.0 * counts[c][s] / sizes[c]
                       for c in counts for s in STAGES) * 1.28)
    ax.legend(frameon=False, loc='upper left', handlelength=1.1, borderaxespad=0.2)
    ax.tick_params(axis='x', pad=2)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'stage_distribution_combined.{ext}', dpi=600,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'-> {out}/stage_distribution_combined.pdf (+.png)')


def main(input_dir='.', output_dir='outputs/figures'):
    counts, sizes, per_split = load_counts(input_dir)

    rows = []
    for s in STAGES:
        rows.append({'stage': s,
                     'UMM_n': counts['UMM'][s],
                     'UMM_pct': round(100.0 * counts['UMM'][s] / sizes['UMM'], 2),
                     'MAINZ_n': counts['MAINZ'][s],
                     'MAINZ_pct': round(100.0 * counts['MAINZ'][s] / sizes['MAINZ'], 2)})
    table = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(output_dir) / 'stage_distribution_combined.csv', index=False)

    print(f"UMM n = {sizes['UMM']} (train {per_split['train']} / "
          f"val {per_split['val']} / test {per_split['test']}), MAINZ n = {sizes['MAINZ']}")
    print(table.to_string(index=False))
    plot(counts, sizes, output_dir)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['.']))
