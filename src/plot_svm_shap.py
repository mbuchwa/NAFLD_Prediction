"""
plot_svm_shap.py
================
Supplement figure and table for the SVM ensemble: the model whose discrimination
is closest to FIB-4, used in the manuscript to contrast the attribution profile of
a transaminase-driven score against the tree ensembles.

Run:  python plot_svm_shap.py [shap_all_features.csv] [output_dir]
Defaults to ./shap_all_features.csv and ./outputs/figures.

WHY SHARE-OF-TOTAL AND NOT ABSOLUTE mean|SHAP|
----------------------------------------------
The SVM attributions span four orders of magnitude (0.21 down to 0.00002, a
factor of 5,000-10,000 depending on task and cohort). On a linear axis only the
first three bars are visible; on a log axis the bars stop being proportional to
what they represent and invite misreading. Each cohort's values are therefore
shown as a share of that cohort's total attribution, which is also what makes the
two cohorts comparable at all -- absolute SHAP magnitudes depend on the predicted
probability distribution, and the two cohorts differ in prevalence.

The absolute values with standard deviations are in the accompanying table, so
nothing is lost.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS = ['fibrosis', 'two_stage', 'cirrhosis']
TASK_LABEL = {'fibrosis': 'Moderate fibrosis', 'two_stage': 'Severe fibrosis',
              'cirrhosis': 'Cirrhosis'}
PANEL_LETTERS = ['a', 'b', 'c']
UMM_COL, MAINZ_COL = '#2B4C6F', '#A2583D'
TOP_N = 10
SINGLE, DOUBLE = 89 / 25.4, 183 / 25.4


def pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
        'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 0,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _tex(s):
    s = str(s)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}')):
        s = s.replace(a, b)
    return s


def load(path):
    df = pd.read_csv(path)
    df = df[(df.model == 'SVM') & (df.task.isin(TASKS))].copy()
    if df.empty:
        raise SystemExit('no SVM rows found in the input file')
    df['share'] = df.groupby(['task', 'cohort'])['mean_abs_shap'].transform(
        lambda s: s / s.sum() * 100)
    df['share_sd'] = df.apply(
        lambda r: r['sd'] / df[(df.task == r.task) & (df.cohort == r.cohort)]
        ['mean_abs_shap'].sum() * 100, axis=1)
    return df


def marker_order(df, top_n=TOP_N):
    """One fixed marker order for all panels, by mean UMM share across tasks.

    Panels that share a y-axis can be read across; per-panel ordering would force
    the reader to re-find every marker in every panel.
    """
    m = (df[df.cohort == 'UMM'].groupby('feature')['share'].mean()
         .sort_values(ascending=False))
    return list(m.index[:top_n]), list(m.index[top_n:])


def figure(df, order, rest, out_dir):
    pub_style()
    y = np.arange(len(order) + 1)[::-1]        # +1 for the aggregated remainder
    labels = [f.split(' (')[0] for f in order] + [f'Other ({len(rest)} markers)']

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 0.22 * len(y) + 1.15),
                             sharey=True)
    xmax = 0
    for k, task in enumerate(TASKS):
        ax = axes[k]
        for off, cohort, col in ((0.19, 'UMM', UMM_COL), (-0.19, 'MAINZ', MAINZ_COL)):
            sub = df[(df.task == task) & (df.cohort == cohort)].set_index('feature')
            vals = [float(sub.loc[f, 'share']) for f in order]
            vals.append(float(sub.loc[rest, 'share'].sum()) if rest else 0.0)
            errs = [float(sub.loc[f, 'share_sd']) for f in order] + [0.0]
            ax.barh(y + off, vals, 0.36, color=col, edgecolor='none',
                    label=cohort if k == 0 else None,
                    xerr=errs, error_kw=dict(ecolor='0.45', lw=0.5, capsize=1.0))
            xmax = max(xmax, max(v + e for v, e in zip(vals, errs)))
        ax.set_title(f'{PANEL_LETTERS[k]}  {TASK_LABEL[task]}', loc='left',
                     fontsize=7.6, fontweight='semibold', color='0.1', pad=6)
        ax.set_xlabel('Share of total mean |SHAP| (%)', labelpad=2, color='0.15')
        ax.grid(True, axis='x', lw=0.35, color='0.94')
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('0.55')
        ax.tick_params(axis='x', colors='0.35', labelcolor='0.15')

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=6.4)
    axes[0].set_ylim(-0.6, len(y) - 0.4)
    for ax in axes:
        ax.set_xlim(0, xmax * 1.06)
    axes[0].legend(frameon=False, loc='lower right', handlelength=1.2,
                   labelspacing=0.35)
    fig.suptitle('SVM ensemble — biomarker attribution', x=0.004, y=0.995,
                 ha='left', fontsize=8.5, fontweight='semibold', color='0.1')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'shap_svm_panel.{ext}')
    plt.close(fig)
    print(f'-> {out_dir}/shap_svm_panel.pdf (+.png)')


def table(df, order, rest, out_dir):
    rows = []
    for task in TASKS:
        for f in order:
            r = {'task': TASK_LABEL[task], 'biomarker': f}
            for cohort in ('UMM', 'MAINZ'):
                s = df[(df.task == task) & (df.cohort == cohort) &
                       (df.feature == f)].iloc[0]
                r[f'{cohort}_rank'] = int(s['rank'])
                r[f'{cohort}_shap'] = s['mean_abs_shap']
                r[f'{cohort}_sd'] = s['sd']
                r[f'{cohort}_share'] = s['share']
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / 'shap_svm_table.csv', index=False)

    lines = [r'\begin{table*}[htbp]', r'    \centering',
             r'    \caption{\small{Biomarker attribution of the SVM ensemble, the model whose',
             r'    discrimination is closest to FIB-4. The ten highest-ranked of the 20',
             r'    biomarkers are listed, ordered by their mean share of total attribution in',
             r'    the UMM test partition. Values are the mean absolute SHAP value $\pm$',
             r'    standard deviation across the $m=10$ ensemble members, with the share of',
             r'    that cohort\'s total attribution in parentheses; rank refers to all 20',
             r'    biomarkers. Absolute magnitudes are not comparable between cohorts because',
             r'    the predicted-probability distributions differ with prevalence; the shares',
             r'    and ranks are.}}',
             r'    \label{tab:shap_svm}',
             r'    \begin{tabular}{llcccc}', r'        \toprule',
             r'        & & \multicolumn{2}{c}{\textbf{UMM test partition}}'
             r' & \multicolumn{2}{c}{\textbf{MAINZ cohort}}\\',
             r'        \cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r'        \textbf{Task} & \textbf{Biomarker} & \textbf{Rank}'
             r' & \textbf{mean $|$SHAP$|$ (share)} & \textbf{Rank}'
             r' & \textbf{mean $|$SHAP$|$ (share)}\\',
             r'        \midrule']
    for task in TASKS:
        sub = tab[tab.task == TASK_LABEL[task]]
        for i, (_, r) in enumerate(sub.iterrows()):
            first = f'\\textit{{{TASK_LABEL[task]}}}' if i == 0 else ''
            lines.append(
                f'        {first} & {_tex(r.biomarker)} & {r.UMM_rank} & '
                f'{r.UMM_shap:.4f} $\\pm$ {r.UMM_sd:.4f} ({r.UMM_share:.1f}\\%) & '
                f'{r.MAINZ_rank} & '
                f'{r.MAINZ_shap:.4f} $\\pm$ {r.MAINZ_sd:.4f} ({r.MAINZ_share:.1f}\\%)\\\\')
        lines.append(r'        \midrule')
    lines[-1] = r'        \bottomrule'
    lines += [r'    \end{tabular}', r'\end{table*}']
    (out_dir / 'shap_svm_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    print(f'-> {out_dir}/shap_svm_table.tex (+.csv)')
    return tab


def main(csv_path='outputs/figures/shap_all_features.csv', out='outputs/figures'):
    out_dir = Path(out)
    df = load(csv_path)
    order, rest = marker_order(df)
    figure(df, order, rest, out_dir)
    tab = table(df, order, rest, out_dir)

    print('\nShare of total attribution, UMM test partition:')
    for task in TASKS:
        s = df[(df.task == task) & (df.cohort == 'UMM')].nlargest(6, 'share')
        print(f'  {TASK_LABEL[task]:18s} ' +
              ', '.join(f'{r.feature.split(" (")[0]} {r.share:.1f}%'
                        for _, r in s.iterrows()))
    print(f'\n  remaining {len(rest)} markers combined: ' + ', '.join(
        f'{TASK_LABEL[t]} '
        f'{df[(df.task == t) & (df.cohort == "UMM") & df.feature.isin(rest)].share.sum():.1f}%'
        for t in TASKS))
    return tab


if __name__ == '__main__':
    main(*(sys.argv[1:3] or []))