"""
cohort_figures.py
=================
Replacement for `export_cohort_figures`, split into the two things the old
function conflated:

  export_label_audit(df_umm, df_pro)    -- RAW frames, called BEFORE the split.
      The 654 -> 526 label cascade and the free-text/missing counts. This is the
      top stage of the CONSORT diagram and can only be computed on the raw data.

  export_cohort_figures(xs_train, ys_train, ..., classification_type)
      -- ANALYTIC cohort, called AFTER the four preprocess() calls, straight
      from their return values. Stage histogram, per-split class prevalence,
      cohort sizes.

Copy both functions into preprocess.py (or import them from here) and apply the
one-line patch to preprocess() described under ANALYTIC_GRADES below.

--------------------------------------------------------------------------
WHY THE ONE-LINE PATCH IS NEEDED
--------------------------------------------------------------------------
By the time preprocess() returns, `ys` has already been through
categorize_micro(): 0/1 for the binary tasks, 0/1/2 for three_stage. The F0-F4
grades are gone. A stage histogram therefore cannot be built from xs/ys alone.

The grade still exists inside preprocess(), one line before the binarisation.
Capture it there into a module-level registry:

    # --- in preprocess(), immediately BEFORE this existing line: ---
    #     df['Micro'] = df['Micro'].apply(lambda x: categorize_micro(...))
    ANALYTIC_GRADES[data_type] = df['Micro'].astype(int).to_numpy()
    # --------------------------------------------------------------

Do NOT add the grade as a column on `df`. The mice() loop takes
`x = df.drop('Micro', axis=1)`, so any extra grade column would be imputed and
handed to the models as a feature -- i.e. the label itself as a predictor.
The registry keeps it out of the feature matrix entirely.

The arrays land in the same row order as the returned xs/ys, because the
registry is written after every row-dropping step in preprocess().
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# data_type -> np.ndarray of F0-F4 grades, in the row order of the returned xs/ys.
# Written by preprocess(); each call overwrites its own key.
ANALYTIC_GRADES = {}

SINGLE, DOUBLE = 89 / 25.4, 183 / 25.4
UMM_COL, MAINZ_COL = '#4878A8', '#D65F5F'

TASK_CLASS_NAMES = {
    'fibrosis': {0: 'F0/1', 1: 'F2/3/4'},
    'two_stage': {0: 'F0/1/2', 1: 'F3/4'},
    'cirrhosis': {0: 'F0/1/2/3', 1: 'F4'},
    'three_stage': {0: 'F0/1', 1: 'F2/3', 2: 'F4'},
}


def _pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _save(fig, output_dir, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{output_dir}/{stem}.{ext}', dpi=600,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'[figures] -> {output_dir}/{stem}.pdf (+.png)')


def _first(seq):
    """xs/ys are lists of m imputations; labels are identical across them."""
    a = seq[0] if isinstance(seq, (list, tuple)) else seq
    return np.asarray(a).ravel()


def _n_rows(seq):
    a = seq[0] if isinstance(seq, (list, tuple)) else seq
    return int(np.asarray(a).shape[0])


# ============================================================ label audit ===
def export_label_audit(df_umm, df_pro, output_dir='outputs/figures', label_col='Micro'):
    """Label-exclusion cascade on the RAW frames. Call BEFORE the split.

    This is the only place the 654 -> 526 numbers and the free-text/missing
    counts exist; they are gone once preprocess() has run.
    """
    os.makedirs(output_dir, exist_ok=True)

    def audit(df):
        col = df[label_col]
        num = pd.to_numeric(col, errors='coerce')
        n_freetext = int(col.apply(lambda v: isinstance(v, str)).sum())
        n_nan = int(col.isna().sum())
        grades = num.dropna().astype(int)
        grades = grades[(grades >= 0) & (grades <= 4)]
        return len(grades), n_freetext, n_nan

    umm_n, umm_ft, umm_na = audit(df_umm)
    pro_n, pro_ft, pro_na = audit(df_pro)

    table = pd.DataFrame([
        {'cohort': 'UMM', 'n_total': len(df_umm), 'n_numeric_grade': umm_n,
         'n_freetext_report': umm_ft, 'n_missing': umm_na},
        {'cohort': 'MAINZ', 'n_total': len(df_pro), 'n_numeric_grade': pro_n,
         'n_freetext_report': pro_ft, 'n_missing': pro_na},
    ])
    table.to_csv(f'{output_dir}/label_exclusion_summary.csv', index=False)
    print(f'[audit] label cascade -> {output_dir}/label_exclusion_summary.csv')
    print(f'[audit]   UMM: {umm_n} graded, {umm_ft} free-text, {umm_na} missing (of {len(df_umm)})')
    print(f'[audit]   MAINZ: {pro_n} graded, {pro_ft} free-text, {pro_na} missing (of {len(df_pro)})')
    return table


# ======================================================== cohort figures ====
def export_cohort_figures(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test,
                          xs_pro, ys_pro, classification_type,
                          output_dir='outputs/figures', grades=None):
    """Figures and tables for the ANALYTIC cohort. Call AFTER the preprocess() calls.

    Args:
        xs_*/ys_*: exactly what preprocess() returned (lists of m imputations).
        classification_type: needed to name the task classes.
        grades: dict data_type -> F0-F4 array. Defaults to ANALYTIC_GRADES,
            which preprocess() fills via the one-line patch (see module docstring).
            If empty, the stage histogram is replaced by the task-class
            distribution and labelled as such rather than mislabelled as F0-F4.

    Returns:
        (stage_table, prevalence_table) as DataFrames; both are also written to CSV.
    """
    _pub_style()
    os.makedirs(output_dir, exist_ok=True)
    grades = ANALYTIC_GRADES if grades is None else grades

    splits = {'train': (xs_train, ys_train), 'val': (xs_val, ys_val),
              'test': (xs_test, ys_test), 'prospective': (xs_pro, ys_pro)}

    # ---- cohort sizes -----------------------------------------------------
    sizes = {k: _n_rows(v[0]) for k, v in splits.items()}
    n_umm = sizes['train'] + sizes['val'] + sizes['test']
    size_table = pd.DataFrame(
        [{'split': k, 'n': v} for k, v in sizes.items()]
        + [{'split': 'UMM total (train+val+test)', 'n': n_umm}])
    size_table.to_csv(f'{output_dir}/cohort_sizes.csv', index=False)
    print(f"[figures] analytic cohort: UMM n={n_umm} "
          f"(train {sizes['train']} / val {sizes['val']} / test {sizes['test']}), "
          f"MAINZ n={sizes['prospective']}")

    # ---- class prevalence per split --------------------------------------
    names = TASK_CLASS_NAMES.get(classification_type, {})
    prev_rows = []
    for split, (xs, ys) in splits.items():
        y = _first(ys).astype(int)
        for cls in sorted(np.unique(y)):
            cnt = int((y == cls).sum())
            prev_rows.append({
                'split': split, 'task': classification_type, 'class': int(cls),
                'class_label': names.get(int(cls), str(cls)),
                'count': cnt, 'n': len(y),
                'prevalence_pct': round(100.0 * cnt / len(y), 2)})
    prevalence_table = pd.DataFrame(prev_rows)
    prevalence_table.to_csv(
        f'{output_dir}/class_prevalence_{classification_type}.csv', index=False)
    pos = prevalence_table[prevalence_table['class'] == prevalence_table['class'].max()]
    print(f'[figures] positive-class prevalence -- '
          + ', '.join(f"{r['split']} {r['prevalence_pct']}%" for _, r in pos.iterrows()))

    # ---- stage histogram --------------------------------------------------
    have_grades = all(k in grades and grades[k] is not None
                      for k in ('train', 'val', 'test', 'prospective'))
    if have_grades:
        umm_vals = np.concatenate([np.asarray(grades[k]).ravel()
                                   for k in ('train', 'val', 'test')]).astype(int)
        mainz_vals = np.asarray(grades['prospective']).ravel().astype(int)
        if len(umm_vals) != n_umm or len(mainz_vals) != sizes['prospective']:
            print(f'[figures] !! grade registry out of sync with xs/ys '
                  f'({len(umm_vals)} vs {n_umm}, {len(mainz_vals)} vs {sizes["prospective"]}) '
                  f'-- the capture line must sit after every row-dropping step')
            have_grades = False

    if have_grades:
        ticks, labels, xlabel, stem = list(range(5)), [f'F{i}' for i in range(5)], \
            'Fibrosis stage (histological)', 'stage_histograms_analytic_cohort'
    else:
        umm_vals = np.concatenate([_first(splits[k][1]) for k in ('train', 'val', 'test')]).astype(int)
        mainz_vals = _first(ys_pro).astype(int)
        ticks = sorted(set(umm_vals.tolist()) | set(mainz_vals.tolist()))
        labels = [names.get(t, str(t)) for t in ticks]
        xlabel = f'Class ({classification_type})'
        stem = f'class_histograms_{classification_type}'
        print('[figures] NOTE: no grade registry -- plotting task classes, not F0-F4. '
              'Apply the one-line patch in preprocess() for a true stage histogram.')

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
    for ax, vals, title, col in [(axes[0], umm_vals, 'UMM', UMM_COL),
                                 (axes[1], mainz_vals, 'MAINZ', MAINZ_COL)]:
        counts = pd.Series(vals).value_counts().reindex(ticks, fill_value=0)
        ax.bar(range(len(ticks)), counts.values, color=col,
               edgecolor='white', linewidth=0.6, width=0.8)
        for i, v in enumerate(counts.values):
            if v:
                ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=6)
        ax.set_title(f'{title} (n={int(counts.sum())})', loc='left', fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Number of patients')
    fig.tight_layout()
    _save(fig, output_dir, stem)

    stage_table = pd.DataFrame({
        'stage': labels,
        'UMM': [int((umm_vals == t).sum()) for t in ticks],
        'MAINZ': [int((mainz_vals == t).sum()) for t in ticks]})
    stage_table.to_csv(f'{output_dir}/stage_distribution_analytic_cohort.csv', index=False)
    print('[figures] stage distribution of the analytic cohort:')
    print(stage_table.to_string(index=False))

    return stage_table, prevalence_table
