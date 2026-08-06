"""
run_missingness_sensitivity_all.py
==================================
Runs the missingness sensitivity analysis for every classification task and
writes ONE combined CSV (one row per task x cohort), instead of overwriting.

Place in:  src/            Run from:  src/  ->  python run_missingness_sensitivity_all.py
Output:    outputs/robustness/missingness_sensitivity_all.csv
           outputs/robustness/missingness_sensitivity_table.tex

Reuses the existing functions in missingness_sensitivity.py; only the driver is
new. Binary tasks report AUROC; the three-stage task reports macro one-vs-rest
AUROC so it fits the same table (noted in the CSV).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.preprocess import prepare_data
from src.missingness_sensitivity import (
    _ensemble_proba, _train_ensemble, _rank_biomarkers_by_missingness,
    _select_features,
)

TASKS = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']
TASK_LABEL = {'fibrosis': 'Moderate fibrosis', 'two_stage': 'Severe fibrosis',
              'cirrhosis': 'Cirrhosis', 'three_stage': 'Three-stage'}
TOP_N = 3


def _auroc(y_true, proba, three_stage):
    """Binary AUROC, or macro one-vs-rest AUROC for the three-stage task."""
    y_true = np.asarray(y_true)
    if three_stage:
        # macro OvR; guard against a class missing in a tiny split
        try:
            return float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro'))
        except ValueError:
            return float('nan')
    p = proba[:, 1] if proba.ndim > 1 else proba
    return float(roc_auc_score(y_true, p))


def run_task(classification_type, top_n=TOP_N):
    three_stage = classification_type == 'three_stage'
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = repo_root / 'outputs' / 'data_qc' / 'missingness_profile.csv'
    if not profile_path.exists():
        raise FileNotFoundError(f'Missing {profile_path}; run preprocessing first.')

    (xs_train, ys_train, xs_val, ys_val, xs_test, ys_test,
     xs_pro, ys_pro, df_cols) = prepare_data(
        classification_type=classification_type, shap_selected=False, scaling=False)

    ranked = _rank_biomarkers_by_missingness(profile_path, set(df_cols))
    excluded = ranked.head(top_n)['biomarker'].tolist()
    reduced = [f for f in df_cols if f not in excluded]

    full = _train_ensemble(xs_train, ys_train, xs_val, ys_val, classification_type)
    reduced_models = _train_ensemble(
        _select_features(xs_train, df_cols, reduced), ys_train,
        _select_features(xs_val, df_cols, reduced), ys_val, classification_type)

    rxs_test = _select_features(xs_test, df_cols, reduced)
    rxs_pro = _select_features(xs_pro, df_cols, reduced)

    rows = []
    for cohort, xf, xr, y in [
        ('internal', xs_test[0], rxs_test[0], ys_test[0]),
        ('external', xs_pro[0], rxs_pro[0], ys_pro[0])]:
        a_full = _auroc(y, _ensemble_proba(full, xf), three_stage)
        a_red = _auroc(y, _ensemble_proba(reduced_models, xr), three_stage)
        rows.append({
            'classification_type': classification_type,
            'cohort': cohort,
            'excluded_top_missing_biomarkers': '; '.join(excluded),
            'auroc_metric': 'macro OvR' if three_stage else 'binary',
            'n_full_features': len(df_cols),
            'n_reduced_features': len(reduced),
            'auroc_full': a_full,
            'auroc_reduced': a_red,
            'delta_full_minus_reduced': a_full - a_red,
        })
    print(f'[{classification_type}] excluded {excluded} -> '
          f'internal Δ {rows[0]["delta_full_minus_reduced"]:.4f}, '
          f'external Δ {rows[1]["delta_full_minus_reduced"]:.4f}')
    return rows


def build_latex(df, out_path):
    lines = [r'\begin{table*}[htbp]', r'    \centering',
             r'    \caption{\small{Sensitivity of model discrimination to the most '
             r'incomplete biomarkers. For each task the three biomarkers with the highest '
             r'missingness were removed and the ensemble retrained; AUROC of the reduced '
             r'model is compared with the full model. $\Delta$AUROC is full minus reduced '
             r'(positive = loss). Three-stage AUROC is macro one-vs-rest.}}',
             r'    \label{tab:missingness_sensitivity}',
             r'    \begin{tabular}{llccc}', r'        \toprule',
             r'        \textbf{Task} & \textbf{Cohort} & \textbf{AUROC (full)} '
             r'& \textbf{AUROC (reduced)} & \textbf{$\Delta$AUROC}\\',
             r'        \midrule']
    for i, r in df.iterrows():
        sh = r'\rowcolor{gray!10} ' if i % 2 == 0 else ''
        task = TASK_LABEL.get(r['classification_type'], r['classification_type'])
        coh = 'UMM (internal)' if r['cohort'] == 'internal' else 'MAINZ (external)'
        lines.append(f"        {sh}{task} & {coh} & {r['auroc_full']:.3f} & "
                     f"{r['auroc_reduced']:.3f} & {r['delta_full_minus_reduced']:.3f}\\\\")
    lines += [r'        \bottomrule', r'    \end{tabular}', r'\end{table*}']
    out_path.write_text('\n'.join(lines))


def main():
    all_rows = []
    for task in TASKS:
        try:
            all_rows.extend(run_task(task))
        except Exception as exc:
            print(f'[{task}] failed: {exc}')
    if not all_rows:
        print('No results produced.')
        return
    df = pd.DataFrame(all_rows)
    out_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'robustness'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'missingness_sensitivity_all.csv'
    df.to_csv(csv_path, index=False)
    build_latex(df.reset_index(drop=True), out_dir / 'missingness_sensitivity_table.tex')
    print(f'\nCombined CSV  -> {csv_path}')
    print(f'LaTeX table   -> {out_dir / "missingness_sensitivity_table.tex"}')
    print('\nSummary:')
    print(df[['classification_type', 'cohort', 'auroc_full', 'auroc_reduced',
              'delta_full_minus_reduced']].to_string(index=False))


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
