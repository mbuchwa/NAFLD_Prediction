"""
aggregate_results.py
====================
Collect every <model>_<task>_metrics.json produced by evaluate_performance
(after patch A) into one tidy long CSV plus publication-ready wide tables
(one xlsx sheet per task, internal + external side by side).

Place in:  src/aggregate_results.py
Run from:  src/    ->   python aggregate_results.py

It scans outputs/<model>/ and outputs/<model>/prospective/ for the JSON records,
so it works whether or not you used run_all_experiments.py.
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ORDER = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']
TASK_LABEL = {
    'fibrosis': 'Moderate Fibrosis (F0-1 vs F2-4)',
    'two_stage': 'Severe Fibrosis (F0-2 vs F3-4)',
    'cirrhosis': 'Cirrhosis (F0-3 vs F4)',
    'three_stage': 'Three-Stage (F0-1 / F2-3 / F4)',
}
MODEL_LABEL = {
    'svm': 'SVM', 'rf': 'Random Forest', 'xgb': 'XGBoost',
    'light_gbm': 'LightGBM', 'ffn': 'MLP', 'tab_transformer': 'TabTransformer',
    'vi_bnn': 'VI-BNN', 'gandalf': 'GANDALF',
}


def _fmt(v, lo=None, hi=None, pct=False, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    scale = 100.0 if pct else 1.0
    s = f'{v * scale:.{nd}f}'
    if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi)):
        s += f' ({lo * scale:.{nd}f}\u2013{hi * scale:.{nd}f})'
    return s


def load_records():
    rows = []
    for path in glob.glob('outputs/**/*_metrics.json', recursive=True):
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception:
            continue
        rows.append(rec)
    return rows


def to_long(records):
    """One row per (model, task, split, metric)."""
    long_rows = []
    for rec in records:
        base = dict(model=rec.get('model_name'),
                    task=rec.get('classification_type'),
                    split=rec.get('split'),
                    n=rec.get('n_samples'),
                    prevalence=rec.get('positive_prevalence'))
        # pooled Rubin metrics
        for m, d in (rec.get('pooled_rubin') or {}).items():
            long_rows.append({**base, 'metric': f'{m}_pooled',
                              'value': d['estimate'],
                              'ci_lower': d['ci_lower'], 'ci_upper': d['ci_upper']})
        # binary threshold metrics
        b = rec.get('binary')
        if b:
            long_rows.append({**base, 'metric': 'AUROC', 'value': b['auroc'],
                              'ci_lower': b['auroc_ci_lower'], 'ci_upper': b['auroc_ci_upper']})
            for mn in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
                lo, hi = b['operating_cis'][mn]
                long_rows.append({**base, 'metric': mn,
                                  'value': b['operating_metrics'][mn],
                                  'ci_lower': lo, 'ci_upper': hi})
        # multiclass OvR
        for r in (rec.get('multiclass_ovr') or []):
            long_rows.append({**base, 'metric': f"AUROC_{r['class_name']}",
                              'value': r['auroc_ovr'],
                              'ci_lower': r['auroc_ci_lower'],
                              'ci_upper': r['auroc_ci_upper']})
        # ordinal
        o = rec.get('ordinal')
        if o:
            for k in ['cohen_kappa_linear', 'cohen_kappa_quadratic', 'mae']:
                long_rows.append({**base, 'metric': k, 'value': o.get(k),
                                  'ci_lower': None, 'ci_upper': None})
    return pd.DataFrame(long_rows)


def build_wide_tables(records, writer):
    """One sheet per task: rows = models, columns = metrics x {internal, external}."""
    by_key = {(r.get('model_name'), r.get('classification_type'), r.get('split')): r
              for r in records}
    models = sorted({r.get('model_name') for r in records},
                    key=lambda m: list(MODEL_LABEL).index(m) if m in MODEL_LABEL else 99)

    for task in TASK_ORDER:
        binary = task != 'three_stage'
        table = []
        for model in models:
            row = {'Model': MODEL_LABEL.get(model, model)}
            for split, tag in [('internal_test', 'UMM'), ('prospective', 'MAINZ')]:
                rec = by_key.get((model, task, split))
                if not rec:
                    continue
                pooled = rec.get('pooled_rubin') or {}
                row[f'ACC {tag}'] = _fmt(pooled.get('ACC', {}).get('estimate'),
                                         pooled.get('ACC', {}).get('ci_lower'),
                                         pooled.get('ACC', {}).get('ci_upper'), pct=True)
                if binary and rec.get('binary'):
                    b = rec['binary']
                    row[f'AUROC {tag}'] = _fmt(b['auroc'], b['auroc_ci_lower'], b['auroc_ci_upper'], nd=3)
                    row[f'Sens {tag}'] = _fmt(b['operating_metrics']['Sensitivity'],
                                              *b['operating_cis']['Sensitivity'], pct=True)
                    row[f'Spec {tag}'] = _fmt(b['operating_metrics']['Specificity'],
                                              *b['operating_cis']['Specificity'], pct=True)
                    row[f'PPV {tag}'] = _fmt(b['operating_metrics']['PPV'],
                                             *b['operating_cis']['PPV'], pct=True)
                    row[f'NPV {tag}'] = _fmt(b['operating_metrics']['NPV'],
                                             *b['operating_cis']['NPV'], pct=True)
                elif not binary and rec.get('ordinal'):
                    o = rec['ordinal']
                    row[f'kappa_lin {tag}'] = _fmt(o.get('cohen_kappa_linear'), nd=3)
                    row[f'kappa_quad {tag}'] = _fmt(o.get('cohen_kappa_quadratic'), nd=3)
                    row[f'MAE {tag}'] = _fmt(o.get('mae'), nd=3)
            table.append(row)
        df = pd.DataFrame(table)
        sheet = task[:31]
        df.to_excel(writer, sheet_name=sheet, index=False)


def main():
    records = load_records()
    if not records:
        print('No *_metrics.json found. Did you apply patch A and run the sweep?')
        return

    Path('outputs/results').mkdir(parents=True, exist_ok=True)

    long_df = to_long(records)
    long_path = 'outputs/results/all_metrics_long.csv'
    long_df.to_csv(long_path, index=False)
    print(f'Tidy long table  -> {long_path}  ({len(long_df)} rows)')

    xlsx_path = 'outputs/results/manuscript_tables.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        build_wide_tables(records, writer)
    print(f'Wide tables      -> {xlsx_path}  (one sheet per task)')


if __name__ == '__main__':
    main()
