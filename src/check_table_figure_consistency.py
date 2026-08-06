"""
check_table_figure_consistency.py
=================================
Diagnoses B4: which model checkpoints still reproduce the AUROCs printed in the
manuscript tables, and which do not.

Run from:  src/   ->   python check_table_figure_consistency.py

--------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------
For every model / task / cohort it

  1. loads models/<model>/model_<task>.pickle,
  2. records the file's mtime, size and SHA-256 (so two tasks accidentally
     sharing one file, or a checkpoint older than the tables, become visible),
  3. checks the feature count the model expects -- a 3-feature reduced model
     sitting in a full-model slot is a common cause of a single task being off,
  4. recomputes AUROC under BOTH ensemble conventions:
        paired    member i evaluated on imputation i   (what the manuscript
                  describes and what the figure scripts now do)
        imp0      every member evaluated on imputation 0  (what the older
                  scripts did)
  5. compares both against TABLE_AUROC below.

The point is to find out which of the two numbers in B4 — 0.948 (Table 2) or
0.897 (Figure 4) — the checkpoints on disk actually produce. Whichever it is,
one of the two artefacts was generated from a different model than the other,
and only a re-run of the losing artefact fixes it.

--------------------------------------------------------------------------
READING THE OUTPUT
--------------------------------------------------------------------------
  match(paired)     the tables were made with the current checkpoints and the
                    correct pairing -> regenerate the FIGURES, tables stay.
  match(imp0)       the tables were made with the current checkpoints but the
                    old pairing -> regenerate the TABLES with the fixed
                    ensemble function; the figures are right.
  no match          the checkpoint is not the model the tables were made from.
                    Check mtime: if it is newer than tables_revised.tex, the
                    model was retrained after the tables were written and the
                    tables must be regenerated.

--------------------------------------------------------------------------
THE STRUCTURAL FIX
--------------------------------------------------------------------------
This script diagnoses; it does not prevent recurrence. Tables and figures
diverge because they are produced by two scripts that each load the checkpoints
independently. The durable fix is one script that loads every checkpoint once,
computes every number once, and emits both the .tex tables and the figures from
that single in-memory result. Until then, always regenerate both together.
"""

import hashlib
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TASKS = ['fibrosis', 'two_stage', 'cirrhosis']
MODEL_DIRS = {'SVM': 'svm', 'Random Forest': 'rf', 'XGBoost': 'xgb',
              'LightGBM': 'light_gbm', 'VI-BNN': 'vi_bnn', 'GANDALF': 'gandalf'}
TOLERANCE = 0.005

# AUROCs exactly as printed in the manuscript. Edit here if the tables change.
TABLE_AUROC = {
    ('fibrosis', 'UMM'):    {'SVM': 0.714, 'Random Forest': 0.923, 'XGBoost': 0.934,
                             'LightGBM': 0.888, 'VI-BNN': 0.781, 'GANDALF': 0.730},
    ('two_stage', 'UMM'):   {'SVM': 0.703, 'Random Forest': 0.923, 'XGBoost': 0.903,
                             'LightGBM': 0.924, 'VI-BNN': 0.764, 'GANDALF': 0.692},
    ('cirrhosis', 'UMM'):   {'SVM': 0.684, 'Random Forest': 0.877, 'XGBoost': 0.807,
                             'LightGBM': 0.882, 'VI-BNN': 0.754, 'GANDALF': 0.615},
    ('fibrosis', 'MAINZ'):  {'SVM': 0.833, 'Random Forest': 0.859, 'XGBoost': 0.889,
                             'LightGBM': 0.860, 'VI-BNN': 0.835, 'GANDALF': 0.606},
    ('two_stage', 'MAINZ'): {'SVM': 0.869, 'Random Forest': 0.905, 'XGBoost': 0.911,
                             'LightGBM': 0.896, 'VI-BNN': 0.861, 'GANDALF': 0.678},
    ('cirrhosis', 'MAINZ'): {'SVM': 0.848, 'Random Forest': 0.876, 'XGBoost': 0.948,
                             'LightGBM': 0.878, 'VI-BNN': 0.854, 'GANDALF': 0.690},
}

# Reference artefacts whose mtime is compared against the checkpoints.
ARTEFACTS = ['outputs/tables/tables_revised.tex', 'tables_revised.tex',
             'outputs/figures/roc_panel_mainz.pdf']


def _sha(path, limit=8):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:limit]


def _n_features(mdl):
    for attr in ('n_features_in_', 'n_features_', 'num_feature'):
        v = getattr(mdl, attr, None)
        if callable(v):
            try:
                return int(v())
            except Exception:
                continue
        if v is not None:
            return int(v)
    booster = getattr(mdl, 'booster_', None)
    if booster is not None and hasattr(booster, 'num_feature'):
        return int(booster.num_feature())
    return None


def _proba(models, xs, paired):
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    out = []
    for i, mdl in enumerate(models):
        x = np.asarray(xs[i] if (paired and i < len(xs)) else xs[0])
        p = np.asarray(mdl.predict_proba(x))
        out.append(p)
    p = np.mean(out, axis=0)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    print('Reference artefact timestamps:')
    for a in ARTEFACTS:
        p = Path(a)
        if p.exists():
            print(f'  {a}: {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}')
    print()

    rows = []
    for task in TASKS:
        d = prepare_data(classification_type=task, shap_selected=False, scaling=False)
        data = {'UMM': (d[4], np.asarray(d[5][0]).ravel()),
                'MAINZ': (d[6], np.asarray(d[7][0]).ravel())}

        for label, sub in MODEL_DIRS.items():
            path = Path(f'models/{sub}/model_{task}.pickle')
            if not path.exists():
                # vi_bnn / gandalf save one artefact per ensemble member
                try:
                    from src.neural_loaders import load_any_ensemble, LOADERS
                except ImportError:
                    try:
                        from neural_loaders import load_any_ensemble, LOADERS
                    except ImportError:
                        LOADERS = {}
                if sub in LOADERS:
                    try:
                        nmodels = load_any_ensemble(sub, task)
                    except Exception as exc:
                        print(f'  {label}/{task}: loader failed — {exc}')
                        continue
                    base = {'task': task, 'model': label, 'n_members': len(nmodels),
                            'n_features': None, 'sha': f'{sub}-per-member',
                            'mtime': 'per-member'}
                    for cohort, (xs, y) in data.items():
                        exp = TABLE_AUROC.get((task, cohort), {}).get(label, np.nan)
                        r = dict(base, cohort=cohort, table_auroc=exp)
                        for tag, paired in (('paired', True), ('imp0', False)):
                            try:
                                r[f'auroc_{tag}'] = round(float(
                                    roc_auc_score(y, _proba(nmodels, xs, paired))), 4)
                            except Exception as exc:
                                r[f'auroc_{tag}'] = np.nan
                                r['error'] = str(exc)[:70]
                        if not np.isnan(exp):
                            dp = abs(r.get('auroc_paired', np.nan) - exp)
                            di = abs(r.get('auroc_imp0', np.nan) - exp)
                            r['verdict'] = ('match(paired)' if dp <= TOLERANCE else
                                            'match(imp0)' if di <= TOLERANCE else 'NO MATCH')
                            r['delta_paired'] = round(float(dp), 4)
                        rows.append(r)
                    continue
                print(f'  {label}/{task}: no checkpoint')
                continue
            st = path.stat()
            try:
                with open(path, 'rb') as fh:
                    models = pickle.load(fh)
            except Exception as exc:
                print(f'  {label}/{task}: unreadable ({exc})')
                continue

            nfeat = _n_features(models[0]) if len(models) else None
            base = {'task': task, 'model': label, 'n_members': len(models),
                    'n_features': nfeat, 'sha': _sha(path),
                    'mtime': datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M')}

            for cohort, (xs, y) in data.items():
                exp = TABLE_AUROC.get((task, cohort), {}).get(label, np.nan)
                r = dict(base, cohort=cohort, table_auroc=exp)
                for tag, paired in (('paired', True), ('imp0', False)):
                    try:
                        r[f'auroc_{tag}'] = round(float(
                            roc_auc_score(y, _proba(models, xs, paired))), 4)
                    except Exception as exc:
                        r[f'auroc_{tag}'] = np.nan
                        r['error'] = str(exc)[:70]
                if not np.isnan(exp):
                    dp = abs(r.get('auroc_paired', np.nan) - exp)
                    di = abs(r.get('auroc_imp0', np.nan) - exp)
                    r['verdict'] = ('match(paired)' if dp <= TOLERANCE else
                                    'match(imp0)' if di <= TOLERANCE else 'NO MATCH')
                    r['delta_paired'] = round(float(dp), 4)
                rows.append(r)

    df = pd.DataFrame(rows)
    out = Path('outputs/robustness'); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'table_figure_consistency.csv', index=False)

    cols = [c for c in ['task', 'cohort', 'model', 'table_auroc', 'auroc_paired',
                        'auroc_imp0', 'delta_paired', 'verdict', 'n_features',
                        'n_members', 'mtime'] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f'\n-> {out}/table_figure_consistency.csv')

    if 'verdict' in df.columns:
        bad = df[df.verdict == 'NO MATCH']
        print(f'\nSummary: {(df.verdict == "match(paired)").sum()} match(paired), '
              f'{(df.verdict == "match(imp0)").sum()} match(imp0), {len(bad)} NO MATCH')
        if len(bad):
            print('\nCheckpoints that do not reproduce their table value:')
            print(bad[['task', 'cohort', 'model', 'table_auroc', 'auroc_paired',
                       'n_features', 'mtime']].to_string(index=False))

    dupes = df.groupby('sha')['task'].nunique()
    if (dupes > 1).any():
        print('\nWARNING: identical checkpoint file used for more than one task:')
        for sha, n in dupes[dupes > 1].items():
            print(f'  {sha}: {sorted(df[df.sha == sha].task.unique())}')

    odd = df[(df.n_features.notna()) & (df.n_features != df.n_features.mode().iloc[0])]
    if len(odd):
        print('\nWARNING: checkpoints expecting an unusual number of features '
              '(reduced-feature model in a full-model slot?):')
        print(odd[['task', 'cohort', 'model', 'n_features']].drop_duplicates()
              .to_string(index=False))


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
