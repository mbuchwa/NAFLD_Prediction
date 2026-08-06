"""
paired_model_comparison.py
==========================
Paired bootstrap comparison of AUROCs on identical patients.

Run from:  src/   ->   python -m src.paired_model_comparison

Two questions it answers:

  1. Is the leading model actually better than the runner-up, or is the gap
     within noise? (VI-BNN 0.906 vs Random Forest 0.888 for cirrhosis/MAINZ)

  2. Is the leading model better than FIB-4? This is the comparison the
     manuscript's central claim rests on.

--------------------------------------------------------------------------
WHY PAIRED AND NOT TWO INTERVALS
--------------------------------------------------------------------------
Both models are evaluated on the same patients, so their errors are correlated:
a patient who is hard for one model tends to be hard for the other. Comparing
two independent confidence intervals throws that correlation away and adds
uncertainty that cancels when the difference is resampled directly. With
n=284 the two approaches routinely disagree -- overlapping intervals next to a
paired difference that excludes zero.

Each bootstrap replicate resamples PATIENTS, then recomputes both AUROCs on the
same resample and takes the difference. The reported interval is the 2.5/97.5
percentile of those differences.

A caveat worth keeping: this quantifies sampling uncertainty on one fixed pair
of trained models. It says nothing about how much the ranking would move if the
models were retrained -- for that, see the run-to-run variation discussed in
test_training_determinism.py.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TASKS = ['fibrosis', 'two_stage', 'cirrhosis']
MODELS = [('SVM', 'svm'), ('Random Forest', 'rf'), ('XGBoost', 'xgb'),
          ('LightGBM', 'light_gbm'), ('MLP', 'ffn'),
          ('TabTransformer', 'tab_transformer'), ('VI-BNN', 'vi_bnn'),
          ('GANDALF', 'gandalf')]
SCALED_MODELS = {'VI-BNN'}
COHORTS = ['UMM', 'MAINZ']
N_BOOT, SEED = 2000, 0
OUT_DIR = Path('outputs/tables')


def _proba_one(mdl, x):
    x = np.asarray(x)
    if hasattr(mdl, 'predict_proba'):
        return np.asarray(mdl.predict_proba(x))
    if hasattr(mdl, 'decision_function'):
        s = np.asarray(mdl.decision_function(x)).ravel()
        p = 1 / (1 + np.exp(-s))
        return np.c_[1 - p, p]
    out = np.asarray(mdl.predict(x))
    return np.c_[1 - out, out] if out.ndim == 1 else out


def ensemble_score(models, xs):
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    p = np.mean([_proba_one(m, xs[i] if i < len(xs) else xs[0])
                 for i, m in enumerate(models)], axis=0)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def paired_delta(y, a, b, n_boot=N_BOOT, seed=SEED):
    """AUROC(a) - AUROC(b) with a paired percentile bootstrap over patients."""
    y = np.asarray(y).ravel()
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    delta = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    if not vals:
        return delta, np.nan, np.nan, np.nan
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    # two-sided bootstrap p: how often the difference crosses zero
    v = np.asarray(vals)
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return delta, lo, hi, float(min(p, 1.0))


def load_models(dirname, task):
    p = Path(f'models/{dirname}/model_{task}.pickle')
    if p.exists():
        with open(p, 'rb') as fh:
            return pickle.load(fh)
    try:
        from src.neural_loaders import load_any_ensemble, LOADERS
    except ImportError:
        from neural_loaders import load_any_ensemble, LOADERS
    if dirname in LOADERS:
        return load_any_ensemble(dirname, task)
    return None


def load_score(task, split, name='FIB-4'):
    p = Path(f'../data/preprocessed_mice_fib_{split}/{split}_{task}_0.csv')
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    col = {'FIB-4': 'Fib4', 'APRI': 'APRI'}[name]
    if col not in df.columns:
        return None, None
    y = df['Micro'].to_numpy(dtype=int) if 'Micro' in df.columns else None
    return df[col].to_numpy(dtype=float), y


def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for task in TASKS:
        d_raw = prepare_data(task, False, False)
        d_scl = None
        for cohort, split, i_x, i_y in (('UMM', 'test', 4, 5),
                                        ('MAINZ', 'prospective', 6, 7)):
            scores, y = {}, np.asarray(d_raw[i_y][0]).ravel()

            for label, dirname in MODELS:
                models = load_models(dirname, task)
                if models is None:
                    continue
                if label in SCALED_MODELS:
                    if d_scl is None:
                        d_scl = prepare_data(task, False, True)
                    xs = d_scl[i_x]
                else:
                    xs = d_raw[i_x]
                try:
                    scores[label] = ensemble_score(models, xs)
                except Exception as exc:
                    print(f'  {label}/{task}/{cohort}: {exc}')

            s_fib, y_csv = load_score(task, split, 'FIB-4')
            if s_fib is not None and len(s_fib) == len(y):
                scores['FIB-4'] = s_fib
                if y_csv is not None:
                    y = y_csv

            if len(scores) < 2:
                continue

            ranked = sorted(scores, key=lambda k: roc_auc_score(y, scores[k]), reverse=True)
            best = ranked[0]
            print(f'\n=== {task} / {cohort} (n={len(y)}) — reference: {best} '
                  f'(AUROC {roc_auc_score(y, scores[best]):.3f}) ===')
            for other in ranked[1:]:
                d, lo, hi, p = paired_delta(y, scores[best], scores[other])
                sig = 'yes' if (lo > 0 or hi < 0) else 'no '
                print(f'  vs {other:15s} ΔAUROC {d:+.4f} ({lo:+.4f}, {hi:+.4f})  '
                      f'p={p:.4f}  separates: {sig}')
                rows.append(dict(task=task, cohort=cohort, reference=best,
                                 comparator=other, delta_auroc=round(d, 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                 p_value=round(p, 4),
                                 excludes_zero=(lo > 0 or hi < 0)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'paired_model_comparison.csv', index=False)
    print(f'\n-> {OUT_DIR}/paired_model_comparison.csv')

    fib = df[df.comparator == 'FIB-4']
    if len(fib):
        print('\nBest model vs FIB-4 — the comparison the manuscript rests on:')
        print(fib[['task', 'cohort', 'reference', 'delta_auroc', 'ci_lo', 'ci_hi',
                   'p_value', 'excludes_zero']].to_string(index=False))


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
