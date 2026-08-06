"""
recompute_three_stage.py
========================
Regenerates Table 3, the ordinal three-stage task (F0/1 vs. F2/3 vs. F4).

Run from:  src/   ->   python -m src.recompute_three_stage

--------------------------------------------------------------------------
WHY THIS IS A SEPARATE SCRIPT
--------------------------------------------------------------------------
recompute_tables.py reports AUROC, sensitivity, specificity, PPV and NPV at a
single operating threshold. None of that transfers to three classes: there is no
single ROC curve, no single threshold, and accuracy alone ignores that a
F0/1-as-F4 error is worse than a F0/1-as-F2/3 error. This task therefore needs
its own metrics.

--------------------------------------------------------------------------
METRICS
--------------------------------------------------------------------------
ACC (Rubin-pooled)  Accuracy is computed separately on each of the m=10
                    imputations -- model i on imputation i -- and pooled with
                    Rubin's rules on the logit scale, so the interval cannot
                    leave [0,1]. The pooled variance is the within-imputation
                    variance plus (1 + 1/m) times the between-imputation
                    variance, with the Barnard-Rubin degrees of freedom.
                    Reporting the accuracy of the soft-voted ensemble instead
                    would understate the uncertainty, because it ignores the
                    spread across imputations entirely.

kappa_lin           Linearly weighted Cohen's kappa. Penalises a two-stage error
                    twice as heavily as a one-stage error, and corrects for
                    agreement expected by chance -- which matters here because
                    F4 alone accounts for 55% of the external cohort, so a
                    trivial always-F4 classifier would already reach 55%
                    accuracy but kappa near zero.

kappa_quad          Quadratically weighted kappa, reported alongside because
                    Reviewer 3 asked for weighted error measures and the two
                    weightings can disagree.

MAE                 Mean absolute error in fibrosis stages, on the ordinal
                    coding 0/1/2. Directly interpretable: how many stages the
                    prediction is off on average.

AUROC (macro OvR)   Macro-average one-vs-rest AUROC, for continuity with the
                    binary tables. Threshold-free and therefore comparable
                    across cohorts with different class mixes.

All intervals are 95% bootstrap percentile intervals on the evaluation partition
(1,000 resamples), except the Rubin-pooled accuracy interval, which comes from
the pooling itself.

--------------------------------------------------------------------------
FIB-4 AND APRI
--------------------------------------------------------------------------
Both are read from the CSVs preprocess() writes and staged with the same
cut-offs the manuscript lists: FIB-4 at 1.45 and 3.25, APRI at 1.5 and 2.0.
They produce hard class labels, so they get ACC, kappa and MAE but no AUROC --
a two-cut-off rule has no ranking to build a curve from. The dash in that column
is the honest entry, not a missing value.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

TASK = 'three_stage'
MODELS = [('SVM', 'svm'), ('Random Forest', 'rf'), ('XGBoost', 'xgb'),
          ('LightGBM', 'light_gbm'), ('MLP', 'ffn'),
          ('TabTransformer', 'tab_transformer'), ('VI-BNN', 'vi_bnn'),
          ('GANDALF', 'gandalf')]
SCALED_MODELS = {'VI-BNN'}
N_BOOT, SEED = 1000, 0
OUT_DIR = Path('outputs/tables')
CLASSES = [0, 1, 2]          # F0/1, F2/3, F4


# ----------------------------------------------------------- predictions ---
def _proba_one(mdl, x):
    x = np.asarray(x)
    if hasattr(mdl, 'predict_proba'):
        p = np.asarray(mdl.predict_proba(x))
    elif hasattr(mdl, 'decision_function'):
        s = np.asarray(mdl.decision_function(x))
        e = np.exp(s - s.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
    else:
        out = np.asarray(mdl.predict(x))
        if out.ndim == 1:
            p = np.zeros((len(out), len(CLASSES)))
            p[np.arange(len(out)), out.astype(int)] = 1.0
        else:
            e = np.exp(out - out.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
    if p.shape[1] != len(CLASSES):
        raise ValueError(f'model returned {p.shape[1]} classes, expected {len(CLASSES)}')
    return p


def ensemble_proba(models, xs):
    """Soft vote, member i on imputation i -- the ensemble the Methods define."""
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    return np.mean([_proba_one(m, xs[i] if i < len(xs) else xs[0])
                    for i, m in enumerate(models)], axis=0)


# --------------------------------------------------------------- metrics ---
def _ordinal_metrics(y, pred):
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    return {
        'acc': float((y == pred).mean()) * 100,
        'kappa_lin': float(cohen_kappa_score(y, pred, weights='linear', labels=CLASSES)),
        'kappa_quad': float(cohen_kappa_score(y, pred, weights='quadratic', labels=CLASSES)),
        'mae': float(np.abs(y - pred).mean()),
    }


def _macro_ovr_auroc(y, proba):
    y = np.asarray(y).ravel()
    aucs = []
    for c in CLASSES:
        yb = (y == c).astype(int)
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, proba[:, c]))
    return float(np.mean(aucs)) if aucs else np.nan


def bootstrap_ci(y, pred, proba, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)
    keys = ('kappa_lin', 'kappa_quad', 'mae', 'auroc')
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = _ordinal_metrics(y[idx], pred[idx])
        for k in ('kappa_lin', 'kappa_quad', 'mae'):
            boot[k].append(m[k])
        if proba is not None:
            boot['auroc'].append(_macro_ovr_auroc(y[idx], proba[idx]))
    out = {}
    for k in keys:
        v = [x for x in boot[k] if np.isfinite(x)]
        out[f'{k}_lo'] = float(np.percentile(v, 2.5)) if v else np.nan
        out[f'{k}_hi'] = float(np.percentile(v, 97.5)) if v else np.nan
    return out


def rubin_pool_accuracy(accs, n):
    """Pool per-imputation accuracies with Rubin's rules on the logit scale.

    Working on the logit avoids intervals that run past 0 or 1, which happens
    with a naive pooling when accuracy is high and n small.
    """
    a = np.asarray([x / 100.0 for x in accs], dtype=float)
    m = len(a)
    eps = 0.5 / n                                   # keeps logit finite at 0 or 1
    a = np.clip(a, eps, 1 - eps)
    theta = np.log(a / (1 - a))                     # logit
    var_w = a * (1 - a) / n                         # binomial variance ...
    var_w = var_w / (a * (1 - a)) ** 2              # ... delta-method to logit
    qbar = float(theta.mean())
    ubar = float(var_w.mean())                      # within-imputation
    b = float(theta.var(ddof=1)) if m > 1 else 0.0  # between-imputation
    total = ubar + (1 + 1 / m) * b
    if b > 0 and ubar > 0:
        r = (1 + 1 / m) * b / ubar
        df = (m - 1) * (1 + 1 / r) ** 2
        df = min(df, (n - 1) * (1 + ubar / ((1 + 1 / m) * b)) / (1 + (n - 1) / (n + 1)))
    else:
        df = max(m - 1, 1)
    t = stats.t.ppf(0.975, max(df, 1))
    lo, hi = qbar - t * np.sqrt(total), qbar + t * np.sqrt(total)
    inv = lambda z: 100.0 / (1 + np.exp(-z))
    return inv(qbar), inv(lo), inv(hi), float(df)


# --------------------------------------------------------------- scores ----
def load_score_stages(split):
    """FIB-4 / APRI three-stage labels from the CSVs preprocess() writes."""
    p = Path(f'../data/preprocessed_mice_fib_{split}/{split}_{TASK}_0.csv')
    if not p.exists():
        return None
    df = pd.read_csv(p)
    out = {'y': df['Micro'].to_numpy(dtype=int) if 'Micro' in df.columns else None}
    for name, col in (('FIB-4', 'Fib4 Stages'), ('APRI', 'APRI Stages')):
        if col in df.columns:
            out[name] = df[col].to_numpy(dtype=int)
    return out


# ------------------------------------------------------------------ main ---
def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d_raw = prepare_data(TASK, False, False)
    d_scl = None
    rows, missing = [], []

    cohorts_raw = {'UMM': (d_raw[4], d_raw[5]), 'MAINZ': (d_raw[6], d_raw[7])}

    for label, dirname in MODELS:
        path = Path(f'models/{dirname}/model_{TASK}.pickle')
        models = None
        if path.exists():
            with open(path, 'rb') as fh:
                models = pickle.load(fh)
        else:
            # vi_bnn and gandalf save one artefact per ensemble member
            try:
                from src.neural_loaders import load_any_ensemble, LOADERS
            except ImportError:
                try:
                    from neural_loaders import load_any_ensemble, LOADERS
                except ImportError:
                    LOADERS = {}
            if dirname in LOADERS:
                try:
                    models = load_any_ensemble(dirname, TASK)
                    path = Path(f'models/{dirname}/ (per-member)')
                except Exception as exc:
                    missing.append({'model': label, 'path': str(path), 'note': str(exc)[:70]})
                    print(f'  {label:15s} -- loader failed: {exc}')
                    continue
        if models is None:
            missing.append({'model': label, 'path': str(path), 'note': 'no checkpoint'})
            print(f'  {label:15s} -- no checkpoint')
            continue

        if label in SCALED_MODELS:
            if d_scl is None:
                d_scl = prepare_data(TASK, False, True)
            cohorts = {'UMM': (d_scl[4], d_scl[5]), 'MAINZ': (d_scl[6], d_scl[7])}
        else:
            cohorts = cohorts_raw

        for cohort, (xs, ys) in cohorts.items():
            y0 = np.asarray(ys[0]).ravel()
            try:
                proba = ensemble_proba(models, xs)
            except Exception as exc:
                missing.append({'model': label, 'path': str(path),
                                'note': f'{cohort}: {exc}'})
                print(f'  {label:15s} {cohort:6s} -- failed: {exc}')
                continue
            pred = proba.argmax(1)

            r = {'task': TASK, 'cohort': cohort, 'model': label, 'n': len(y0)}
            r.update(_ordinal_metrics(y0, pred))
            r['auroc'] = _macro_ovr_auroc(y0, proba)
            r.update(bootstrap_ci(y0, pred, proba))

            # per-imputation accuracies -> Rubin
            accs = []
            for i, mdl in enumerate(models):
                xi = np.asarray(xs[i] if i < len(xs) else xs[0])
                yi = np.asarray(ys[i] if i < len(ys) else ys[0]).ravel()
                accs.append(float((yi == _proba_one(mdl, xi).argmax(1)).mean()) * 100)
            acc, lo, hi, df_ = rubin_pool_accuracy(accs, len(y0))
            r.update({'acc_pooled': acc, 'acc_lo': lo, 'acc_hi': hi,
                      'rubin_df': round(df_, 1),
                      'acc_per_imputation_sd': float(np.std(accs, ddof=1))})
            rows.append(r)
            print(f'  {label:15s} {cohort:6s} ACC {acc:5.2f} ({lo:.2f}-{hi:.2f})  '
                  f'k_lin {r["kappa_lin"]:.3f}  MAE {r["mae"]:.3f}  '
                  f'AUROC {r["auroc"]:.3f}')

    # FIB-4 / APRI
    for cohort, split in (('UMM', 'test'), ('MAINZ', 'prospective')):
        sc = load_score_stages(split)
        if not sc:
            print(f'  scores CSV for {split} not found -- FIB-4/APRI skipped')
            continue
        y = sc['y'] if sc['y'] is not None else np.asarray(cohorts_raw[cohort][1][0]).ravel()
        for name in ('FIB-4', 'APRI'):
            if name not in sc:
                continue
            pred = sc[name]
            r = {'task': TASK, 'cohort': cohort, 'model': name, 'n': len(y)}
            r.update(_ordinal_metrics(y, pred))
            r['auroc'] = np.nan                     # no ranking from fixed cut-offs
            r.update(bootstrap_ci(y, pred, None))
            r.update({'acc_pooled': r['acc'], 'acc_lo': np.nan, 'acc_hi': np.nan,
                      'rubin_df': np.nan, 'acc_per_imputation_sd': np.nan})
            rows.append(r)
            print(f'  {name:15s} {cohort:6s} ACC {r["acc"]:5.2f}  '
                  f'k_lin {r["kappa_lin"]:.3f}  MAE {r["mae"]:.3f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'table3_three_stage_recomputed.csv', index=False)
    write_latex(df)

    if missing:
        print('\nROWS THAT COULD NOT BE RECOMPUTED — retrain or delete them:')
        print(pd.DataFrame(missing).to_string(index=False))

    print('\nBest model by external macro-AUROC:')
    ext = df[(df.cohort == 'MAINZ') & df.auroc.notna()]
    if len(ext):
        b = ext.loc[ext.auroc.idxmax()]
        print(f'  {b.model} (AUROC {b.auroc:.3f}, kappa_lin {b.kappa_lin:.3f})')
        print('  -> set this in BEST_MODEL_PER_TASK["three_stage"] in '
              'shap_publication_figures.py')
    print(f'\n-> {OUT_DIR}/table3_three_stage_recomputed.csv, .tex')


def write_latex(df):
    lines = [r'\begin{table*}[htbp]', r'    \centering',
             r'    \caption{\small{Ordinal-aware performance for the \textit{Three Stage}',
             r'    task (F0/1 vs.\ F2/3 vs.\ F4). Accuracy is pooled across the $m=10$',
             r"    imputations using Rubin's rules on the logit scale; the remaining",
             r'    intervals are 95\% bootstrap percentile intervals (1{,}000 resamples).',
             r"    Linearly and quadratically weighted Cohen's $\kappa$ and the mean",
             r'    absolute error (MAE) in fibrosis stages penalise distant',
             r'    misclassifications more strongly than adjacent ones. AUROC is the',
             r'    macro-average of the three one-vs-rest curves; FIB-4 and APRI produce',
             r'    hard labels from fixed cut-offs and therefore have no AUROC.}}',
             r'    \label{tab:three_stage}',
             r'    \begin{tabular}{lccccc}', r'        \toprule']
    for cohort in ('UMM', 'MAINZ'):
        sub = df[df.cohort == cohort]
        if sub.empty:
            continue
        n = int(sub.iloc[0]['n'])
        lines += [f'        \\multicolumn{{6}}{{l}}{{\\textbf{{{cohort}}} (n={n})}}\\\\',
                  r'        \midrule',
                  r'        \textbf{Model} & \textbf{ACC (\%)} $\uparrow$ & '
                  r'\textbf{$\kappa_{lin}$} $\uparrow$ & \textbf{$\kappa_{quad}$} $\uparrow$ & '
                  r'\textbf{MAE} $\downarrow$ & \textbf{AUROC} $\uparrow$\\',
                  r'        \midrule']
        models = sub[~sub.model.isin(['FIB-4', 'APRI'])]
        scores = sub[sub.model.isin(['FIB-4', 'APRI'])]
        best = models.auroc.idxmax() if models.auroc.notna().any() else None
        for i, (idx, r) in enumerate(models.iterrows()):
            sh = r'\rowcolor{customgray!40} ' if i % 2 == 0 else ''
            name = f'\\textbf{{{r.model}}}' if idx == best else r.model
            acc = (f'{r.acc_pooled:.2f} ({r.acc_lo:.2f}--{r.acc_hi:.2f})'
                   if np.isfinite(r.acc_lo) else f'{r.acc_pooled:.2f}')
            auroc = f'{r.auroc:.3f} ({r.auroc_lo:.3f}--{r.auroc_hi:.3f})' \
                if np.isfinite(r.auroc) else '--'
            lines.append(f'        {sh}{name} & {acc} & {r.kappa_lin:.3f} & '
                         f'{r.kappa_quad:.3f} & {r.mae:.3f} & {auroc}\\\\')
        if len(scores):
            lines.append(r'        \midrule')
            for _, r in scores.iterrows():
                lines.append(f'        \\rowcolor{{gray!10}} {r.model} & {r.acc:.2f} & '
                             f'{r.kappa_lin:.3f} & {r.kappa_quad:.3f} & {r.mae:.3f} & --\\\\')
        lines.append(r'        \midrule')
    lines[-1] = r'        \bottomrule'
    lines += [r'    \end{tabular}', r'\end{table*}']
    (OUT_DIR / 'table3_three_stage_recomputed.tex').write_text(
        '\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
