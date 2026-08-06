"""
recompute_tables.py
===================
Regenerates Tables 1 and 2 from the checkpoints currently on disk, so that
tables, figures and models share one provenance.

Run from:  src/   ->   python recompute_tables.py

--------------------------------------------------------------------------
WHAT CHANGES AGAINST THE PRINTED TABLES
--------------------------------------------------------------------------
1. Ensemble convention. Member i is evaluated on imputation i -- the soft vote
   the Methods section describes. The printed tables were produced with every
   member evaluated on imputation 0; the diagnostic showed that all six rows
   where the two conventions differ match the imp0 value.

2. Current checkpoints. Rows whose model was retrained after the tables were
   written now report what that model actually does.

Everything else is unchanged: same Youden threshold rule fixed on the
validation partition, same 1,000-resample bootstrap, same clinical cut-offs
for FIB-4 and APRI.

--------------------------------------------------------------------------
MODELS WITHOUT A CHECKPOINT
--------------------------------------------------------------------------
Neural models save differently from the tree/SVM ensembles (torch state dicts,
pytorch_tabular directories) and are not found by the plain
models/<dir>/model_<task>.pickle lookup. NEURAL_GLOBS below is searched as a
fallback and every hit is reported; extend it to match your layout.

Whatever is still not found cannot be recomputed and must not stay in the
table with its old value -- retrain it or drop the row. The script prints an
explicit list at the end so nothing is forgotten silently.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

TASKS = ['fibrosis', 'two_stage', 'cirrhosis']
TASK_TITLE = {'fibrosis': ('Moderate Fibrosis', 'F0/1 vs.\\ F2/3/4'),
              'two_stage': ('Severe Fibrosis', 'F0/1/2 vs.\\ F3/4'),
              'cirrhosis': ('Cirrhosis', 'F0/1/2/3 vs.\\ F4')}
# MLP and TabTransformer were previously reported for the three-stage task only,
# on the grounds that their binary outputs "did not yield usable discrimination".
# That verdict came from the old evaluation; include them here so the claim can be
# rechecked against the current checkpoints, and drop the rows again if it holds.
MODELS = [('SVM', 'svm'), ('Random Forest', 'rf'), ('XGBoost', 'xgb'),
          ('LightGBM', 'light_gbm'), ('MLP', 'ffn'),
          ('TabTransformer', 'tab_transformer'), ('VI-BNN', 'vi_bnn'),
          ('GANDALF', 'gandalf')]
SCALED_MODELS = {'VI-BNN'}          # trained on scaled features (see train.py)

NEURAL_GLOBS = ['models/{d}/model_{t}.pt', 'models/{d}/model_{t}.pth',
                'models/{d}/{t}/*.pt', 'models/{d}/model_{t}*.pkl',
                'models/{d}/{t}*']

FIB4_CUTOFF = {'fibrosis': 1.45, 'two_stage': 2.67, 'cirrhosis': 3.25}
APRI_CUTOFF = {'fibrosis': 1.50, 'two_stage': 1.50, 'cirrhosis': 2.00}

N_BOOT, SEED = 1000, 0
OUT_DIR = Path('outputs/tables')

# AUROCs as printed, for the change report only.
PRINTED = {
    ('fibrosis', 'UMM'):    {'SVM': .714, 'Random Forest': .923, 'XGBoost': .934,
                             'LightGBM': .888, 'VI-BNN': .781, 'GANDALF': .730},
    ('two_stage', 'UMM'):   {'SVM': .703, 'Random Forest': .923, 'XGBoost': .903,
                             'LightGBM': .924, 'VI-BNN': .764, 'GANDALF': .692},
    ('cirrhosis', 'UMM'):   {'SVM': .684, 'Random Forest': .877, 'XGBoost': .807,
                             'LightGBM': .882, 'VI-BNN': .754, 'GANDALF': .615},
    ('fibrosis', 'MAINZ'):  {'SVM': .833, 'Random Forest': .859, 'XGBoost': .889,
                             'LightGBM': .860, 'VI-BNN': .835, 'GANDALF': .606},
    ('two_stage', 'MAINZ'): {'SVM': .869, 'Random Forest': .905, 'XGBoost': .911,
                             'LightGBM': .896, 'VI-BNN': .861, 'GANDALF': .678},
    ('cirrhosis', 'MAINZ'): {'SVM': .848, 'Random Forest': .876, 'XGBoost': .948,
                             'LightGBM': .878, 'VI-BNN': .854, 'GANDALF': .690},
}


# ------------------------------------------------------------- prediction ---
def _proba_one(mdl, x):
    """predict_proba where available, otherwise a softmax/sigmoid fallback."""
    x = np.asarray(x)
    if hasattr(mdl, 'predict_proba'):
        return np.asarray(mdl.predict_proba(x))
    if hasattr(mdl, 'decision_function'):
        s = np.asarray(mdl.decision_function(x)).ravel()
        p = 1 / (1 + np.exp(-s))
        return np.c_[1 - p, p]
    out = np.asarray(mdl.predict(x))
    if out.ndim == 1:
        return np.c_[1 - out, out]
    e = np.exp(out - out.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ensemble_score(models, xs):
    """Member i on imputation i, then average -- the manuscript's soft vote."""
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    ps = []
    for i, mdl in enumerate(models):
        ps.append(_proba_one(mdl, xs[i] if i < len(xs) else xs[0]))
    p = np.mean(ps, axis=0)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def load_checkpoint(dirname, task):
    p = Path(f'models/{dirname}/model_{task}.pickle')
    if p.exists():
        with open(p, 'rb') as fh:
            return pickle.load(fh), str(p)
    # vi_bnn and gandalf save one artefact per ensemble member, not one pickle
    try:
        from src.neural_loaders import load_any_ensemble, LOADERS
    except ImportError:
        try:
            from neural_loaders import load_any_ensemble, LOADERS
        except ImportError:
            return None, None
    if dirname in LOADERS:
        try:
            return load_any_ensemble(dirname, task), f'models/{dirname}/ (per-member)'
        except Exception as exc:
            print(f'    {dirname}/{task}: loader failed — {exc}')
    return None, None


# ---------------------------------------------------------------- metrics ---
def youden_threshold(y, score):
    fpr, tpr, thr = roc_curve(np.asarray(y).ravel(), np.asarray(score).ravel())
    return float(thr[int(np.argmax(tpr - fpr))])


def _point_metrics(y, score, thr):
    y = np.asarray(y).ravel().astype(int)
    pred = (np.asarray(score).ravel() >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f = lambda a, b: 100.0 * a / (a + b) if (a + b) else np.nan
    return {'sens': f(tp, fn), 'spec': f(tn, fp), 'ppv': f(tp, fp), 'npv': f(tn, fn)}


def evaluate(y, score, thr, n_boot=N_BOOT, seed=SEED):
    y, score = np.asarray(y).ravel(), np.asarray(score).ravel()
    res = {'auroc': float(roc_auc_score(y, score))}
    res.update(_point_metrics(y, score, thr))
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in ('auroc', 'sens', 'spec', 'ppv', 'npv')}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot['auroc'].append(roc_auc_score(y[idx], score[idx]))
        for k, v in _point_metrics(y[idx], score[idx], thr).items():
            boot[k].append(v)
    for k, vals in boot.items():
        vals = [v for v in vals if not np.isnan(v)]
        res[f'{k}_lo'] = float(np.percentile(vals, 2.5)) if vals else np.nan
        res[f'{k}_hi'] = float(np.percentile(vals, 97.5)) if vals else np.nan
    return res


def load_scores(task, split):
    """FIB-4 and APRI from the CSVs preprocess writes, row order as in xs."""
    p = Path(f'../data/preprocessed_mice_fib_{split}/{split}_{task}_0.csv')
    if not p.exists():
        return None
    df = pd.read_csv(p)
    out = {}
    for name, col, stage in (('FIB-4', 'Fib4', 'Fib4 Stages'),
                             ('APRI', 'APRI', 'APRI Stages')):
        if col in df.columns:
            out[name] = df[col].to_numpy(dtype=float)
        elif stage in df.columns:
            out[name] = df[stage].to_numpy(dtype=float)
    out['y'] = df['Micro'].to_numpy(dtype=int) if 'Micro' in df.columns else None
    return out


# ------------------------------------------------------------------ main ----
def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []

    for task in TASKS:
        d_raw = prepare_data(classification_type=task, shap_selected=False, scaling=False)
        d_scl = None
        cohorts = {'UMM': (d_raw[4], np.asarray(d_raw[5][0]).ravel()),
                   'MAINZ': (d_raw[6], np.asarray(d_raw[7][0]).ravel())}
        val = (d_raw[2], np.asarray(d_raw[3][0]).ravel())

        for label, dirname in MODELS:
            models, path = load_checkpoint(dirname, task)
            if models is None:
                missing.append({'task': task, 'model': label,
                                'found': path or '-',
                                'note': 'not loadable as pickle' if path else 'no file'})
                print(f'  {label:14s} {task:10s} -- no usable checkpoint')
                continue

            if label in SCALED_MODELS:
                if d_scl is None:
                    d_scl = prepare_data(classification_type=task,
                                         shap_selected=False, scaling=True)
                src = d_scl
                use = {'UMM': (src[4], np.asarray(src[5][0]).ravel()),
                       'MAINZ': (src[6], np.asarray(src[7][0]).ravel())}
                use_val = (src[2], np.asarray(src[3][0]).ravel())
            else:
                use, use_val = cohorts, val

            try:
                thr = youden_threshold(use_val[1], ensemble_score(models, use_val[0]))
            except Exception as exc:
                print(f'  {label:14s} {task:10s} -- threshold failed: {exc}')
                missing.append({'task': task, 'model': label, 'found': path,
                                'note': f'evaluation failed: {exc}'})
                continue

            for cohort, (xs, y) in use.items():
                r = evaluate(y, ensemble_score(models, xs), thr)
                r.update({'task': task, 'cohort': cohort, 'model': label,
                          'threshold': round(thr, 4), 'n': len(y),
                          'checkpoint': path,
                          'printed_auroc': PRINTED.get((task, cohort), {}).get(label, np.nan)})
                r['delta_vs_printed'] = round(r['auroc'] - r['printed_auroc'], 4) \
                    if not np.isnan(r['printed_auroc']) else np.nan
                rows.append(r)

        # FIB-4 / APRI at their clinical cut-offs
        for cohort, split in (('UMM', 'test'), ('MAINZ', 'prospective')):
            sc = load_scores(task, split)
            if not sc:
                print(f'  scores CSV for {task}/{split} not found -- FIB-4/APRI skipped')
                continue
            y = sc['y'] if sc['y'] is not None else cohorts[cohort][1]
            for name, cut in (('FIB-4', FIB4_CUTOFF[task]), ('APRI', APRI_CUTOFF[task])):
                if name not in sc:
                    continue
                r = evaluate(y, sc[name], cut)
                r.update({'task': task, 'cohort': cohort, 'model': f'{name} (cut-off {cut})',
                          'threshold': cut, 'n': len(y), 'checkpoint': 'score',
                          'printed_auroc': np.nan, 'delta_vs_printed': np.nan})
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'tables_recomputed.csv', index=False)
    write_latex(df)

    print('\nAUROC change against the printed tables:')
    ch = df[df.delta_vs_printed.notna()].copy()
    ch['flag'] = np.where(ch.delta_vs_printed.abs() > 0.01, '  <== >0.01', '')
    print(ch[['task', 'cohort', 'model', 'printed_auroc', 'auroc',
              'delta_vs_printed', 'flag']].to_string(index=False))

    if missing:
        print('\nROWS THAT COULD NOT BE RECOMPUTED — retrain or delete them:')
        print(pd.DataFrame(missing).to_string(index=False))
        pd.DataFrame(missing).to_csv(OUT_DIR / 'tables_missing_rows.csv', index=False)
    print(f'\n-> {OUT_DIR}/tables_recomputed.csv, .tex')


def write_latex(df):
    def fmt(r, k):
        return (f'{r[k]:.3f} ({r[k + "_lo"]:.3f}--{r[k + "_hi"]:.3f})' if k == 'auroc'
                else f'{r[k]:.2f} ({r[k + "_lo"]:.2f}--{r[k + "_hi"]:.2f})')

    for cohort, num in (('UMM', 1), ('MAINZ', 2)):
        lines = []
        for task in TASKS:
            sub = df[(df.task == task) & (df.cohort == cohort)]
            if sub.empty:
                continue
            title, defn = TASK_TITLE[task]
            lines += [r'\begin{table*}[htbp]', r'    \centering',
                      f'    \\caption{{\\small{{{title} ({defn}) on the '
                      f'\\textit{{{cohort}}} cohort (n={int(sub.iloc[0]["n"])}). '
                      r'Point estimates with 95\% bootstrap confidence intervals '
                      r'(1{,}000 resamples). Sensitivity, specificity, PPV and NPV are '
                      r'evaluated at a single operating threshold fixed on the validation '
                      r'partition (Youden index); FIB-4 and APRI at their clinical '
                      r'cut-offs.}}',
                      f'    \\label{{tab:{task}_{cohort.lower()}}}',
                      r'    \begin{tabular}{lccccc}', r'        \toprule',
                      r'        \textbf{Model} & \textbf{AUROC} $\uparrow$ & '
                      r'\textbf{Sens.\ (\%)} $\uparrow$ & \textbf{Spec.\ (\%)} $\uparrow$ & '
                      r'\textbf{PPV (\%)} $\uparrow$ & \textbf{NPV (\%)} $\uparrow$\\',
                      r'        \midrule']
            models = sub[~sub.model.str.contains('cut-off')]
            scores = sub[sub.model.str.contains('cut-off')]
            best = models.auroc.idxmax() if len(models) else None
            for i, (_, r) in enumerate(models.iterrows()):
                sh = r'\rowcolor{customgray!40} ' if i % 2 == 0 else ''
                cells = ' & '.join(fmt(r, k) for k in ('auroc', 'sens', 'spec', 'ppv', 'npv'))
                name = f'\\textbf{{{r.model}}}' if r.name == best else r.model
                lines.append(f'        {sh}{name} & {cells}\\\\')
            if len(scores):
                lines += [r'        \midrule', r'        \midrule']
                for _, r in scores.iterrows():
                    cells = ' & '.join(fmt(r, k) for k in ('auroc', 'sens', 'spec', 'ppv', 'npv'))
                    lines.append(f'        \\rowcolor{{gray!10}} {r.model} & {cells}\\\\')
            lines += [r'        \bottomrule', r'    \end{tabular}', r'\end{table*}', '']
        (OUT_DIR / f'table{num}_{cohort.lower()}_recomputed.tex').write_text(
            '\n'.join(lines), encoding='utf-8')
        print(f'-> {OUT_DIR}/table{num}_{cohort.lower()}_recomputed.tex')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()