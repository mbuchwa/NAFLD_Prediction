"""
recompute_reduced_tables.py
===========================
Regenerates Table 5 (reduced-feature models) directly from the checkpoints, so
that the "full" and "3 feat." columns are computed the same way.

Run from:  src/   ->   python -m src.recompute_reduced_tables

--------------------------------------------------------------------------
WHY THIS REPLACES make_reduced_tables.py
--------------------------------------------------------------------------
The previous table read the metric JSONs written by evaluate_performance, while
Table 1 comes from recompute_tables.py. The two disagree for two reasons:

  1. Ensemble pairing. evaluate_performance scores ensemble_pred_probas[0], i.e.
     every member on imputation 0. The Methods section describes member i on
     imputation i, which is what recompute_tables and this script do. At n=31
     that accounts for differences of 0.017-0.042.

  2. VI-BNN sampling. evaluate_performance takes a single stochastic forward
     pass; neural_loaders averages 200 posterior samples. That is the 0.137 gap
     seen for moderate fibrosis (0.705 vs 0.842).

Taking only the "full" column from tables_recomputed.csv would not fix this -- a
row would then hold one paired and one imp0 number, and their difference would be
the difference between two conventions rather than between two feature sets.
Both columns are therefore computed here, in one pass, under one convention.

--------------------------------------------------------------------------
MODEL SELECTION
--------------------------------------------------------------------------
The previous table showed a different set of models per task, which makes rows
incomparable: a reader cannot tell whether a task differs because of the feature
reduction or because a different model was picked. MODELS below is applied to
every task and cohort; models without a reduced checkpoint are reported as
missing rather than silently replaced.

--------------------------------------------------------------------------
WHAT THE REDUCED RUN NEEDS
--------------------------------------------------------------------------
    models/<name>_shap_selected/model_<task>.pickle      (or the per-member
                                                          layout for the neural
                                                          models)
    outputs/shap_top_features.json                       written by
                                                          derive_shap_top_features.py

The reduced feature matrices come from prepare_data(..., shap_selected=True),
which reads that JSON. If the JSON changed after the reduced models were trained,
the columns no longer match the weights -- the check below compares the expected
feature count against what each model reports and refuses to score on a mismatch.
"""

import json
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

# One fixed set for every task and cohort, so rows stay comparable.
MODELS = [('Random Forest', 'rf'), ('XGBoost', 'xgb'), ('LightGBM', 'light_gbm')]
SCALED_MODELS = {'VI-BNN', 'MLP'}

N_BOOT, SEED = 1000, 0
OUT_DIR = Path('outputs/tables')
TOP_FEATURES_JSON = Path('outputs/shap_top_features.json')


def _tex(s):
    s = str(s)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('_', r'\_')):
        s = s.replace(a, b)
    return s


def _english(names):
    """Marker names for the caption. The JSON holds the German column names."""
    try:
        from src.utils.ger_eng_dict import dict_germ_eng
    except ImportError:
        try:
            from utils.ger_eng_dict import dict_germ_eng
        except ImportError:
            return list(names)
    return [dict_germ_eng.get(n, n) for n in names]


# ------------------------------------------------------------ prediction ---
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
    """Member i on imputation i -- the soft vote the Methods define."""
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    p = np.mean([_proba_one(m, xs[i] if i < len(xs) else xs[0])
                 for i, m in enumerate(models)], axis=0)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def load_models(dirname, task):
    p = Path(f'models/{dirname}/model_{task}.pickle')
    if p.exists():
        with open(p, 'rb') as fh:
            return pickle.load(fh)
    try:
        from src.neural_loaders import load_any_ensemble, LOADERS
    except ImportError:
        try:
            from neural_loaders import load_any_ensemble, LOADERS
        except ImportError:
            return None
    base = dirname.replace('_shap_selected', '')
    if base in LOADERS:
        try:
            return load_any_ensemble(base, task, model_dir=f'models/{dirname}')
        except Exception as exc:
            print(f'    {dirname}/{task}: {exc}')
    return None


def _n_features(mdl):
    for attr in ('n_features_in_', 'n_features_'):
        v = getattr(mdl, attr, None)
        if v is not None:
            return int(v)
    b = getattr(mdl, 'booster_', None)
    if b is not None and hasattr(b, 'num_feature'):
        return int(b.num_feature())
    b = getattr(mdl, 'get_booster', None)
    if callable(b):
        try:
            return int(b().num_features())
        except Exception:
            pass
    return None


# --------------------------------------------------------------- metrics ---
def youden_threshold(y, score):
    fpr, tpr, thr = roc_curve(np.asarray(y).ravel(), np.asarray(score).ravel())
    return float(thr[int(np.argmax(tpr - fpr))])


def point_metrics(y, score, thr):
    y = np.asarray(y).ravel().astype(int)
    pred = (np.asarray(score).ravel() >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f = lambda a, b: 100.0 * a / (a + b) if (a + b) else np.nan
    return {'sens': f(tp, fn), 'spec': f(tn, fp)}


def paired_delta(y, a, b, n_boot=N_BOOT, seed=SEED):
    """AUROC(reduced) - AUROC(full) on the same patients."""
    y = np.asarray(y).ravel()
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    d = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    if not vals:
        return d, np.nan, np.nan
    return d, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ main ---
def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top = json.loads(TOP_FEATURES_JSON.read_text(encoding='utf-8')) \
        if TOP_FEATURES_JSON.exists() else {}
    if not top:
        print(f'NOTE: {TOP_FEATURES_JSON} not found -- feature names in the caption '
              f'will be missing. Run derive_shap_top_features.py first.')

    rows, missing = [], []
    for task in TASKS:
        feats = top.get(task, [])
        d_full = prepare_data(task, False, False)
        d_red = prepare_data(task, True, False)
        scl_full = scl_red = None

        for label, dirname in MODELS:
            m_full = load_models(dirname, task)
            m_red = load_models(f'{dirname}_shap_selected', task)
            if m_full is None or m_red is None:
                missing.append({'task': task, 'model': label,
                                'full': m_full is not None,
                                'reduced': m_red is not None})
                print(f'  {label:15s} {task:10s} -- '
                      f'{"reduced" if m_full is not None else "full"} checkpoint missing')
                continue

            if label in SCALED_MODELS:
                if scl_full is None:
                    scl_full = prepare_data(task, False, True)
                    scl_red = prepare_data(task, True, True)
                df_full, df_red = scl_full, scl_red
            else:
                df_full, df_red = d_full, d_red

            # guard: do the reduced weights match the reduced feature matrix?
            nf = _n_features(m_red[0])
            n_cols = np.asarray(df_red[4][0]).shape[1]
            if nf is not None and nf != n_cols:
                print(f'  {label:15s} {task:10s} -- reduced model expects {nf} features, '
                      f'prepare_data gives {n_cols}. shap_top_features.json probably '
                      f'changed after training. Skipped.')
                missing.append({'task': task, 'model': label, 'full': True,
                                'reduced': False, 'note': f'{nf} vs {n_cols} features'})
                continue

            for cohort, ix, iy in (('UMM', 4, 5), ('MAINZ', 6, 7)):
                y = np.asarray(df_full[iy][0]).ravel()
                s_full = ensemble_score(m_full, df_full[ix])
                s_red = ensemble_score(m_red, df_red[ix])

                thr = youden_threshold(np.asarray(df_red[3][0]).ravel(),
                                       ensemble_score(m_red, df_red[2]))
                d, lo, hi = paired_delta(y, s_red, s_full)
                r = {'task': task, 'cohort': cohort, 'model': label, 'n': len(y),
                     'auroc_full': float(roc_auc_score(y, s_full)),
                     'auroc_reduced': float(roc_auc_score(y, s_red)),
                     'delta_auroc': d, 'delta_lo': lo, 'delta_hi': hi,
                     'separates': bool(lo > 0 or hi < 0),
                     'features': ', '.join(_english(feats))}
                r.update(point_metrics(y, s_red, thr))
                rows.append(r)
                print(f'  {label:15s} {task:10s} {cohort:6s} '
                      f'full {r["auroc_full"]:.3f} -> reduced {r["auroc_reduced"]:.3f}  '
                      f'delta {d:+.3f} ({lo:+.3f},{hi:+.3f})')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'table5_reduced_recomputed.csv', index=False)
    write_latex(df, top)

    if missing:
        print('\nROWS THAT COULD NOT BE COMPUTED:')
        print(pd.DataFrame(missing).to_string(index=False))

    if len(df):
        print('\nCross-check against Table 1 -- the full column must match '
              'tables_recomputed.csv exactly (same convention, same checkpoints):')
        print(df[['task', 'cohort', 'model', 'auroc_full']].round(4).to_string(index=False))
    print(f'\n-> {OUT_DIR}/table5_reduced_recomputed.csv, .tex')


def write_latex(df, top):
    lines = [r'\begin{table*}[htbp]', r'    \centering',
             r'    \caption{\small{Performance of reduced-feature models across',
             r'    fibrosis-classification tasks and cohorts. Each model was retrained using',
             r'    only the three most influential biomarkers for the respective task, taken',
             r'    from the SHAP ranking of the full-biomarker models on the development',
             r'    cohort. Both columns were computed from the model checkpoints under the',
             r'    same ensemble convention, so the full column matches Table 1 exactly.',
             r'    $\Delta$AUROC is the paired difference (reduced minus full) on identical',
             r'    patients, with a 95\% bootstrap interval from 1{,}000 resamples;',
             r'    sensitivity and specificity refer to the reduced model at the threshold',
             r'    fixed on the validation partition.}}',
             r'    \label{tab:reduced}',
             r'    \begin{tabular}{llccccc}', r'        \toprule',
             r'        \textbf{Cohort} & \textbf{Model} & \textbf{AUROC full} & '
             r'\textbf{AUROC 3 feat.} & \textbf{$\Delta$AUROC (95\% CI)} & '
             r'\textbf{Sens.\ (\%)} & \textbf{Spec.\ (\%)}\\',
             r'        \midrule']
    for task in TASKS:
        sub = df[df.task == task]
        if sub.empty:
            continue
        title, defn = TASK_TITLE[task]
        feats = ', '.join(_tex(f) for f in _english(top.get(task, [])))
        lines.append(f'        \\multicolumn{{7}}{{l}}{{\\textit{{{title}}} ({defn}; '
                     f'{feats})}}\\\\')
        for i, (_, r) in enumerate(sub.iterrows()):
            sh = r'\rowcolor{customgray!40} ' if i % 2 == 0 else ''
            star = r'$^{*}$' if r.separates else ''
            lines.append(
                f'        {sh}{r.cohort} & {r.model} & {r.auroc_full:.3f} & '
                f'{r.auroc_reduced:.3f} & {r.delta_auroc:+.3f}{star} '
                f'({r.delta_lo:+.3f}, {r.delta_hi:+.3f}) & '
                f'{r.sens:.1f} & {r.spec:.1f}\\\\')
        lines.append(r'        \midrule')
    lines[-1] = r'        \bottomrule'
    lines += [r'    \end{tabular}',
              r'    \\[2pt]\footnotesize{$^{*}$ paired interval excludes zero.}',
              r'\end{table*}']
    (OUT_DIR / 'table5_reduced_recomputed.tex').write_text('\n'.join(lines),
                                                           encoding='utf-8')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
