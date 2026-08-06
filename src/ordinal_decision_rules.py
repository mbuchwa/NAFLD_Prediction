"""
ordinal_decision_rules.py
=========================
Compares decision rules for the ordinal three-stage task (F0/1 vs F2/3 vs F4)
without retraining anything.

Run from:  src/   ->   python -m src.ordinal_decision_rules

--------------------------------------------------------------------------
THE PROBLEM
--------------------------------------------------------------------------
The three-stage models are fitted as unordered multiclass problems: the softmax
does not know that F0/1 < F2/3 < F4. FIB-4, in contrast, is defined by two
ascending cut-offs and can barely produce a two-stage error by construction --
and two-stage errors are exactly what the reported metrics punish. In the
external cohort FIB-4 therefore reaches kappa_lin 0.505 and MAE 0.377 against
0.522 and 0.380 for the best model.

A second mismatch sits in the prediction rule. `proba.argmax(1)` returns the MODE
of the predictive distribution, which is optimal for 0/1 loss. For MAE the
optimal point prediction is the MEDIAN; for quadratically weighted kappa it is
closer to the rounded expected value. The models are thus decided by one loss and
reported under another.

--------------------------------------------------------------------------
WHAT IS COMPARED
--------------------------------------------------------------------------
Three rules applied to the native three-stage probabilities:

    mode        argmax                       -- the current rule
    median      first k with P(Y<=k) >= 0.5  -- minimises MAE
    expected    round(sum k * p_k)           -- closer to quadratic kappa

The same three rules applied to an ORDINAL DECOMPOSITION built from the existing
binary models (Frank & Hall):

    P(F0/1) = 1 - P(F>=2)          from the `fibrosis` model
    P(F2/3) = P(F>=2) - P(F>=4)
    P(F4)   = P(F>=4)              from the `cirrhosis` model

Both binary models are trained on the same split as the three-stage model, so no
information crosses partitions. Their independent fitting can violate
monotonicity (P(F>=4) > P(F>=2)); this is repaired by enforcing a non-increasing
survival function before differencing.

--------------------------------------------------------------------------
HOW THE CHOICE IS MADE
--------------------------------------------------------------------------
Every rule is scored on the VALIDATION partition, and only the validation column
may drive the decision. Test and external columns are reported for all rules so
the effect is visible, but choosing on them would make the held-out partition
part of model development -- the objection Reviewer 3 raised elsewhere.

Whatever wins, report it as a pre-specified consequence of the metric, not as the
rule that happened to score best externally.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

MODELS = [('SVM', 'svm'), ('Random Forest', 'rf'), ('XGBoost', 'xgb'),
          ('LightGBM', 'light_gbm'), ('MLP', 'ffn'),
          ('TabTransformer', 'tab_transformer'), ('VI-BNN', 'vi_bnn'),
          ('GANDALF', 'gandalf')]
SCALED_MODELS = {'VI-BNN', 'MLP'}
CLASSES = [0, 1, 2]
OUT_DIR = Path('outputs/tables')

# split index in the prepare_data tuple
SPLITS = {'validation': (2, 3), 'test': (4, 5), 'external': (6, 7)}


# ----------------------------------------------------------- probabilities --
def _proba_one(mdl, x):
    x = np.asarray(x)
    if hasattr(mdl, 'predict_proba'):
        return np.asarray(mdl.predict_proba(x))
    if hasattr(mdl, 'decision_function'):
        s = np.asarray(mdl.decision_function(x))
        if s.ndim == 1:
            p = 1 / (1 + np.exp(-s))
            return np.c_[1 - p, p]
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    out = np.asarray(mdl.predict(x))
    return np.c_[1 - out, out] if out.ndim == 1 else out


def ensemble_proba(models, xs):
    """Soft vote, member i on imputation i."""
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    return np.mean([_proba_one(m, xs[i] if i < len(xs) else xs[0])
                    for i, m in enumerate(models)], axis=0)


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
        try:
            return load_any_ensemble(dirname, task)
        except Exception as exc:
            print(f'    {dirname}/{task}: {exc}')
    return None


# --------------------------------------------------------- decision rules ---
def rule_mode(p):
    return p.argmax(1)


def rule_median(p):
    """First class k with P(Y <= k) >= 0.5 -- the predictive median."""
    return (np.cumsum(p, axis=1) < 0.5).sum(axis=1).clip(0, p.shape[1] - 1)


def rule_expected(p):
    """Rounded expected stage."""
    return np.rint(p @ np.arange(p.shape[1])).astype(int).clip(0, p.shape[1] - 1)


RULES = {'mode': rule_mode, 'median': rule_median, 'expected': rule_expected}


def ordinal_decomposition(p_ge2, p_ge4):
    """Frank & Hall: turn two binary survival probabilities into three classes.

    Enforces a non-increasing survival function first, because the two binary
    models are fitted independently and can return P(F>=4) > P(F>=2).
    """
    s = np.column_stack([np.ones_like(p_ge2), p_ge2, p_ge4])
    s = np.minimum.accumulate(s, axis=1)          # monotone survival
    p = np.column_stack([s[:, 0] - s[:, 1], s[:, 1] - s[:, 2], s[:, 2]])
    p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- metrics ---
def evaluate(y, p, pred):
    y = np.asarray(y).ravel().astype(int)
    aucs = [roc_auc_score((y == c).astype(int), p[:, c])
            for c in CLASSES if len(np.unique(y == c)) > 1]
    return {
        'acc': 100 * float((y == pred).mean()),
        'kappa_lin': float(cohen_kappa_score(y, pred, weights='linear', labels=CLASSES)),
        'kappa_quad': float(cohen_kappa_score(y, pred, weights='quadratic', labels=CLASSES)),
        'mae': float(np.abs(y - pred).mean()),
        'auroc': float(np.mean(aucs)) if aucs else np.nan,
        'n_two_stage_errors': int((np.abs(y - pred) == 2).sum()),
    }


def score_stages(split):
    """FIB-4 / APRI three-stage labels, for the comparator rows."""
    p = Path(f'../data/preprocessed_mice_fib_{split}/{split}_three_stage_0.csv')
    if not p.exists():
        return None
    df = pd.read_csv(p)
    out = {'y': df['Micro'].to_numpy(int) if 'Micro' in df.columns else None}
    for name, col in (('FIB-4', 'Fib4 Stages'), ('APRI', 'APRI Stages')):
        if col in df.columns:
            out[name] = df[col].to_numpy(int)
    return out


# ------------------------------------------------------------------- main ---
def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {sc: {t: prepare_data(t, False, sc)
                 for t in ('three_stage', 'fibrosis', 'cirrhosis')}
            for sc in (False, True)}

    rows = []
    for label, dirname in MODELS:
        sc = label in SCALED_MODELS
        m3 = load_models(dirname, 'three_stage')
        if m3 is None:
            print(f'  {label:15s} -- no three-stage checkpoint')
            continue
        m_fib = load_models(dirname, 'fibrosis')
        m_cir = load_models(dirname, 'cirrhosis')
        has_dec = m_fib is not None and m_cir is not None
        print(f'  {label:15s} native{"" if not has_dec else " + decomposition"}')

        for split, (ix, iy) in SPLITS.items():
            d3 = data[sc]['three_stage']
            y = np.asarray(d3[iy][0]).ravel()
            p_native = ensemble_proba(m3, d3[ix])

            sources = {'native': p_native}
            if has_dec:
                p2 = ensemble_proba(m_fib, data[sc]['fibrosis'][ix])[:, 1]
                p4 = ensemble_proba(m_cir, data[sc]['cirrhosis'][ix])[:, 1]
                if len(p2) == len(y) and len(p4) == len(y):
                    sources['decomposed'] = ordinal_decomposition(p2, p4)

            for src, p in sources.items():
                for rname, rule in RULES.items():
                    r = evaluate(y, p, rule(p))
                    r.update(model=label, split=split, source=src, rule=rname,
                             n=len(y))
                    rows.append(r)

    # comparators
    for split, csv_split in (('test', 'test'), ('external', 'prospective')):
        sc = score_stages(csv_split)
        if not sc:
            continue
        y = sc['y']
        for name in ('FIB-4', 'APRI'):
            if name not in sc:
                continue
            pred = sc[name]
            p = np.zeros((len(pred), 3))
            p[np.arange(len(pred)), pred] = 1.0        # hard labels, no ranking
            r = evaluate(y, p, pred)
            r['auroc'] = np.nan
            r.update(model=name, split=split, source='score', rule='cut-offs',
                     n=len(y))
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'ordinal_decision_rules.csv', index=False)

    # ---- what may drive the choice: validation only ----
    val = df[df.split == 'validation']
    print('\n' + '=' * 78)
    print('SELECTION — validation partition only')
    print('=' * 78)
    if len(val):
        best = (val.groupby(['source', 'rule'])[['kappa_lin', 'mae']]
                .agg({'kappa_lin': 'mean', 'mae': 'mean'})
                .sort_values('kappa_lin', ascending=False))
        print(best.round(4).to_string())
        top = best.index[0]
        print(f'\n  -> best on validation: source={top[0]}, rule={top[1]}')
        print('     Use this for the manuscript. Do not switch to whatever wins '
              'on test or external.')

    print('\n' + '=' * 78)
    print('REPORTING — all rules, external cohort')
    print('=' * 78)
    ext = df[(df.split == 'external')].copy()
    show = ext.sort_values('kappa_lin', ascending=False)[
        ['model', 'source', 'rule', 'acc', 'kappa_lin', 'kappa_quad', 'mae',
         'n_two_stage_errors', 'auroc']]
    print(show.head(25).round(4).to_string(index=False))

    print('\nCurrent rule (native + mode), external:')
    cur = ext[(ext.source == 'native') & (ext.rule == 'mode')]
    print(cur[['model', 'acc', 'kappa_lin', 'mae']].round(4).to_string(index=False))

    fib = ext[ext.model == 'FIB-4']
    if len(fib):
        f = fib.iloc[0]
        print(f'\nFIB-4 external: ACC {f.acc:.2f}, kappa_lin {f.kappa_lin:.3f}, '
              f'MAE {f.mae:.3f}, two-stage errors {int(f.n_two_stage_errors)}')
        beat = ext[(ext.kappa_lin > f.kappa_lin) & (ext.source != 'score')]
        print(f'Model/rule combinations above FIB-4 on kappa_lin: {len(beat)}')
        if len(beat):
            print(beat.sort_values('kappa_lin', ascending=False)
                  [['model', 'source', 'rule', 'kappa_lin', 'mae']]
                  .head(8).round(4).to_string(index=False))

    print(f'\n-> {OUT_DIR}/ordinal_decision_rules.csv')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()