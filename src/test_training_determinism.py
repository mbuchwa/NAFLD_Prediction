"""
test_training_determinism.py
============================
Answers one question before any retraining: is the run-to-run variation in the
LightGBM results caused by the training code, or by the data it was given?

Place in:  src/test_training_determinism.py
Run from:  src/    ->    python test_training_determinism.py

Runs on ONE imputation of ONE task (default: cirrhosis, imputation 0), which is
where the largest swing was observed (AUROC 0.797 vs 0.866 between two runs).

WHAT IT CHECKS
--------------
  A  Determinism as-is        train the current hypertrain function twice on
                              identical data, compare AUROC and the selected
                              hyperparameters. Identical -> the variation came
                              from the DATA, and something changed in the
                              preprocessing between the two runs. Different ->
                              the training itself is non-deterministic.

  B  subsample is inert       fits LightGBM with subsample=0.3 and 1.0 and
                              compares predictions bit for bit.

  C  early stopping           inspects whether the fitted LightGBM used fewer
                              trees than n_estimators, i.e. whether early
                              stopping actually fired.

  D  scoring granularity      how many of the sampled hyperparameter candidates
                              share the same CV score. Ties are broken by
                              position, so a coarse scorer makes the selection
                              sensitive to tiny data changes.

  E  candidate spread         AUROC range across the sampled candidates -- how
                              much the choice between them actually matters.

Nothing is written to models/ or outputs/; existing checkpoints are untouched.
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

TASK = 'cirrhosis'
IMPUTATION = 0
SEP = '=' * 72


def _auc(model, x, y):
    p = model.predict_proba(np.asarray(x))
    return float(roc_auc_score(np.asarray(y).ravel(), p[:, 1] if p.ndim > 1 else p))


def _params(model):
    return {k: getattr(model, k, None) for k in
            ('max_depth', 'learning_rate', 'subsample', 'subsample_freq',
             'n_estimators', 'random_state')}


def _n_trees(model):
    b = getattr(model, 'booster_', None)
    if b is not None and hasattr(b, 'num_trees'):
        return int(b.num_trees())
    b = getattr(model, 'get_booster', None)
    if callable(b):
        try:
            return len(b().get_dump())
        except Exception:
            return None
    return None


def main():
    try:
        from src.preprocess import prepare_data
        from src.models.light_gmb import hypertrain_light_gbm_model
    except ImportError:
        from preprocess import prepare_data
        from models.light_gmb import hypertrain_light_gbm_model
    import lightgbm as lgb

    print(f'{SEP}\nLoading data: task={TASK}, imputation={IMPUTATION}\n{SEP}')
    d = prepare_data(TASK, False, False)
    xt, yt = np.asarray(d[0][IMPUTATION]), np.asarray(d[1][IMPUTATION]).ravel()
    xv, yv = np.asarray(d[2][IMPUTATION]), np.asarray(d[3][IMPUTATION]).ravel()
    xs, ys = np.asarray(d[4][IMPUTATION]), np.asarray(d[5][IMPUTATION]).ravel()
    xp, yp = np.asarray(d[6][IMPUTATION]), np.asarray(d[7][IMPUTATION]).ravel()
    print(f'train {xt.shape}  val {xv.shape}  test {xs.shape}  mainz {xp.shape}')
    print(f'data fingerprint (train): {float(xt.sum()):.6f}')

    # ------------------------------------------------------------ A ---------
    print(f'\n{SEP}\nA  Determinism of the current training function\n{SEP}')
    runs = []
    for i in (1, 2):
        m = hypertrain_light_gbm_model(xt, yt, xv, yv, xs, ys, xp, yp,
                                       classification_type=TASK)
        runs.append(m)
        print(f'  run {i}: test AUROC {_auc(m, xs, ys):.4f}   '
              f'mainz AUROC {_auc(m, xp, yp):.4f}   {_params(m)}')

    same_pred = np.array_equal(runs[0].predict_proba(xs), runs[1].predict_proba(xs))
    same_par = _params(runs[0]) == _params(runs[1])
    print(f'\n  identical predictions : {same_pred}')
    print(f'  identical hyperparams : {same_par}')
    if same_pred:
        print('  => Training is deterministic. The variation between your two runs\n'
              '     came from the DATA, not the seed. Find what changed in the\n'
              '     preprocessing before finalising any table.')
    else:
        print('  => Training is NOT deterministic. Set random_state on the\n'
              '     estimator (see the revised light_gmb.py) and re-check.')

    # ------------------------------------------------------------ B ---------
    print(f'\n{SEP}\nB  Does subsample do anything?\n{SEP}')
    kw = dict(objective='binary', verbosity=-1, random_state=42, n_estimators=100)
    a = lgb.LGBMClassifier(subsample=0.3, **kw).fit(xt, yt)
    b = lgb.LGBMClassifier(subsample=1.0, **kw).fit(xt, yt)
    c = lgb.LGBMClassifier(subsample=0.3, subsample_freq=1, **kw).fit(xt, yt)
    print(f'  subsample 0.3 vs 1.0, subsample_freq=0 -> identical: '
          f'{np.array_equal(a.predict_proba(xs), b.predict_proba(xs))}')
    print(f'  subsample 0.3 vs 1.0, subsample_freq=1 -> identical: '
          f'{np.array_equal(c.predict_proba(xs), b.predict_proba(xs))}')
    print('  => If the first line says True, subsample is inert in the current\n'
          '     search space: one of three hyperparameters has no effect.')

    # ------------------------------------------------------------ C ---------
    print(f'\n{SEP}\nC  Did early stopping fire?\n{SEP}')
    n = _n_trees(runs[0])
    ne = getattr(runs[0], 'n_estimators', None)
    print(f'  trees in the fitted model : {n}')
    print(f'  n_estimators requested    : {ne}')
    if n is not None and ne is not None:
        print(f'  => {"early stopping fired" if n < ne else "NO early stopping — all trees were built"}')
        if n >= ne:
            print('     The Methods section claims early stopping on the validation\n'
                  '     partition. For LightGBM that is currently not the case.')

    # ------------------------------------------------------------ D+E -------
    print(f'\n{SEP}\nD  Scoring granularity and E  candidate spread\n{SEP}')
    grid = {'max_depth': np.arange(1, 40),
            'learning_rate': np.linspace(0.5, 0.01, 5),
            'subsample': np.linspace(1, 0.3, 5)}
    for scoring in ('neg_mean_squared_error', 'roc_auc'):
        rs = RandomizedSearchCV(
            lgb.LGBMClassifier(objective='binary', verbosity=-1, random_state=42),
            grid, scoring=scoring, cv=5, random_state=42, n_jobs=1).fit(xt, yt)
        sc = np.round(rs.cv_results_['mean_test_score'], 6)
        ties = int(list(sc).count(sc.max()))
        aucs = []
        for p in rs.cv_results_['params']:
            m = lgb.LGBMClassifier(objective='binary', verbosity=-1,
                                   random_state=42, **p).fit(xt, yt)
            aucs.append(_auc(m, xs, ys))
        print(f'  scoring={scoring:24s} distinct scores {len(set(sc)):2d}/10 | '
              f'tied at best {ties} | chosen max_depth {rs.best_estimator_.max_depth}')
        print(f'  {"":33s} candidate test AUROC {min(aucs):.3f}-{max(aucs):.3f} '
              f'(spread {max(aucs) - min(aucs):.3f})')
    print('\n  => A coarse scorer with ties makes the selection flip on small data\n'
          '     changes; the spread shows how much that flip is worth in AUROC.')

    print(f'\n{SEP}\nNothing was written to models/ or outputs/.\n{SEP}')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()