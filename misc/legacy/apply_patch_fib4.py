"""
apply_patch_fib4.py
===================
Adds FIB-4 as a fully-fledged comparator to evaluate_performance, so that every
metrics JSON carries a `fib4` block with the same metrics as the models:
AUROC + sensitivity / specificity / PPV / NPV with 95% bootstrap CIs (binary
tasks), or weighted kappa + MAE (three-stage task).

FIB-4 is evaluated at its established clinical cut-off for each task
(1.45 moderate fibrosis, 2.67 severe fibrosis, 3.25 cirrhosis), which is how the
score is used in practice; AUROC is threshold-independent and therefore
unaffected by that choice.

Place in:  src/            Run from:  src/  ->  python apply_patch_fib4.py
Requires:  Patch A already applied (structured_record exists).
Idempotent; writes a .bak backup and reverts on syntax error.
"""

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path('utils/validation_tools.py')

HELPER = '''

FIB4_CUTOFFS = {'fibrosis': 1.45, 'two_stage': 2.67, 'cirrhosis': 3.25}
FIB4_THREE_STAGE_CUTOFFS = (1.45, 3.25)


def _evaluate_fib4_binary(y_true, fib4_scores, cutoff, bootstrap_n=1000):
    """AUROC + operating metrics for the continuous FIB-4 score at a fixed cut-off."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(fib4_scores, dtype=float)
    n = y_true.shape[0]
    if len(np.unique(y_true)) < 2:
        return None

    auroc = float(roc_auc_score(y_true, scores))

    def auc_bootstrap(rng):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            return None
        return roc_auc_score(y_true[idx], scores[idx])

    auc_ci = _bootstrap_ci(auc_bootstrap, n=bootstrap_n)
    op = _calc_binary_operating_metrics(y_true, scores, cutoff)
    op_cis = {}
    for metric_name in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
        def metric_bootstrap(rng, mn=metric_name):
            idx = rng.integers(0, n, n)
            return _calc_binary_operating_metrics(y_true[idx], scores[idx], cutoff)[mn]
        op_cis[metric_name] = _bootstrap_ci(metric_bootstrap, n=bootstrap_n)

    return {
        'threshold': float(cutoff),
        'auroc': auroc,
        'auroc_ci_lower': auc_ci[0],
        'auroc_ci_upper': auc_ci[1],
        'operating_metrics': op,
        'operating_cis': op_cis,
    }


def _evaluate_fib4_three_stage(y_true, fib4_scores):
    """Ordinal metrics for FIB-4 staged with the two established cut-offs."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(fib4_scores, dtype=float)
    lo, hi = FIB4_THREE_STAGE_CUTOFFS
    staged = np.digitize(scores, [lo, hi])  # 0: <1.45, 1: 1.45-3.25, 2: >3.25
    return {
        'cutoffs': [float(lo), float(hi)],
        'accuracy': float(np.mean(staged == y_true)),
        'cohen_kappa_linear': float(cohen_kappa_score(y_true, staged, weights='linear')),
        'cohen_kappa_quadratic': float(cohen_kappa_score(y_true, staged, weights='quadratic')),
        'mae': float(mean_absolute_error(y_true, staged)),
    }
'''

ANCHOR_HELPER = "def _get_eval_output_dir(model_name, prospective):"

ANCHOR_REC = """    # ---- machine-readable record for downstream table building ----
    structured_record = {"""

INSERT_REC = """    # ---- FIB-4 comparator on the very same split ----
    fib4_record = None
    _fib4_raw = _extract_fib4_scores_from_features(xs_test[0], df_cols)
    if _fib4_raw is not None:
        try:
            if eval_proba_ref.shape[1] == 2 and classification_type in FIB4_CUTOFFS:
                fib4_record = _evaluate_fib4_binary(
                    eval_y_ref, _fib4_raw, FIB4_CUTOFFS[classification_type])
            elif classification_type == 'three_stage':
                fib4_record = _evaluate_fib4_three_stage(eval_y_ref, _fib4_raw)
        except Exception as exc:  # never let the comparator break the run
            print(f'FIB-4 comparator skipped: {exc}')
    else:
        print('FIB-4 comparator skipped: required biomarker columns unavailable.')

    # ---- machine-readable record for downstream table building ----
    structured_record = {"""

ANCHOR_FIELD = "        'ordinal': ordinal_metrics,"
INSERT_FIELD = "        'ordinal': ordinal_metrics,\n        'fib4': fib4_record,"


def main():
    if not TARGET.exists():
        sys.exit(f'ERROR: {TARGET} not found. Run this from the src/ directory.')
    text = TARGET.read_text()

    if 'structured_record' not in text:
        sys.exit('ERROR: Patch A not applied yet - run apply_patch_a.py first '
                 '(or use the patched validation_tools.py).')
    if '_evaluate_fib4_binary' in text:
        print('FIB-4 patch already applied - nothing to do.')
        return

    for name, anchor in [('helper', ANCHOR_HELPER), ('record', ANCHOR_REC),
                         ('field', ANCHOR_FIELD)]:
        if text.count(anchor) != 1:
            sys.exit(f'ERROR: anchor {name} found {text.count(anchor)}x (expected 1).')

    shutil.copy2(TARGET, TARGET.with_suffix('.py.bak'))

    text = text.replace(ANCHOR_HELPER, HELPER.strip('\n') + '\n\n\n' + ANCHOR_HELPER)
    text = text.replace(ANCHOR_REC, INSERT_REC)
    text = text.replace(ANCHOR_FIELD, INSERT_FIELD)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        sys.exit(f'ERROR: patched source would not parse ({exc}); nothing written.')

    TARGET.write_text(text)
    print(f'FIB-4 patch applied. Backup: {TARGET.with_suffix(".py.bak")}')
    print('Next:  python run_all_experiments.py   (evaluation only, no retraining)')
    print('       python make_latex_tables.py')


if __name__ == '__main__':
    main()