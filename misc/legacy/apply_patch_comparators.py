"""
apply_patch_comparators.py
==========================
Wires BOTH FIB-4 and APRI into evaluate_performance as fully-evaluated
comparators, so every metrics JSON carries a 'fib4' and an 'apri' block with the
same metrics as the models (AUROC + Sens/Spec/PPV/NPV with bootstrap CIs for the
binary tasks; weighted kappa + MAE for three-stage). make_latex_tables.py then
renders both as comparator rows.

Place in:  src/            Run from:  src/  ->  python apply_patch_comparators.py
Requires:  validation_tools.py already writes `structured_record` (Patch A).
Idempotent; writes a .bak and reverts on syntax error.
"""

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path('utils/validation_tools.py')

# --- helper functions inserted before _get_eval_output_dir -------------------
HELPERS = '''
FIB4_CUTOFFS = {'fibrosis': 1.45, 'two_stage': 2.67, 'cirrhosis': 3.25}
APRI_CUTOFFS = {'fibrosis': 1.5, 'two_stage': 1.5, 'cirrhosis': 2.0}
FIB4_THREE_STAGE_CUTOFFS = (1.45, 3.25)
APRI_THREE_STAGE_CUTOFFS = (1.5, 2.0)
APRI_AST_ULN = 35.0  # keep in sync with AST_ULN in preprocess.py


def _extract_apri_scores_from_features(x_matrix, df_cols):
    """APRI = (AST / AST_ULN) / platelets * 100, from the translated feature matrix."""
    required = ['ASAT (U/l)', 'Platelets (Billion/l)']
    if not all(c in df_cols for c in required):
        return None
    f = pd.DataFrame(x_matrix, columns=df_cols)
    denom = np.clip(f['Platelets (Billion/l)'], 1e-6, None)
    return (((f['ASAT (U/l)'] / APRI_AST_ULN) / denom) * 100.0).to_numpy(dtype=float)


def _evaluate_score_binary(y_true, scores, cutoff, bootstrap_n=1000):
    """AUROC + operating metrics for a continuous score at a fixed cut-off."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
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
    for mn in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
        def mb(rng, _mn=mn):
            idx = rng.integers(0, n, n)
            return _calc_binary_operating_metrics(y_true[idx], scores[idx], cutoff)[_mn]
        op_cis[mn] = _bootstrap_ci(mb, n=bootstrap_n)
    return {'threshold': float(cutoff), 'auroc': auroc,
            'auroc_ci_lower': auc_ci[0], 'auroc_ci_upper': auc_ci[1],
            'operating_metrics': op, 'operating_cis': op_cis}


def _evaluate_score_three_stage(y_true, scores, cutoffs):
    """Ordinal metrics for a continuous score staged with two cut-offs."""
    y_true = np.asarray(y_true).astype(int)
    staged = np.digitize(np.asarray(scores, dtype=float), list(cutoffs))
    return {'cutoffs': [float(c) for c in cutoffs],
            'accuracy': float(np.mean(staged == y_true)),
            'cohen_kappa_linear': float(cohen_kappa_score(y_true, staged, weights='linear')),
            'cohen_kappa_quadratic': float(cohen_kappa_score(y_true, staged, weights='quadratic')),
            'mae': float(mean_absolute_error(y_true, staged))}


def _comparator_record(name, raw_scores, eval_y_ref, eval_proba_ref, classification_type):
    """Build one comparator block (binary or three-stage) or None."""
    if raw_scores is None:
        return None
    try:
        if eval_proba_ref.shape[1] == 2:
            cut = (FIB4_CUTOFFS if name == 'fib4' else APRI_CUTOFFS).get(classification_type)
            return _evaluate_score_binary(eval_y_ref, raw_scores, cut) if cut else None
        if classification_type == 'three_stage':
            cuts = FIB4_THREE_STAGE_CUTOFFS if name == 'fib4' else APRI_THREE_STAGE_CUTOFFS
            return _evaluate_score_three_stage(eval_y_ref, raw_scores, cuts)
    except Exception as exc:
        print(f'{name.upper()} comparator skipped: {exc}')
    return None

'''

ANCHOR_HELPERS = "def _get_eval_output_dir(model_name, prospective):"

# --- build both comparator records before structured_record ------------------
ANCHOR_REC = "    structured_record = {"
INSERT_REC = """    # ---- FIB-4 and APRI comparators on the very same split ----
    _fib4_raw = _extract_fib4_scores_from_features(xs_test[0], df_cols)
    _apri_raw = _extract_apri_scores_from_features(xs_test[0], df_cols)
    fib4_record = _comparator_record('fib4', _fib4_raw, eval_y_ref, eval_proba_ref, classification_type)
    apri_record = _comparator_record('apri', _apri_raw, eval_y_ref, eval_proba_ref, classification_type)

    structured_record = {"""

# --- add the two fields into the dict ----------------------------------------
ANCHOR_FIELD = "        'ordinal': ordinal_metrics,"
INSERT_FIELD = "        'ordinal': ordinal_metrics,\n        'fib4': fib4_record,\n        'apri': apri_record,"


def main():
    if not TARGET.exists():
        sys.exit(f'ERROR: {TARGET} not found. Run from the src/ directory.')
    text = TARGET.read_text()

    if 'structured_record' not in text:
        sys.exit('ERROR: Patch A not applied (no structured_record). Apply it first.')
    if '_evaluate_score_binary' in text:
        print('Comparator patch already applied - nothing to do.')
        return

    for name, anchor in [('helpers', ANCHOR_HELPERS), ('record', ANCHOR_REC),
                         ('field', ANCHOR_FIELD)]:
        if text.count(anchor) != 1:
            sys.exit(f'ERROR: anchor {name} found {text.count(anchor)}x (expected 1).')

    # required prerequisites from Patch A / base file
    for needed in ['_extract_fib4_scores_from_features', '_bootstrap_ci',
                   '_calc_binary_operating_metrics', 'roc_auc_score',
                   'cohen_kappa_score', 'mean_absolute_error']:
        if needed not in text:
            sys.exit(f'ERROR: expected helper/import {needed!r} not found in the file.')

    shutil.copy2(TARGET, TARGET.with_suffix('.py.bak'))
    text = text.replace(ANCHOR_HELPERS, HELPERS.strip('\n') + '\n\n\n' + ANCHOR_HELPERS)
    text = text.replace(ANCHOR_REC, INSERT_REC)
    text = text.replace(ANCHOR_FIELD, INSERT_FIELD)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        sys.exit(f'ERROR: patched source would not parse ({exc}); nothing written.')

    TARGET.write_text(text)
    print(f'FIB-4 + APRI comparators wired in. Backup: {TARGET.with_suffix(".py.bak")}')
    print('Next:  python run_all_experiments.py   (evaluation only, no retraining)')
    print('       python aggregate_results.py')
    print('       python make_latex_tables.py')


if __name__ == '__main__':
    main()