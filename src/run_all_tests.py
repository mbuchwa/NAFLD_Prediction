"""
run_all_experiments.py
======================
Consecutive evaluation of every trained ensemble across every classification
task, with per-task data-QC snapshots and machine-readable metric records.

Place in:  src/run_all_experiments.py
Run from:  src/         ->  python run_all_experiments.py

What it does
------------
1. For each (task, scaling) it calls `prepare_data` exactly once and reuses the
   splits across all models that share that scaling setting (big speed-up).
2. For each (model, task) it calls `testing(...)` -> evaluate_performance, which
   writes ROC / confusion-matrix / calibration / DCA artefacts AND (after the
   small JSON patch, see patch_notes.md) one structured metrics JSON per split.
3. After each task it snapshots outputs/data_qc/* into
   outputs/results/<run_tag>/data_qc_<task>/ so nothing is overwritten between
   tasks.
4. Every (model, task) is wrapped in try/except: one failing model never aborts
   the whole sweep; failures are logged to outputs/results/<run_tag>/failures.log

Configure MODELS / TASKS below. Deep-learning models are OFF by default because
they need trained checkpoints and a Python >=3.10 import chain; switch them on
once those are in place.
"""

import os
import sys
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

# --- repo imports (must run from src/) -------------------------------------
from src.preprocess import prepare_data
from src.utils.ger_eng_dict import dict_germ_eng
from src.test import testing, export_external_class_prevalence


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TASKS = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']

# ---- model groups ---------------------------------------------------------
# Tree / classical models: safe, fast, no special import chain.
TREE_MODELS = ['light_gbm', 'rf', 'xgb', 'svm']

# Deep-learning / advanced models: need trained checkpoints AND a working import
# chain (tab_transformer / gandalf / vi_bnn require Python >=3.10 for their
# dependencies; see the lazy-import fix in test.py). 'mcmc_bnn' additionally
# needs a branch in test.testing() - see note in ENABLE_ADVANCED below.
ADVANCED_MODELS = ['ffn', 'vi_bnn', 'tab_transformer', 'gandalf']

# Flip to True once checkpoints exist and the imports load without error.
# Even when False you lose nothing: failures are caught per (model, task) and
# logged, so you *can* set the full list and let the sweep skip what's missing.
ENABLE_ADVANCED = True

MODELS = TREE_MODELS + (ADVANCED_MODELS if ENABLE_ADVANCED else [])

# Models that require standardised features (scaling=True in prepare_data):
SCALING_MODELS = {'vi_bnn', 'ffn', 'mcmc_bnn'}

SHAP_SELECTED = True
SELECT_PATIENTS = False          # keep False: closest-patient selection biases external eval
SMOTE = False                    # keep False: no oversampling on external cohort

# Optionally repeat the whole sweep once more with borderline MAINZ findings
# included (requires the create_scores(include_borderline=...) patch).
RUN_BORDERLINE_SENSITIVITY = False


# ---------------------------------------------------------------------------
def _run_tag():
    return datetime.now().strftime('run_%Y%m%d_%H%M%S')


def _snapshot_data_qc(dst_dir):
    """Copy the data-QC artefacts produced during preprocessing into dst_dir."""
    src_qc = Path('outputs/data_qc')
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    if src_qc.exists():
        for f in src_qc.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    # external prevalence lives elsewhere
    ext_prev = Path('outputs/external/class_prevalence.csv')
    if ext_prev.exists():
        shutil.copy2(ext_prev, dst / 'class_prevalence.csv')


def _prepare_once(task, scaling, cache):
    """prepare_data is expensive; cache by (task, scaling)."""
    key = (task, scaling)
    if key not in cache:
        print(f'\n=== preprocessing  task={task}  scaling={scaling} ===')
        data = prepare_data(
            task, SHAP_SELECTED, scaling,
            select_patients=SELECT_PATIENTS, smote=SMOTE
        )
        cache[key] = data
    return cache[key]


def sweep(run_tag, include_borderline=False):
    results_root = Path(f'outputs/results/{run_tag}')
    results_root.mkdir(parents=True, exist_ok=True)
    fail_log = results_root / 'failures.log'

    # Models present in the assert list but WITHOUT a branch in test.testing():
    # they would silently produce no evaluation. Warn loudly instead.
    UNWIRED_IN_TESTING = {'mcmc_bnn'}

    cache = {}
    for task in TASKS:
        for model in MODELS:
            if model in UNWIRED_IN_TESTING:
                warn = (f'[SKIP] model={model} task={task}: no branch in '
                        f'test.testing() - add an `elif model_name == "{model}"` '
                        f'block there before enabling it.')
                print(warn)
                with open(fail_log, 'a') as f:
                    f.write(warn + '\n')
                continue
            scaling = model in SCALING_MODELS
            try:
                (_, _, xs_val, ys_val, xs_test, ys_test,
                 xs_pro, ys_pro, df_cols) = _prepare_once(task, scaling, cache)

                # Export empirical external prevalence for this task
                export_external_class_prevalence(
                    ys_pro,
                    output_path=f'outputs/external/class_prevalence_{task}.csv'
                )

                df_cols_en = [dict_germ_eng[c] for c in df_cols]

                print(f'\n########## MODEL {model} | TASK {task} '
                      f'| borderline={include_borderline} ##########')
                testing(
                    xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val,
                    df_cols=df_cols_en, classification_type=task,
                    model_name=model, shap_selected=SHAP_SELECTED
                )
            except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                msg = (f'[FAIL] model={model} task={task} '
                       f'borderline={include_borderline}: {exc}')
                print(msg)
                with open(fail_log, 'a') as f:
                    f.write(msg + '\n')
                    f.write(traceback.format_exc() + '\n')

        # snapshot QC once per task (after last model of that task)
        _snapshot_data_qc(results_root / f'data_qc_{task}')

    print(f'\nDone. Structured artefacts under: {results_root}')
    print('Next: python aggregate_results.py '
          f'--run {run_tag}   # builds tidy CSV + xlsx tables')
    return results_root


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)  # ensure we run from src/
    tag = _run_tag()

    print(f'>>> primary sweep  ({tag})')
    sweep(tag, include_borderline=False)

    if RUN_BORDERLINE_SENSITIVITY:
        # Requires create_scores(include_borderline=True) wired through
        # prepare_data. See patch_notes.md, section B.
        print('\n>>> borderline-sensitivity sweep')
        os.environ['NAFLD_INCLUDE_BORDERLINE'] = '1'
        sweep(tag + '_borderline', include_borderline=True)
        os.environ.pop('NAFLD_INCLUDE_BORDERLINE', None)