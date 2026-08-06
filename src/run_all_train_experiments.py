"""
run_all_train_experiments.py
============================
Retrain the deep-learning ensembles (ffn / vi_bnn / tab_transformer / gandalf)
across every task, so their checkpoints match the current environment's library
versions. Tree models (light_gbm/rf/xgb/svm) do NOT need retraining - their
pickled sklearn models are version-stable.

Place in:  src/run_all_train_experiments.py
Run from:  src/    ->   python run_all_train_experiments.py

IMPORTANT - must match run_all_experiments.py:
  * SHAP_SELECTED here == SHAP_SELECTED in the eval sweep (both False),
    otherwise the checkpoint expects a different feature dimension than
    evaluation provides.
  * scaling policy here == scaling policy at eval time. In the original repo
    ONLY vi_bnn uses scaling=True; every other model uses scaling=False.
    Training and evaluation MUST agree on this per model.

Delete old DL checkpoints first (as you planned):
    rm -f models/ffn/*.pth models/vi_bnn/*.pth \
          models/tab_transformer/*.pth models/gandalf/*.pth \
          models/*/model_params_* models/*/df_cols.txt
(keep the tree-model pickles under models/light_gbm, models/rf, ...)
"""

import os
import traceback
from datetime import datetime
from pathlib import Path

from src.preprocess import prepare_data
from src.train import hypertrain


# ---------------------------------------------------------------------------
# CONFIGURATION  (keep in sync with run_all_experiments.py)
# ---------------------------------------------------------------------------
TASKS = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']

# Only the DL models need retraining. Add/remove as needed.
TRAIN_MODELS = ['svm', 'rf', 'xgb', 'light_gbm', 'ffn', 'gandalf', 'tab_transformer', 'vi_bnn']

# Must equal the eval sweep. Original repo: only vi_bnn scales.
SCALING_MODELS = {'vi_bnn'}

# Must equal the eval sweep (run_all_experiments.py uses False).
SHAP_SELECTED = True


# ---------------------------------------------------------------------------
def _prepare_once(task, scaling, cache):
    key = (task, scaling)
    if key not in cache:
        print(f'\n=== preprocessing  task={task}  scaling={scaling} ===')
        cache[key] = prepare_data(task, SHAP_SELECTED, scaling)
    return cache[key]


def train_sweep():
    log_root = Path('outputs/results')
    log_root.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime('train_%Y%m%d_%H%M%S')
    fail_log = log_root / f'{tag}_train_failures.log'

    for task in TASKS:
        d = prepare_data(task, False, False)
        print('UMM test+val+train:', sum(len(x[0]) for x in [d[0], d[2], d[4]]))
        print('MAINZ:', len(d[7][0]))

    cache = {}
    for task in TASKS:
        for model in TRAIN_MODELS:
            scaling = model in SCALING_MODELS
            try:
                (xs_train, ys_train, xs_val, ys_val, xs_test, ys_test,
                 xs_pro, ys_pro, df_cols) = _prepare_once(task, scaling, cache)

                print(f'\n########## TRAIN {model} | TASK {task} '
                      f'| scaling={scaling} | shap_selected={SHAP_SELECTED} ##########')
                hypertrain(
                    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test,
                    xs_pro, ys_pro, df_cols=df_cols,
                    classification_type=task, model_name=model,
                    shap_selected=SHAP_SELECTED
                )
            except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                msg = f'[TRAIN FAIL] model={model} task={task}: {exc}'
                print(msg)
                with open(fail_log, 'a') as f:
                    f.write(msg + '\n')
                    f.write(traceback.format_exc() + '\n')

    print(f'\nTraining sweep done. Failures (if any): {fail_log}')
    print('Next: set ENABLE_ADVANCED=True in run_all_experiments.py, then')
    print('      python run_all_experiments.py && python aggregate_results.py')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    train_sweep()
