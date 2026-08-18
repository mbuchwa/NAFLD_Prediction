# Retraining playbook after reviewer comments

This playbook gives the exact order of commands to regenerate preprocessing outputs and retrain/evaluate models with the updated methodological safeguards.

## 0) Environment

From repository root:

```bash
cd /workspace/NAFLD_Prediction
python --version
```

## 1) Regenerate preprocessing artifacts (temporal filtering, attrition, missingness, censored-value tracking)

Run any script that calls `prepare_data(...)` once to force preprocessing outputs to be rebuilt.

```bash
cd src
python train.py
```

This refreshes the data-QC artifacts under `outputs/data_qc/`, including:
- `lab_timing_summary.csv/json`
- `patient_attrition_summary.csv/json`
- `missingness_profile.csv`
- `censored_values_summary.csv`

## 2) Lab-window robustness sweep (Issue 2)

```bash
cd src
python lab_window_robustness.py
python print_recommended_lab_window.py
```

This creates:
- `outputs/robustness/lab_window_auroc_comparison.csv`

Window definitions currently included:
- pre-only: `-90/+0`
- symmetric: `±30`
- symmetric: `±60`

## 3) Main model training with chosen window

Edit `src/train.py` main block and set:
- `classification_type` (`two_stage`, `fibrosis`, `cirrhosis`, or `three_stage`)
- `model_name` (e.g., `light_gbm`)

Then run:

```bash
cd src
python train.py
```

## 4) One-time held-out + external evaluation (no test-set CV reuse)

Edit `src/test.py` main block to match the task/model and ensure:
- `smote = False` (required for external validation)

Then run:

```bash
cd src
python test.py
```

This exports external prevalence in:
- `outputs/external/class_prevalence.csv`

## 5) Optional sensitivity run: remove top-missing biomarkers (Issue 11)

```bash
cd src
python missingness_sensitivity.py
```

Creates:
- `outputs/robustness/missingness_sensitivity.csv`

## 6) Quick checklist before manuscript export

- Verify timing leakage control outputs exist in `outputs/data_qc/`.
- Verify robustness table exists in `outputs/robustness/lab_window_auroc_comparison.csv`.
- Verify external prevalence report exists and reflects true class distribution (no SMOTE).
- Keep train/test split discipline: CV only in training/tuning, held-out test evaluated once.

## 7) Authoritative manuscript training settings (execution record)

The settings below reconstruct the final manuscript runs from the checked-in
analysis scripts and the configuration changes in Git history. They are an
**execution record, not a request to change today's checked-in defaults**.
`src/run_all_train_experiments.py`, `src/train.py`, and `src/test.py` are
manually configured entry points; consequently, reproducing a historical run
requires making the listed temporary edits and reverting them afterward.

### Settings shared by the manuscript runs

- The standard preprocessing window was `window_days_pre=7` and
  `window_days_post=0` (nearest eligible pre-biopsy laboratory value).
- `SCALING_MODELS = {'vi_bnn'}`: `vi_bnn` used `scaling=True`; every other
  model used `scaling=False`. Training and evaluation used the same value.
- `select_patients=False` and `smote=False` were used for held-out/external
  evaluation. In particular, external data were not SMOTE-resampled.
- Training was dispatched through `hypertrain(...)`; evaluation was dispatched
  through `testing(...)`. `shap_selected` had to match between the two.

### 7.1 Full-feature binary-task training

For the three binary manuscript endpoints, the sweep was run with:

```python
TASKS = ['fibrosis', 'two_stage', 'cirrhosis']
TRAIN_MODELS = [
    'svm', 'rf', 'xgb', 'light_gbm',
    'ffn', 'gandalf', 'tab_transformer', 'vi_bnn',
]
SCALING_MODELS = {'vi_bnn'}
SHAP_SELECTED = False
```

The equivalent per-run settings in `src/train.py` and `src/test.py` were
`classification_type=<one binary task>`, `model_name=<one model>`,
`shap_selected=False`, and `scaling=(model_name == 'vi_bnn')`. Evaluation also
used `select_patients=False` and `smote=False`.

The all-model historical snapshot included `three_stage` in `TASKS`; separating
the binary endpoints above reflects the manuscript reporting groups rather than
a different preprocessing or model policy.

### 7.2 Three-stage training

The three-stage models used the same full-feature sweep settings, with:

```python
TASKS = ['three_stage']  # historically included alongside the three binary tasks
TRAIN_MODELS = [
    'svm', 'rf', 'xgb', 'light_gbm',
    'ffn', 'gandalf', 'tab_transformer', 'vi_bnn',
]
SCALING_MODELS = {'vi_bnn'}
SHAP_SELECTED = False
```

Thus the manual single-run equivalent was
`classification_type='three_stage'`, the selected `model_name`,
`shap_selected=False`, and scaling only for `vi_bnn`. Three-stage training was
not a SHAP-reduced run.

### 7.3 Development-only SHAP feature selection

`src/shap_feature_selection_development.py` is the authoritative selection
procedure for the final reduced binary models:

- `TASKS = ['fibrosis', 'two_stage', 'cirrhosis']`; `three_stage` remained
  commented out.
- It called `prepare_data(task, False, False)`, i.e. full features,
  `shap_selected=False`, and `scaling=False`.
- The ranking model was the corresponding full-feature **LightGBM** ensemble.
  For `two_stage`, the prespecified `pre7_post0` checkpoint was the fallback if
  the normal LightGBM checkpoint was absent.
- Each ensemble member was explained only on its matching UMM training
  imputation. Validation, held-out UMM test, and MAINZ observations did not
  enter selection.
- Global importance was mean absolute SHAP across patients (and across classes
  for multiclass output), then averaged across ensemble members. The top three
  features per task were selected.

Do not use `src/derive_shap_top_features.py` to reconstruct this final selection:
that older helper reads publication SHAP records and can therefore represent a
different analysis path.

### 7.4 Reduced-feature retraining

After producing/checking the development-only top-three feature sets, the
historical runner was manually changed to the currently visible reduced-run
configuration:

```python
TASKS = ['fibrosis', 'two_stage', 'cirrhosis']
TRAIN_MODELS = ['rf', 'xgb']
SCALING_MODELS = {'vi_bnn'}
SHAP_SELECTED = True
```

Because neither selected model is in `SCALING_MODELS`, both reduced RF and XGB
runs actually used `scaling=False`. Evaluation had to use the same three binary
tasks/models with `shap_selected=True`, `scaling=False`,
`select_patients=False`, and `smote=False`. The `*_shap_selected` model/output
names distinguish these checkpoints from full-feature models.

This configuration is still checked in because it was the last manual run; it
must not be interpreted as a universal default, nor silently changed back to
the full-feature sweep.

### 7.5 Fine-tuning and the final manuscript

No fine-tuned checkpoint or fine-tuned output is present in the manuscript
analysis records, and none of the final table/figure recomputation scripts load
a `*_finetuned` model. Therefore **no fine-tuning run is evidenced as part of
the final manuscript results**.

For completeness, `src/finetuning.py` is a separate, manual experimental entry
point. Its checked-in example is:

```python
model_name = 'light_gbm'
classification_type = 'cirrhosis'
shap_selected = False
scaling = False
# prepare_data(..., finetune=True)
```

It permits only `xgb` or `light_gbm` and swaps the cohort roles via
`finetune=True`. Those example values document how the dormant script is
configured; they are **not evidence that its output was used in the final
manuscript**. Do not run it or substitute its outputs without maintainer
confirmation.

### Historical manual-edit sequence

The relevant configuration snapshots were:

1. The runner initially contained all four tasks/all eight models with
   `SHAP_SELECTED=True` (an intermediate reduced-feature configuration).
2. It was corrected to all four tasks/all eight models with
   `SHAP_SELECTED=False` for the authoritative full-feature and three-stage
   training sweep.
3. It was then manually narrowed to the three binary tasks, RF/XGB, and
   `SHAP_SELECTED=True` for reduced-feature retraining; this is the current
   checked-in state.

When reproducing these runs, record the temporary diff (or use a disposable
branch), run the matching evaluation configuration, and restore the checked-in
state. Do not “clean up” these manual settings without maintainer approval.
