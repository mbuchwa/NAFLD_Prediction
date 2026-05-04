# Retraining playbook after reviewer comments

This playbook gives the exact order of commands to regenerate preprocessing outputs and retrain/evaluate models with the updated methodological safeguards.

## 0) Environment

From repository root:

```bash
cd /workspace/NAFLD_Prediction
python --version
```

## 1) Regenerate preprocessing artifacts (temporal filtering, attrition, missingness, censored-value tracking)

Run any script that calls `preprare_data(...)` once to force preprocessing outputs to be rebuilt.

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
