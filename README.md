# Reproducing laboratory-based NAFLD fibrosis staging

This repository contains the analysis code used to develop and externally evaluate machine-learning models for biopsy-defined liver-fibrosis staging from routine laboratory measurements. It supports reconstruction of preprocessing, model fitting, held-out and external validation, and manuscript tables and figures; it is research software, not a clinical diagnostic product.

## Scope and study overview

The study uses retrospective University Medical Center Mannheim (**UMM**) data for development/internal evaluation and an independent Mainz (**MAINZ**) cohort for external evaluation. Liver-biopsy fibrosis stages define every outcome; routine laboratory variables are the predictors. The eight reported model families are SVM, random forest, XGBoost, LightGBM, multilayer perceptron (FFN), variational-inference Bayesian neural network (VI-BNN), GANDALF, and TabTransformer. Analyses comprise three binary tasks—moderate fibrosis (F0–1 vs F2–4), severe fibrosis (F0–2 vs F3–4), and cirrhosis (F0–3 vs F4)—and one three-stage task (F0–1, F2–3, F4).

The repository covers classification-oriented analyses only. It excludes deployment, clinical decision support, causal inference, and unreported endpoints. **No patient-level data are included in this repository or its Git history.**

## Repository structure

```text
data/                 controlled raw inputs and generated local datasets (ignored)
src/                  preprocessing, models, training, evaluation, and reporting code
src/models/           canonical trained-checkpoint location
src/outputs/          primary results, tables, figures, QC, and clinical-utility outputs
outputs/               repository-root reporting and selected sensitivity/QC outputs
misc/                  exploratory, diagnostic, and legacy material (not primary workflow)
environment.yml       canonical Conda environment
```

See `data/README.md` for the exact data contract and `PATH_CONTRACT_INVENTORY.md` before moving any artifact.

## Installation

Run from the repository root. Prerequisites are Conda (or Mamba), a Linux-compatible environment, and, for the pinned GPU stack, a compatible NVIDIA driver/CUDA runtime.

```bash
conda env create -f environment.yml
conda activate nafl
```

The first command creates the canonical `nafl` environment; the second activates it. `requirements.txt` is retained for compatibility but is not the canonical reproducibility installation.

## Controlled data placement

Access is controlled; consult the manuscript and associated heiDATA record for availability and the request procedure. Authorized researchers must place the three workbooks named in `data/README.md` directly under `data/`. Prerequisites are approved access and the exact original filenames; the main outputs are ignored cleaned tables in `data/preprocessed_no_mice_<split>/` and ten imputed datasets in `data/preprocessed_mice_fib_<split>/`. Never force-add raw or derived patient data to Git.

## Primary workflow (ordered)

Run legacy analysis entry points from `src/` so their relative `models/`, `outputs/`, and `../data/` paths resolve correctly. Configuration is currently by constants in each script, not command-line arguments.

1. **Configure and train one model/task.** In `train.py`, set `model_name`, `classification_type`, and `shap_selected`; preserve the scaling policy described below.

   ```bash
   cd src
   python train.py
   ```

   Prerequisites: active `nafl` environment, controlled workbooks in `data/`, and a reviewed configuration. There is no separate canonical preprocessing command: `train.py` calls `prepare_data(...)`, which performs preprocessing/imputation before training. Main outputs are generated datasets under `data/`, data-QC artifacts, checkpoints under `src/models/<model>/`, and training artifacts under `src/outputs/<model>/`.

2. **Evaluate the matching checkpoint once.** Set the same task, model, feature-selection, and scaling choices in `test.py`; retain `select_patients=False` and `smote=False` for external evaluation.

   ```bash
   python test.py
   ```

   Prerequisites: matching checkpoints and generated data. Main outputs are held-out UMM and external MAINZ metrics/plots under `src/outputs/<model>/` (including `prospective/`) and external prevalence under `src/outputs/external/`. Evaluation preprocessing is combined in this entry point through `prepare_data(...)`.

3. **Recompute checkpoint-based manuscript tables.** Run each command only after all checkpoints required by that script exist.

   ```bash
   python recompute_tables.py
   python recompute_three_stage.py
   python recompute_reduced_tables.py
   ```

   The commands respectively produce binary tables (`src/outputs/tables/tables_recomputed.{csv,tex}`), the ordinal table (`table3_three_stage_recomputed.{csv,tex}`), and reduced-feature results (`table5_reduced_recomputed.{csv,tex}`). The reduced command additionally requires `src/outputs/shap_top_features.json` and matching `*_shap_selected` checkpoints.

4. **Generate publication figures and clinical-utility reporting.** These scripts consume generated datasets and compatible checkpoints directly; they do not consume the recomputed table CSVs.

   ```bash
   python make_publication_figures.py
   python shap_publication_figures.py
   python clinical_utility_from_checkpoints.py
   python check_table_figure_consistency.py
   ```

   Prerequisites: complete checkpoint/data contracts (and SHAP dependencies for SHAP output). Main outputs are publication PNG/PDF/CSV files in `src/outputs/figures/`, calibration metrics/plots and decision curves in `src/outputs/clinical_utility/`, and consistency results in `src/outputs/robustness/`. Missing checkpoints may be skipped by some reporting scripts; inspect warnings and outputs before reporting results.

5. **Optionally aggregate legacy JSON metrics.** This is distinct from checkpoint-recomputed tables.

   ```bash
   python aggregate_results.py
   ```

   Prerequisites: evaluation `*_metrics.json` files under `src/outputs/`. Main outputs are `src/outputs/results/all_metrics_long.csv` and `manuscript_tables.xlsx`.

## Simplified wrappers

The wrappers reduce repeated manual calls but remain configuration-driven:

```bash
cd src
PYTHONPATH=.. python run_all_train_experiments.py
PYTHONPATH=.. python run_all_tests.py
```

The explicit `PYTHONPATH` makes the repository-level `src` package importable while preserving the required working directory. Before either command, inspect and synchronize `TASKS`, model lists, `SCALING_MODELS`, and `SHAP_SELECTED`; ensure raw data are available, and ensure evaluation checkpoints exist. The training wrapper writes checkpoints plus a failure log in `src/outputs/results/`; the testing wrapper writes per-model metrics, prevalence files, QC snapshots, and a timestamped results/failure-log directory.

**Important:** the checked-in `run_all_train_experiments.py` is a reduced-feature RF/XGBoost configuration, whereas `run_all_tests.py` currently selects a broader model set with a different scaling configuration. Therefore, the training wrapper **does not currently train all eight reported models** and the two wrappers must not be run as though they were a matched full-study pipeline. Use the documented historical settings in `RETRAINING_PLAYBOOK.md`, make temporary reviewed edits, record the diff, and restore it afterward.

## Auxiliary, sensitivity, and QC analyses

These are separate from the ordered primary workflow and should not replace it.

| Command (run from `src/`) | Prerequisites | Main outputs |
|---|---|---|
| `python lab_window_robustness.py` | Controlled raw data and active environment | `src/outputs/robustness/lab_window_auroc_comparison.csv` and window checkpoints/artifacts |
| `python print_recommended_lab_window.py` | Completed lab-window sweep | Printed recommendation based on robustness output |
| `python run_missingness_sensitivity_all.py` | Controlled data and compatible checkpoints/dependencies | Repository-root `outputs/robustness/` sensitivity results |
| `python check_split_stratification.py` | Controlled raw/generated data | Split-balance QC reported by the script |
| `python test_training_determinism.py` | Full training dependencies | Determinism check output/failures in the terminal |
| `python paired_model_comparison.py` | Binary checkpoints and imputed evaluation data | `src/outputs/tables/paired_model_comparison.{csv,tex}` |
| `python ordinal_decision_rules.py` | Three-stage checkpoints and imputed data | `src/outputs/tables/ordinal_decision_rules.{csv,tex}` |

`misc/` contains exploratory, diagnostic, and legacy scripts and is not an alternative primary pipeline.

## Fixed reproducibility choices

- The prespecified standard laboratory window is 7 days before biopsy and 0 days after; the nearest eligible pre-biopsy measurement is used.
- Multiple imputation uses ten MICE/`IterativeImputer` datasets with posterior sampling and `random_state=0,...,9`; ensemble member *i* must remain paired with imputation *i*.
- UMM supplies train/validation/held-out test partitions; MAINZ remains the external (`prospective`) cohort. Do not apply SMOTE or patient selection to held-out/external evaluation.
- Full-study scaling is enabled only for VI-BNN and must match between training and evaluation. Feature selection must likewise match its checkpoint.
- Canonical table and ROC/SHAP routines use 1,000 patient-level bootstrap resamples and fixed seeds defined in their scripts (generally seed 0); clinical-utility calibration uses 1,000 resamples with seed 42.
- Preserve task IDs, model storage IDs (notably `light_gbm`), split names, filenames, and working-directory conventions in `PATH_CONTRACT_INVENTORY.md`.

## Reporting and artifact locations

Checkpoints are discovered under `src/models/<model>/`; `src/checkpoints/`, `src/saved_model/`, and `src/saved_models/` are not active fallbacks. Evaluation results are under `src/outputs/<model>/`, aggregate results under `src/outputs/results/`, recomputed tables under `src/outputs/tables/`, figures under `src/outputs/figures/`, and calibration/decision-curve artifacts under `src/outputs/clinical_utility/`. Some file-relative sensitivity and QC scripts intentionally write to repository-root `outputs/data_qc/` or `outputs/robustness/`.

Report the work in accordance with **TRIPOD+AI**: identify development and external cohorts, biopsy-defined outcomes, predictor handling, missing-data procedures, model specification/tuning, and all analysis populations. Report discrimination with point estimates and 95% patient-level bootstrap confidence intervals, not point estimates alone. Include calibration plots plus Brier score, calibration intercept, and calibration slope, and report decision-curve net benefit across clinically relevant thresholds. Preserve both UMM and MAINZ results and disclose any skipped or incompatible checkpoint.

## Citation

Bibliographic details are not yet available. Please cite the accompanying manuscript (authors, title, journal/preprint, year, and DOI to be added when available) and the repository version or commit hash used for reproduction.

## License

No repository license has yet been supplied. Consequently, no permission terms should be inferred; contact the repository maintainers before reuse or redistribution.
