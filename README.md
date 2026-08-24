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

The first command creates the canonical `nafl` environment; the second activates it. `requirements.txt` is an alternative/historical pip snapshot, not a numerically equivalent representation of the frozen Conda environment and not the canonical reproducibility installation. See `ENVIRONMENT_AUDIT.md` for the primary-workflow import comparison and the provenance of dependencies absent from the YAML.

## Controlled data placement

Access is controlled; consult the manuscript and associated heiDATA record for availability and the request procedure. Authorized researchers must place the three workbooks named in `data/README.md` directly under `data/`. Prerequisites are approved access and the exact original filenames; the main outputs are ignored cleaned tables in `data/preprocessed_no_mice_<split>/` and ten imputed datasets in `data/preprocessed_mice_fib_<split>/`. Never force-add raw or derived patient data to Git.

## Reproduction command chain (ordered)

The commands below are the result of auditing every entry point and checking the
training configuration recorded in `RETRAINING_PLAYBOOK.md`. Run them **in this
order from the repository root**. Entry points marked **checkpoint-only** load
existing fitted models (although they may regenerate deterministic preprocessing
files); entries marked **EXPENSIVE TRAINING** fit models and may take hours or
days. Configuration is by constants in the scripts, not command-line arguments.

1. **Create and activate the environment.**

   ```bash
   conda env create -f environment.yml
   conda activate nafl
   python --version
   ```

2. **Place the controlled data.** Obtain authorized copies and put the exact
   filenames below directly in `data/` (do not commit them):

   ```text
   data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx
   data/202403 Lap und Histo Daten von Ines Tuschner.xlsx
   data/20240813-FibrosisDB(302_Patients).xlsx
   ```

3. **Prepare cohorts, create the fixed split, and perform multiple
   imputation.** There is no separate current preprocessing CLI and no
   load-only split CLI. Every call to `prepare_data(...)` first rebuilds the
   eligible UMM cohort, makes the deterministic patient partitions (seed 42),
   prepares MAINZ as the external cohort, and writes ten MICE imputations (seeds
   0--9). Consequently this stage is performed at the start of the training
   command in step 4; there is no extra command to run here. Generated clean
   and imputed files are under
   `data/preprocessed_no_mice_<split>/` and
   `data/preprocessed_mice_fib_<split>/`. Re-running with the same raw inputs
   and settings recreates, rather than loads, the fixed split. Do **not** use
   `python -m src.preprocess` for this chain: its standalone block is a legacy
   NumPy-export path, not the current cohort-preparation contract.

4. **Run hyperparameter optimization and full-feature training.** Apply the
   authoritative temporary configuration from `RETRAINING_PLAYBOOK.md`: binary
   tasks `fibrosis`, `two_stage`, and `cirrhosis`, followed by `three_stage`;
   all eight reported model families; `SHAP_SELECTED=False`; and scaling only
   for `vi_bnn`. Record the temporary diff and restore it afterward.

   ```bash
   # EXPENSIVE TRAINING: optimization plus ten-member/imputation ensembles
   python -m src.run_all_train_experiments
   # QC only, after the prepared split has been written
   python -m src.check_split_stratification
   ```

   The runner's checked-in constants currently describe the later reduced RF/
   XGBoost experiment, so they **must be reviewed before this full sweep**.
   `hypertrain(...)` performs the model-family hyperparameter search and fitting;
   there is no separate supported optimization CLI.

5. **Evaluate binary models and compute patient-bootstrap intervals.** First
   make `TASKS`, `MODELS`, `SCALING_MODELS`, and `SHAP_SELECTED=False` in
   `src/run_all_tests.py` match the full-feature checkpoints. Keep
   `SELECT_PATIENTS=False` and `SMOTE=False`.

   ```bash
   # CHECKPOINT-ONLY evaluation (no model fitting)
   python -m src.run_all_tests
   # CHECKPOINT-ONLY: binary Tables 1--2 and 1,000-resample 95% intervals
   python -m src.recompute_tables
   ```

6. **Evaluate the three-stage endpoint and FIB-4/APRI comparators.** The first
   command pools the three-stage checkpoint predictions and calculates its
   bootstrap intervals; it also reads the FIB-4 and APRI fields written into
   imputation 0. The second performs paired patient-bootstrap model comparisons
   for the binary endpoints (including the leading model versus FIB-4).

   ```bash
   # CHECKPOINT-ONLY: three-stage models plus FIB-4/APRI
   python -m src.recompute_three_stage
   # CHECKPOINT-ONLY: paired binary comparisons with FIB-4/APRI
   python -m src.paired_model_comparison
   ```

   Binary FIB-4/APRI operating-point outputs are also produced automatically by
   preprocessing; there is no current standalone comparator command.

7. **Generate full-feature SHAP attribution.** This canonical publication SHAP
   entry point loads fitted tree and neural artifacts and never trains them.

   ```bash
   # CHECKPOINT-ONLY; computationally substantial attribution/reporting
   python -m src.shap_publication_figures
   ```

8. **Select reduced features using development data only.** This procedure
   explains each full-feature LightGBM ensemble member on its matching UMM
   *training* imputation. It must precede reduced training; it does not use UMM
   validation/test or MAINZ patients.

   ```bash
   # CHECKPOINT-ONLY LightGBM attribution; writes development top-three features
   python -m src.shap_feature_selection_development
   ```

9. **Retrain and evaluate the reduced-feature models.** Confirm the generated
   top-three feature record, then use the checked-in historical reduced setup:
   the three binary tasks, RF and XGBoost, `SHAP_SELECTED=True`, and no scaling.
   Make the testing runner match those choices before evaluating.

   ```bash
   # EXPENSIVE TRAINING: reduced-feature RF/XGBoost optimization and fitting
   python -m src.run_all_train_experiments
   # CHECKPOINT-ONLY after matching run_all_tests.py to the reduced setup
   python -m src.run_all_tests
   # CHECKPOINT-ONLY reduced-feature Table 5 with bootstrap intervals
   python -m src.recompute_reduced_tables
   ```

10. **Compute calibration, Brier metrics, and decision curves.** This single
    checkpoint reporter writes Brier score, calibration intercept/slope and
    bootstrap intervals, calibration plots, and decision-curve net benefit for
    the binary tasks.

    ```bash
    # CHECKPOINT-ONLY
    python -m src.clinical_utility_from_checkpoints
    ```

11. **Create manuscript tables.** The three recomputation commands above are
    the authoritative checkpoint-derived tables. If structured evaluation JSON
    was also generated, the legacy aggregation/export commands use `src`-relative
    paths and do **not** support the repository-root `python -m src.<module>`
    contract; invoke their actual supported form:

    ```bash
    (cd src && python aggregate_results.py)
    (cd src && python make_latex_tables.py)
    (cd src && python make_reduced_tables.py)
    ```

12. **Create publication figures and verify table/figure agreement.** These
    entry points rely on prepared cohorts and existing checkpoints only.

    ```bash
    # CHECKPOINT-ONLY
    python -m src.make_publication_figures
    # CHECKPOINT-ONLY; rerun after reduced checkpoints if reduced SHAP is needed
    python -m src.shap_publication_figures
    # CHECKPOINT-ONLY/report-only consistency check
    python -m src.check_table_figure_consistency
    ```

Most root-level `python -m` forms above are supported because those entry points
normalize their working directory to `src`. The exceptions are shown explicitly
with a root-launched `(cd src && ...)` subshell. Missing controlled workbooks or
a required checkpoint is a prerequisite failure, not a reason to substitute an
invocation form. Inspect every sweep failure log: several reporters deliberately
skip missing or incompatible checkpoints rather than aborting.

## Legacy single-run workflow

The lower-level entry points remain useful for one reviewed model/task. Run them
from the repository root with module execution.

1. **Configure and train one model/task.** In `train.py`, set `model_name`, `classification_type`, and `shap_selected`; preserve the scaling policy described below.

   ```bash
   (cd src && PYTHONPATH=.. python -m src.train)
   ```

   Prerequisites: active `nafl` environment, controlled workbooks in `data/`, and a reviewed configuration. There is no separate canonical preprocessing command: `train.py` calls `prepare_data(...)`, which performs preprocessing/imputation before training. Main outputs are generated datasets under `data/`, data-QC artifacts, checkpoints under `src/models/<model>/`, and training artifacts under `src/outputs/<model>/`.

2. **Evaluate the matching checkpoint once.** Set the same task, model, feature-selection, and scaling choices in `test.py`; retain `select_patients=False` and `smote=False` for external evaluation.

   ```bash
   (cd src && PYTHONPATH=.. python -m src.test)
   ```

   Prerequisites: matching checkpoints and generated data. Main outputs are held-out UMM and external MAINZ metrics/plots under `src/outputs/<model>/` (including `prospective/`) and external prevalence under `src/outputs/external/`. Evaluation preprocessing is combined in this entry point through `prepare_data(...)`.

3. **Recompute checkpoint-based manuscript tables.** Run each command only after all checkpoints required by that script exist.

   ```bash
   python -m src.recompute_tables
   python -m src.recompute_three_stage
   python -m src.recompute_reduced_tables
   ```

   The commands respectively produce binary tables (`src/outputs/tables/tables_recomputed.{csv,tex}`), the ordinal table (`table3_three_stage_recomputed.{csv,tex}`), and reduced-feature results (`table5_reduced_recomputed.{csv,tex}`). The reduced command additionally requires `src/outputs/shap_top_features.json` and matching `*_shap_selected` checkpoints.

4. **Generate publication figures and clinical-utility reporting.** These scripts consume generated datasets and compatible checkpoints directly; they do not consume the recomputed table CSVs.

   ```bash
   python -m src.make_publication_figures
   python -m src.shap_publication_figures
   python -m src.clinical_utility_from_checkpoints
   python -m src.check_table_figure_consistency
   ```

   Prerequisites: complete checkpoint/data contracts (and SHAP dependencies for SHAP output). Main outputs are publication PNG/PDF/CSV files in `src/outputs/figures/`, calibration metrics/plots and decision curves in `src/outputs/clinical_utility/`, and consistency results in `src/outputs/robustness/`. Missing checkpoints may be skipped by some reporting scripts; inspect warnings and outputs before reporting results.

5. **Optionally aggregate legacy JSON metrics.** This is distinct from checkpoint-recomputed tables.

   ```bash
   (cd src && python aggregate_results.py)
   ```

   Prerequisites: evaluation `*_metrics.json` files under `src/outputs/`. Main outputs are `src/outputs/results/all_metrics_long.csv` and `manuscript_tables.xlsx`.

## Simplified wrappers

The wrappers reduce repeated manual calls but remain configuration-driven:

```bash
python -m src.run_all_train_experiments
python -m src.run_all_tests
```

Both wrappers normalize their working directory themselves. Before either command, inspect and synchronize `TASKS`, model lists, `SCALING_MODELS`, and `SHAP_SELECTED`; ensure raw data are available, and ensure evaluation checkpoints exist. The training wrapper writes checkpoints plus a failure log in `src/outputs/results/`; the testing wrapper writes per-model metrics, prevalence files, QC snapshots, and a timestamped results/failure-log directory.

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
