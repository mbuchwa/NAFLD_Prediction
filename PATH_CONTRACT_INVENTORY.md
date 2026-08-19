# Path-contract inventory

This inventory records the filenames that connect training, evaluation, and
publication scripts.  They are compatibility contracts, not suggestions for a
new layout: **do not rename any directory, model identifier, checkpoint,
result, or output listed here without updating every producer and consumer in
the same change**.

## Working-directory rule

Most legacy entry points say to run from `src/`.  Consequently their relative
`models/...`, `outputs/...`, and `../data/...` paths mean `src/models/...`,
`src/outputs/...`, and `data/...`, respectively.  A smaller group resolves the
repository root from `__file__`; those exceptions are called out below.  Using
`python src/<script>.py` from the repository root can otherwise create a second,
incompatible `models/` or `outputs/` tree.

## Stable identifiers

| Kind | Contract values |
|---|---|
| Tasks used in filenames | `fibrosis`, `two_stage`, `cirrhosis`, `three_stage` |
| Storage model IDs | `svm`, `rf`, `xgb`, `light_gbm`, `ffn`, `tab_transformer`, `vi_bnn`, `gandalf`, `mcmc_bnn` |
| Display-only model names | `SVM`, `Random Forest`, `XGBoost`, `LightGBM`, `MLP`, `TabTransformer`, `VI-BNN`, `GANDALF` |
| Reduced-model suffix | `<model>_shap_selected` |
| Fine-tuned-model suffix | `<model>_finetuned` |
| Cohort/split IDs | `train`, `val`, `test` (UMM), `prospective` (MAINZ); reporting labels are `UMM` and `MAINZ` |
| Lab-window ID | `light_gbm_window/pre7_post0` (the prespecified two-stage LightGBM fallback) |

`light_gbm` is the contract spelling even though the implementation module is
`src/models/light_gmb.py`.  Neither spelling should be normalized: changing it
would disconnect existing checkpoints.  Likewise, cohort labels and split
directory names deliberately differ.

## Operation audit

The source search covered `pickle.load`, `torch.load`, `joblib.load`, `open`,
`to_csv`, and `savefig` (including commented legacy operations, which still
document historical contracts).  Active checkpoint deserialization uses
`pickle.load` for tree/SVM ensembles and `torch.load` for neural state; no
`joblib.load` call exists in `src/`.  Generic `open` calls additionally handle
JSON, parameter text, prediction text, data-QC summaries, and legacy CSV
records.  `to_csv` and `savefig` provide the tabular and figure sinks inventoried
below.  Model writers also use `pickle.dump`, `torch.save`, and
`TabularModel.save_model`, so those producers are included even though they were
outside the minimum search terms.

## Checkpoint contracts

### `src/models/<model>/`

All paths in this section are interpreted from `src/`.

| Producer/layout | Primary artifact | Direct consumers |
|---|---|---|
| `src/models/{svm,rf,xgb,light_gbm}/` | `model_<task>.pickle`, a pickled ensemble | `recompute_tables.py`, `recompute_three_stage.py` (XGBoost/LightGBM), `recompute_reduced_tables.py`, `paired_model_comparison.py`, `ordinal_decision_rules.py`, `check_table_figure_consistency.py`, `clinical_utility_from_checkpoints.py`, and the publication/SHAP figure scripts |
| `src/models/{ffn,vi_bnn,tab_transformer}/` | `model_<task>_<index>.pth` plus `model_params_<task>_<index>.txt`; TabTransformer also writes `<task>_df_cols.txt` | `neural_loaders.py`, then the recomputation and publication scripts through `load_any_ensemble`; canonical `shap_publication_figures.py` also loads state dictionaries directly |
| `src/models/gandalf/` | `model_<task>_<index>.pth` directories plus `df_cols.txt` and `model_params_<task>_<index>.txt` | `neural_loaders.py`, then checkpoint-based tables/figures |
| `src/models/mcmc_bnn/` | `model_<task>_<index>.pth`, `bnn_posterior_samples_<task>_<index>.pth`, and parameter text | the MCMC-BNN evaluation routines in `models/mcmc_bnn.py` |
| `src/models/<model>_shap_selected/` | the same pickle/per-member convention as the base model | `recompute_reduced_tables.py`; training is driven with `shap_selected=True` |
| `src/models/<model>_finetuned/` | model-family-specific fine-tuned artifacts | the corresponding family finetuning/evaluation function |
| `src/models/light_gbm_window/pre7_post0/` | `model_two_stage.pickle` | `shap_feature_selection_development.py`, `clinical_utility_from_checkpoints.py`, and canonical `shap_publication_figures.py` when selecting the manuscript checkpoint |

The generic binary lookup is
`src/models/<storage-id>/model_<task>.pickle`.  Per-member neural loaders are a
fallback, not an alternate name for the pickle.  Ensemble indices are part of
the contract because member *i* is evaluated on imputation *i*.

### `src/checkpoints/`

No current source operation reads or writes `src/checkpoints/`.  It is a
reserved/legacy location, **not** a fallback searched by the checkpoint-based
reporting scripts.  Placing a checkpoint there will not make it discoverable.

### `src/saved_model/` and `src/saved_models/`

Neither directory is read or written by current source code.  They are not
aliases for `src/models/`; publication and recomputation scripts do not search
them.

## Data contracts under `data/`

`preprocess.py` is the primary producer.  From `src/`, its `../data/...` paths
resolve to these repository paths:

| Artifact | Producer | Consumers |
|---|---|---|
| `data/preprocessed_no_mice_<split>/<split>_<task>.csv` | `preprocess.py` | preprocessing/merge workflows |
| `data/preprocessed_mice_fib_<split>/<split>_<task>_<imputation>.csv` | `preprocess.py` | `prepare_data`, `recompute_tables.py` (FIB-4/APRI), `recompute_three_stage.py`, `ordinal_decision_rules.py`, `paired_model_comparison.py`, `stats.py`, and dataset diagnostics |
| `data/preprocessed_no_mice_data.csv` and `data/preprocessed_mice_fib_data.csv` | `preprocess.py` merge helpers | legacy exploratory/reporting workflows |
| `data/{xs_train,xs_test,ys_train,ys_test}.npy` and `data/df_cols.pickle` | `preprocess.py` legacy export | legacy model utilities |
| `data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx` and `data/202403 Lap und Histo Daten von Ines Tuschner.xlsx` | external UMM inputs | `preprocess.py`, split checks, statistics, and cohort/data-QC figures |
| `data/20240813-FibrosisDB(302_Patients).xlsx` | external MAINZ input | `preprocess.py` |

The literal column/label contract includes `Micro`, `Fib4` or `Fib4 Stages`,
and `APRI` or `APRI Stages`.  The imputed filename suffix (`_0` through the
available imputations) must remain aligned with ensemble-member numbering.

## Artifact-writing contracts

### `src/outputs/`

This is the main runtime artifact root when commands are run from `src/`.

| Primary output | Producer | Later consumer(s) |
|---|---|---|
| `outputs/<model>/<model>_<task>_metrics.json` (including `prospective/`) | evaluation in `utils/validation_tools.py` | `aggregate_results.py`, `make_latex_tables.py`, `make_reduced_tables.py` |
| `outputs/<model>/<model>_<task>_ensemble_preds.txt` and `model_<task>.csv` (and `prospective/`) | `utils/validation_tools.py` | `stats.py` and manual audit workflows |
| `outputs/results/all_metrics_long.csv` and `outputs/results/manuscript_tables.xlsx` | `aggregate_results.py` | downstream/manual manuscript assembly |
| `outputs/results/tables_revised.tex` | `make_latex_tables.py` | publication manuscript |
| `outputs/results/tables_reduced.tex` | `make_reduced_tables.py` | publication manuscript (legacy JSON-based reduced table) |
| `outputs/results/<run_tag>/...` | `run_all_tests.py`, `run_all_train_experiments.py` | `aggregate_results.py` scans nested metric JSON files; run logs/QC snapshots are audit records |
| `outputs/tables/tables_recomputed.csv`, `.tex`, and `tables_missing_rows.csv` | `recompute_tables.py` | canonical binary publication tables and consistency/manual checks |
| `outputs/tables/table3_three_stage_recomputed.csv` and `.tex` | `recompute_three_stage.py` | canonical ordinal publication table |
| `outputs/tables/table5_reduced_recomputed.csv` and `.tex` | `recompute_reduced_tables.py` | canonical checkpoint-based reduced publication table |
| `outputs/tables/ordinal_decision_rules.csv`/`.tex` | `ordinal_decision_rules.py` | publication supplement/manual assembly |
| `outputs/tables/paired_model_comparison.csv`/`.tex` | `paired_model_comparison.py` | publication supplement/manual assembly |
| `outputs/shap_top_features.json` | `derive_shap_top_features.py` from `outputs/figures/shap_all_features.csv` | `preprocess.py` when `shap_selected=True`; `recompute_reduced_tables.py` for feature labels and checkpoint compatibility |
| `outputs/figures/shap_feature_selection_development_all.csv` and `_top3.csv` | `shap_feature_selection_development.py` | development-only selection audit/manual review; these do **not** replace `outputs/shap_top_features.json` automatically |
| `outputs/figures/shap_all_features.csv`, `shap_top5.csv`, `shap_rank_agreement.csv`, and `shap_values_<model>_<task>_<cohort>.csv` | `shap_both_cohorts_figures.py` and the canonical `shap_publication_figures.py` and historical variants | `derive_shap_top_features.py`, `plot_svm_shap.py`, and publication/manual analysis |
| `outputs/figures/*.{png,pdf}` plus figure CSVs | `make_publication_figures.py`, `make_umm_shap_figures.py`, `shap_both_cohorts_figures.py`, `shap_publication_figures.py` and historical variants, cohort/stage/SVM plotters | publication manuscript |
| `outputs/clinical_utility/predictions_<task>_<cohort>.csv`, `decision_curve_<task>_<cohort>.csv`, calibration metric CSVs, and calibration/decision-curve PNG/PDF files | `clinical_utility_from_checkpoints.py` | clinical-utility publication figures/tables and manual audit |
| `outputs/data_qc/*` | `preprocess.py` and data-QC plotters | `make_publication_figures.py`, missingness scripts, and `run_all_tests.py` snapshots |
| `outputs/robustness/*` | robustness, missingness, and consistency scripts | `print_recommended_lab_window.py` and publication/manual reporting |
| `outputs/fib4/fib4_<task>.csv` and performance text | `preprocess.py` | legacy comparator reporting |

Several plotting helpers also write fixed names under
`outputs/<model>/[prospective/]`; their stems include the storage model ID and
task.  `savefig` extensions (`.png`, `.pdf`, and occasionally other requested
formats) are publication contracts and must not be silently consolidated.

### `src/results/`

No source operation currently reads or writes `src/results/`.  Result artifacts
live under `src/outputs/results/`; `src/results/` is not an alias and is not
searched by the table generators.

### Repository-root `outputs/`

`missingness_sensitivity.py`, `run_missingness_sensitivity_all.py`, and some
robustness helpers resolve the repository root via `__file__` and therefore
write `outputs/data_qc` or `outputs/robustness`, not `src/outputs/...`.  This
coexisting root is intentional in current code and must not be merged or renamed
without an explicit migration.

## Primary consumer traces

1. **Binary checkpoint tables:** `src/models/<model>/model_<task>.pickle` (or
   neural per-member artifacts) + imputed `data/` partitions ->
   `recompute_tables.py` -> `src/outputs/tables/tables_recomputed.{csv,tex}`.
2. **Ordinal table:** XGBoost/LightGBM `model_three_stage.pickle` +
   `data/preprocessed_mice_fib_{test,prospective}/...` ->
   `recompute_three_stage.py` ->
   `src/outputs/tables/table3_three_stage_recomputed.{csv,tex}`.
3. **Publication SHAP:** the same model pickles + data from `prepare_data` ->
   `shap_publication_figures.py` (canonical; historical alternatives are documented in `SHAP_PROVENANCE.md`) -> figure
   files and SHAP CSVs under `src/outputs/figures/`; `plot_svm_shap.py` consumes
   `shap_all_features.csv`, while `derive_shap_top_features.py` converts it to
   `src/outputs/shap_top_features.json`.
4. **Development-only selection:** LightGBM pickles (with the fixed
   `light_gbm_window/pre7_post0` fallback for `two_stage`) + UMM training
   imputations -> `shap_feature_selection_development.py` -> the two
   `shap_feature_selection_development_*.csv` audit files.
5. **Reduced table:** `outputs/shap_top_features.json` -> `preprocess.py` reduced
   matrices; base and `<model>_shap_selected` checkpoints ->
   `recompute_reduced_tables.py` ->
   `src/outputs/tables/table5_reduced_recomputed.{csv,tex}`.
6. **Clinical utility:** manuscript-selected/base pickles + validation, UMM,
   and MAINZ matrices -> `clinical_utility_from_checkpoints.py` -> predictions,
   decision curves, calibration metrics, and publication plots under
   `src/outputs/clinical_utility/`.
7. **Legacy publication tables:** per-run `*_metrics.json` ->
   `aggregate_results.py`, `make_latex_tables.py`, and
   `make_reduced_tables.py` -> `src/outputs/results/`.  These are distinct from
   the checkpoint-recomputed tables under `src/outputs/tables/`.
8. **Publication figures:** `make_publication_figures.py` consumes data-QC
   artifacts and checkpoints directly; SHAP-specific figure generators also
   consume checkpoints directly.  They do not consume the recomputed table CSVs,
   so table/figure consistency depends on using the same checkpoint tree.

## Inventory maintenance check

Before merging cleanup changes, inspect the final diff for path-bearing lines:

```bash
git diff -- src | rg '(models/|checkpoints/|saved_models?/|outputs/|results/|data/|model_)'
```

Every changed contract in that output must either preserve the literal path and
identifier or be reflected in this inventory with all producers and consumers
audited.  Documentation-only changes should produce no path-bearing `src/`
diff, which is the safest possible result.
