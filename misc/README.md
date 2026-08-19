# Non-primary analyses and retained artifacts

The canonical workflow is preprocessing, training/external evaluation, and the
primary manuscript generators identified in [`python_module_audit.md`](python_module_audit.md).
The files below are retained for supplementary analysis, quality control, or
provenance; their presence does **not** make them part of that workflow.

In the tables, **core reference** means a tracked core module imports the file or
consumes its output. **Output effect** says whether running the file can train a
model or create/replace generated artifacts. Commands assume the repository root.

## Auxiliary and sensitivity analyses

| File | Purpose and reason non-primary | Core reference | Model/output effect | Useful invocation |
|---|---|---|---|---|
| `src/clinical_utility_from_checkpoints.py` | Supplementary calibration and decision-curve analysis, rather than the canonical discrimination evaluation. | No; standalone manuscript analysis. | Loads checkpoints; writes clinical-utility CSV/PNG/PDF files, but does not train. | `python -m src.clinical_utility_from_checkpoints` |
| `src/derive_shap_top_features.py` | Derives the reduced feature list from an existing SHAP export; it is a bridge into the reduced analysis, not primary model evaluation. | Yes; its JSON is consumed by reduced preprocessing/reporting. | No training; replaces `src/outputs/shap_top_features.json`. | `python -m src.derive_shap_top_features` |
| `src/make_publication_figures.py` | Produces supplementary cohort, PCA, characteristic, and ROC figures; final performance and SHAP have separate canonical generators. | No direct importer; manuscript outputs use it. | Loads tree checkpoints and writes figures/tables; no training. | `python -m src.make_publication_figures` |
| `src/make_reduced_tables.py` | Formats metrics from the reduced-feature experiment; it does not generate the canonical full-feature results. | No; standalone reporter. | No training; writes a reduced TeX table. | `cd src && python make_reduced_tables.py` |
| `src/ordinal_decision_rules.py` | Supplementary ordinal decision-rule comparison. | No; standalone manuscript analysis. | Loads checkpoints and writes a CSV; no training. | `python -m src.ordinal_decision_rules` |
| `src/paired_model_comparison.py` | Supplementary paired statistical comparison of checkpoint predictions. | No; standalone manuscript analysis. | Loads checkpoints and writes a CSV; no training. | `python -m src.paired_model_comparison` |
| `src/plot_svm_shap.py` | Formats an existing SVM SHAP export for the supplement. | No; consumes canonical/auxiliary SHAP output. | No training; writes CSV/TeX/PNG/PDF. | `python -m src.plot_svm_shap` |
| `src/shap_both_cohorts_figures.py` | Additional cross-cohort SHAP/rank analysis; `src/shap_publication_figures.py` is canonical. | Yes; its all-feature CSV can feed `derive_shap_top_features.py`. | Loads tree checkpoints and writes SHAP tables/figures; no training. | `python -m src.shap_both_cohorts_figures` |
| `src/missingness_sensitivity.py` | Implements one missingness sensitivity run, separate from primary evaluation. | Yes; imported by the all-task sensitivity runner. | May train a missing LightGBM sensitivity model and writes sensitivity CSVs. | `python -m src.missingness_sensitivity` |
| `src/run_missingness_sensitivity_all.py` | Runs missingness sensitivity across tasks, not the canonical experiment sweep. | No; standalone orchestrator. | May train missing sensitivity models and replaces sensitivity CSVs. | `python -m src.run_missingness_sensitivity_all` |
| `src/lab_window_robustness.py` | Compares laboratory collection windows. This is sensitivity/exploratory analysis, not evidence that a window was prospectively fixed. | Yes; the recommendation printer imports its ranking function. | **Trains LightGBM models**, moves/replaces model pickles, and writes the AUROC comparison CSV. | `python -m src.lab_window_robustness` |
| `src/print_recommended_lab_window.py` | Selects the best observed window by mean AUROC and cohort gap. **The performance-based laboratory-window recommendation is exploratory unless manuscript provenance independently establishes otherwise.** | No; standalone consumer of the robustness CSV. | No training and stdout only; selection is performance-based and must not redefine the canonical pipeline. | `python src/print_recommended_lab_window.py` (expects `outputs/robustness/lab_window_auroc_comparison.csv`) |

SMOTE is **not part of the final external evaluation pipeline**. External data
must retain their observed class distribution; `src/run_all_tests.py` fixes
SMOTE to false, and the evaluation/preprocessing guards reject SMOTE for the
external/prospective cohort. The disconnected helper `src/utils/smote.py` does
not change that policy.

## Diagnostics and quality control

These checks diagnose inputs or reproducibility. They are not result-producing
steps required to run the canonical workflow.

| File | Purpose and reason non-primary | Core reference | Model/output effect | Useful invocation |
|---|---|---|---|---|
| `src/check_split_stratification.py` | Reports split balance for QC only. | No. | No training; stdout only. | `python -m src.check_split_stratification` |
| `src/check_table_figure_consistency.py` | Reconciles checkpoint predictions with reference tables/figures. | No. | Loads checkpoints and writes a consistency CSV; no training. | `python -m src.check_table_figure_consistency` |
| `src/plot_prebiopsy_days.py` | Describes laboratory timing as QC/window context, not a primary outcome analysis. | No. | No training; writes summary CSV and PNG/PDF. | `python -m src.plot_prebiopsy_days` |
| `src/test_training_determinism.py` | Temporarily retrains LightGBM to test reproducibility. | No. | **Trains temporary models** but is designed not to persist them. | `python -m src.test_training_determinism` |
| `misc/diagnostics/README.md` | Placeholder explaining why no standalone diagnostic was moved. | Linked only by this index. | Documentation only. | — |

## Exploratory analyses

| File | Purpose and reason non-primary | Core reference | Model/output effect | Useful invocation |
|---|---|---|---|---|
| `misc/exploratory/variance_test_datasets.py` | Archived ad-hoc, interactive UMM/MAINZ cirrhosis variance comparison. | No; only the compatibility wrapper delegates to it. | No training; displays plots/stdout and does not intentionally persist output. | `cd src && python variance_test_datasets.py` |
| `src/variance_test_datasets.py` | Preserves the historical command for the moved implementation; not a pipeline stage. | No. | Same behavior as the archived implementation. | `cd src && python variance_test_datasets.py` |
| `src/shap_feature_selection_development.py` | Development-cohort feature-selection record; it is not the canonical manuscript SHAP analysis. | No. | Loads LightGBM checkpoints and writes development SHAP CSVs; no training. | `python -m src.shap_feature_selection_development` |
| `src/stats.py` | Configured ad-hoc statistical analyses rather than a stable pipeline report. | No. | No model training; may write configured statistical outputs. | `cd src && python stats.py` |
| `src/utils/descriptive_stats.py` | Interactive cross-cohort biomarker distribution review. | No. | No training; displays plots/stdout (the retained PNGs below are historical exports). | `python -m src.utils.descriptive_stats` |
| `misc/exploratory/README.md` | Short location note for the moved variance script. | Linked only by this index. | Documentation only. | — |

## Historical or legacy scripts

Run historical generators only in an isolated output tree: some preserve fixed
names and can overwrite current artifacts.

| File | Purpose and reason non-primary | Core reference | Model/output effect | Useful invocation |
|---|---|---|---|---|
| `misc/legacy/apply_patch_comparators.py` | One-off migration that added FIB-4/APRI code now integrated in validation tools. | No; compatibility wrapper only. | Does not train; **can rewrite source files** and should not normally be run. | Historical reproduction: `cd src && python apply_patch_comparators.py` |
| `src/apply_patch_comparators.py` | Compatibility wrapper for that moved migration. | No. | Delegates the same source-rewriting behavior. | `cd src && python apply_patch_comparators.py` |
| `misc/legacy/apply_patch_fib4.py` | Earlier, superseded FIB-4-only source migration. | No; compatibility wrapper only. | Does not train; **can rewrite source files** and should not normally be run. | Historical reproduction: `cd src && python apply_patch_fib4.py` |
| `src/apply_patch_fib4.py` | Compatibility wrapper for that moved migration. | No. | Delegates the same source-rewriting behavior. | `cd src && python apply_patch_fib4.py` |
| `misc/legacy/shap_publication_figures_v1.py` | Pickle-only predecessor retained for manuscript provenance; it lacks current native neural loading. | No; provenance documentation references it. | Loads checkpoints; no training, but **can overwrite canonical SHAP outputs**. | Historical reproduction only: `cd src && python ../misc/legacy/shap_publication_figures_v1.py` |
| `src/finetuning.py` | Retained older fine-tuning experiment tied to established model paths, not the canonical training sweep. | Core dispatch does not import it. | **Trains/fine-tunes models** and writes checkpoints/evaluation artifacts. | `cd src && python finetuning.py` |
| `src/make_umm_shap_figures.py` | Superseded UMM-only SHAP generator retained because it uses established paths. | No. | Loads tree checkpoints and can replace SHAP artifacts; no training. | `python -m src.make_umm_shap_figures` |
| `src/models/mcmc_bnn.py` | Experimental/legacy model family retained for dispatch/checkpoint compatibility; its evaluator has a known parameter-name mismatch and is absent from the ordinary test sweep. | Yes; training dispatch can import it. | **May train models** and write checkpoints, metrics, and SHAP output. | No supported standalone command; invoke only through a deliberately repaired/configured training call. |
| `src/utils/convert_amainz_dat.py` | Manual legacy workbook conversion with import-time behavior, not routine preprocessing. | No. | No training; writes a converted input workbook. | `cd src/utils && python convert_amainz_dat.py` |
| `src/utils/smote.py` | Retained disconnected resampling helper for import compatibility; it is not the external evaluation path. | No tracked importer. | No model training; returns altered in-memory training arrays if explicitly imported. | Library only; no standalone command. |
| `src/utils/utils.py` | Retained duplicate legacy vote/index helpers for uncertain external imports. | No tracked importer. | No training or persistent output. | Library only; no standalone command. |
| `misc/legacy/README.md` | Short warning and provenance note for moved legacy implementations. | Linked only by this index. | Documentation only. | — |

## Static images already present in `misc/`

These are retained snapshots, not canonical generated artifacts. Neither is
referenced by core code, and neither can train models or alter outputs merely by
being present.

| File | Purpose and reason non-primary | Core reference | Model/output effect | Useful invocation |
|---|---|---|---|---|
| `misc/densities.png` | Historical density-plot export from exploratory biomarker distribution review; provenance is not sufficient to treat it as a manuscript result. | No. | Static image only. | View directly; regenerate interactively with `python -m src.utils.descriptive_stats` if the private inputs are available. |
| `misc/histograms.png` | Historical histogram export from the same exploratory review; provenance is not sufficient to treat it as a manuscript result. | No. | Static image only. | View directly; regenerate interactively with `python -m src.utils.descriptive_stats` if the private inputs are available. |

## Documentation retained in `misc/`

`misc/python_module_audit.md` is the detailed execution/data-flow inventory and
relocation record that supports this concise index. This README and the three
nested README files are navigation/provenance documentation: core code does not
read them, and they cannot train models or change generated output.

For the canonical manuscript SHAP entry point and the collision risk among
historical variants, see [`../SHAP_PROVENANCE.md`](../SHAP_PROVENANCE.md).
