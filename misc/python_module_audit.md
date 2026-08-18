# Python module execution and data-flow audit

This maintainer reference covers **all 57 tracked Python files under `src/`**. It was built with AST import inspection and repository-wide import and string-reference searches, rather than names alone. It records paths and contracts only—no row values, private-data contents, or patient identifiers.

## Reading the inventory

The pipeline's effective working directory is normally `src`: `../data` means the repository's private/untracked `data/`, while `models/` and `outputs/` mean `src/models/` and `src/outputs/`. “Prepared cohorts” means the two dated UMM and MAINZ workbooks named in `preprocess.py`, or split/imputed CSVs and metadata generated from them. `script` means `cd src && python FILE`; `-m` means `python -m src.MODULE` at repository root. Entry points which force `chdir(src)` accept either. Importers below are tracked direct Python importers; “CLI” means none. Dependencies omit the standard library. Modes are **train**, **evaluate** (load checkpoints), and **report**.

## Entry points and analysis modules

| File | Class | Importers / dependencies | Invocation; inputs | Outputs / mode / consumers |
|---|---|---|---|---|
| `aggregate_results.py` | primary | CLI; numpy, pandas | script; metrics JSON under `outputs/`; no checkpoint | results long CSV and workbook; **report**; manuscript tables |
| `apply_patch_comparators.py` | legacy | CLI; AST utilities | script only; `utils/validation_tools.py` | edits target plus `.bak`; **code patch only**; test sweep. Already integrated—do not run normally. |
| `apply_patch_fib4.py` | legacy | CLI; AST utilities | script only; same target | edits target plus `.bak`; **code patch only**; superseded by comparator/current code |
| `check_split_stratification.py` | QC | CLI; preprocess, numpy/pandas/sklearn | `-m`/script, forces `src`; raw workbooks | stdout; **report**; split diagnosis |
| `check_table_figure_consistency.py` | QC | CLI; preprocess, neural_loaders, sklearn | `-m`/script, forces `src`; prepared cohorts, reference table/figure, family checkpoints | robustness consistency CSV; **evaluate/report**; checkpoint reconciliation |
| `clinical_utility_from_checkpoints.py` | auxiliary | CLI; preprocess, neural_loaders, scipy/sklearn/matplotlib/torch | `-m`/script, forces `src`; prepared cohorts; tree/neural checkpoints and prescribed two-stage fallback | calibration, predictions and decision-curve CSV/PNG/PDF under `outputs/clinical_utility`; **evaluate/report**; manuscript |
| `cohort_figures.py` | shared implementation | preprocess; numpy/pandas/matplotlib | module, caller cwd; raw frames/prepared arrays | cohort/label/stage CSV and figures; **report**; preprocess/publication QC |
| `derive_shap_top_features.py` | auxiliary | CLI; preprocess, translation map, pandas | `-m`/script, forces `src`; `outputs/figures/shap_all_features.csv` | `outputs/shap_top_features.json`; **report**; reduced preprocessing/tables |
| `finetuning.py` | legacy | CLI; preprocess, model families | script; prepared cohorts and base checkpoints where loaders require them | finetuned checkpoints/evaluation artifacts; **train/evaluate**; testing |
| `lab_window_robustness.py` | sensitivity | print-window; preprocess, LightGBM | `-m`/script, forces `src`; prepared window cohorts and default/window pickles | window pickles and robustness AUROC CSV; **train missing/evaluate**; window printer/manuscript |
| `make_latex_tables.py` | primary | CLI; numpy | script; metrics JSON; no checkpoint | `outputs/results/tables_revised.tex`; **report**; manuscript |
| `make_publication_figures.py` | auxiliary | CLI; preprocess, scipy/sklearn/matplotlib | `-m`/script, forces `src`; prepared/raw snapshots, split CSVs, optional attrition JSON, tree pickles | cohort/stage/PCA/characteristic/ROC artifacts; **evaluate/report**; manuscript |
| `make_reduced_tables.py` | auxiliary | CLI; numpy | script; reduced-run metrics JSON | reduced TeX table; **report**; manuscript |
| `make_umm_shap_figures.py` | legacy | CLI; preprocess, translation map, SHAP/sklearn | `-m`/script, forces `src`; UMM prepared cohort, tree pickles | SHAP CSV/table/figures; **evaluate/report**; superseded publication variants |
| `missingness_sensitivity.py` | sensitivity | all-task sensitivity runner; preprocess, LightGBM | module/`-m`, direct forces `src`; prepared cohorts, missingness profile, LightGBM checkpoint/training data | sensitivity/ranked CSVs; **train/evaluate**; runner/manuscript |
| `neural_loaders.py` | shared implementation | consistency, clinical, ordinal/paired, recompute tools; networks/helpers, torch, pytorch-tabular | module; prepared feature metadata, params and neural `.pth` artifacts | in-memory adapters; **evaluate library**; listed consumers |
| `ordinal_decision_rules.py` | auxiliary | CLI; preprocess, neural_loaders, sklearn | `-m`/script, forces `src`; prepared/cached three-stage data, family checkpoints | ordinal decision-rules CSV; **evaluate/report**; manuscript |
| `paired_model_comparison.py` | auxiliary | CLI; preprocess, neural_loaders, sklearn | `-m`/script, forces `src`; prepared/cached cohorts, family checkpoints | paired-comparison CSV; **evaluate/report**; manuscript |
| `plot_prebiopsy_days.py` | QC | CLI; pandas/numpy/matplotlib | `-m`/script, forces `src`; raw UMM workbook | QC summary CSV and PNG/PDF; **report**; window justification |
| `plot_stage_distribution.py` | primary | CLI; pandas/numpy/matplotlib | `-m`/script, file-relative roots; emitted stage inputs | combined stage CSV/PNG/PDF; **report**; manuscript |
| `plot_svm_shap.py` | auxiliary | CLI; pandas/numpy/matplotlib | `-m`/script; SHAP all-features CSV | SVM SHAP CSV/TeX/PNG/PDF; **report**; supplement |
| `preprocess.py` | primary | 26 tracked modules; select_test_datasets, plots, cohort_figures, sklearn/pandas | module/direct script from `src`; raw workbooks, optional SHAP-top JSON/caches; no checkpoint | imputed split trees, metadata and QC/cohort artifacts; **data generation**; nearly all trainers/reporters |
| `print_recommended_lab_window.py` | sensitivity | CLI; lab-window module, pandas | repository-root script; literal root `outputs/robustness/...csv` | stdout; **report**. Path disagrees with producer's normal `src/outputs`; copy or point deliberately. |
| `recompute_reduced_tables.py` | primary | CLI; preprocess, neural_loaders, translation map | `-m`/script, forces `src`; full/reduced data, SHAP-top JSON, full/reduced checkpoints | Table 5 CSV/TeX; **evaluate/report**; manuscript |
| `recompute_tables.py` | primary | CLI; preprocess, neural_loaders | `-m`/script, forces `src`; prepared/cached cohorts and family checkpoints | Tables 1–2 CSV/TeX, missing rows CSV; **evaluate/report**; manuscript |
| `recompute_three_stage.py` | primary | CLI; preprocess, neural_loaders, scipy/sklearn | `-m`/script, forces `src`; three-stage cohorts/checkpoints | Table 3 CSV/TeX; **evaluate/report**; manuscript |
| `run_all_tests.py` | primary | CLI; preprocess, test, translation map | `-m`/script, forces `src`; prepared cohorts and configured checkpoints | family metrics/predictions/plots, prevalence and failure logs; **evaluate**; aggregators/tables |
| `run_all_train_experiments.py` | primary | CLI; preprocess, train | `-m`/script, forces `src`; prepared cohorts; no required checkpoint | checkpoints/params and failure logs; **train**; test sweep/reporters |
| `run_missingness_sensitivity_all.py` | sensitivity | CLI; preprocess, missingness module | `-m`/script, forces `src`; cohorts/profile and LightGBM artifacts | all-task sensitivity CSV; **train/evaluate**; manuscript |
| `select_test_datasets.py` | shared implementation | preprocess; numpy/pandas/scipy/sklearn/matplotlib | module only; caller arrays; no checkpoint | selected arrays/indices and optional diagnostics; **data selection**; preprocess |
| `shap_both_cohorts_figures.py` | auxiliary | CLI; preprocess, translation map, SHAP/scipy/sklearn | `-m`/script, forces `src`; both cohorts and tree pickles | SHAP/rank/top-five CSV/TeX/figures; **evaluate/report**; derive-top-features/manuscript |
| `shap_feature_selection_development.py` | exploratory | CLI; preprocess, translation map, SHAP | `-m`/script, forces `src`; development cohorts, LightGBM pickles/prescribed fallback | development all/top3 CSVs; **evaluate/report**; development record only |
| `shap_publication_figures.py` | auxiliary | CLI; preprocess, translation map, SHAP/scipy/sklearn | `-m`/script, forces `src`; cohorts and **pickle ensembles only** | publication SHAP/ROC/confusion artifacts; **evaluate/report**; derive-top-features/manuscript; neural families are skipped |
| `shap_publication_figures_v2.py` | primary | CLI; preprocess, validation/networks/helpers, SHAP/torch | `-m`/script, forces `src`; scaled/unscaled cohorts, tree pickles/fallback and native neural artifacts | same artifact names as v1; **evaluate/report**; current broad loader. Archive output before variant comparisons to avoid overwrite. |
| `stats.py` | exploratory | CLI; preprocess/helpers, scipy/statsmodels/sklearn | script (`src`, bare imports); raw workbooks, imputed test CSVs/text results | stdout/configured outputs; **report**; ad-hoc analysis |
| `test.py` | primary | run_all_tests; preprocess, translation map, selected models | module/direct script from `src`; cohorts/checkpoints | evaluation metrics/predictions/plots and prevalence CSV; **evaluate**; sweep/aggregators |
| `test_training_determinism.py` | QC | CLI; preprocess, LightGBM/sklearn | `-m`/script, forces `src`; prepared data; no checkpoint | stdout, explicitly no persistent write; **temporary train/QC**; retraining decision |
| `train.py` | primary | train-all; preprocess, all model modules | module/direct script from `src`; prepared cohorts | family checkpoints/params/evaluation outputs; **train**; testing/reporters |
| `variance_test_datasets.py` | exploratory | CLI; scipy/statsmodels/sklearn/pandas | **import-time script** from `src`; exact cirrhosis imputation-0 test/prospective CSVs | interactive plots/stdout; **report**; no tracked consumer |

## `src/models/`: shared implementations

These are module APIs (no supported standalone invocation), expect caller cwd `src`, and consume arrays from preprocessing. `train`, `test`, finetuning and checkpoint reporters consume their artifacts.

| File | Importers / dependencies | Required/written artifacts; mode |
|---|---|---|
| `models/ffn.py` | train/test/finetuning; helpers, validation, networks, torch/Lightning | per-task/member `.pth` plus params and outputs; **train/evaluate** |
| `models/gandalf.py` | train/test/finetuning; helpers/validation, pytorch-tabular | pytorch-tabular per-member artifacts, `df_cols.txt`, params; **train/evaluate**; not plain state dict format |
| `models/light_gmb.py` | train/test/finetuning and window/missingness/determinism; LightGBM/sklearn | task pickle, optional finetuned/window pickles; **train/evaluate** |
| `models/mcmc_bnn.py` | train only; helpers, Pyro/torch/SHAP | model state **and separate posterior-sample** `.pth`, params, MCMC metrics/SHAP; **train/evaluate, legacy/experimental**. Its evaluator reads `model_params__...` (double underscore) although training writes `model_params_...`; repair before relying on evaluation. It is absent from `test.py` and `neural_loaders`. |
| `models/rf.py` | train/test/finetuning; helpers/validation/sklearn | task and finetuned pickles; **train/evaluate** |
| `models/svm.py` | train/test/finetuning; helpers/validation/sklearn | task pickle; **train/evaluate** |
| `models/tab_transformer.py` | train/test/finetuning; helpers/validation/networks, torch | member `.pth`, column/parameter metadata; **train/evaluate** |
| `models/vi_bnn.py` | train/test/finetuning; helpers/validation/networks, torch | member `.pth` and params; **train/evaluate** |
| `models/xgb.py` | train/test/finetuning; helpers/validation, XGBoost/te2rules | task and finetuned pickles plus rule/evaluation output; **train/evaluate** |

## `src/utils/`

| File | Class | Importers / dependencies; invocation and inputs | Outputs / mode / consumers |
|---|---|---|---|
| `convert_amainz_dat.py` | legacy | CLI; pandas; **import-time script from `src/utils`**, raw MAINZ workbook | converted workbook in `data`; **data conversion**; manual preprocessing input |
| `descriptive_stats.py` | exploratory | CLI; preprocess/map, scipy/seaborn; module/script from `src`, prepared cohorts | density/histogram images; **report**; exploratory review |
| `ger_eng_dict.py` | shared implementation | SHAP/derive/recompute/test/descriptive importers; module, no files | translation mapping; **library** |
| `helper_functions.py` | shared implementation | train/test/models/loaders/stats/validation; plots/sklearn; module, arrays and optional legacy singular `model.pickle` | predictions/metrics/delegated plots; **evaluate helper** |
| `networks.py` | shared implementation | neural models/loaders/SHAP-v2/validation; torch/Lightning | network objects/state; **train/evaluate library** |
| `plots.py` | shared implementation | preprocess/helpers/utils; seaborn/matplotlib/torch/sklearn | plots in `outputs/<name>` or `plots/<name>`; **report library** |
| `smote.py` | shared implementation | **no tracked importer**; imbalanced-learn/numpy; module, caller X/y lists | resampled arrays/stdout; **data transform**; currently disconnected even though preprocessing has an SMOTE option |
| `utils.py` | legacy | no tracked importer; plots; module, in-memory predictions | duplicate vote/index helpers; **library** |
| `validation_tools.py` | shared implementation | conventional models and SHAP-v2; helpers/networks, SHAP/pytorch-tabular/sklearn | metrics JSON/text/CSV, predictions, calibration/decision/subgroup CSV and plots; **evaluate/report library**; sweep, aggregators, table makers |

## Operational cautions

1. Both `apply_patch_*` scripts are migration aids, not experiments; current comparator code is already integrated.
2. `select_test_datasets.py` and `utils/smote.py` are libraries, not dataset scripts. Only the former is connected.
3. `variance_test_datasets.py` and `utils/convert_amainz_dat.py` execute on import.
4. SHAP variants collide on output names. Prefer v2 for native tree and neural formats, and isolate outputs when comparing variants.
5. MCMC-BNN requires paired posterior files and has a parameter-filename mismatch; it is not part of the ordinary evaluation sweep.
