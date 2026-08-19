# SHAP manuscript figure provenance and entry points

## Maintainer/artifact-history conclusion

The canonical manuscript SHAP and associated ROC/confusion-matrix generator is
[`src/shap_publication_figures.py`](src/shap_publication_figures.py). Run it from
`src/`:

```bash
cd src
python shap_publication_figures.py
```

This conclusion is based on the maintainer's committed artifact history, not on
a numerical comparison of SHAP calculations:

1. Commit `efd7072` (2026-08-06, authored by maintainer `mbuchwa`) introduced
   the UMM-only generator, the both-cohort generator, and the original
   publication generator together.
2. Commits `3fc5eb3` and `d7eced4` subsequently changed the publication
   generator's manuscript task/model mappings and combined publication panels.
3. Commit `9a62435` (2026-08-18, also authored by `mbuchwa`) added the last
   publication implementation as `shap_publication_figures_v2.py`. Its own
   provenance notes that Figures 3--6 follow the table-consistent evaluation
   path, and it uniquely supports the selected native neural checkpoints and
   the prespecified two-stage LightGBM fallback. The same commit added the
   development-only feature-selection record used for the final reduced models.
4. No generated manuscript PDF/TeX figure bundle is tracked in this repository,
   so there is no stronger in-repository file timestamp to contradict that
   maintainer history. The later implementation is therefore retained at the
   stable canonical path above. Its former `_v2` filename is removed to prevent
   two scripts from overwriting the same output names.

The earlier implementation formerly at the canonical path is archived, without
algorithmic changes, as
[`misc/legacy/shap_publication_figures_v1.py`](misc/legacy/shap_publication_figures_v1.py).
Repository-wide Python import and string-reference inspection found no importer
or required invocation of that implementation. The stable
`src/shap_publication_figures.py` path remains in place for documented consumers.

## Script comparison

| Script | Provenance and role | Tasks and model mapping | Checkpoints | Cohorts | Principal outputs / consumers |
|---|---|---|---|---|---|
| `src/shap_publication_figures.py` | **Canonical final manuscript entry point.** Latest maintainer implementation (`9a62435`), formerly suffixed `_v2`. | Four tasks. Figures 3--6 use per-cohort AUROC winners: UMM RF/MLP/TabTransformer/SVM and MAINZ LightGBM/LightGBM/VI-BNN/XGBoost. Table 4 and Figures 7--10 use LightGBM for all four tasks. | Tree ensembles at `models/<model>/model_<task>.pickle`; two-stage LightGBM fallback at `models/light_gbm_window/pre7_post0/model_two_stage.pickle`; native FFN, TabTransformer, and VI-BNN artifacts under their model directories. | Held-out UMM and external MAINZ, prepared in scaled and unscaled forms as required. The same UMM-trained attribution model is explained in each cohort. | Combined ROC/confusion and SHAP panels, per-member SHAP CSVs, `shap_all_features.csv`, `shap_top5.csv`, `shap_rank_agreement.csv`, and `shap_top5_table.tex` under `outputs/figures/`. Publication/manual consumers; `plot_svm_shap.py` and the historical derivation helper read `shap_all_features.csv`. |
| `misc/legacy/shap_publication_figures_v1.py` | Historical predecessor from `efd7072`, updated through `d7eced4`; archived because it had no importer and shared output names with the final script. | Same four tasks; broader SHAP list (LightGBM/XGBoost/RF/SVM), with RF for cirrhosis attribution. Evaluation winner mapping matches the later script. | Pickled `models/<model>/model_<task>.pickle` ensembles only. Selected neural evaluation families are skipped because this version has no native loader. | Held-out UMM and MAINZ. | Same fixed output namespace as the canonical script; running it can overwrite canonical artifacts. Historical reproduction only. |
| `src/shap_both_cohorts_figures.py` | Earlier both-cohort, tree-only publication report introduced in `efd7072`; historical but left in place because its established CLI/path role is documented. | Four tasks; LightGBM/XGBoost/RF SHAP. Its selected attribution/panel mapping is cohort-specific: UMM XGBoost/XGBoost/LightGBM/XGBoost and MAINZ XGBoost for all four tasks. | Pickled `models/<model>/model_<task>.pickle` ensembles. | Held-out UMM and MAINZ. | Collides with canonical SHAP CSV/table/panel names. Historical/manual publication path and old source for `derive_shap_top_features.py`. |
| `src/make_umm_shap_figures.py` | First UMM-only publication variant from `efd7072`; retained in place as an established legacy CLI. | Four tasks; LightGBM/XGBoost/RF. | Pickled `models/<model>/model_<task>.pickle` ensembles. | SHAP uses held-out UMM only; confusion matrices cover held-out UMM and external MAINZ. | Per-model SHAP figures/CSVs, confusion figures, `shap_top5.csv`, and `shap_top5_table.tex`; names overlap newer generators. Historical/manual only. |
| `src/derive_shap_top_features.py` | Historical bridge from publication explanations to reduced-feature JSON, introduced in `efd7072`; **not** the final feature-selection procedure. | Expects UMM rows for XGBoost across all four tasks. This no longer matches the canonical generator's LightGBM-only attribution export. | No checkpoint; reads `outputs/figures/shap_all_features.csv`. | Filters the publication output to held-out UMM. | Writes `outputs/shap_top_features.json`, which can be read by `preprocess.py` and `recompute_reduced_tables.py`. Do not use it to reconstruct the final reduced models. |
| `src/shap_feature_selection_development.py` | Separate development-only selection record added with the final publication implementation in `9a62435`; authoritative for final Table 5 feature selection, but **not** a manuscript figure generator. | Binary tasks only (moderate, severe, cirrhosis); LightGBM for every task. Three-stage is intentionally commented out. | `models/light_gbm/model_<task>.pickle`, with the prespecified `models/light_gbm_window/pre7_post0/model_two_stage.pickle` fallback. | Matching UMM training imputation only; excludes UMM validation/test and MAINZ. | `shap_feature_selection_development_all.csv` and `_top3.csv` for the selection audit/manual transfer. It does not write `shap_top_features.json`. |

## Consumer and safety notes

* All relative `models/` and `outputs/` paths above assume the documented
  working directory, `src/`.
* The publication generators share fixed filenames. Do not run a historical
  variant into the canonical `outputs/figures/` directory when preserving a
  manuscript artifact bundle.
* Publication explanation and development feature selection answer different
  questions and use different cohort partitions. Neither
  `derive_shap_top_features.py` nor publication SHAP should be substituted for
  the final development-only selection record.
* This cleanup changes provenance, filenames, and documentation only. It does
  not merge, rewrite, or numerically compare SHAP implementations.
