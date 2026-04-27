import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocess import preprare_data
from src.models.light_gmb import hypertrain_light_gbm_model


def _binary_auroc(y_true, proba):
    if proba.ndim > 1:
        proba = proba[:, 1]
    return float(roc_auc_score(y_true, proba))


def _ensemble_proba(models, x):
    probs = [model.predict_proba(x) for model in models]
    return np.mean(probs, axis=0)


def _train_ensemble(xs_train, ys_train, xs_val, ys_val, classification_type):
    models = []
    for x_train, y_train, x_val, y_val in zip(xs_train, ys_train, xs_val, ys_val):
        models.append(
            hypertrain_light_gbm_model(
                x_train,
                y_train,
                x_val,
                y_val,
                classification_type=classification_type,
            )
        )
    return models


def _rank_biomarkers_by_missingness(profile_path, available_features):
    missingness_df = pd.read_csv(profile_path)
    if 'biomarker' not in missingness_df.columns:
        raise ValueError(
            f"Expected `biomarker` column in {profile_path}, got columns: {list(missingness_df.columns)}"
        )
    if 'missingness_before_cleaning_pct' not in missingness_df.columns:
        raise ValueError(
            "Expected `missingness_before_cleaning_pct` column in missingness profile."
        )

    ranked = missingness_df[missingness_df['biomarker'].isin(available_features)].copy()
    ranked = ranked.sort_values('missingness_before_cleaning_pct', ascending=False).reset_index(drop=True)
    if ranked.empty:
        raise ValueError(
            "No overlap between profiling biomarkers and model features. "
            "Cannot build reduced feature set."
        )
    return ranked


def _select_features(xs_list, df_cols, selected_cols):
    selected_indices = [df_cols.index(col) for col in selected_cols]
    return [x[:, selected_indices] for x in xs_list]


def run_missingness_sensitivity(classification_type='two_stage', top_n=3):
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = repo_root / 'outputs' / 'data_qc' / 'missingness_profile.csv'
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing profiling output at {profile_path}. Run preprocessing first to generate missingness_profile.csv."
        )

    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols = preprare_data(
        classification_type=classification_type,
        shap_selected=False,
        scaling=False,
    )

    ranked_missing = _rank_biomarkers_by_missingness(profile_path, set(df_cols))
    excluded = ranked_missing.head(top_n)['biomarker'].tolist()
    reduced_features = [feature for feature in df_cols if feature not in excluded]

    if len(reduced_features) == 0:
        raise ValueError("Reduced feature set is empty after removing top missing biomarkers.")

    full_models = _train_ensemble(xs_train, ys_train, xs_val, ys_val, classification_type)
    reduced_models = _train_ensemble(
        _select_features(xs_train, df_cols, reduced_features),
        ys_train,
        _select_features(xs_val, df_cols, reduced_features),
        ys_val,
        classification_type,
    )

    full_internal = _binary_auroc(ys_test[0], _ensemble_proba(full_models, xs_test[0]))
    full_external = _binary_auroc(ys_pro[0], _ensemble_proba(full_models, xs_pro[0]))

    reduced_xs_test = _select_features(xs_test, df_cols, reduced_features)
    reduced_xs_pro = _select_features(xs_pro, df_cols, reduced_features)
    reduced_internal = _binary_auroc(ys_test[0], _ensemble_proba(reduced_models, reduced_xs_test[0]))
    reduced_external = _binary_auroc(ys_pro[0], _ensemble_proba(reduced_models, reduced_xs_pro[0]))

    output_dir = repo_root / 'outputs' / 'robustness'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'missingness_sensitivity.csv'

    comparison = pd.DataFrame(
        [
            {
                'classification_type': classification_type,
                'excluded_top_missing_biomarkers': '; '.join(excluded),
                'n_reduced_features': len(reduced_features),
                'internal_auroc_full': full_internal,
                'internal_auroc_reduced': reduced_internal,
                'internal_delta_full_minus_reduced': full_internal - reduced_internal,
                'external_auroc_full': full_external,
                'external_auroc_reduced': reduced_external,
                'external_delta_full_minus_reduced': full_external - reduced_external,
            }
        ]
    )
    comparison.to_csv(output_path, index=False)

    print('Top biomarkers by missingness:')
    print(ranked_missing[['biomarker', 'missingness_before_cleaning_pct']].head(10).to_string(index=False))
    print(f'\nReduced feature set excludes: {excluded}')
    print(f'Exported sensitivity comparison to: {output_path}')

    return output_path


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    run_missingness_sensitivity(classification_type='two_stage', top_n=3)
