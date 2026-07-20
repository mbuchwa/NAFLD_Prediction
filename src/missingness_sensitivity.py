import os
from pathlib import Path

import numpy as np
import pandas as pd


def _binary_auroc(y_true, proba):
    """
    Compute binary AUROC without requiring sklearn at import time.
    """
    if proba.ndim > 1:
        proba = proba[:, 1]

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    positive = y_true == 1
    negative = y_true == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError('AUROC is undefined when a cohort has one observed class.')

    order = np.argsort(proba)
    sorted_scores = proba[order]
    ranks = np.empty(len(proba), dtype=float)
    start = 0
    while start < len(proba):
        end = start + 1
        while end < len(proba) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[positive].sum()
    return float((positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def _ensemble_proba(models, x):
    probs = [model.predict_proba(x) for model in models]
    return np.mean(probs, axis=0)


def _train_ensemble(xs_train, ys_train, xs_val, ys_val, classification_type):
    from src.models.light_gmb import hypertrain_light_gbm_model

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
    required_columns = {'biomarker', 'missingness_before_cleaning_pct'}
    missing_columns = required_columns.difference(missingness_df.columns)
    if missing_columns:
        raise ValueError(
            f"Expected columns {sorted(required_columns)} in {profile_path}, "
            f"missing: {sorted(missing_columns)}. Got columns: {list(missingness_df.columns)}"
        )

    ranked = missingness_df[missingness_df['biomarker'].isin(available_features)].copy()
    ranked = ranked.sort_values('missingness_before_cleaning_pct', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'missingness_rank', np.arange(1, len(ranked) + 1))
    if ranked.empty:
        raise ValueError(
            "No overlap between profiling biomarkers and model features. "
            "Cannot build reduced feature set. Regenerate outputs/data_qc/missingness_profile.csv "
            "from the same preprocessing run used for model features."
        )
    return ranked


def _select_features(xs_list, df_cols, selected_cols):
    selected_indices = [df_cols.index(col) for col in selected_cols]
    return [x[:, selected_indices] for x in xs_list]


def run_missingness_sensitivity(classification_type='two_stage', top_n=3):
    from preprocess import preprare_data

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

    full_scores = {
        'internal': _binary_auroc(ys_test[0], _ensemble_proba(full_models, xs_test[0])),
        'external': _binary_auroc(ys_pro[0], _ensemble_proba(full_models, xs_pro[0])),
    }

    reduced_xs_test = _select_features(xs_test, df_cols, reduced_features)
    reduced_xs_pro = _select_features(xs_pro, df_cols, reduced_features)
    reduced_scores = {
        'internal': _binary_auroc(ys_test[0], _ensemble_proba(reduced_models, reduced_xs_test[0])),
        'external': _binary_auroc(ys_pro[0], _ensemble_proba(reduced_models, reduced_xs_pro[0])),
    }

    output_dir = repo_root / 'outputs' / 'robustness'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'missingness_sensitivity.csv'

    comparison = pd.DataFrame(
        [
            {
                'classification_type': classification_type,
                'cohort': cohort,
                'excluded_top_missing_biomarkers': '; '.join(excluded),
                'n_full_features': len(df_cols),
                'n_reduced_features': len(reduced_features),
                'auroc_full': full_scores[cohort],
                'auroc_reduced': reduced_scores[cohort],
                'delta_full_minus_reduced': full_scores[cohort] - reduced_scores[cohort],
            }
            for cohort in ['internal', 'external']
        ]
    )
    comparison.to_csv(output_path, index=False)

    ranked_output_path = output_dir / 'missingness_ranked_biomarkers.csv'
    ranked_missing.to_csv(ranked_output_path, index=False)

    print('Top biomarkers by missingness:')
    print(ranked_missing[['missingness_rank', 'biomarker', 'missingness_before_cleaning_pct']].head(10).to_string(index=False))
    print(f'\nReduced feature set excludes: {excluded}')
    print(f'Exported sensitivity comparison to: {output_path}')
    print(f'Exported ranked missingness table to: {ranked_output_path}')

    return output_path


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    run_missingness_sensitivity(classification_type='two_stage', top_n=3)
