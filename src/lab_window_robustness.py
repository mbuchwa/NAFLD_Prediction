import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocess import preprare_data
from src.models.light_gmb import hypertrain_ensemble_light_gbm


def _ensemble_proba(models, x):
    probs = [model.predict_proba(x) for model in models]
    return np.mean(probs, axis=0)


def _binary_auroc(y_true, proba):
    if proba.ndim > 1:
        proba = proba[:, 1]
    return float(roc_auc_score(y_true, proba))


def run_window_experiment(window_label, window_days_pre, window_days_post, classification_type='two_stage',
                          shap_selected=True, scaling=False):
    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, _ = preprare_data(
        classification_type=classification_type,
        shap_selected=shap_selected,
        scaling=scaling,
        window_days_pre=window_days_pre,
        window_days_post=window_days_post
    )

    model_name = f'light_gbm_window_{window_label}'
    hypertrain_ensemble_light_gbm(
        xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, [],
        classification_type=classification_type,
        shap_selected=False,
        interpret_model=False,
        testing=True
    )

    default_model_path = Path(f'models/light_gbm/model_{classification_type}.pickle')
    if default_model_path.exists():
        window_model_dir = Path(f'models/{model_name}')
        window_model_dir.mkdir(parents=True, exist_ok=True)
        target_model_path = window_model_dir / f'model_{classification_type}.pickle'
        default_model_path.replace(target_model_path)
    else:
        target_model_path = default_model_path

    with open(target_model_path, 'rb') as f:
        models = pickle.load(f)

    internal_proba = _ensemble_proba(models, xs_test[0])
    external_proba = _ensemble_proba(models, xs_pro[0])

    return {
        'window_label': window_label,
        'window_days_pre': window_days_pre,
        'window_days_post': window_days_post,
        'internal_auroc': _binary_auroc(ys_test[0], internal_proba),
        'external_auroc': _binary_auroc(ys_pro[0], external_proba),
    }


def choose_recommended_window(results_df):
    ranked = results_df.copy()
    ranked['mean_auroc'] = ranked[['internal_auroc', 'external_auroc']].mean(axis=1)
    ranked['cohort_gap'] = (ranked['internal_auroc'] - ranked['external_auroc']).abs()
    ranked = ranked.sort_values(['mean_auroc', 'cohort_gap'], ascending=[False, True]).reset_index(drop=True)
    return ranked.iloc[0]


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)

    windows = [
        ('pre_only_m90_p0', 90, 0),
        ('symmetric_30', 30, 30),
        ('symmetric_60', 60, 60),
    ]

    results = []
    for label, pre_days, post_days in windows:
        print(f'Running experiment for {label} (pre={pre_days}, post={post_days})')
        results.append(run_window_experiment(label, pre_days, post_days))

    output_dir = Path('../outputs/robustness')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'lab_window_auroc_comparison.csv'

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    recommendation = choose_recommended_window(results_df)
    print('\nRecommended final window:')
    print(
        f"{recommendation['window_label']} | "
        f"internal={recommendation['internal_auroc']:.4f}, "
        f"external={recommendation['external_auroc']:.4f}, "
        f"mean={recommendation['mean_auroc']:.4f}, "
        f"gap={recommendation['cohort_gap']:.4f}"
    )
