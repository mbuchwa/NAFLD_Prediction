import warnings
from pathlib import Path

import numpy as np
import os
import pandas as pd
from utils.helper_functions import *
from preprocess import preprare_data
from src.models.tab_transformer import evaluate_ensemble_tab_transformer
from src.models.xgb import evaluate_ensemble_xgboost
from src.models.ffn import evaluate_ensemble_ffn
from src.models.svm import evaluate_ensemble_svm
from src.models.rf import evaluate_ensemble_rf
from src.models.gandalf import evaluate_ensemble_gandalf
from src.models.vi_bnn import evaluate_ensemble_vi_bnn
from src.models.light_gmb import evaluate_ensemble_light_gbm
from src.utils.ger_eng_dict import dict_germ_eng
warnings.filterwarnings('ignore')


def testing(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, model_name, shap_selected):
    """Run one-time held-out evaluation without any test-set re-splitting."""
    if not os.path.exists(f'outputs/{model_name}'):
        os.makedirs(f'outputs/{model_name}')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # Evaluate TabTransformer
    if model_name == 'tab_transformer':
        evaluate_ensemble_tab_transformer(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    # Evaluate XGBoost
    if model_name == 'xgb':
        evaluate_ensemble_xgboost(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    # Evaluate LightGBM
    if model_name == 'light_gbm':
        evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    # Evaluate VI_BNN
    elif model_name == 'vi_bnn':
        evaluate_ensemble_vi_bnn(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    # Evaluate Feed Forward Neural Network
    elif model_name == 'ffn':
        evaluate_ensemble_ffn(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    # Evaluate SVM
    elif model_name == 'svm':
        evaluate_ensemble_svm(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    # Evaluate RF
    elif model_name == 'rf':
        evaluate_ensemble_rf(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)

    elif model_name == 'gandalf':
        evaluate_ensemble_gandalf(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected)


def export_external_class_prevalence(ys_external, output_path='outputs/external/class_prevalence.csv'):
    """
    Export prevalence summary for external labels:
    - class 0: F0-F1
    - class 1: F2-F4
    """
    if isinstance(ys_external, list):
        if len(ys_external) == 0:
            raise ValueError("Cannot report prevalence for external labels: `ys_external` is empty.")
        labels = np.asarray(ys_external[0]).reshape(-1)
    else:
        labels = np.asarray(ys_external).reshape(-1)

    total = labels.size
    if total == 0:
        raise ValueError("Cannot report prevalence for external labels: no samples available.")

    class_0_count = int(np.sum(labels == 0))
    class_1_count = int(np.sum(labels == 1))
    prevalence_df = pd.DataFrame(
        [
            {'class_group': 'F0-F1', 'label': 0, 'count': class_0_count, 'prevalence_percent': (class_0_count / total) * 100},
            {'class_group': 'F2-F4', 'label': 1, 'count': class_1_count, 'prevalence_percent': (class_1_count / total) * 100},
        ]
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prevalence_df.to_csv(output_file, index=False)
    print(f'External class prevalence exported to {output_file.resolve()}')
    print(prevalence_df)


if __name__ == '__main__':
    model_name = 'light_gbm'
    classification_type = 'fibrosis'
    shap_selected = False
    scaling = False
    select_patients = False
    smote = False

    assert model_name in ['svm', 'rf', 'xgb', 'light_gbm', 'ffn', 'gandalf', 'tab_transformer', 'mcmc_bnn', 'vi_bnn']
    assert classification_type in ['cirrhosis', 'fibrosis', 'three_stage', 'two_stage']

    if model_name == 'vi_bnn':
        scaling = True

    if smote:
        raise ValueError(
            "SMOTE cannot be enabled for external/prospective data in this evaluation pipeline. "
            "Set `smote = False` in src/test.py."
        )

    _, _, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols = preprare_data(classification_type, shap_selected, scaling,
                                                                          select_patients=select_patients, smote=smote)

    export_external_class_prevalence(ys_pro, output_path='outputs/external/class_prevalence.csv')

    df_cols = [dict_germ_eng[biomarker] for biomarker in df_cols]

    print(f'\n ----- Testing model {model_name} | Task {classification_type} ----- \n')
    print('-------------------------------------------')

    testing(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols=df_cols, classification_type=classification_type,
            model_name=model_name, shap_selected=shap_selected)
