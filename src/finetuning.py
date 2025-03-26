import warnings
from utils.helper_functions import *
from preprocess import preprare_data
from src.models.tab_transformer import evaluate_ensemble_tab_transformer
from src.models.xgb import finetune_ensemble_xgb
from src.models.ffn import evaluate_ensemble_ffn
from src.models.svm import evaluate_ensemble_svm
from src.models.rf import finetune_ensemble_rf
from src.models.gandalf import evaluate_ensemble_gandalf
from src.models.vi_bnn import evaluate_ensemble_vi_bnn
from src.models.light_gmb import finetune_ensemble_light_gbm
warnings.filterwarnings('ignore')


def finetuning(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_test_umm, ys_test_umm, df_cols,
               classification_type, model_name, shap_selected):
    if not os.path.exists(f'outputs/{model_name}'):
        os.makedirs(f'outputs/{model_name}')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # Finetune XGBoost
    if model_name == 'xgb':
        finetune_ensemble_xgb(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_test_umm, ys_test_umm, df_cols,
                              classification_type, shap_selected)

    # Finetune LightGBM
    if model_name == 'light_gbm':
        finetune_ensemble_light_gbm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_test_umm, ys_test_umm,
                                    df_cols, classification_type, shap_selected)


if __name__ == '__main__':
    model_name = 'light_gbm'
    classification_type = 'cirrhosis'
    shap_selected = False
    scaling = False

    assert model_name in ['xgb', 'light_gbm']
    assert classification_type in ['cirrhosis', 'fibrosis', 'three_stage', 'two_stage']

    if model_name == 'vi_bnn':
        scaling = True

    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_test_umm, ys_test_umm, df_cols = \
        preprare_data(classification_type, shap_selected, scaling, finetune=True)

    print(f'\n ----- Testing model {model_name} | Task {classification_type} ----- \n')
    print('-------------------------------------------')

    finetuning(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_test_umm, ys_test_umm, df_cols=df_cols,
               classification_type=classification_type, model_name=model_name, shap_selected=shap_selected)
