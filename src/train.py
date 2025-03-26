import warnings
from utils.helper_functions import *
from preprocess import preprare_data
from src.models.tab_transformer import hypertrain_ensemble_tab_transformer
from src.models.xgb import hypertrain_ensemble_xgboost
from src.models.mcmc_bnn import hypertrain_ensemble_mcmc_bnn
from src.models.ffn import hypertrain_ensemble_ffn
from src.models.svm import hypertrain_ensemble_svm
from src.models.rf import hypertrain_ensemble_rf
from src.models.gandalf import hypertrain_ensemble_gandalf
from src.models.vi_bnn import hypertrain_ensemble_vi_bnn
from src.models.light_gmb import hypertrain_ensemble_light_gbm
warnings.filterwarnings('ignore')


def hypertrain(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, model_name, shap_selected):
    if not os.path.exists(f'outputs/{model_name}'):
        os.makedirs(f'outputs/{model_name}')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # Hypertrain TabTransformer
    if model_name == 'tab_transformer':
        hypertrain_ensemble_tab_transformer(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                            shap_selected)

    # Hypertrain XGBoost
    if model_name == 'xgb':
        hypertrain_ensemble_xgboost(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                    shap_selected)

    # Hypertrain LightGBM
    if model_name == 'light_gbm':
        hypertrain_ensemble_light_gbm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                      shap_selected)

    # Hypertrain MCMC_BNN
    elif model_name == 'mcmc_bnn':
        hypertrain_ensemble_mcmc_bnn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                     shap_selected)

    # Hypertrain VI_BNN
    elif model_name == 'vi_bnn':
        hypertrain_ensemble_vi_bnn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                   shap_selected)

    # Hypertrain Feed Forward Neural Networkf
    elif model_name == 'ffn':
        hypertrain_ensemble_ffn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                shap_selected)

    # Hypertrain SVM
    elif model_name == 'svm':
        hypertrain_ensemble_svm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                shap_selected)

    # Hypertrain RF
    elif model_name == 'rf':
        hypertrain_ensemble_rf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                               shap_selected)

    elif model_name == 'gandalf':
        hypertrain_ensemble_gandalf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                    shap_selected)


if __name__ == '__main__':
    model_name = 'light_gbm'
    classification_type = 'two_stage'
    shap_selected = True
    scaling = False

    assert model_name in ['svm', 'rf', 'xgb', 'light_gbm', 'ffn', 'gandalf', 'tab_transformer', 'mcmc_bnn', 'vi_bnn']
    assert classification_type in ['cirrhosis', 'fibrosis', 'three_stage', 'two_stage']

    if model_name == 'vi_bnn':
        scaling = True

    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols = preprare_data(classification_type, shap_selected, scaling)

    print(f'\n ----- Hypertraining model {model_name} | Task {classification_type} ----- \n')
    print('-------------------------------------------')

    hypertrain(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols=df_cols,
               classification_type=classification_type, model_name=model_name, shap_selected=shap_selected)
