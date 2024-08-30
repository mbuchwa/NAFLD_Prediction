import shap
from src.utils.helper_functions import *


def interpret(x_train, x_test, df_cols):
    """
    Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.

    Parameters:
        x_train (array-like): Training data features.
        x_test (array-like): Test data features.
        df_cols (list): List of column names (features).

    Returns:
        None
    """
    explainer = shap.KernelExplainer(model=predict_pickle_model, data=shap.sample(x_train, 50))
    shap_values = explainer.shap_values(x_test)

    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        x_test,
        feature_names=df_cols,
        show=False)
    shap.save_html('outputs/xgb/force_plot.htm', f)
    plt.close()

    fig, ax = plt.subplots()
    shap_values2 = explainer(x_test)
    shap.plots.bar(shap_values2, show=False)

    f = plt.gcf()
    f.savefig('outputs/XGBoost/summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots()

    shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)

    f = plt.gcf()
    f.savefig('outputs/XGBoost/beeswarm.png', bbox_inches='tight', dpi=300)
    plt.close()