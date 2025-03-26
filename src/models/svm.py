from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit


def hypertrain_ensemble_svm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                            classification_type, shap_selected, interpret_model=True, testing=True):
    models = []

    # Directory setup
    model_name = 'svm_shap_selected' if shap_selected else 'svm'

    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        models.append(hypertrain_svm_model(X_train, y_train, X_val, y_val))

    model_path = f'models/{model_name}/model_{classification_type}.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    print('------ Finished Training Ensemble ------')

    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols, classification_type=classification_type, model_name=model_name)

    if testing:
        # Optionally test immediately after training
        evaluate_ensemble_svm(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected)


def evaluate_ensemble_svm(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                          model_name='svm'):
    model_name = f'{model_name}_shap_selected' if shap_selected else model_name

    checkpoint_file = [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')
                       and classification_type in f][0]

    # Load the models from the saved pickle file
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)
    print('----- Prospective Evaluation ------')
    prospective = True
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)

#
# def interpret_svm(x_train, x_test, df_cols, classification_type='fibrosis'):
#     """
#     Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.
#
#     Parameters:
#         x_train (array-like): Training data features.
#         x_test (array-like): Test data features.
#         df_cols (list): List of column names (features).
#         classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
#
#     Returns:
#         None
#     """
#     explainer = shap.KernelExplainer(model=lambda x: predict_svm_model(x, classification_type=classification_type), data=shap.sample(x_train, 50), feature_names=df_cols)
#
#     shap_values = explainer.shap_values(x_test)
#
#     f = shap.force_plot(
#         explainer.expected_value,
#         shap_values,
#         x_test,
#         feature_names=df_cols,
#         show=False)
#     shap.save_html(f'outputs/xgb/{classification_type}_force_plot.htm', f)
#     plt.close()
#
#     fig, ax = plt.subplots()
#     shap_values2 = explainer(x_test)
#     shap.plots.bar(shap_values2, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/svm/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     fig, ax = plt.subplots()
#
#     shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/svm/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#
# def predict_svm_model(data, classification_type='fibrosis'):
#     """
#     Predictions of the model for certain data. Model is saved in output/models.pickle
#
#     Args:
#         data: A numpy array to predict on.
#         classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
#
#     Returns:
#         A numpy array of class predictions
#     """
#
#     with open(f'models/svm/model_{classification_type}.pickle', "rb") as f:
#         models = pickle.load(f)
#
#     y_preds = []
#     for model in models:
#         y_pred = model.predict_proba(data)
#         y_preds.append(y_pred)
#
#     maj_preds = majority_vote(y_preds, rule='soft')
#     indices, _ = get_index_and_proba(maj_preds)
#
#     return np.array(indices)


def hypertrain_svm_model(x_train, y_train, x_val, y_val):
    """
    Trains a model on the provided features (x_train) and labels (y_train).

    Args:
        x_train (pandas.DataFrame or numpy.ndarray): The features used for training.
        y_train (pandas.Series or numpy.ndarray): The labels used for training.
        x_val (pandas.DataFrame or numpy.ndarray): The features used for validation.
        y_val (pandas.Series or numpy.ndarray): The labels used for validation.

    Returns:
        sklearn model: The trained model.
    """
    # Concatenate x_train and x_val to X to use for PredefinedSplit. Reason: RF does not take eval_set as input
    X = np.concatenate((x_train, x_val))
    y = np.concatenate((y_train, y_val))

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1] * x_train.shape[0]
    val_index_list = [0] * x_val.shape[0]
    split_index.extend(val_index_list)

    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold=split_index)

    model = SVC(random_state=42, probability=True)

    hp_space = {
        'degree': np.arange(2, 5),  # Degree of polynomial kernel (if applicable)
    }

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=hp_space,
        scoring='neg_log_loss',
        cv=pds,
        random_state=42,
    )

    clf.fit(X, y, sample_weight=classes_weights)

    return clf.best_estimator_
