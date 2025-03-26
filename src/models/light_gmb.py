from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from sklearn.model_selection import RandomizedSearchCV
import shap
import lightgbm as lgb
import te2rules
from te2rules.explainer import ModelExplainer


def hypertrain_ensemble_light_gbm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                                  classification_type, shap_selected, interpret_model=True, testing=True):
    models = []
    model_name = 'light_gbm_shap_selected' if shap_selected else 'light_gbm'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Train models
    for idx, (X_train, y_train, X_val, y_val, X_test, y_test, X_pro, y_pro) in enumerate(zip(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro)):
        print(f'Training model {idx}')
        models.append(hypertrain_light_gbm_model(X_train, y_train, X_val, y_val, X_test, y_test, X_pro, y_pro, classification_type=classification_type))

    # Save models
    model_path = f'models/{model_name}/model_{classification_type}.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    # Optionally interpret models
    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols, classification_type=classification_type, model_name=model_name)

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                    model_name=model_name)


def evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                model_name='light_gbm'):

    if 'shap_selected' not in model_name and shap_selected:
        model_name = f'{model_name}_shap_selected'

    checkpoint_file = [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')
                       and classification_type in f][0]

    # Load the models from the saved pickle file
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)

    # Prospective evaluation
    prospective = True
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)


def finetune_ensemble_light_gbm(xs_finetune, ys_finetune, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                                classification_type, shap_selected, interpret_model=True, testing=True):
    """
    Fine-tunes a saved ensemble of LGBMClassifier models on new data.

    Parameters:
    xs_finetune (list of np.array): List of feature arrays for fine-tuning.
    ys_finetune (list of np.array): List of target arrays for fine-tuning.
    xs_val (list of np.array): List of feature arrays for validation.
    ys_val (list of np.array): List of target arrays for validation.
    xs_test (list of np.array): List of feature arrays for testing.
    ys_test (list of np.array): List of target arrays for testing.
    xs_pro (list of np.array): List of feature arrays for prospective evaluation.
    ys_pro (list of np.array): List of target arrays for prospective evaluation.
    df_cols (list): List of column names used for feature interpretation.
    classification_type (str): Type of classification (e.g., binary, multi-class).
    shap_selected (bool): Whether SHAP-selected features were used.
    interpret_model (bool): Whether to interpret the model using SHAP.
    testing (bool): Whether to evaluate the model after fine-tuning.
    """
    model_name = 'light_gbm_shap_selected' if shap_selected else 'light_gbm'

    # Load the model
    model_path = f'./models/{model_name}/model_{classification_type}.pickle'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    # Ensure models is a list of LGBMClassifier instances
    if not all(isinstance(model, lgb.LGBMClassifier) for model in models):
        raise TypeError("Loaded models are not of type LGBMClassifier")

    # Fine-tune models
    for idx, (model, X_finetune, y_finetune, X_val, y_val) in enumerate(
            zip(models, xs_finetune, ys_finetune, xs_val, ys_val)):
        print(f'Fine-tuning model {idx}')
        # model.fit(X_finetune, y_finetune, eval_set=[(X_val, y_val)],
        #           eval_metric='multi_logloss' if classification_type == 'three_stage' else 'rmse',
        #           init_model=model)
        models[idx] = model

    print('------ Finished Fine-Tuning Ensemble ------')

    # Save fine-tuned models
    finetuned_model_dir = f'./models/{model_name}_finetuned/'
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = f'{finetuned_model_dir}/model_{classification_type}.pickle'

    with open(finetuned_model_path, "wb") as f:
        pickle.dump(models, f)

    # Optionally interpret models
    if interpret_model:
        interpret(xs_finetune[0], xs_test[0], df_cols, classification_type=classification_type,
                  model_name=model_name + '_finetuned')

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                    model_name=model_name + '_finetuned')


# def interpret_light_gbm(x_train, x_test, df_cols, classification_type='fibrosis', model_name='light_gbm'):
#     """
#     Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.
#
#     Parameters:
#         x_train (array-like): Training data features.
#         x_test (array-like): Test data features.
#         df_cols (list): List of column names (features)
#         classification_type (str): 'fibrosis' or 'cirrhosis'.
#         model_name (str): name of output model
#
#     Returns:
#         None
#     """
#     explainer = shap.KernelExplainer(model=lambda x: predict_light_gbm_model(x, classification_type=classification_type,
#                                                                              model_name=model_name),
#                                      data=shap.sample(x_train, 50), feature_names=df_cols)
#
#     shap_values = explainer.shap_values(x_test)
#
#     f = shap.force_plot(explainer.expected_value, shap_values, x_test, feature_names=df_cols, show=False)
#     shap.save_html(f'outputs/{model_name}/{classification_type}_force_plot.htm', f)
#     plt.close()
#
#     fig, ax = plt.subplots()
#     shap_values2 = explainer(x_test)
#     shap.plots.bar(shap_values2, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/{model_name}/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     fig, ax = plt.subplots()
#
#     shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/{model_name}/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#
# def predict_light_gbm_model(data, classification_type='fibrosis', model_name='light_gbm'):
#     """
#     Predictions of the model for certain data. Model is saved in output/models.pickle
#
#     Args:
#         data: A numpy array to predict on.
#         classification_type (str): 'fibrosis' or 'cirrhosis'.
#         model_name (str): name of output model
#
#     Returns:
#         A numpy array of class predictions
#     """
#
#     with open(f'models/{model_name}/model_{classification_type}.pickle', "rb") as f:
#         models = pickle.load(f)
#
#     y_preds = []
#     for model in models:
#
#         # # ---------------------------------------------------------
#         #
#         # # Extract rules from the LightGBM model
#         # def traverse_tree(node, depth=0):
#         #     if "split_index" in node:
#         #         split_feature = node["split_feature"]
#         #         threshold = node["threshold"]
#         #         left_child = node["left_child"]
#         #         right_child = node["right_child"]
#         #
#         #         rule_left = traverse_tree(left_child, depth + 1)
#         #         rule_right = traverse_tree(right_child, depth + 1)
#         #
#         #         feature_name = feature_mapping.get(split_feature, f"Feature {split_feature}")
#         #
#         #         rule = f"{feature_name} <= {np.round(threshold, 2)}"
#         #         rules_left = [f"{rule} AND {r}" for r in rule_left]
#         #
#         #         rule = f"{feature_name} > {np.round(threshold, 2)}"
#         #         rules_right = [f"{rule} AND {r}" for r in rule_right]
#         #
#         #         return rules_left + rules_right
#         #     else:
#         #         return [f"class: {node['leaf_value']}"]
#         #
#         # def extract_rules_from_model(model):
#         #     booster = model.booster_
#         #     trees = booster.dump_model()["tree_info"]
#         #     rules = []
#         #     for tree_index, tree in enumerate(trees):
#         #         tree_structure = tree["tree_structure"]
#         #         tree_rules = traverse_tree(tree_structure)
#         #         tree_rules = [f"Tree {tree_index}: {rule}" for rule in tree_rules]
#         #         rules.extend(tree_rules)
#         #     return rules
#         #
#         # # Define feature mapping
#         # feature_mapping = {
#         #     0: 'Thrombozyten (Mrd/l)',
#         #     1: 'MCV (fl)',
#         #     2: 'INR'
#         # }
#         #
#         # # Get rules
#         # rules = extract_rules_from_model(model)
#         # for rule in rules[:10]:  # Display first 10 rules for brevity
#         #     print(rule)
#         #
#         # # ---------------------------------------------------------
#
#         y_pred = model.predict_proba(data)
#         y_preds.append(y_pred)
#
#     maj_preds = majority_vote(y_preds, rule='soft')
#     indices, _ = get_index_and_proba(maj_preds)
#
#     return np.array(indices)


def hypertrain_light_gbm_model(x_train, y_train, x_val, y_val, x_test=None, y_test=None, x_pro=None, y_pro=None, classification_type='fibrosis'):
    """
    Trains a model on the provided features (x_train) and labels (y_train) with early stopping based on validation loss.

    Args:
        x_train (pandas.DataFrame or numpy.ndarray): The features used for training.
        y_train (pandas.Series or numpy.ndarray): The labels used for training.
        x_val (pandas.DataFrame or numpy.ndarray): The features used for validation.
        y_val (pandas.Series or numpy.ndarray): The labels used for validation.
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'

    Returns:
        sklearn model: The trained model.
    """
    # ---------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Extract coefficients and intercept
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    # Map coefficients to feature names
    feature_names = ['Thrombozyten (Mrd/l)', 'MCV (fl)', 'INR']
    coef_dict = dict(zip(feature_names, coefficients))

    # Print the logistic regression equation
    print("Logistic Regression Equation:")
    print(
        f"log-odds = {intercept:.4f} + ({coef_dict['Thrombozyten (Mrd/l)']:.4f} * Thrombozyten (Mrd/l)) + ({coef_dict['MCV (fl)']:.4f} * MCV (fl)) + ({coef_dict['INR']:.4f} * INR)")

    # To get the probability, use the sigmoid function
    print("\nProbability of positive class (P):")
    print("P = 1 / (1 + exp(-log-odds))")

    breakpoint()
    # Make predictions on the test set
    y_test_pred_proba = model.predict_proba(x_test)[:, 1]  # Probabilities for the positive class
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary prediction

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    ppv = tp / (tp + fp)  # Positive Predictive Value (Precision)
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Positive Predictive Value (PPV): {ppv:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    
    # Make predictions on the prospective set
    y_pro_pred_proba = model.predict_proba(x_pro)[:, 1]  # Probabilities for the positive class
    y_pro_pred = (y_pro_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary prediction

    # Calculate metrics
    accuracy = accuracy_score(y_pro, y_pro_pred)
    f1 = f1_score(y_pro, y_pro_pred)
    tn, fp, fn, tp = confusion_matrix(y_pro, y_pro_pred).ravel()
    ppv = tp / (tp + fp)  # Positive Predictive Value (Precision)
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Positive Predictive Value (PPV): {ppv:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    breakpoint()

    # ---------------------------------------------------------

    grid_params = {
        'max_depth': np.arange(1, 40),
        'learning_rate': np.linspace(0.5, 0.01, 5),
        'subsample': np.linspace(1, 0.3, 5)
    }

    if classification_type in ['fibrosis', 'cirrhosis', 'two_stage']:
        lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='regression', verbosity=-1)

    elif classification_type == 'three_stage':
        lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=3, verbosity=-1)

    else:
        raise ValueError(f'classification_type {classification_type} is not implemented!')

    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=grid_params,
        scoring='accuracy' if classification_type == 'three_stage' else 'neg_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    random_search.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                      eval_metric='multi_logloss' if classification_type == 'three_stage' else 'rmse')

    # print()
    # print(f'max_depth: {random_search.best_estimator_.max_depth}')
    # print(f'learning_rate: {random_search.best_estimator_.learning_rate}')
    # print(f'subsample: {random_search.best_estimator_.subsample}')
    # print()

    return random_search.best_estimator_
