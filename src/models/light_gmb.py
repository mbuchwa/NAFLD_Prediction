from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import make_scorer, cohen_kappa_score
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Reproducibility settings. Keep these in sync with xgb.py.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_ESTIMATORS = 500          # upper bound; early stopping decides the actual size
N_ITER = 10                 # candidates drawn by the randomized search
EARLY_STOPPING_ROUNDS = 10




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

    # Optionally run single-pass held-out evaluation
    if testing:
        evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected,
                                    model_name=model_name)


def evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected,
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
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective, xs_val=xs_val, ys_val=ys_val)

    # Prospective evaluation
    prospective = True
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective, xs_val=xs_val, ys_val=ys_val)


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

    # Optionally run single-pass held-out evaluation
    if testing:
        evaluate_ensemble_light_gbm(xs_test, ys_test, xs_pro, ys_pro, xs_val, ys_val, df_cols, classification_type, shap_selected,
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


def hypertrain_light_gbm_model(x_train, y_train, x_val, y_val, x_test=None, y_test=None, x_pro=None, y_pro=None,
                               classification_type='fibrosis', random_state=RANDOM_STATE,
                               early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    """
    Trains a LightGBM model with a randomized hyperparameter search on the training
    partition and early stopping on the validation partition.

    Changes against the previous version, all of which affect the reported numbers:

    1. random_state / deterministic on the ESTIMATOR. Previously only
       RandomizedSearchCV was seeded, which fixes the choice of candidates but not
       the fitting itself. Without this the same data can yield different models
       between runs.

    2. subsample_freq=1. LightGBM ignores `subsample` (bagging_fraction) unless
       `subsample_freq` (bagging_freq) is greater than zero. In the previous
       version one of the three searched hyperparameters therefore had no effect
       at all; subsample=0.3 and subsample=1.0 produced bit-identical models.

    3. Early stopping via callback. Passing eval_set alone does nothing in
       lightgbm >= 4 -- it only records metrics. The Methods section states that
       early stopping was performed on the validation partition, which was true
       for XGBoost but not for LightGBM. n_estimators is raised so that early
       stopping, not the iteration cap, determines the model size.

    4. scoring='roc_auc'. The previous scorer, neg_mean_squared_error, calls
       predict() on a classifier and therefore scores hard 0/1 labels, i.e. the
       misclassification rate. That is a step function on ~214 training samples:
       candidates tie, ties are broken by position, and the selection flips on
       small data changes even though AUROC is the metric the manuscript reports.

    Args:
        x_train, y_train: training features and labels.
        x_val, y_val: validation features and labels, used for early stopping only.
        x_test, y_test, x_pro, y_pro: accepted for signature compatibility, unused.
        classification_type: 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
        random_state: seed for the estimator and the search.
        early_stopping_rounds: patience on the validation partition.

    Returns:
        The refitted best estimator of the randomized search.
    """
    if classification_type in ['fibrosis', 'cirrhosis', 'two_stage']:
        objective_kwargs = dict(objective='binary')
        scoring = 'roc_auc'
        eval_metric = 'auc'
    elif classification_type == 'three_stage':
        objective_kwargs = dict(objective='multiclass', num_class=3)
        # Select on the metric that is reported. The primary metric for the
        # ordinal task is the linearly weighted kappa, which penalises a
        # two-stage error twice as heavily as an adjacent one. Selecting on
        # roc_auc_ovr instead optimises class ranking and is indifferent to how
        # far a misclassification lands -- the same mismatch that motivated the
        # comparison of decision rules.
        scoring = make_scorer(cohen_kappa_score, weights='linear', labels=[0, 1, 2])
        eval_metric = 'multi_logloss'
    else:
        raise ValueError(f'classification_type {classification_type} is not implemented!')

    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        verbosity=-1,
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
        deterministic=True,
        force_row_wise=True,      # avoids the auto-detection that can vary between runs
        n_jobs=1,                 # multithreaded histogram building is not bit-reproducible
        **objective_kwargs
    )

    grid_params = {
        'max_depth': np.arange(1, 40),
        'learning_rate': np.linspace(0.5, 0.01, 5),
        'subsample': np.linspace(1, 0.3, 5),
        'subsample_freq': [1],    # required for subsample to take effect
    }

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=grid_params,
        n_iter=N_ITER,
        scoring=scoring,
        cv=5,
        verbose=0,
        random_state=random_state,
        n_jobs=1
    )

    # lightgbm >= 4.6 deprecates eval_set in favour of eval_X / eval_y; older
    # versions do not know the new names. Try the new signature first and fall
    # back, so the same file works across the versions in the group.
    fit_common = dict(sample_weight=classes_weights,
                      eval_metric=eval_metric,
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(period=0)])
    try:
        random_search.fit(x_train, y_train, eval_X=x_val, eval_y=y_val, **fit_common)
    except TypeError:
        random_search.fit(x_train, y_train, eval_set=[(x_val, y_val)], **fit_common)

    best = random_search.best_estimator_
    n_trees = best.booster_.num_trees() if hasattr(best, 'booster_') else None
    print(f'    best: max_depth={best.max_depth}, learning_rate={best.learning_rate:.4f}, '
          f'subsample={best.subsample:.2f}, trees={n_trees}/{N_ESTIMATORS}, '
          f'cv_{scoring}={random_search.best_score_:.4f}')

    return best