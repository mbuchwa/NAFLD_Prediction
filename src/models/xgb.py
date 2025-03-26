import os

from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight
import xgboost as xgb
import te2rules
from te2rules.explainer import ModelExplainer
import numpy as np
import matplotlib.pyplot as plt


def hypertrain_ensemble_xgboost(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                                classification_type, shap_selected, interpret_model=True, testing=True):
    models = []
    model_name = 'xgb_shap_selected' if shap_selected else 'xgb'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Train models
    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        models.append(hypertrain_xgb_model(X_train, y_train, X_val, y_val, early_stopping_rounds=10))

    # Save models
    model_path = f'models/{model_name}/model_{classification_type}.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    # Optionally interpret models
    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols, classification_type=classification_type, model_name=model_name)

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_xgboost(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                  model_name=model_name)


def evaluate_ensemble_xgboost(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                              model_name='xgb'):
    if 'shap_selected' not in model_name and shap_selected:
        model_name = f'{model_name}_shap_selected'

    checkpoint_file = \
    [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')
     and classification_type in f][0]

    # Load the models from the saved pickle file
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)

    prospective = True
    # # Prospective evaluation
    # print('----- Prospective Evaluation with same distribution as test------')
    # xs_pro_sub, ys_pro_sub, all_sampled_indices = zip(
    #     *[sample_to_same_dist(xs_test[i], ys_test[i], xs_pro[i], ys_pro[i]) for i in range(10)])
    # plot_and_compare_distributions(ys_test[0], ys_pro[0], '')
    # plot_and_compare_distributions(ys_test[0], ys_pro_sub[0], 'downsampled')
    #
    # dfs = []
    # for i in range(10):
    #     dfs.append(pd.read_csv(f'outputs/fib4/prospective_{i}_fib4.csv').iloc[all_sampled_indices[i]])
    #
    # analyze_fib4(dfs, classification_type, 'sub_prospective')
    # evaluate_performance(models, xs_pro_sub, ys_pro_sub, df_cols, model_name, classification_type, prospective,
    #                      shap_selected)

    # Prospective evaluation
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)


def finetune_ensemble_xgb(xs_finetune, ys_finetune, xs_val, ys_val, xs_test, ys_test, xs_umm, ys_umm, df_cols,
                          classification_type, shap_selected, interpret_model=False, testing=True):
    """
    Fine-tunes a saved XGBoost model on new data.

    Parameters:
    xs_finetune (list of np.array): List of feature arrays for fine-tuning.
    ys_finetune (list of np.array): List of target arrays for fine-tuning.
    xs_val (list of np.array): List of feature arrays for validation.
    ys_val (list of np.array): List of target arrays for validation.
    xs_test (list of np.array): List of feature arrays for testing.
    ys_test (list of np.array): List of target arrays for testing.
    xs_umm (list of np.array): List of feature arrays for prospective evaluation.
    ys_umm (list of np.array): List of target arrays for prospective evaluation.
    df_cols (list): List of column names used for feature interpretation.
    classification_type (str): Type of classification (e.g., binary, multi-class).
    shap_selected (bool): Whether SHAP-selected features were used.
    interpret_model (bool): Whether to interpret the model using SHAP.
    testing (bool): Whether to evaluate the model after fine-tuning.
    """
    model_name = 'xgb_shap_selected' if shap_selected else 'xgb'

    # Load the model
    model_path = f'./models/{model_name}/model_{classification_type}.pickle'
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    # Fine-tune models
    for idx, (model, X_finetune, y_finetune, X_val, y_val) in enumerate(zip(models, xs_finetune, ys_finetune, xs_val, ys_val)):
        print(f'Fine-tuning model {idx}')
        model.fit(X_finetune, y_finetune, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True,
                  xgb_model=model)
        models[idx] = model

    # Save fine-tuned models
    finetuned_model_dir = f'./models/{model_name}_finetuned/'
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = f'{finetuned_model_dir}/model_{classification_type}_finetuned.pickle'

    with open(finetuned_model_path, "wb") as f:
        pickle.dump(models, f)

    # Optionally interpret models
    if interpret_model:
        interpret(xs_finetune[0], xs_test[0], df_cols, classification_type=classification_type, model_name=model_name + '_finetuned')

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_xgboost(xs_test, ys_test, xs_umm, ys_umm, df_cols, classification_type, shap_selected, model_name=model_name + '_finetuned')


# def sample_to_same_dist(xs_test, ys_test, xs_pro, ys_pro):
#     # Create DataFrames for easier manipulation
#     df_pro = pd.DataFrame({'xs': list(xs_pro), 'ys': ys_pro})
#     df_test = pd.DataFrame({'xs': list(xs_test), 'ys': ys_test})
#
#     # Get the distribution of ys_test
#     test_distribution = df_test['ys'].value_counts()
#
#     # Subsample df_pro to match the distribution of ys_test
#     subsampled_pro = pd.DataFrame(columns=['xs', 'ys'])
#     all_sampled_indices = []  # To store all sampled indices
#
#     for label, count in test_distribution.items():
#         # Sample indices for the given label
#         indices = df_pro[df_pro['ys'] == label].index
#
#         # Ensure the count does not exceed the available number of indices
#         sample_size = min(count, len(indices))
#         sampled_indices = np.random.choice(indices, size=sample_size, replace=False)
#
#         # Append the sampled indices to the list
#         all_sampled_indices.extend(sampled_indices)
#
#         # Concatenate the sampled data to the subsampled DataFrame
#         subsampled_pro = pd.concat([subsampled_pro, df_pro.loc[sampled_indices]])
#
#     # Reset index for final DataFrame
#     subsampled_pro = subsampled_pro.reset_index(drop=True)
#
#     # Separate subsampled xs and ys
#     xs_subsampled = np.array(subsampled_pro['xs'].tolist())
#     ys_subsampled = np.array(subsampled_pro['ys'].tolist())
#
#     return xs_subsampled, ys_subsampled, np.array(all_sampled_indices)
#
#
# def plot_and_compare_distributions(ys_test, ys_pro, name):
#     unique_ys_test, counts_ys_test = np.unique(ys_test, return_counts=True)
#     unique_ys_mainz_test, counts_ys_mainz_test = np.unique(ys_pro, return_counts=True)
#
#     # Ensure the values are the same for both arrays (for accurate comparison)
#     assert np.array_equal(unique_ys_test, unique_ys_mainz_test), "Arrays have different unique values"
#
#     # Assuming your data is in the form of numpy arrays
#     data = pd.DataFrame({
#         'Value': np.concatenate([unique_ys_test, unique_ys_mainz_test]),
#         'Frequency': np.concatenate([counts_ys_test, counts_ys_mainz_test]),
#         'Category': ['ys_test'] * len(unique_ys_test) + ['ys_mainz_test'] * len(unique_ys_mainz_test)
#     })
#
#     # Create the bar plot
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Value', y='Frequency', hue='Category', data=data, alpha=0.7)
#
#     # Add labels, title, and legend
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.title('Distribution Comparison of ys_test and ys_mainz_test')
#     plt.legend(title='Category')
#
#     # Ensure x-axis ticks are integers
#     plt.xticks(np.arange(int(data['Value'].min()), int(data['Value'].max()) + 1))
#
#     # Save the plot
#     plt.savefig(f'../data/dists_{name}.png')
#
#     # Show plot
#     plt.show()
#
#
# def analyze_fib4(dfs, classification_type='fibrosis', name='test'):
#     """
#     Computes the FIB4 classification performances on the data sets.
#     Args:
#         dfs (pandas.Dataframe): list of dataframes.
#
#     Returns:
#         None
#     """
#     if not os.path.exists(f'outputs/fib4'):
#         os.makedirs(f'outputs/fib4')
#
#     cms = []
#     reports = []
#     for i, df in enumerate(dfs):
#         df.to_csv(f'outputs/fib4/{name}_{i}_fib4.csv')
#     for df in dfs:
#         cms.append(confusion_matrix(df['Micro'], df['Fib4 Stages']))
#         reports.append(
#             classification_report(
#                 df['Micro'],
#                 df['Fib4 Stages'],
#                 output_dict=True))
#
#     plot_cm(cms[0], 'fib4', classification_type=classification_type)
#
#     with open(f'outputs/fib4/fib4_{classification_type}.csv', 'w') as f:
#         for (cm, report) in zip(cms, reports):
#             f.write(str(report))
#             f.write(str(cm))
#
#     acc, f1, ppv, tpr = [], [], [], []
#     for report in reports:
#         acc.append(report['accuracy'])
#         f1.append(report['macro avg']['f1-score'])
#         ppv.append(report['macro avg']['precision'])
#         tpr.append(report['macro avg']['recall'])
#
#     performance_string = f'\nFIB4\n' \
#                          f'Average ACC: {np.round(np.average(acc) * 100, 2)} | Std ACC: {np.round(np.std(acc) * 100, 2)},\n' \
#                          f'Average F1: {np.round(np.average(f1) * 100, 2)} | Std F1: {np.round(np.std(f1) * 100, 2)},\n' \
#                          f'Average PPV: {np.round(np.average(ppv) * 100, 2)} | Std PPV: {np.round(np.std(ppv) * 100, 2)},\n' \
#                          f'Average TPR: {np.round(np.average(tpr) * 100, 2)} | Std TPR: {np.round(np.std(tpr) * 100, 2)}\n'
#     print(performance_string)
#
#     with open(f'outputs/fib4/fib4_{classification_type}_performance_metrics.txt', 'w') as f:
#         f.write(performance_string)
#         f.close()


# def interpret_xgb(x_train, x_test, df_cols, classification_type='fibrosis', model_name='xgb'):
#     """
#     Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.
#
#     Parameters:
#         x_train (array-like): Training data features.
#         x_test (array-like): Test data features.
#         df_cols (list): List of column names (features).
#         classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
#         model_name (str): name of output model
#
#     Returns:
#         None
#     """
#     explainer = shap.KernelExplainer(model=lambda x: predict_xgb_model(x, classification_type=classification_type,
#                                                                        model_name=model_name),
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
# def predict_xgb_model(data, classification_type='fibrosis', model_name='xgb'):
#     """
#     Predictions of the model for certain data. Model is saved in output/models.pickle
#
#     Args:
#         data: A numpy array to predict on.
#         classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
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
#         y_pred = model.predict_proba(data)
#         y_preds.append(y_pred)
#
#     maj_preds = majority_vote(y_preds, rule='soft')
#     indices, _ = get_index_and_proba(maj_preds)
#
#     return np.array(indices)


def hypertrain_xgb_model(x_train, y_train, x_val, y_val, early_stopping_rounds=10):
    """
    Trains a model on the provided features (x_train) and labels (y_train) with early stopping based on validation loss.

    Args:
        x_train (pandas.DataFrame or numpy.ndarray): The features used for training.
        y_train (pandas.Series or numpy.ndarray): The labels used for training.
        x_val (pandas.DataFrame or numpy.ndarray): The features used for validation.
        y_val (pandas.Series or numpy.ndarray): The labels used for validation.
        early_stopping_rounds (int): Number of rounds to wait for validation loss to improve before stopping.

    Returns:
        sklearn model: The trained model.
    """

    """Te2Rules Explainer of a single trained XGB"""

    # # ---------------------------------------------------------
    #
    # # model = xgb.XGBClassifier(tree_method='hist', max_depth=1, learning_rate=0.3775, subsample=0.3)
    # model = xgb.XGBClassifier(tree_method='hist', max_depth=2, learning_rate=0.01, subsample=0.3)
    # model.fit(x_train, y_train)
    #
    # accuracy = model.score(x_val, y_val)
    #
    # print("Accuracy")
    # print(accuracy)
    #
    # model_explainer = ModelExplainer(
    #     model=model,
    #     feature_names=['Thrombozyten', 'MCV', 'INR']
    # )
    #
    # rules = model_explainer.explain(
    #     X=x_train, y=y_train,
    #     num_stages=1,  # stages can be between 1 and max_depth
    #     min_precision=0.80,  # higher min_precision can result in rules with more terms overfit on training data
    #     jaccard_threshold=0.8  # lower jaccard_threshold speeds up the rule exploration, but can miss some good rules
    # )
    #
    # print(str(len(rules)) + " rules found:")
    # print()
    # for i in range(len(rules)):
    #     print("Rule " + str(i) + ": " + str(rules[i]))
    #
    # fidelity, positive_fidelity, negative_fidelity = model_explainer.get_fidelity()
    #
    # print("The rules explain " + str(round(fidelity * 100, 2)) + "% of the overall predictions of the model")
    # print("The rules explain " + str(round(positive_fidelity * 100, 2)) + "% of the positive predictions of the model")
    # print("The rules explain " + str(round(negative_fidelity * 100, 2)) + "% of the negative predictions of the model")
    #
    # breakpoint()
    # # ---------------------------------------------------------

    model = xgb.XGBClassifier(tree_method='hist', early_stopping_rounds=early_stopping_rounds)

    hp_space = {
        'max_depth': np.arange(1, 40),
        'learning_rate': np.linspace(0.5, 0.01, 5),
        'subsample': np.linspace(1, 0.3, 5)
    }

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=hp_space,
        scoring='neg_log_loss',
        cv=5,
        random_state=42,
    )

    clf.fit(x_train, y_train, sample_weight=classes_weights, eval_set=[(x_val, y_val)], verbose=0)

    print()
    print(f'max_depth: {clf.best_estimator_.max_depth}')
    print(f'learning_rate: {clf.best_estimator_.learning_rate}')
    print(f'subsample: {clf.best_estimator_.subsample}')
    print()

    return clf.best_estimator_
