from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret

from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import os


def hypertrain_ensemble_gandalf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                                classification_type, shap_selected, interpret_model=True, testing=True):
    models = []
    models_params = []
    model_name = 'gandalf_shap_selected' if shap_selected else 'gandalf'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Save df_cols for future reference
    with open(f'./models/{model_name}/df_cols.txt', 'w') as txt_f:
        txt_f.write(str(df_cols))

    # Train models
    for idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(zip(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test)):
        print(f'Training model {idx}')
        best_model, model_params = hypertrain_gandalf_model(X_train, y_train, X_val, y_val, df_cols)
        models.append(best_model)
        models_params.append(model_params)

        # Save model
        model_path = f"./models/{model_name}/model_{classification_type}_{idx}.pth"
        best_model.save_model(model_path)

        # Save model parameters
        with open(f'./models/{model_name}/model_params_{classification_type}_{idx}.txt', 'w') as txt_f:
            txt_f.write(str(model_params))

    # Optionally interpret models
    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols, classification_type=classification_type, model_name=model_name)

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_gandalf(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected)


def evaluate_ensemble_gandalf(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                              model_name='gandalf'):
    models = []
    model_name = f'{model_name}_shap_selected' if shap_selected else model_name

    checkpoint_files = [f for f in os.listdir(f"./models/{model_name}/") if
                        f.endswith('.pth') and 'model' in f and classification_type in f]
    for checkpoint_file in checkpoint_files:
        model = TabularModel.load_model(f'models/{model_name}/{checkpoint_file}')
        models.append(model)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)

    # Prospective evaluation
    prospective = True
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)



def hypertrain_gandalf_model(x_train, y_train, x_val, y_val, df_cols, cv=5, n_iter=30, classification_type='fibrosis'):
    """
    Hyperparameter tuning and training a GANDALF model on the provided features (x_train) and labels (y_train) using random search.

    Args:
        x_train (np.array): The features used for training.
        y_train (np.array): The labels used for training.
        x_val (np.array): The features used for validation.
        y_val (np.array): The labels used for validation.
        cv (int): Number of cross-validation folds (default=3).
        n_iter (int): Number of parameter settings that are sampled (default=10).
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'


    Returns:
        GANDALF Model: Best trained model found during random search.
    """
    # Merge train and validation data
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    best_model = None
    best_params = None
    best_accuracy = 0

    hp_space = {
        'learning_rate': np.linspace(0.01, 0.0001, 5),
        'gflu_dropout': np.linspace(0.5, 0.0, 5),
        'gflu_stages': np.arange(2, 7)
    }

    # Initialize cross-validation splitter
    kf = KFold(n_splits=cv)

    # sklearn KFold does not return same length of fold x and fold y if x.shape[0] % cv != 0 !
    # get the remainder
    b = x.shape[0] % cv
    # drop the remainder samples
    x = x[:-1 * b]
    y = y[:-1 * b]

    for _ in range(n_iter):
        # Set seed
        np.random.seed(42)
        # Sample hyperparameters from the search space
        sampled_params = {param: np.random.choice(values) for param, values in hp_space.items()}

        # Perform cross-validation
        scores = []

        for train_index, val_index in kf.split(x):
            x_train_fold, x_val_fold = x[train_index], x[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            train_data = pd.DataFrame(data=x_train_fold, columns=df_cols)
            train_data['target'] = y_train_fold

            validation_data = pd.DataFrame(data=x_val_fold, columns=df_cols)
            validation_data['target'] = y_val_fold

            trainer_config = TrainerConfig(
                auto_lr_find=True,
                batch_size=32,
                max_epochs=100,
            )
            optimizer_config = OptimizerConfig()

            data_config = DataConfig(
                target=[
                    "target"
                ],
                continuous_cols=df_cols,
                categorical_cols=[],
            )

            model_config = GANDALFConfig(
                task="classification",
                gflu_feature_init_sparsity=0.3,
                learning_rate=float(sampled_params["learning_rate"]),
                gflu_stages=int(sampled_params["gflu_stages"]),
                gflu_dropout=float(sampled_params["gflu_dropout"]),
                # num_classes=3 if classification_type == 'three_stage' else 2
            )

            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )
            tabular_model.fit(train=train_data, validation=validation_data)
            pred_df = tabular_model.predict(validation_data)

            predicted_labels = pred_df['prediction']
            actual_labels = validation_data['target']

            # Calculate accuracy
            accuracy = accuracy_score(actual_labels, predicted_labels)

            scores.append(accuracy)

        # Calculate average score across folds
        avg_score = np.mean(scores)

        # Save the best model
        if avg_score > best_accuracy:
            best_accuracy = avg_score
            best_model = tabular_model
            best_params = sampled_params

        return best_model, best_params


# def make_ensemble_preds_gandalf(x_test, y_test, df_cols, models, classification_type='fibrosis'):
#     """
#     Trains m models individually on m data sets
#     Args:
#         x_test (list): List of Test matrices.
#         y_test (list): List of Test labels.
#         df_cols (list): List of column names
#         models (list): List of m PyTorch models.
#         classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'
#
#     Returns:
#         list: A list containing dictionaries of classification reports.
#         list: A list of m confusion matrices.
#         list: A list of predictions of m data sets.
#         list: A list of probabilities of m data sets.
#     """
#
#     ensemble_cms = []
#     ensemble_reports = []
#     ensemble_preds = []  # stores the index of predicted class
#     ensemble_probas = []  # stores the probability of the predicted class
#     ensemble_pred_probas = []  # stores the probability of both classes
#
#     for (X, y) in zip(x_test, y_test):
#
#         test_data = pd.DataFrame(data=X, columns=df_cols)
#         test_data['target'] = y
#
#         y_preds = []
#         for model in models:
#             # Perform forward pass to get probabilities
#             probas_df = model.predict(test_data)
#
#             # Convert to list of lists containing class proababilities
#             if classification_type == 'three_stage':
#                 y_preds.append(probas_df[['0_probability', '1_probability', '2_probability']].values.tolist())
#             else:
#                 y_preds.append(probas_df[['0_probability', '1_probability']].values.tolist())
#
#         # Take majority vote or soft voting
#         ensemble_pred = majority_vote(y_preds, rule='soft')
#
#         ensemble_pred_probas.append(ensemble_pred)
#
#         ensemble_pred, probas = get_index_and_proba(ensemble_pred)
#
#         # Convert predicted labels to numpy array
#         ensemble_pred = np.array(ensemble_pred)
#
#         # Calculate classification report and confusion matrix
#         maj_report = classification_report(y, ensemble_pred, output_dict=True)
#         cm = confusion_matrix(y, ensemble_pred)
#
#         ensemble_reports.append(maj_report)
#         ensemble_cms.append(cm)
#         ensemble_preds.append(ensemble_pred)
#         ensemble_probas.append(probas)
#
#     return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas
#
#
# def predict_gandalf_models(data, classification_type='fibrosis'):
#     """
#     Predictions of the GANDALF model for certain data. Models are saved as checkpoint files in a directory.
#
#     Args:
#         data: A numpy array to predict on.
#         classification_type (str): 'fibrosis' or 'cirrhosis'.
#     Returns:
#         A numpy array of class predictions
#     """
#
#     # Get a list of all checkpoint files in the directory
#     checkpoint_files = [f for f in os.listdir(f"./models/gandalf/") if f.endswith('.pth') and 'model' in f and classification_type in f]
#
#     # Load each model checkpoint and make predictions
#     y_preds = []
#
#     # Load the df cols
#     with open(f'./models/gandalf/df_cols.txt', 'r') as file:
#         df_cols_string = file.read()
#
#     # Convert the string to a list using ast.literal_eval
#     df_cols = ast.literal_eval(df_cols_string)
#
#     for checkpoint_file in checkpoint_files:
#         model = TabularModel.load_model(f'models/gandalf/{checkpoint_file}')
#         test_data = pd.DataFrame(data=data, columns=df_cols)
#         probas_df = model.predict(test_data)
#         # Convert to list of lists containing class proababilities
#         if classification_type == 'three_stage':
#             y_preds.append(probas_df[['0_probability', '1_probability', '2_probability']].values.tolist())
#         else:
#             y_preds.append(probas_df[['0_probability', '1_probability']].values.tolist())
#
#     # Perform majority voting
#     maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
#     indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function
#
#     return np.array(indices)
#
#
# def interpret_gandalf_model(x_train, x_test, df_cols, classification_type='fibrosis'):
#     """
#     Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.
#
#     Parameters:
#         x_train (array-like): Training data features.
#         x_test (array-like): Test data features.
#         df_cols (list): List of column names (features).
#         classification_type (str): 'fibrosis' or 'cirrhosis'.
#
#     Returns:
#         None
#     """
#     explainer = shap.KernelExplainer(model=lambda x: predict_gandalf_models(x, classification_type=classification_type), data=shap.sample(x_train, 50), feature_names=df_cols)
#
#     shap_values = explainer.shap_values(x_test)
#
#     f = shap.force_plot(
#         explainer.expected_value,
#         shap_values,
#         x_test,
#         feature_names=df_cols,
#         show=False)
#     shap.save_html(f'outputs/gandalf/{classification_type}_force_plot.htm', f)
#     plt.close()
#
#     fig, ax = plt.subplots()
#     shap_values2 = explainer(x_test)
#     shap.plots.bar(shap_values2, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/gandalf/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     fig, ax = plt.subplots()
#
#     shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/gandalf/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
#     plt.close()

