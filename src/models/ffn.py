from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret, load_pytorch_model
from src.utils.networks import NeuralNetwork
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def hypertrain_ensemble_ffn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                            classification_type, shap_selected, interpret_model=True, testing=True):
    models = []
    models_params = []
    model_name = 'ffn_shap_selected' if shap_selected else 'ffn'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Train models
    for idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(zip(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test)):
        print(f'Training model {idx}')
        best_model, model_params = hypertrain_pytorch_model(X_train, y_train, X_val, y_val, classification_type=classification_type)
        models.append(best_model)
        models_params.append(model_params)

        # Save model state dict
        model_path = f"./models/{model_name}/model_{classification_type}_{idx}.pth"
        torch.save(best_model.state_dict(), model_path)

        # Save model parameters
        with open(f'./models/{model_name}/model_params_{classification_type}_{idx}.txt', 'w') as txt_f:
            txt_f.write(str(model_params))

    # Optionally interpret models
    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols, classification_type=classification_type, model_name=model_name)

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_ffn(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected)


def evaluate_ensemble_ffn(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                          model_name='ffn'):
    model_name = f'{model_name}_shap_selected' if shap_selected else model_name

    models = []
    checkpoint_files = [f for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pth') and classification_type in f]
    for checkpoint_file in checkpoint_files:
        model = load_pytorch_model(checkpoint_file, classification_type, model_name)
        models.append(model)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)

    # Prospective evaluation
    prospective = True
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)


def hypertrain_pytorch_model(x_train, y_train, x_val, y_val, cv=5, n_iter=10, max_epochs=100, early_stopping_rounds=30,
                             classification_type='fibrosis'):
    """
    Hyperparameter tuning and training a PyTorch Lightning model on the provided features (x_train) and labels (y_train) using random search.

    Args:
        x_train (np.array): The features used for training.
        y_train (np.array): The labels used for training.
        x_val (np.array): The features used for validation.
        y_val (np.array): The labels used for validation.
        x_test (np.array): The features used for testing.
        y_test (np.array): The labels used for testing.
        scoring (str): Scoring metric for evaluation (default='neg_log_loss').
        cv (int): Number of cross-validation folds (default=3).
        n_iter (int): Number of parameter settings that are sampled (default=10).
        max_epochs (int): Maximum number of epochs to train (default=100).
        early_stopping_rounds (int): Number of epochs to wait for validation loss to improve before stopping early (default=10).
        classification_type (str): 'fibrosis', 'cirrhosis' or 'three_stage'


    Returns:
        PyTorchModel: Best trained model found during random search.
    """
    # Merge train and validation data
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    best_model = None
    best_params = None
    best_score = float('-inf')

    hp_space = {
        'lr': np.linspace(0.005, 0.0005, 5),
        'dropout': np.linspace(0.3, 0.0, 5),
        'num_layers': np.arange(3, 7),
        'hidden_dim': np.array([32, 64, 128, 256])
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
        sampled_params['input_dim'] = x_train.shape[1]
        sampled_params['num_classes'] = 3 if classification_type == 'three_stage' else 2

        # Perform cross-validation
        scores = []

        # sklearn KFold does not return same length of fold x and fold y if x.shape[0] % cv != 0 !
        for train_index, val_index in kf.split(x):
            x_train_fold, x_val_fold = x[train_index], x[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Convert fold data to PyTorch tensors
            fold_train_dataset = TensorDataset(torch.tensor(x_train_fold), torch.tensor(y_train_fold))
            fold_val_dataset = TensorDataset(torch.tensor(x_val_fold), torch.tensor(y_val_fold))
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True)
            fold_val_loader = DataLoader(fold_val_dataset, batch_size=32)

            # Create model instance with sampled hyperparameters
            model = NeuralNetwork(**sampled_params, classification_type=classification_type)

            # Initialize Lightning Callbacks
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, mode='min')

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                callbacks=[early_stop_callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,  # Disable the progress bar
                log_every_n_steps=0,  # Disable logging during training steps
                enable_model_summary=False  # Disable model summary printing)
            )

            # Train the model
            trainer.fit(model, fold_train_loader, fold_val_loader)

            # Evaluate the model
            val_acc = trainer.test(model, dataloaders=[fold_val_loader])[0]['test_acc']

            # Update scores based on scoring metric
            scores.append(val_acc)

        # Calculate average score across folds
        avg_score = np.mean(scores)

        # Update best model if the score is better
        if avg_score > best_score:
            print(f'\n best model has acc: {avg_score}')
            best_score = avg_score
            best_model = model
            best_params = sampled_params

    return best_model, best_params


# def make_ensemble_preds_pytorch(xs_test, ys_test, models):
#     """
#     Trains m models individually on m data sets
#     Args:
#         xs_test (list): List of Test matrices.
#         ys_test (list): List of Test labels.
#         models (list): List of m PyTorch models.
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
#     for (X, y) in zip(xs_test, ys_test):
#         y_preds = []
#         for model in models:
#             model.eval()  # Set model to evaluation mode
#             with torch.no_grad():
#                 # Convert X to torch tensor if it's not already
#                 if not torch.is_tensor(X):
#                     X = torch.tensor(X)
#                 # Perform forward pass to get probabilities
#                 probas = model(X.float())  # Assuming input is float
#                 y_preds.append(probas.numpy())
#
#         # Take majority vote - hard or soft voting
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
# def predict_pytorch_models(data, classification_type='fibrosis'):
#     """
#     Predictions of the model for certain data. PyTorch models are saved as checkpoint files in a directory.
#
#     Args:
#         model_directory: Directory where the PyTorch model checkpoints are stored.
#         data: A numpy array to predict on.
#     Returns:
#         A numpy array of class predictions
#     """
#
#     # Get a list of all checkpoint files in the directory
#     checkpoint_files = [f for f in os.listdir("./models/ffn/") if f.endswith('.pth') and classification_type in f]
#
#     # Load each model checkpoint and make predictions
#     y_preds = []
#     for checkpoint_file in checkpoint_files:
#         # Read the respective hyperparam file
#         model_index = checkpoint_file.split('_')[-1].rstrip('.pth')
#         with open(f"./models/ffn/model_params_{classification_type}_{model_index}.txt", "r") as file:
#             # Read the first line
#             dict_str = file.readline().strip()
#
#             # Convert the string representation of the dictionary to a Python dictionary
#             param_dict = eval(dict_str)
#
#         # Load the model checkpoint
#         model = NeuralNetwork(**param_dict)  # Replace YourModelClass with your actual model class
#         model.load_state_dict(torch.load(os.path.join("./models/ffn/", checkpoint_file)))
#         model.eval()
#
#         # Make predictions
#         with torch.no_grad():
#             inputs = torch.tensor(data, dtype=torch.float)  # Assuming data is a numpy array
#             outputs = model(inputs)
#             y_pred = torch.softmax(outputs, dim=1).numpy()  # Assuming output is probability distribution
#             y_preds.append(y_pred)
#
#     # Perform majority voting
#     maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
#     indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function
#
#     return np.array(indices)
#
#
# def interpret_pytorch_model(x_train, x_test, df_cols, classification_type='fibrosis'):
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
#     explainer = shap.KernelExplainer(model=lambda x: predict_pytorch_models(x, classification_type=classification_type), data=shap.sample(x_train, 50), feature_names=df_cols)
#
#     shap_values = explainer.shap_values(x_test)
#
#     f = shap.force_plot(
#         explainer.expected_value,
#         shap_values,
#         x_test,
#         feature_names=df_cols,
#         show=False)
#     shap.save_html(f'outputs/ffn/{classification_type}_force_plot.htm', f)
#     plt.close()
#
#     fig, ax = plt.subplots()
#     shap_values2 = explainer(x_test)
#     shap.plots.bar(shap_values2, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/ffn/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     fig, ax = plt.subplots()
#
#     shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/ffn/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
#     plt.close()


def predict_pytorch_models(data, classification_type='fibrosis', model_name='ffn'):
    """
    Predictions of the model for certain data. PyTorch models are saved as checkpoint files in a directory.

    Args:
        model_directory: Directory where the PyTorch model checkpoints are stored.
        data: A numpy array to predict on.
    Returns:
        A numpy array of class predictions
    """

    # Get a list of all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pth') and classification_type in f]

    # Load each model checkpoint and make predictions
    y_preds = []
    for checkpoint_file in checkpoint_files:
        model = load_pytorch_model(checkpoint_file, classification_type, model_name)

        # Make predictions
        with torch.no_grad():
            inputs = torch.tensor(data, dtype=torch.float)  # Assuming data is a numpy array
            outputs = model(inputs)
            y_pred = torch.softmax(outputs, dim=1).numpy()  # Assuming output is probability distribution
            y_preds.append(y_pred)

    # Perform majority voting
    maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
    indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function

    return np.array(indices)
