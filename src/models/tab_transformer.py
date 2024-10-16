from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret, load_pytorch_model
from src.utils.networks import PLTabTransformer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import os


def hypertrain_ensemble_tab_transformer(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                                        classification_type, shap_selected, interpret_model=False, testing=False):
    models = []
    models_params = []
    model_name = 'tab_transformer_shap_selected' if shap_selected else 'tab_transformer'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Save df_cols for future reference
    with open(f'./models/{model_name}/{classification_type}_df_cols.txt', 'w') as txt_f:
        txt_f.write(str(df_cols))

    # Train models
    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        best_model, model_params = hypertrain_tab_transformer_model(X_train, y_train, X_val, y_val, df_cols,
                                                                    classification_type=classification_type)
        models.append(best_model)
        models_params.append(model_params)

        # Save model
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
        evaluate_ensemble_tab_transformer(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected)


def evaluate_ensemble_tab_transformer(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                      model_name='tab_transformer'):
    models = []
    model_name = f'{model_name}_shap_selected' if shap_selected else model_name

    checkpoint_files = [f for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pth') and classification_type in f]
    for checkpoint_file in checkpoint_files:
        model = load_pytorch_model(checkpoint_file, classification_type, model_name, df_cols=df_cols)
        models.append(model)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)

    # Prospective evaluation
    prospective = True
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)


def finetune_ensemble_tab_transformer(xs_finetune, ys_finetune, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro,
                                      df_cols,
                                      classification_type, shap_selected, interpret_model=True, testing=False):
    """
    Fine-tunes a saved ensemble of TabTransformer models on new data.

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
    models = []
    model_name = 'tab_transformer_shap_selected' if shap_selected else 'tab_transformer'

    checkpoint_files = [f for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pth') and classification_type in f]

    # Load each model checkpoint and make predictions
    y_preds = []
    for checkpoint_file in checkpoint_files:
        # Read the respective hyperparam file
        model_index = checkpoint_file.split('_')[-1].rstrip('.pth')
        with open(f"./models/tab_transformer/model_params_{classification_type}_{model_index}.txt", "r") as file:
            # Read the first line
            dict_str = file.readline().strip()

            # Convert the string representation of the dictionary to a Python dictionary
            param_dict = eval(dict_str)

        # Load the model checkpoint
        model = PLTabTransformer(**param_dict,
                                 df_cols=df_cols)  # Replace YourModelClass with your actual model class
        model.load_state_dict(torch.load(os.path.join("./models/tab_transformer/", checkpoint_file)))
        model.train()

        # Fine-tune models
    for idx, (model, X_finetune, y_finetune, X_val, y_val) in enumerate(
            zip(models, xs_finetune, ys_finetune, xs_val, ys_val)):
        print(f'Fine-tuning model {idx}')
        finetune_tab_transformer_model(model, X_finetune, y_finetune, X_val, y_val, df_cols,
                                       classification_type=classification_type)

        # Save fine-tuned model
        finetuned_model_dir = f'./models/{model_name}_finetuned/'
        os.makedirs(finetuned_model_dir, exist_ok=True)
        model_path = f"{finetuned_model_dir}/model_{classification_type}_{idx}.pth"
        torch.save(model.state_dict(), model_path)

        # Save model parameters
        with open(f'{finetuned_model_dir}/model_params_{classification_type}_{idx}.txt', 'w') as txt_f:
            txt_f.write(str(model_params[idx]))

    print('------ Finished Fine-Tuning Ensemble ------')

    # Optionally interpret models
    if interpret_model:
        interpret(xs_finetune[0], xs_test[0], df_cols, classification_type=classification_type,
                  model_name=model_name + '_finetuned')

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_tab_transformer(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                          model_name=model_name + '_finetuned')


def hypertrain_tab_transformer_model(x_train, y_train, x_val, y_val, df_cols, scoring='neg_log_loss',
                                     max_epochs=200, cv=5, n_iter=30, early_stopping_rounds=30,
                                     classification_type='fibrosis'):
    """
    Hyperparameter tuning and training a tab_transformer model on the provided features (x_train) and labels (y_train) using random search.

    Args:
        x_train (np.array): The features used for training.
        y_train (np.array): The labels used for training.
        x_val (np.array): The features used for validation.
        y_val (np.array): The labels used for validation.
        x_test (np.array): The features used for testing.
        y_test (np.array): The labels used for testing.
        df_cols (list): list of column names
        scoring (str): Scoring metric for evaluation (default='neg_log_loss').
        cv (int): Number of cross-validation folds (default=3).
        n_iter (int): Number of parameter settings that are sampled (default=10).
        max_epochs (int): Maximum number of epochs to train (default=100).
        early_stopping_rounds (int): Number of epochs to wait for validation loss to improve before stopping early (default=10).
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'

    Returns:
        tab_transformer Model: Best trained model found during random search.
    """
    # Merge train and validation data
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    best_model = None
    best_params = None
    best_score = float('-inf') if scoring == 'neg_log_loss' else float('inf')

    hp_space = {
        'lr': np.linspace(0.005, 0.0001, 5),
        'dropout': np.linspace(0.5, 0.0, 5),
        'num_layers': np.arange(2, 7),
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

    # Loop for hyperparameter search
    for _ in range(n_iter):
        # Set seed
        np.random.seed(42)
        # Sample hyperparameters from the search space
        sampled_params = {param: np.random.choice(values) for param, values in hp_space.items()}
        sampled_params['out_dim'] = 3 if classification_type == 'three_stage' else 2

        # Perform cross-validation
        scores = []

        for train_index, val_index in kf.split(x):
            x_train_fold, x_val_fold = torch.tensor(x[train_index], device=get_device(i=0)), torch.tensor(x[val_index], device=get_device(i=0))
            y_train_fold, y_val_fold = torch.tensor(y[train_index], device=get_device(i=0)), torch.tensor(y[val_index], device=get_device(i=0))

            fold_train_dataset = TensorDataset(torch.tensor(x_train_fold), torch.tensor(y_train_fold))
            fold_val_dataset = TensorDataset(torch.tensor(x_val_fold), torch.tensor(y_val_fold))
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True)
            fold_val_loader = DataLoader(fold_val_dataset, batch_size=32)

            model = PLTabTransformer(**sampled_params, df_cols=df_cols, classification_type=classification_type)

            # Initialize Lightning Callbacks
            early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_rounds,
                                                             mode='min')

            # Initialize Lightning Trainer
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                callbacks=[early_stop_callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,  # Disable the progress bar
                log_every_n_steps=0,  # Disable logging during training steps
                enable_model_summary=False  # Disable model summary printing
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


def make_ensemble_preds_tab_transformer(x_test, y_test, df_cols, models):
    """
    Trains m models individually on m data sets
    Args:
        x_test (list): List of Test matrices.
        y_test (list): List of Test labels.
        models (list): List of m PyTorch models.

    Returns:
        list: A list containing dictionaries of classification reports.
        list: A list of m confusion matrices.
        list: A list of predictions of m data sets.
        list: A list of probabilities of m data sets.
    """

    ensemble_cms = []
    ensemble_reports = []
    ensemble_preds = []  # stores the index of predicted class
    ensemble_probas = []  # stores the probability of the predicted class
    ensemble_pred_probas = []  # stores the probability of both classes

    for (X, y) in zip(x_test, y_test):
        y_preds = []
        for model in models:
            # Perform forward pass to get probabilities
            probas = model(torch.tensor(X, dtype=torch.float32))

            # Convert to list of lists containing class proababilities
            y_preds.append(probas.detach().cpu().numpy())

        # Take majority vote or soft voting
        ensemble_pred = majority_vote(y_preds, rule='soft')

        ensemble_pred_probas.append(ensemble_pred)

        ensemble_pred, probas = get_index_and_proba(ensemble_pred)

        # Convert predicted labels to numpy array
        ensemble_pred = np.array(ensemble_pred)

        # Calculate classification report and confusion matrix
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)

        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred)
        ensemble_probas.append(probas)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas
