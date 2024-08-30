from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from src.utils.networks import VI_BNN
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import torch.optim as optim


def hypertrain_ensemble_vi_bnn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                               classification_type, shap_selected, interpret_model=True, testing=False):
    models = []
    models_params = []
    model_name = 'vi_bnn_shap_selected' if shap_selected else 'vi_bnn'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Train models
    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        best_model, model_params = hypertrain_vi_bnn_model(X_train, y_train, X_val, y_val,
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
        evaluate_ensemble_vi_bnn(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                                 model_name)


def evaluate_ensemble_vi_bnn(xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type, shap_selected,
                             model_name='vi_bnn'):
    models = []
    model_name = f'{model_name}_shap_selected' if shap_selected else model_name
    checkpoint_files = [f for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pth') and classification_type in f]

    # Load each model checkpoint and make predictions
    for checkpoint_file in checkpoint_files:
        # Read the respective hyperparam file
        model_index = checkpoint_file.split('_')[-1].rstrip('.pth')
        with open(f"./models/{model_name}/model_params_{classification_type}_{model_index}.txt", "r") as file:
            # Read the first line
            dict_str = file.readline().strip()

            # Convert the string representation of the dictionary to a Python dictionary
            param_dict = eval(dict_str)

        # Load the model checkpoint
        model = VI_BNN(**param_dict, prior_var=1.0).to(get_device(i=0))

        # load the model here
        model.load_state_dict(torch.load(os.path.join(f"./models/{model_name}/", checkpoint_file)))
        model.eval()  # Set the model to evaluation mode
        models.append(model)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective)

    # Prospective evaluation
    prospective = True
    print('----- Prospective Evaluation ------')
    evaluate_performance(models, xs_pro, ys_pro, df_cols, model_name, classification_type, prospective)


def hypertrain_vi_bnn_model(x_train, y_train, x_val, y_val, cv=1, n_iter=5, max_epochs=100, early_stopping_rounds=10,
                            samples=100, classification_type='fibrosis'):
    """
    Hyperparameter tuning and training a PyTorch Lightning model on the provided features (x_train) and labels (y_train) using random search.

    Args:
        x_train (np.array): The features used for training.
        y_train (np.array): The labels used for training.
        x_val (np.array): The features used for validation.
        y_val (np.array): The labels used for validation.
        cv (int): Number of cross-validation folds (default=3).
        n_iter (int): Number of parameter settings that are sampled (default=10).
        max_epochs (int): Maximum number of epochs to train (default=100).
        early_stopping_rounds (int): Number of epochs to wait for validation loss to improve before stopping early (default=10).
        samples (int): Number of BNN samples (default=100).
        classification_type (str): 'fibrosis', 'cirrhosis' or 'three_stage'

    Returns:
        PyTorchModel: Best trained model found during random search.
    """
    # # Merge train and validation data
    # x = np.concatenate((x_train, x_val), axis=0)
    # y = np.concatenate((y_train, y_val), axis=0)

    best_model = None
    best_params = None
    best_score = float('-inf')

    hp_space = {
        'lr': np.linspace(0.01, 0.001, 2),
        'dropout': np.linspace(0.2, 0.0, 2),
        'num_layers': np.arange(1, 3),
        'hidden_dim': np.array([32, 64])
    }

    # # Initialize cross-validation splitter
    # kf = KFold(n_splits=cv)
    #
    # # sklearn KFold does not return same length of fold x and fold y if x.shape[0] % cv != 0 !
    # # get the remainder
    # b = x.shape[0] % cv
    # # drop the remainder samples
    # x = x[:-1 * b]
    # y = y[:-1 * b]

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
        # for train_index, val_index in kf.split(x):
        #     x_train_fold, x_val_fold = x[train_index], x[val_index]
        #     y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Convert fold data to PyTorch tensors
            # fold_train_dataset = TensorDataset(torch.tensor(x_train_fold), torch.tensor(y_train_fold))
            # fold_val_dataset = TensorDataset(torch.tensor(x_val_fold), torch.tensor(y_val_fold))

        # Starting from here everything is indented --------------------------

        fold_train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        fold_val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=64, shuffle=True)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=64)

        # Create model instance with sampled hyperparameters
        model = VI_BNN(**sampled_params)
        epochs = max_epochs
        optimizer = optim.Adam(model.parameters(), lr=sampled_params['lr'], weight_decay=1e-7)
        model = model.to(get_device(i=0))

        best_val_loss = float('inf')
        patience = early_stopping_rounds  # Number of epochs to wait for improvement
        counter = 0

        # Training loop
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            total_loss = 0
            progress_bar_train = tqdm(enumerate(fold_train_loader), total=len(fold_train_loader), desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
            for step_train, (x_batch, y_batch) in progress_bar_train:
                x_batch, y_batch = x_batch.to(get_device(i=0)), y_batch.to(get_device(i=0))
                optimizer.zero_grad()
                loss = model.sample_elbo(x_batch.to(torch.float32), y_batch, samples)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar_train.set_postfix({'Train Loss': total_loss / (step_train + 1)})
            avg_train_loss = total_loss / len(fold_train_loader)

            # Validation
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_loss = 0
                progress_bar_val = tqdm(enumerate(fold_val_loader), total=len(fold_val_loader), desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
                for step_val, (x_val_batch, y_val_batch) in progress_bar_val:
                    x_val_batch, y_val_batch = x_val_batch.to(get_device(i=0)), y_val_batch.to(get_device(i=0))
                    val_loss += model.sample_elbo(x_val_batch.to(torch.float32), y_val_batch, samples).item()
                    progress_bar_val.set_postfix({'Validation Loss': val_loss / (step_val + 1)})
                avg_val_loss = val_loss / len(fold_val_loader)

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                    break

        # Set model to evaluation mode
        model.eval()

        # Iterate through the val loader
        with torch.no_grad():
            for x_val, y_val in fold_val_loader:
                x_val, y_val = x_val.to(get_device(i=0)), y_val.to(get_device(i=0))
                outputs = model(x_val.to(torch.float32))
                preds = torch.argmax(outputs, dim=1)
                if classification_type == 'three_stage':
                    accuracy = Accuracy(task='multiclass', num_classes=3).to(get_device(i=0))
                else:
                    accuracy = BinaryAccuracy().to(get_device(i=0))
                test_acc = accuracy(preds, y_val)

        # Update scores based on scoring metric
        scores.append(test_acc.detach().cpu().numpy())

        # Up until here everything is indented --------------------------

        # Calculate average score across folds
        avg_score = np.mean(scores)

        # Update best model if the score is better
        if avg_score > best_score:
            print(f'\n best model has acc: {avg_score}')
            best_score = avg_score
            best_model = model
            best_params = sampled_params

        del model

    return best_model, best_params
