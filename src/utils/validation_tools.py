from src.utils.helper_functions import *
from src.utils.networks import PLTabTransformer, NeuralNetwork, VI_BNN
import shap
from pytorch_tabular import TabularModel
import ast
from sklearn.model_selection import StratifiedKFold


def evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective):
    # Ensemble Prediction
    if any(substring in model_name for substring in ['svm', 'rf', 'xgb', 'light_gbm']):
        ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas = \
            make_ensemble_preds(xs_test, ys_test, models)
    elif 'ffn' in model_name:
        ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas = \
            make_ensemble_preds_pytorch(xs_test, ys_test, models)
    elif 'gandalf' in model_name:
        ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas = \
            make_ensemble_preds_gandalf(xs_test, ys_test, df_cols, models)
    elif 'tab_transformer' in model_name:
        ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas = \
            make_ensemble_preds_tab_transformer(xs_test, ys_test, models)
    elif 'vi_bnn' in model_name:
        ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas = \
            make_ensemble_preds_vi_bnn(xs_test, ys_test, models)

    plot_accuracies(ensemble_reports, 'Ensemble', name=f'{model_name}', prospective=prospective,
                    classification_type=classification_type)

    plot_combined_roc_curve(ys_test, ensemble_pred_probas, f'{model_name}', prospective=prospective,
                            classification_type=classification_type)  # Plot combined ROC and show AUC

    plot_cm(ensemble_cms[0], name=f'{model_name}', prospective=prospective, classification_type=classification_type)

    r, _ = test(y=ys_test[0], y_pred=ensemble_preds[0])
    print('Results on one data set: ', r)

    if prospective is True:
        with open(f'outputs/{model_name}/prospective/{model_name}_{classification_type}_ensemble_preds.txt', 'w') as f:
            for model_ensemble_pred in ensemble_preds:
                f.write(str(model_ensemble_pred) + '\n')

        with open(f'outputs/{model_name}/prospective/model_{classification_type}.csv', 'w') as f:
            for (cm, report) in zip(ensemble_cms, ensemble_reports):
                f.write(str(report))
                f.write(str(cm))
    else:
        with open(f'outputs/{model_name}/{model_name}_{classification_type}_ensemble_preds.txt', 'w') as f:
            for model_ensemble_pred in ensemble_preds:
                f.write(str(model_ensemble_pred) + '\n')

        with open(f'outputs/{model_name}/model_{classification_type}.csv', 'w') as f:
            for (cm, report) in zip(ensemble_cms, ensemble_reports):
                f.write(str(report))
                f.write(str(cm))

    acc, f1, ppv, tpr = [], [], [], []
    for report in ensemble_reports:
        acc.append(report['accuracy'])
        f1.append(report['macro avg']['f1-score'])
        ppv.append(report['macro avg']['precision'])
        tpr.append(report['macro avg']['recall'])

    performance_string = f'\n Ensemble {model_name} \n' \
                         f'Average ACC: {np.round(np.average(acc) * 100, 2)} | Std ACC: {np.round(np.std(acc) * 100, 2)},\n' \
                         f'Average F1: {np.round(np.average(f1) * 100, 2)} | Std F1: {np.round(np.std(f1) * 100, 2)},\n' \
                         f'Average PPV: {np.round(np.average(ppv) * 100, 2)} | Std PPV: {np.round(np.std(ppv) * 100, 2)},\n' \
                         f'Average TPR: {np.round(np.average(tpr) * 100, 2)} | Std TPR: {np.round(np.std(tpr) * 100, 2)}\n'

    print(performance_string)

    if prospective is True:
        with open(f'outputs/{model_name}/prospective/rf_{classification_type}_performance_metrics.txt', 'w') as f:
            f.write(performance_string)
    else:
        with open(f'outputs/{model_name}/rf_{classification_type}_performance_metrics.txt', 'w') as f:
            f.write(performance_string)


def interpret(x_train, x_test, df_cols, prospective=False, classification_type='fibrosis', model_name='rf'):
    """
    Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.

    Parameters:
        x_train (array-like): Training data features.
        x_test (array-like): Test data features.
        df_cols (list): List of column names (features).
        prospective (bool): whether to use the validation data from Mainz
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
        model_name (str): name of output model

    Returns:
        None
    """
    if any(substring in model_name for substring in ['svm', 'rf', 'xgb', 'light_gbm']):
        explainer = shap.KernelExplainer(model=lambda x: predict_model(x, classification_type=classification_type,
                                                                       model_name=model_name),
                                         data=shap.sample(x_train, 50), feature_names=df_cols)
    elif 'ffn' in model_name:
        explainer = shap.KernelExplainer(
            model=lambda x: predict_pytorch_models(x, classification_type=classification_type,
                                                   model_name=model_name),
            data=shap.sample(x_train, 50), feature_names=df_cols)
    elif 'gandalf' in model_name:
        explainer = shap.KernelExplainer(
            model=lambda x: predict_gandalf_models(x, classification_type=classification_type),
            data=shap.sample(x_train, 50), feature_names=df_cols)
    elif 'tab_transformer' in model_name:
        explainer = shap.KernelExplainer(
            model=lambda x: predict_tab_transformer_models(x, classification_type=classification_type),
            data=shap.sample(x_train, 50), feature_names=df_cols)
    elif 'vi_bnn' in model_name:
        explainer = shap.KernelExplainer(
            model=lambda x: predict_vi_bnn_models(x, classification_type=classification_type),
            data=shap.sample(x_train, 50), feature_names=df_cols)

    shap_values = explainer.shap_values(x_test)

    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        x_test,
        feature_names=df_cols,
        show=False)

    if not os.path.isdir(f'outputs/{model_name}/'):
        os.mkdir(f'outputs/{model_name}/')

    if prospective is True:
        shap.save_html(f'outputs/{model_name}/prospective/{classification_type}_force_plot.htm', f)
        plt.close()
    else:
        shap.save_html(f'outputs/{model_name}/{classification_type}_force_plot.htm', f)
        plt.close()

    fig, ax = plt.subplots()
    shap_values2 = explainer(x_test)
    shap.plots.bar(shap_values2, show=False)

    f = plt.gcf()
    if prospective is True:
        f.savefig(f'outputs/{model_name}/prospective/{classification_type}_summary_bar.png', bbox_inches='tight',
                  dpi=300)
    else:
        f.savefig(f'outputs/{model_name}/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots()

    shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)

    f = plt.gcf()
    if prospective is True:
        f.savefig(f'outputs/{model_name}/prospective/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
    else:
        f.savefig(f'outputs/{model_name}/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)

    plt.close()


def interpret_tab_transformer_model(x_train, x_test, df_cols, classification_type='fibrosis'):
    """
    Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.

    Parameters:
        x_train (array-like): Training data features.
        x_test (array-like): Test data features.
        df_cols (list): List of column names (features).
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.

    Returns:
        None
    """
    explainer = shap.KernelExplainer(model=lambda x: predict_tab_transformer_models(x, classification_type=classification_type), data=shap.sample(x_train, 50), feature_names=df_cols)

    shap_values = explainer.shap_values(x_test)

    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        x_test,
        feature_names=df_cols,
        show=False)
    shap.save_html(f'outputs/tab_transformer/{classification_type}_force_plot.htm', f)
    plt.close()

    fig, ax = plt.subplots()
    shap_values2 = explainer(x_test)
    shap.plots.bar(shap_values2, show=False)

    f = plt.gcf()
    f.savefig(f'outputs/tab_transformer/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots()

    shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)

    f = plt.gcf()
    f.savefig(f'outputs/tab_transformer/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
    plt.close()


def interpret_vi_bnn_model(x_train, x_test, df_cols, classification_type='fibrosis'):
    """
    Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.

    Parameters:
        x_train (array-like): Training data features.
        x_test (array-like): Test data features.
        df_cols (list): List of column names (features).
        classification_type (str): 'fibrosis' or 'cirrhosis'

    Returns:
        None
    """
    explainer = shap.KernelExplainer(model=lambda x: predict_vi_bnn_models(x, classification_type=classification_type),
                                     data=shap.sample(x_train, 50), feature_names=df_cols)

    shap_values = explainer.shap_values(x_test)

    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        x_test,
        feature_names=df_cols,
        show=False)

    shap.save_html(f'outputs/vi_bnn/{classification_type}_force_plot.htm', f)
    plt.close()

    fig, ax = plt.subplots()
    shap_values2 = explainer(x_test)
    shap.plots.bar(shap_values2, show=False)

    f = plt.gcf()
    f.savefig(f'outputs/vi_bnn/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots()

    shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)

    f = plt.gcf()
    f.savefig(f'outputs/vi_bnn/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
    plt.close()


def predict_model(data, classification_type='fibrosis', model_name='rf'):
    """
    Predictions of the model for certain data. Model is saved in output/models.pickle

    Args:
        data: A numpy array to predict on.
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
        model_name (str): name of output model

    Returns:
        A numpy array of class predictions
    """

    with open(f'models/{model_name}/model_{classification_type}.pickle', "rb") as f:
        models = pickle.load(f)

    y_preds = []
    for model in models:
        y_pred = model.predict_proba(data)
        y_preds.append(y_pred)

    maj_preds = majority_vote(y_preds, rule='soft')
    indices, _ = get_index_and_proba(maj_preds)

    return np.array(indices)


def make_ensemble_preds_pytorch(xs_test, ys_test, models, intra_model_preds=False):
    """
    Trains m models individually on m data sets
    Args:
        xs_test (list): List of Test matrices.
        ys_test (list): List of Test labels.
        models (list): List of m PyTorch models.
        intra_model_preds (bool): If True, then redefine the ensemble preds to intra-model preds on xs[0] instead of
        majority-model vote for intra-dataset preds as xs[0] == xs[1] == xs[2] (set True for shap_selected)

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

    # Take first imputed dataset
    X_all, y_all = xs_test[0], ys_test[0]

    skf = StratifiedKFold(n_splits=5)

    # Perform cross-validation
    # Use for stratified_split the train_index (length is 80% of all samples) to calculate the CMs
    for split_index, _ in skf.split(X_all, y_all):
        X = X_all[split_index]
        y = y_all[split_index]

        y_preds = []
        y_preds_all = []

        for model in models:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Convert X to torch tensor if it's not already
                if not torch.is_tensor(X):
                    X = torch.tensor(X)
                if not torch.is_tensor(X_all):
                    X_all = torch.tensor(X_all)
                # Perform forward pass to get probabilities
                probas = model(X.float())  # Assuming input is float
                probas_all = model(X_all.float())

                # Convert to list of lists containing class proababilities
                probas = probas.detach().cpu().numpy()
                probas_all = probas_all.detach().cpu().numpy()
                y_preds.append(probas)
                y_preds_all.append(probas_all)

            """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
            instead ensemble prediction on each xs_test"""
            if intra_model_preds:
                # ensemble_pred is now model probability instead of majority vote of models
                ensemble_pred = probas.tolist()
                ensemble_pred_probas.append(probas)

                ensemble_pred, probas = get_index_and_proba(ensemble_pred)
                maj_report, cm = test(y=y, y_pred=ensemble_pred)

                ensemble_reports.append(maj_report)
                ensemble_cms.append(cm)
                ensemble_preds.append(ensemble_pred)
                ensemble_probas.append(probas)

        """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
        instead ensemble prediction on each xs_test"""
        if intra_model_preds:
            return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

        """get ensemble_preds_probas on stratisfied preds for ROC curve and test"""
        # Take majority vote or soft voting
        ensemble_pred = majority_vote(y_preds, rule='soft')
        ensemble_pred, probas = get_index_and_proba(ensemble_pred)
        # Convert predicted labels to numpy array
        ensemble_pred = np.array(ensemble_pred)

        """get ensemble_preds_probas and ensemble_preds on all preds for ROC curve and test"""
        ensemble_pred_all_probas = majority_vote(y_preds_all, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_all_probas)
        ensemble_pred_all, probas_all = get_index_and_proba(ensemble_pred_all_probas)
        # Convert predicted labels to numpy array
        ensemble_pred_all = np.array(ensemble_pred_all)

        # Calculate classification report and confusion matrix
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)

        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred_all)
        ensemble_probas.append(probas_all)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


def make_ensemble_preds_gandalf(xs_test, ys_test, df_cols, models, classification_type='fibrosis', intra_model_preds=False):
    """
    Trains m models individually on m data sets
    Args:
        x_test (list): List of Test matrices.
        y_test (list): List of Test labels.
        df_cols (list): List of column names
        models (list): List of m PyTorch models.
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'
        intra_model_preds (bool): If True, then redefine the ensemble preds to intra-model preds on xs[0] instead of
        majority-model vote for intra-dataset preds as xs[0] == xs[1] == xs[2] (set True for shap_selected)

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

    # Take first imputed dataset
    X_all, y_all = xs_test[0], ys_test[0]

    skf = StratifiedKFold(n_splits=5)

    # Perform cross-validation
    # Use for stratified_split the train_index (length is 80% of all samples) to calculate the CMs
    for split_index, _ in skf.split(X_all, y_all):
        X = X_all[split_index]
        y = y_all[split_index]

        test_data_all = pd.DataFrame(data=X_all, columns=df_cols)
        test_data_all['target'] = y_all

        test_data = pd.DataFrame(data=X, columns=df_cols)
        test_data['target'] = y

        y_preds = []
        y_preds_all = []

        for model in models:
            # Perform forward pass to get probabilities
            probas_df = model.predict(test_data)
            probas_df_all = model.predict(test_data_all)

            # Convert to list of lists containing class proababilities
            if classification_type == 'three_stage':
                probas = probas_df[['0_probability', '1_probability', '2_probability']].values.tolist()
                probas_all = probas_df_all[['0_probability', '1_probability', '2_probability']].values.tolist()
            else:
                probas = probas_df[['0_probability', '1_probability']].values.tolist()
                probas_all = probas_df_all[['0_probability', '1_probability']].values.tolist()

            y_preds.append(probas)
            y_preds_all.append(probas_all)

            """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
            instead ensemble prediction on each xs_test"""
            if intra_model_preds:
                # ensemble_pred is now model probability instead of majority vote of models
                ensemble_pred_probas.append(probas)

                ensemble_pred, probas = get_index_and_proba(probas)
                maj_report, cm = test(y=y, y_pred=ensemble_pred)

                ensemble_reports.append(maj_report)
                ensemble_cms.append(cm)
                ensemble_preds.append(ensemble_pred)
                ensemble_probas.append(probas)

        """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
        instead ensemble prediction on each xs_test"""
        if intra_model_preds:
            return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

        """get ensemble_preds_probas on stratisfied preds for ROC curve and test"""
        # Take majority vote or soft voting
        ensemble_pred = majority_vote(y_preds, rule='soft')
        ensemble_pred, probas = get_index_and_proba(ensemble_pred)
        # Convert predicted labels to numpy array
        ensemble_pred = np.array(ensemble_pred)

        """get ensemble_preds_probas and ensemble_preds on all preds for ROC curve and test"""
        ensemble_pred_all_probas = majority_vote(y_preds_all, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_all_probas)
        ensemble_pred_all, probas_all = get_index_and_proba(ensemble_pred_all_probas)
        # Convert predicted labels to numpy array
        ensemble_pred_all = np.array(ensemble_pred_all)

        # Calculate classification report and confusion matrix
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)

        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred_all)
        ensemble_probas.append(probas_all)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


def make_ensemble_preds_vi_bnn(xs_test, ys_test, models, intra_model_preds=False):
    """
    Trains m models individually on m data sets
    Args:
        xs_test (list): List of Test matrices.
        ys_test (list): List of Test labels.
        models (list): List of m PyTorch models.
        intra_model_preds (bool): If True, then redefine the ensemble preds to intra-model preds on xs[0] instead of
        majority-model vote for intra-dataset preds as xs[0] == xs[1] == xs[2] (set True for shap_selected)

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

    # Take first imputed dataset
    X_all, y_all = xs_test[0], ys_test[0]

    skf = StratifiedKFold(n_splits=5)

    # Perform cross-validation
    # Use for stratified_split the train_index (length is 80% of all samples) to calculate the CMs
    for split_index, _ in skf.split(X_all, y_all):
        X = X_all[split_index]
        y = y_all[split_index]

        y_preds = []
        y_preds_all = []

        for model in models:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Convert X to torch tensor if it's not already
                if not torch.is_tensor(X):
                    X = torch.tensor(X, device=get_device(i=0))
                if not torch.is_tensor(X_all):
                    X_all = torch.tensor(X_all, device=get_device(i=0))
                # Perform forward pass to get probabilities
                probas = model(X.float())  # Assuming input is float
                probas_all = model(X_all.float())

                # Convert to list of lists containing class proababilities
                probas = probas.detach().cpu().numpy()
                probas_all = probas_all.detach().cpu().numpy()
                y_preds.append(probas)
                y_preds_all.append(probas_all)

            """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
            instead ensemble prediction on each xs_test"""
            if intra_model_preds:
                # ensemble_pred is now model probability instead of majority vote of models
                ensemble_pred = probas.tolist()
                ensemble_pred_probas.append(probas)

                ensemble_pred, probas = get_index_and_proba(ensemble_pred)
                maj_report, cm = test(y=y, y_pred=ensemble_pred)

                ensemble_reports.append(maj_report)
                ensemble_cms.append(cm)
                ensemble_preds.append(ensemble_pred)
                ensemble_probas.append(probas)

        """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
        instead ensemble prediction on each xs_test"""
        if intra_model_preds:
            return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

        """get ensemble_preds_probas on stratisfied preds for ROC curve and test"""
        # Take majority vote or soft voting
        ensemble_pred = majority_vote(y_preds, rule='soft')
        ensemble_pred, probas = get_index_and_proba(ensemble_pred)
        # Convert predicted labels to numpy array
        ensemble_pred = np.array(ensemble_pred)

        """get ensemble_preds_probas and ensemble_preds on all preds for ROC curve and test"""
        ensemble_pred_all_probas = majority_vote(y_preds_all, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_all_probas)
        ensemble_pred_all, probas_all = get_index_and_proba(ensemble_pred_all_probas)
        # Convert predicted labels to numpy array
        ensemble_pred_all = np.array(ensemble_pred_all)

        # Calculate classification report and confusion matrix
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)

        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred_all)
        ensemble_probas.append(probas_all)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


def load_pytorch_model(checkpoint_file, classification_type, model_name, df_cols=None):
    # Read the respective hyperparam file
    model_index = checkpoint_file.split('_')[-1].rstrip('.pth')
    with open(f"./models/{model_name}/model_params_{classification_type}_{model_index}.txt", "r") as file:
        # Read the first line
        dict_str = file.readline().strip()

        # Convert the string representation of the dictionary to a Python dictionary
        param_dict = eval(dict_str)

    # Load the model checkpoint
    if 'ffn' in model_name:
        model = NeuralNetwork(**param_dict)  # Replace YourModelClass with your actual model class
    elif 'tab_transformer' in model_name:
        model = PLTabTransformer(**param_dict, df_cols=df_cols)

    model.load_state_dict(torch.load(os.path.join(f"./models/{model_name}/", checkpoint_file)))
    model.eval()
    return model


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


def predict_gandalf_models(data, classification_type='fibrosis'):
    """
    Predictions of the GANDALF model for certain data. Models are saved as checkpoint files in a directory.

    Args:
        data: A numpy array to predict on.
        classification_type (str): 'fibrosis' or 'cirrhosis'.
    Returns:
        A numpy array of class predictions
    """

    # Get a list of all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(f"./models/gandalf/") if
                        f.endswith('.pth') and 'model' in f and classification_type in f]

    # Load each model checkpoint and make predictions
    y_preds = []

    # Load the df cols
    with open(f'./models/gandalf/df_cols.txt', 'r') as file:
        df_cols_string = file.read()

    # Convert the string to a list using ast.literal_eval
    df_cols = ast.literal_eval(df_cols_string)

    for checkpoint_file in checkpoint_files:
        model = TabularModel.load_model(f'models/gandalf/{checkpoint_file}')
        test_data = pd.DataFrame(data=data, columns=df_cols)
        probas_df = model.predict(test_data)
        # Convert to list of lists containing class proababilities
        if classification_type == 'three_stage':
            y_preds.append(probas_df[['0_probability', '1_probability', '2_probability']].values.tolist())
        else:
            y_preds.append(probas_df[['0_probability', '1_probability']].values.tolist())

    # Perform majority voting
    maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
    indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function

    return np.array(indices)


def predict_tab_transformer_models(data, classification_type='fibrosis'):
    """
    Predictions of the tab_transformer model for certain data. Models are saved as checkpoint files in a directory.

    Args:
        data: A numpy array to predict on.
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.

    Returns:
        A numpy array of class predictions
    """

    # Get a list of all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir("./models/tab_transformer/") if f.endswith('.pth') and 'model' in f and classification_type in f]

    # Load the df cols
    with open(f'./models/tab_transformer/{classification_type}_df_cols.txt', 'r') as file:
        df_cols_string = file.read()

    # Convert the string to a list using ast.literal_eval
    df_cols = ast.literal_eval(df_cols_string)

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
        model = PLTabTransformer(**param_dict, df_cols=df_cols)  # Replace YourModelClass with your actual model class
        model.load_state_dict(torch.load(os.path.join("./models/tab_transformer/", checkpoint_file)))
        model.eval()

        # Make predictions
        with torch.no_grad():
            inputs = torch.tensor(data, dtype=torch.float)  # Assuming data is a numpy array
            outputs = model(inputs)
            y_pred = outputs.numpy()  # Assuming output is probability distribution
            y_preds.append(y_pred)

    # Perform majority voting
    maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
    indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function

    return np.array(indices)


def predict_vi_bnn_models(data, classification_type='fibrosis'):
    """
    Predictions of the model for certain data. PyTorch models are saved as checkpoint files in a directory.

    Args:
        model_directory: Directory where the PyTorch model checkpoints are stored.
        data: A numpy array to predict on.
        classification_type (str): 'fibrosis', 'cirrhosis' or 'three_stage'.

    Returns:
        A numpy array of class predictions
    """

    # Get a list of all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir("./models/vi_bnn/") if f.endswith('.pth') and classification_type in f]

    num_classes = 3 if classification_type == 'three_stage' else 2

    # Load each model checkpoint and make predictions
    y_preds = []
    for checkpoint_file in checkpoint_files:
        # Read the respective hyperparam file
        model_index = checkpoint_file.split('_')[-1].rstrip('.pth')
        with open(f"./models/vi_bnn/model_params_{classification_type}_{model_index}.txt", "r") as file:
            # Read the first line
            dict_str = file.readline().strip()

            # Convert the string representation of the dictionary to a Python dictionary
            param_dict = eval(dict_str)

        # Load the model checkpoint
        model = VI_BNN(**param_dict, prior_var=1.0).to(get_device(i=0))

        # load the model here
        model.load_state_dict(torch.load(os.path.join("./models/vi_bnn/", checkpoint_file)))
        model.eval()

        # test
        samples = 100  # Set the number of samples
        with torch.no_grad():
            if not torch.is_tensor(data):
                data = torch.tensor(data, device=get_device(i=0))
            outputs = torch.zeros(samples, data.shape[0], num_classes).to(get_device(i=0))
            for i in range(samples):
                outputs[i] = model(data.float())
            output = outputs.mean(0)
            y_pred = output.detach().cpu().numpy()  # Assuming output is probability distribution
            y_preds.append(y_pred)

    # Perform majority voting
    maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
    indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function

    return np.array(indices)


def make_ensemble_preds_tab_transformer(xs_test, ys_test, models, intra_model_preds=False):
    """
    Trains m models individually on m data sets
    Args:
        x_test (list): List of Test matrices.
        y_test (list): List of Test labels.
        models (list): List of m PyTorch models.
        intra_model_preds (bool): If True, then redefine the ensemble preds to intra-model preds on xs[0] instead of
        majority-model vote for intra-dataset preds as xs[0] == xs[1] == xs[2] (set True for shap_selected)

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

    # Take first imputed dataset
    X_all, y_all = xs_test[0], ys_test[0]

    skf = StratifiedKFold(n_splits=5)

    # Perform cross-validation
    # Use for stratified_split the train_index (length is 80% of all samples) to calculate the CMs
    for split_index, _ in skf.split(X_all, y_all):
        X = X_all[split_index]
        y = y_all[split_index]

        y_preds = []
        y_preds_all = []

        for model in models:
            # Perform forward pass to get probabilities
            probas = model(torch.tensor(X, dtype=torch.float32))
            probas_all = model(torch.tensor(X_all, dtype=torch.float32))

            # Convert to list of lists containing class proababilities
            probas = probas.detach().cpu().numpy()
            probas_all = probas_all.detach().cpu().numpy()
            y_preds.append(probas)
            y_preds_all.append(probas_all)

            """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
            instead ensemble prediction on each xs_test"""
            if intra_model_preds:
                # ensemble_pred is now model probability instead of majority vote of models
                ensemble_pred = probas.tolist()
                ensemble_pred_probas.append(probas)

                ensemble_pred, probas = get_index_and_proba(ensemble_pred)
                maj_report, cm = test(y=y, y_pred=ensemble_pred)

                ensemble_reports.append(maj_report)
                ensemble_cms.append(cm)
                ensemble_preds.append(ensemble_pred)
                ensemble_probas.append(probas)

        """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
        instead ensemble prediction on each xs_test"""
        if intra_model_preds:
            return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

        """get ensemble_preds_probas on stratisfied preds for ROC curve and test"""
        # Take majority vote or soft voting
        ensemble_pred = majority_vote(y_preds, rule='soft')
        ensemble_pred, probas = get_index_and_proba(ensemble_pred)
        # Convert predicted labels to numpy array
        ensemble_pred = np.array(ensemble_pred)

        """get ensemble_preds_probas and ensemble_preds on all preds for ROC curve and test"""
        ensemble_pred_all_probas = majority_vote(y_preds_all, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_all_probas)
        ensemble_pred_all, probas_all = get_index_and_proba(ensemble_pred_all_probas)
        # Convert predicted labels to numpy array
        ensemble_pred_all = np.array(ensemble_pred_all)

        # Calculate classification report and confusion matrix
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)

        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred_all)
        ensemble_probas.append(probas_all)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas
