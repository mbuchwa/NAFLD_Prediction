from sklearn.metrics import classification_report, confusion_matrix
import pickle
from collections import Counter
from src.utils.plots import *


def test_sklearn_models(models, xs_test, ys_test):
    """
    Trains m models individually on m data sets
    Args:
        models (list): List of XBoost.classifier.
        xs_test (list): list of Test matrices.
        ys_test (list): List of Test labels.

    Returns:
        list: A list containing dictionaries of classifiction reports.
        list: A list of m confusion matrices.
    """

    reports = []
    cms = []
    y_preds = []
    for (model, X, y) in zip(models, xs_test, ys_test):
        y_pred = model.predict(X)
        report, cm = test(y, y_pred)
        reports.append(report)
        cms.append(cm)
        y_preds.append(y_pred)

    return reports, cms


def test(y, y_pred):
    """
    Evaluates a classification prediction.

    Args:
        y (pandas.Series or numpy.ndarray): The true labels.
        y_pred (pandas.Series or numpy.ndarray): The predicted labels.

    Returns:
        dict: A dictionary containing the classification report.
        numpy.ndarray: The confusion matrix.
    """
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return report, cm


def make_individual_preds(xs_test, ys_test, models, model_name='xgb'):

    cms = []
    reports = []
    y_preds = []  # stores the index of predicted class
    probas = []  # stores the probability of the predicted class
    pred_probas = []  # stores the probability of both classes

    for (X, y, model) in zip(xs_test, ys_test, models):
        if model_name == 'xgb' or model_name == 'rf' or model_name == 'svm':
            probas = model.predict_proba(X)
            preds = model.predict(X)
        else:
            probas = model(torch.tensor(X, dtype=torch.float32, device=get_device(i=0)))
        y_preds.append(preds)

        maj_report, cm = test(y=y, y_pred=y_preds)

        reports.append(maj_report)
        cms.append(cm)
        preds.append(y_preds)
        probas.append(probas)

    return reports, cms, preds, probas, pred_probas


def make_ensemble_preds(xs_test, ys_test, models, intra_model_preds=False):
    """
    Run ensemble inference once on the full held-out dataset.
    Args:
        xs_test (list): List of Test matrices.
        ys_test (list): List of Test labels.
        models (list): e.g. m xgboost.XGBClassifier.
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

    X_all, y_all = xs_test[0], ys_test[0]
    y_preds_all = []

    for model in models:
        probas_all = model.predict_proba(X_all)
        y_preds_all.append(probas_all)

        if intra_model_preds:
            ensemble_pred_probas.append(probas_all)
            ensemble_pred, probas = get_index_and_proba(probas_all.tolist())
            maj_report, cm = test(y=y_all, y_pred=ensemble_pred)
            ensemble_reports.append(maj_report)
            ensemble_cms.append(cm)
            ensemble_preds.append(ensemble_pred)
            ensemble_probas.append(probas)

    if intra_model_preds:
        return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

    ensemble_pred_all_probas = majority_vote(y_preds_all, rule='soft')
    ensemble_pred_probas.append(ensemble_pred_all_probas)
    ensemble_pred_all, probas_all = get_index_and_proba(ensemble_pred_all_probas)
    ensemble_pred_all = np.array(ensemble_pred_all)

    maj_report = classification_report(y_all, ensemble_pred_all, output_dict=True)
    cm = confusion_matrix(y_all, ensemble_pred_all)

    ensemble_reports.append(maj_report)
    ensemble_cms.append(cm)
    ensemble_preds.append(ensemble_pred_all)
    ensemble_probas.append(probas_all)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


def get_device(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def predict_pickle_model(data, model_name='xgb'):
    """
    Predictions of the model for certain data. Model is saved in output/models.pickle

    Args:
        data: A numpy array to predict on.
    Returns:
        A numpy array of class predictions
    """

    with open(f'models/{model_name}/model.pickle', "rb") as f:
        models = pickle.load(f)

    y_preds = []
    for model in models:
        y_pred = model.predict_proba(data)
        y_preds.append(y_pred)

    maj_preds = majority_vote(y_preds, rule='soft')
    indices, _ = get_index_and_proba(maj_preds)

    return np.array(indices)


def majority_vote(predictions, rule="hard"):
    """
    Performs majority voting on a list of predictions.

    Args:
        predictions: A list of lists, where each inner list contains predictions (classes) from one model.
        rule: "hard" for hard voting, "soft" for soft voting (default).

    Returns:
        A list of size equal to the number of inner lists, where each element is the majority class for the corresponding prediction across all models.
    """
    majority_classes = []
    for i in range(
            len(predictions[0])):

        if rule == 'hard':
            class_counts = Counter([prediction[i]
                                   for prediction in predictions])
            majority_class = class_counts.most_common(1)[0][0]
        elif rule == 'soft':
            majority_class = [0] * len(predictions[0][0])

            for model_predictions in predictions:
                majority_class = [
                    a + b for a,
                    b in zip(
                        majority_class,
                        model_predictions[i])]

            if sum(majority_class) > 0:
                majority_class = [p / sum(majority_class)
                                  for p in majority_class]

        else:
            raise ValueError("Invalid rule. Choose 'hard' or 'soft'.")
        majority_classes.append(majority_class)

    return majority_classes


def get_index_and_proba(data):
    """
    Finds the index and value of the highest element in each sublist.

    Args:
        data: A list of lists, where each inner list contains numerical values.

    Returns:
        A tuple containing two lists:
            - indices: A list containing the index of the highest element in each sublist.
            - values: A list containing the corresponding highest elements.
    """
    indices = []
    values = []

    for _, sublist in enumerate(data):
        # Find the index of the maximum value
        max_index = sublist.index(max(sublist))
        # Append the index and corresponding value
        indices.append(max_index)
        values.append(max(sublist))

    return indices, values


def remove_files_in_directory(directory):
    # empty all files in the mcmc_bnn
    if not len(os.listdir(directory)) == 0:
        print(f'deleting files in {directory}')
        files = os.listdir(directory)

        # Iterate over each file and remove it
        for file in files:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    else:
        print(f'{directory} already empty')
