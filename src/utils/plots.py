import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def plot_cm(cm, name='xgb', iteration=None, prospective=False, classification_type='fibrosis'):
    """
    Plot the confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix array.
        name (str): Name of the model or context to be included in the saved plot filename.

    Returns:
        None
    """

    if not os.path.isdir(f'./outputs/{name}/'):
        os.mkdir(f'./outputs/{name}/')

    if classification_type == 'three_stage':
        df_cm = pd.DataFrame(
            cm,
            index=[
                i for i in [
                    'Stage 0',
                    'Stage 1',
                    'Stage 2'
                ]],
            columns=[
                i for i in [
                    'Stage 0',
                    'Stage 1',
                    'Stage 2'
                ]])
    else:
        df_cm = pd.DataFrame(
            cm,
            index=[
                i for i in [
                    'Stage 0',
                    'Stage 1'
                ]],
            columns=[
                i for i in [
                    'Stage 0',
                    'Stage 1'
                ]])
    sns.heatmap(df_cm, annot=True, cmap='coolwarm')
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    if prospective:
        plt.savefig(f"outputs/{name}/prospective/{name}_{classification_type}_cm_{iteration}.png")
    else:
        plt.savefig(f"outputs/{name}/{name}_{classification_type}_cm_{iteration}.png")

    plt.close()


def plot_auc(y_true, y_pred_proba):
    """
    Plot the ROC curve and calculate the AUC.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred_proba : array-like of shape (n_samples, 2)
        Predicted probabilities of each class.
    """
    y_pred = np.array(y_pred_proba)[:, 1].tolist()  # Select the probabilities for the positive class

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.fill_between(fpr, tpr, color='orange', alpha=0.3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_combined_roc_curve(y_true, y_pred_ensemble, model_name='xgb', prospective=False, classification_type='fibrosis'):
    """
    Plot combined ROC curve for ensemble model prediction on each test dataset.

    Args:
    - y_true: True class labels (array-like, shape: (n_datasets, n_samples, ))
    - y_pred_ensemble: Predicted probabilities for each model in the ensemble
                       (3D array-like, shape: (n_datasets, n_samples, 2))

    Returns:
    - None (plots the ROC curve)
    """
    # Initialize lists to store FPR, TPR, and AUC for each model
    fpr_list = []
    tpr_list = []
    auc_list = []

    # convert to array
    y_true = np.array(y_true)
    y_pred_ensemble = np.array(y_pred_ensemble)

    # Compute ROC curve and AUC for each model
    for i in range(y_pred_ensemble.shape[0]):  # Assuming y_pred_ensemble has shape (n_datasets, n_samples, 2)
        fpr, tpr, _ = roc_curve(y_true[i], y_pred_ensemble[i, :, 1], pos_label=1)
        auc_val = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_val)

    # Initialize figure and axis
    plt.figure(figsize=(10, 8))

    # Plot individual ROC curves and AUC values
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], label=f'Ensemble Prediction on Test Dataset {i + 1} (AUC = {auc_list[i]:.2f})')

    # Plot ROC curve for random guessing (50% probability)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=2.5, alpha=0.5)

    # Set labels and title
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)

    # Set axis ticks font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the figure
    if prospective is True:
        plt.savefig(f'outputs/{model_name}/prospective/{model_name}_{classification_type}_ensemble_ROCs_on_each_test_data.png',
                    bbox_inches='tight')
    else:
        plt.savefig(f'outputs/{model_name}/{model_name}_{classification_type}_ensemble_ROCs_on_each_test_data.png',
                    bbox_inches='tight')

    plt.close()


def plot_roc(tpr, fpr, roc_auc, model, name, iteration):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.title(f"{name} Comparison")
    plt.savefig(f"outputs/{model}/{model}_{name}_{iteration}.png")
    plt.close()


def plot_metric(metric, model, name):
    """
    Creates and saves a bar chart comparing the accuracies of two sets of methods (models).

    Args:
        metric (list): Metric to be plotted.
    Returns:
        None
    """

    x_positions = np.arange(len(metric))
    plt.figure(figsize=(16, 9), dpi=300)
    # Create the bar plot
    plt.bar(x_positions, metric, color='skyblue')
    plt.xticks(x_positions)

    # Customize and save the plot
    plt.xlabel("Imputed Data Set")
    plt.ylabel("Percentage")
    plt.title(f"{name} Comparison")
    plt.savefig(f"outputs/{model}/{model}_{name}.png")
    plt.close()


def plot_accuracies(reports, report_label, name, prospective, classification_type='fibrosis'):
    """
    Creates and saves a bar chart comparing the accuracies of two sets of methods (models).

    Args:
        reports1 (list): A list of reports from the first imputed data set.
        report_1_label (str): The label for the first set of reports in the plot.
        reports2 (list): A list of reports from the second imputed data set.
        report_2_label (str): The label for the second set of reports in the plot.
        name (str): The title of the accuracy comparison plot.
    """
    accuracies = [report['accuracy'] for report in reports]

    x_positions = np.arange(len(accuracies))
    labels = [f"Imputed data set {i}" for i in x_positions]
    plt.figure(figsize=(16, 9), dpi=300)
    # Create the bar plot
    plt.bar(
        x_positions,
        accuracies,
        label=f'{report_label}',
        color='skyblue',
        width=0.4)

    # Customize and save the plot
    plt.xlabel("Imputed Data Set")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy Comparison{' '.join(word.capitalize() for word in name.split('_'))} ")
    plt.xticks(x_positions + 0.2, labels=labels, rotation=20, ha='right')
    plt.legend()
    if prospective is True:
        if not os.path.isdir(f"outputs/{name}/prospective/"):
            os.mkdir(f"outputs/{name}/prospective/")
        plt.savefig(f"outputs/{name}/prospective/{name}_{classification_type}_accuracies_comparison.png")
    else:
        plt.savefig(f"outputs/{name}/{name}_{classification_type}_accuracies_comparison.png")
    plt.close()


def plot_diff_of_accuracies(
        reports1,
        report_1_label,
        reports2,
        report_2_label,
        name):
    """
    Creates and saves a bar chart comparing the difference of two accuracy lists.

    Args:
        reports1 (list): A list of reports from the first imputed data set.
        report_1_label (str): The label for the first set of reports in the plot.
        reports2 (list): A list of reports from the second imputed data set.
        report_2_label (str): The label for the second set of reports in the plot.
    """
    accuracies_1 = [report['accuracy'] for report in reports1]
    accuracies_2 = [report['accuracy'] for report in reports2]
    accuracy_diff = np.subtract(accuracies_1, accuracies_2)
    x_positions = np.arange(len(accuracy_diff))

    plt.figure(figsize=(16, 9), dpi=300)
    # Create the bar plot
    plt.bar(
        x_positions,
        accuracy_diff,
        label='Accuracy Difference',
        color='skyblue',
        width=0.4)

    # Customize and save the plot
    plt.xlabel("Imputed Data Set")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy Difference ({report_1_label} - {report_2_label}) {' '.join(word.capitalize() for word in name.split('_'))}")
    plt.xticks(x_positions, rotation=45, ha='right')
    plt.legend()
    plt.savefig(f"plots/{name}/{name}_accuracies_difference.png")
    plt.close()


def evaluate_model(logits_samples, mean_logits, y_test_tensor):
    """
    :param logits_samples:
    :param mean_logits:
    :param y_test_tensor:
    :param label_encoder:
    :return:
    """
    # calculate the mean and std
    mean_probs = logits_samples.mean(axis=0)
    std_probs = logits_samples.std(axis=0)
    confidence_interval = 1.96 * std_probs
    lower_bound = mean_probs - confidence_interval
    upper_bound = mean_probs + confidence_interval
    lower_bound.clamp_(min=0, max=1)
    upper_bound.clamp_(min=0, max=1)

    num_classes = mean_logits.shape[1]
    num_samples_to_plot = 200
    ground_truth = y_test_tensor[:num_samples_to_plot].cpu().numpy()
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes))

    for i in range(num_classes):
        mean = mean_logits[:num_samples_to_plot, i]
        lower = lower_bound[:num_samples_to_plot, i]
        upper = upper_bound[:num_samples_to_plot, i]
        if torch.is_tensor(lower):
            lower = lower.cpu().numpy()
        if torch.is_tensor(upper):
            upper = upper.cpu().numpy()
        corrected_lower = np.maximum(lower, 0)
        corrected_upper = np.minimum(upper, 1)
        ci_lower = mean - corrected_lower
        ci_upper = corrected_upper - mean
        axes[i].errorbar(
            range(num_samples_to_plot),
            mean,
            yerr=[
                ci_lower,
                ci_upper],
            fmt='o',
            color='blue',
            ecolor='red',
            alpha=0.7,
            label='Prediction with 95% CI')
        for j, gt in enumerate(ground_truth):
            if gt == i:
                axes[i].axvline(
                    j,
                    color='green',
                    alpha=0.5,
                    linestyle='--',
                    linewidth=0.5)

        axes[i].set_title(f'{i}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
