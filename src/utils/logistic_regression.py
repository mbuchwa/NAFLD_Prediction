from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np


def logistic_regression_analysis(x_train, y_train, x_test, y_test, feature_names, test_cohort_name):
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Extract and map coefficients to feature names
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    coef_dict = dict(zip(feature_names, coefficients))

    # Print logistic regression equation
    print("Logistic Regression Equation:")
    print(f"log-odds = {intercept:.4f} + "
          f"({coef_dict[feature_names[0]]:.4f} * {feature_names[0]}) + "
          f"({coef_dict[feature_names[1]]:.4f} * {feature_names[1]}) + "
          f"({coef_dict[feature_names[2]]:.4f} * {feature_names[2]})")
    print("\nProbability of positive class (P):")
    print("P = 1 / (1 + exp(-log-odds))")

    # Cross-validation on test set
    skf = StratifiedKFold(n_splits=5)
    y_test_pred_proba = model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

    cms = []
    reports = []
    for _, split_index in skf.split(np.zeros(len(y_test)), y_test):
        y_true_split = y_test[split_index]
        y_pred_split = y_test_pred[split_index]

        # Confusion matrix and classification report
        cm = confusion_matrix(y_true_split, y_pred_split)
        report = classification_report(y_true_split, y_pred_split, output_dict=True)
        cms.append(cm)
        reports.append(report)

    # Compute performance metrics
    acc, f1, ppv, tpr = [], [], [], []
    for report in reports:
        acc.append(report['accuracy'])
        f1.append(report['macro avg']['f1-score'])
        ppv.append(report['macro avg']['precision'])
        tpr.append(report['macro avg']['recall'])

    performance_string = f'\nLogistic Regression - {test_cohort_name}\n' \
                         f'Average ACC: {np.round(np.mean(acc) * 100, 2)} | Std ACC: {np.round(np.std(acc) * 100, 2)},\n' \
                         f'Average F1: {np.round(np.mean(f1) * 100, 2)} | Std F1: {np.round(np.std(f1) * 100, 2)},\n' \
                         f'Average PPV: {np.round(np.mean(ppv) * 100, 2)} | Std PPV: {np.round(np.std(ppv) * 100, 2)},\n' \
                         f'Average TPR: {np.round(np.mean(tpr) * 100, 2)} | Std TPR: {np.round(np.std(tpr) * 100, 2)}\n'
    print(performance_string)
