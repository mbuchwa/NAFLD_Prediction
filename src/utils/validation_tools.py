from src.utils.helper_functions import *
from src.utils.networks import PLTabTransformer, NeuralNetwork, VI_BNN
import shap
from pytorch_tabular import TabularModel
import ast
import json
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _compute_external_calibration_stats(y_true, y_pred_proba):
    """
    Compute calibration diagnostics for binary predictions on external data.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    positive_proba = y_pred_proba[:, 1]
    clipped_proba = np.clip(positive_proba, 1e-6, 1 - 1e-6)
    logit_proba = np.log(clipped_proba / (1 - clipped_proba)).reshape(-1, 1)

    recalibration_model = LogisticRegression(
        fit_intercept=True,
        penalty='l2',
        C=1e6,
        solver='lbfgs',
        max_iter=2000
    )
    recalibration_model.fit(logit_proba, y_true)

    calibration_slope = float(recalibration_model.coef_[0][0])
    calibration_intercept = float(recalibration_model.intercept_[0])
    brier = float(brier_score_loss(y_true, positive_proba))

    frac_pos, mean_pred = calibration_curve(y_true, positive_proba, n_bins=10, strategy='quantile')

    return {
        'brier_score': brier,
        'calibration_slope': calibration_slope,
        'calibration_intercept': calibration_intercept,
        'mean_predicted_probability': mean_pred.tolist(),
        'fraction_of_positives': frac_pos.tolist()
    }


def _save_external_calibration_artifacts(calibration_stats, model_name, classification_type):
    output_dir = f'outputs/{model_name}/prospective'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot(
        calibration_stats['mean_predicted_probability'],
        calibration_stats['fraction_of_positives'],
        marker='o',
        linewidth=2,
        label='Model'
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed event rate')
    plt.title(f'Calibration curve ({classification_type})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_curve.png', dpi=300)
    plt.close()

    calibration_stats_path = f'{output_dir}/{classification_type}_calibration_stats.json'
    with open(calibration_stats_path, 'w') as f:
        json.dump(calibration_stats, f, indent=2)


def _calc_binary_operating_metrics(y_true, positive_scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(positive_scores).astype(float)
    y_pred = (scores >= float(threshold)).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return {
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'PPV': float(ppv),
        'NPV': float(npv),
    }


def _bootstrap_ci(metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        sampled_value = metric_fn(rng)
        if sampled_value is None or np.isnan(sampled_value):
            continue
        values.append(float(sampled_value))
    if len(values) == 0:
        return float('nan'), float('nan')
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _youden_threshold(y_true, positive_scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(positive_scores).astype(float)
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return 0.5

    best_thr = 0.5
    best_youden = -np.inf
    for thr in thresholds:
        metrics = _calc_binary_operating_metrics(y_true, scores, thr)
        if np.isnan(metrics['Sensitivity']) or np.isnan(metrics['Specificity']):
            continue
        youden = metrics['Sensitivity'] + metrics['Specificity'] - 1.0
        if youden > best_youden:
            best_youden = youden
            best_thr = float(thr)
    return best_thr


def _format_metric_line(name, value, ci_lower, ci_upper):
    return f'{name}: {value:.4f} (95% bootstrap CI {ci_lower:.4f}-{ci_upper:.4f})'


def _evaluate_binary_with_threshold(y_true, y_pred_proba, threshold, bootstrap_n=1000):
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba)
    positive_scores = y_pred_proba[:, 1]
    n = y_true.shape[0]

    auc_val = float(roc_auc_score(y_true, positive_scores))

    def auc_bootstrap(rng):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            return None
        return roc_auc_score(y_true[idx], positive_scores[idx])

    auc_ci = _bootstrap_ci(auc_bootstrap, n=bootstrap_n)
    op_metrics = _calc_binary_operating_metrics(y_true, positive_scores, threshold)
    op_cis = {}

    for metric_name in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
        def metric_bootstrap(rng, mn=metric_name):
            idx = rng.integers(0, n, n)
            return _calc_binary_operating_metrics(y_true[idx], positive_scores[idx], threshold)[mn]
        op_cis[metric_name] = _bootstrap_ci(metric_bootstrap, n=bootstrap_n)

    return {
        'threshold': float(threshold),
        'auroc': auc_val,
        'auroc_ci_lower': auc_ci[0],
        'auroc_ci_upper': auc_ci[1],
        'operating_metrics': op_metrics,
        'operating_cis': op_cis,
    }


def _evaluate_multiclass_ovr(y_true, y_pred_proba, thresholds, bootstrap_n=1000):
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba)
    n = y_true.shape[0]
    n_classes = y_pred_proba.shape[1]
    rows = []
    for class_idx in range(n_classes):
        class_name = {0: 'F0-1', 1: 'F2', 2: 'F3-4'}.get(class_idx, f'class_{class_idx}')
        y_bin = (y_true == class_idx).astype(int)
        scores = y_pred_proba[:, class_idx]
        threshold = float(thresholds[class_idx])

        if len(np.unique(y_bin)) < 2:
            auroc = np.nan
            auroc_ci = (np.nan, np.nan)
        else:
            auroc = float(roc_auc_score(y_bin, scores))

            def auc_bootstrap(rng):
                idx = rng.integers(0, n, n)
                if len(np.unique(y_bin[idx])) < 2:
                    return None
                return roc_auc_score(y_bin[idx], scores[idx])

            auroc_ci = _bootstrap_ci(auc_bootstrap, n=bootstrap_n)

        metrics = _calc_binary_operating_metrics(y_bin, scores, threshold)
        metric_cis = {}
        for metric_name in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
            def metric_bootstrap(rng, mn=metric_name):
                idx = rng.integers(0, n, n)
                return _calc_binary_operating_metrics(y_bin[idx], scores[idx], threshold)[mn]
            metric_cis[metric_name] = _bootstrap_ci(metric_bootstrap, n=bootstrap_n)

        row = {
            'class_index': class_idx,
            'class_name': class_name,
            'threshold': threshold,
            'auroc_ovr': auroc,
            'auroc_ci_lower': auroc_ci[0],
            'auroc_ci_upper': auroc_ci[1],
        }
        for metric_name, metric_val in metrics.items():
            row[metric_name.lower()] = metric_val
            row[f'{metric_name.lower()}_ci_lower'] = metric_cis[metric_name][0]
            row[f'{metric_name.lower()}_ci_upper'] = metric_cis[metric_name][1]
        rows.append(row)
    return rows


def _get_eval_output_dir(model_name, prospective):
    return f'outputs/{model_name}/prospective' if prospective else f'outputs/{model_name}'


def evaluate_performance(models, xs_test, ys_test, df_cols, model_name, classification_type, prospective,
                         xs_val=None, ys_val=None):
    # Single-pass ensemble prediction on the full held-out split.
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

    # Determine a fixed operating threshold on validation data once, then apply to held-out data.
    if xs_val is not None and ys_val is not None:
        if any(substring in model_name for substring in ['svm', 'rf', 'xgb', 'light_gbm']):
            _, _, _, _, val_pred_probas = make_ensemble_preds(xs_val, ys_val, models)
        elif 'ffn' in model_name:
            _, _, _, _, val_pred_probas = make_ensemble_preds_pytorch(xs_val, ys_val, models)
        elif 'gandalf' in model_name:
            _, _, _, _, val_pred_probas = make_ensemble_preds_gandalf(xs_val, ys_val, df_cols, models)
        elif 'tab_transformer' in model_name:
            _, _, _, _, val_pred_probas = make_ensemble_preds_tab_transformer(xs_val, ys_val, models)
        elif 'vi_bnn' in model_name:
            _, _, _, _, val_pred_probas = make_ensemble_preds_vi_bnn(xs_val, ys_val, models)
        else:
            val_pred_probas = None
    else:
        val_pred_probas = None

    if val_pred_probas is None:
        print('Validation probabilities unavailable; defaulting threshold estimation to current evaluation split.')
        val_y_ref = ys_test[0]
        val_proba_ref = np.asarray(ensemble_pred_probas[0])
    else:
        val_y_ref = ys_val[0]
        val_proba_ref = np.asarray(val_pred_probas[0])

    eval_y_ref = ys_test[0]
    eval_proba_ref = np.asarray(ensemble_pred_probas[0])
    output_dir = _get_eval_output_dir(model_name, prospective)
    os.makedirs(output_dir, exist_ok=True)

    threshold_metrics_lines = []
    if eval_proba_ref.shape[1] == 2:
        operating_threshold = _youden_threshold(val_y_ref, val_proba_ref[:, 1])
        binary_eval = _evaluate_binary_with_threshold(eval_y_ref, eval_proba_ref, operating_threshold, bootstrap_n=1000)
        threshold_metrics_lines.append(f'Operating threshold (Youden, validation): {binary_eval["threshold"]:.4f}')
        threshold_metrics_lines.append(_format_metric_line('AUROC', binary_eval['auroc'],
                                                           binary_eval['auroc_ci_lower'],
                                                           binary_eval['auroc_ci_upper']))
        for metric_name in ['Sensitivity', 'Specificity', 'PPV', 'NPV']:
            val = binary_eval['operating_metrics'][metric_name]
            ci_lower, ci_upper = binary_eval['operating_cis'][metric_name]
            threshold_metrics_lines.append(_format_metric_line(metric_name, val, ci_lower, ci_upper))
    else:
        class_thresholds = {class_idx: _youden_threshold((np.asarray(val_y_ref) == class_idx).astype(int),
                                                         val_proba_ref[:, class_idx])
                            for class_idx in range(eval_proba_ref.shape[1])}
        multiclass_rows = _evaluate_multiclass_ovr(eval_y_ref, eval_proba_ref, class_thresholds, bootstrap_n=1000)
        threshold_metrics_lines.append('One-vs-rest AUROC and operating-point metrics (thresholds from validation Youden):')
        for row in multiclass_rows:
            threshold_metrics_lines.append(
                f"Class {row['class_name']} (thr={row['threshold']:.4f}) "
                f"AUROC={row['auroc_ovr']:.4f} "
                f"[{row['auroc_ci_lower']:.4f}, {row['auroc_ci_upper']:.4f}] "
                f"Sens={row['sensitivity']:.4f} Spec={row['specificity']:.4f} "
                f"PPV={row['ppv']:.4f} NPV={row['npv']:.4f}"
            )
        if classification_type == 'three_stage':
            subgroup_df = pd.DataFrame(multiclass_rows)
            subgroup_path = f'{output_dir}/{model_name}_{classification_type}_subgroup_metrics_table.csv'
            subgroup_df.to_csv(subgroup_path, index=False)

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

    if prospective and classification_type != 'three_stage':
        calibration_stats = _compute_external_calibration_stats(
            y_true=ys_test[0],
            y_pred_proba=ensemble_pred_probas[0]
        )
        _save_external_calibration_artifacts(
            calibration_stats=calibration_stats,
            model_name=model_name,
            classification_type=classification_type
        )

    pooled_metrics = pool_classification_metrics_with_rubins_rules(
        reports=ensemble_reports,
        sample_sizes=[len(y) for y in ys_test]
    )
    performance_lines = [f'\n Ensemble {model_name} (Rubin pooled across imputations)\n']
    for metric in ['ACC', 'F1', 'PPV', 'TPR']:
        values = pooled_metrics[metric]
        performance_lines.append(
            f"{metric}: {values['estimate'] * 100:.2f}% "
            f"(95% CI {values['ci_lower'] * 100:.2f}%-{values['ci_upper'] * 100:.2f}%; "
            f"within_var={values['within_var']:.6f}, between_var={values['between_var']:.6f})"
        )
    performance_string = '\n'.join(performance_lines) + '\n'
    performance_string += '\n'.join(threshold_metrics_lines) + '\n'

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
    Run ensemble inference once on the full held-out dataset.
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
    ensemble_preds = []
    ensemble_probas = []
    ensemble_pred_probas = []

    X_all, y_all = xs_test[0], ys_test[0]
    if intra_model_preds:
        if not torch.is_tensor(X_all):
            X_all = torch.tensor(X_all)
        for model in models:
            model.eval()
            with torch.no_grad():
                probas_all = model(X_all.float()).detach().cpu().numpy()
            ensemble_pred_probas.append(probas_all)
            ensemble_pred, probas = get_index_and_proba(probas_all.tolist())
            maj_report, cm = test(y=y_all, y_pred=ensemble_pred)
            ensemble_reports.append(maj_report)
            ensemble_cms.append(cm)
            ensemble_preds.append(ensemble_pred)
            ensemble_probas.append(probas)
        return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

    for X, y in zip(xs_test, ys_test):
        x_tensor = X if torch.is_tensor(X) else torch.tensor(X)
        y_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                y_preds.append(model(x_tensor.float()).detach().cpu().numpy())
        ensemble_pred_proba = majority_vote(y_preds, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_proba)
        ensemble_pred, probas = get_index_and_proba(ensemble_pred_proba)
        ensemble_pred = np.array(ensemble_pred)
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)
        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred)
        ensemble_probas.append(probas)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


def make_ensemble_preds_gandalf(xs_test, ys_test, df_cols, models, classification_type='fibrosis', intra_model_preds=False):
    """
    Run ensemble inference once on the full held-out dataset.
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
    ensemble_preds = []
    ensemble_probas = []
    ensemble_pred_probas = []

    X_all, y_all = xs_test[0], ys_test[0]
    if intra_model_preds:
        test_data_all = pd.DataFrame(data=X_all, columns=df_cols)
        test_data_all['target'] = y_all
        for model in models:
            probas_df_all = model.predict(test_data_all)
            if classification_type == 'three_stage':
                probas_all = probas_df_all[['0_probability', '1_probability', '2_probability']].values.tolist()
            else:
                probas_all = probas_df_all[['0_probability', '1_probability']].values.tolist()
            ensemble_pred_probas.append(probas_all)
            ensemble_pred, probas = get_index_and_proba(probas_all)
            maj_report, cm = test(y=y_all, y_pred=ensemble_pred)
            ensemble_reports.append(maj_report)
            ensemble_cms.append(cm)
            ensemble_preds.append(ensemble_pred)
            ensemble_probas.append(probas)
        return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

    for X, y in zip(xs_test, ys_test):
        test_data = pd.DataFrame(data=X, columns=df_cols)
        test_data['target'] = y
        y_preds = []
        for model in models:
            probas_df = model.predict(test_data)
            if classification_type == 'three_stage':
                y_preds.append(probas_df[['0_probability', '1_probability', '2_probability']].values.tolist())
            else:
                y_preds.append(probas_df[['0_probability', '1_probability']].values.tolist())
        ensemble_pred_proba = majority_vote(y_preds, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_proba)
        ensemble_pred, probas = get_index_and_proba(ensemble_pred_proba)
        ensemble_pred = np.array(ensemble_pred)
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)
        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred)
        ensemble_probas.append(probas)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


def make_ensemble_preds_vi_bnn(xs_test, ys_test, models, intra_model_preds=False):
    """
    Run ensemble inference once on the full held-out dataset.
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
    ensemble_preds = []
    ensemble_probas = []
    ensemble_pred_probas = []

    X_all, y_all = xs_test[0], ys_test[0]
    if intra_model_preds:
        if not torch.is_tensor(X_all):
            X_all = torch.tensor(X_all, device=get_device(i=0))
        for model in models:
            model.eval()
            with torch.no_grad():
                probas_all = model(X_all.float()).detach().cpu().numpy()
            ensemble_pred_probas.append(probas_all)
            ensemble_pred, probas = get_index_and_proba(probas_all.tolist())
            maj_report, cm = test(y=y_all, y_pred=ensemble_pred)
            ensemble_reports.append(maj_report)
            ensemble_cms.append(cm)
            ensemble_preds.append(ensemble_pred)
            ensemble_probas.append(probas)
        return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

    for X, y in zip(xs_test, ys_test):
        x_tensor = X if torch.is_tensor(X) else torch.tensor(X, device=get_device(i=0))
        y_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                y_preds.append(model(x_tensor.float()).detach().cpu().numpy())
        ensemble_pred_proba = majority_vote(y_preds, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_proba)
        ensemble_pred, probas = get_index_and_proba(ensemble_pred_proba)
        ensemble_pred = np.array(ensemble_pred)
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)
        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred)
        ensemble_probas.append(probas)

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
    Run ensemble inference once on the full held-out dataset.
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
    ensemble_preds = []
    ensemble_probas = []
    ensemble_pred_probas = []

    X_all, y_all = xs_test[0], ys_test[0]
    if intra_model_preds:
        x_tensor_all = torch.tensor(X_all, dtype=torch.float32)
        for model in models:
            probas_all = model(x_tensor_all).detach().cpu().numpy()
            ensemble_pred_probas.append(probas_all)
            ensemble_pred, probas = get_index_and_proba(probas_all.tolist())
            maj_report, cm = test(y=y_all, y_pred=ensemble_pred)
            ensemble_reports.append(maj_report)
            ensemble_cms.append(cm)
            ensemble_preds.append(ensemble_pred)
            ensemble_probas.append(probas)
        return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

    for X, y in zip(xs_test, ys_test):
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_preds = [model(x_tensor).detach().cpu().numpy() for model in models]
        ensemble_pred_proba = majority_vote(y_preds, rule='soft')
        ensemble_pred_probas.append(ensemble_pred_proba)
        ensemble_pred, probas = get_index_and_proba(ensemble_pred_proba)
        ensemble_pred = np.array(ensemble_pred)
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)
        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred)
        ensemble_probas.append(probas)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas
