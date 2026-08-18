"""
clinical_utility_from_checkpoints.py
====================================
Post-hoc calibration and decision-curve analysis for the FINAL binary-model
checkpoints used in the liver-fibrosis manuscript.

NO MODEL RETRAINING IS PERFORMED.

The script is designed to be placed in ``src/`` and run from the project root:

    python -m src.clinical_utility_from_checkpoints

It uses the same data-preparation function and the same ensemble convention as
``recompute_reduced_tables.py``: ensemble member i is evaluated on imputation i,
and predicted probabilities are averaged across the m ensemble members.

Outputs
-------
    src/outputs/clinical_utility/calibration_metrics_all.csv
    src/outputs/clinical_utility/calibration_metrics_manuscript_best.csv
    src/outputs/clinical_utility/predictions_<task>_<cohort>.csv
    src/outputs/clinical_utility/calibration_<task>_<cohort>_<model>.png/.pdf
    src/outputs/clinical_utility/decision_curve_<task>_<cohort>.csv
    src/outputs/clinical_utility/decision_curve_<task>_<cohort>.png/.pdf

Notes
-----
* Binary tasks only. Calibration slope/intercept and standard decision-curve
  analysis are not directly transferable to the three-stage task without a
  separate multiclass formulation.
* The script cross-checks AUROC against the values in Tables 1 and 2. A warning
  is printed if the reconstructed prediction convention does not reproduce the
  manuscript AUROC within AUC_TOLERANCE.
* Calibration-in-the-large (intercept) is fitted with the calibration slope
  fixed at 1. A value of 0 is ideal. Calibration slope is fitted jointly with
  an intercept; a value of 1 is ideal.
* Decision curves use the models' predicted probabilities and compare them with
  treat-all and treat-none. FIB-4/APRI are NOT plotted as probability curves,
  because their established cut-offs are decision rules rather than calibrated
  risk probabilities. Their discrimination/operating-point performance remains
  reported separately in Tables 1 and 2.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASKS = ["fibrosis", "two_stage", "cirrhosis"]
TASK_LABEL = {
    "fibrosis": "Moderate fibrosis",
    "two_stage": "Severe fibrosis",
    "cirrhosis": "Cirrhosis",
}

MODELS = [
    ("SVM", "svm"),
    ("Random Forest", "rf"),
    ("XGBoost", "xgb"),
    ("LightGBM", "light_gbm"),
    ("MLP", "ffn"),
    ("TabTransformer", "tab_transformer"),
    ("VI-BNN", "vi_bnn"),
    ("GANDALF", "gandalf"),
]

# Keep this in sync with the preprocessing used for the final manuscript run.
SCALING_MODELS = {"vi_bnn"}

# The laboratory-window sweep may have moved the manuscript LightGBM severe-
# fibrosis checkpoint. Use only the 7-day reference checkpoint as fallback.
PRIMARY_LIGHTGBM_FALLBACK = {
    "two_stage": Path("models/light_gbm_window/pre7_post0/model_two_stage.pickle"),
}

# Best-AUROC models actually reported in the current manuscript. These are used
# only to create compact manuscript-ready calibration/DCA figures. The CSV files
# contain results for every model that could be loaded.
MANUSCRIPT_BEST = {
    "UMM": {
        "fibrosis": "rf",
        "two_stage": "ffn",
        "cirrhosis": "tab_transformer",
    },
    "MAINZ": {
        "fibrosis": "light_gbm",
        "two_stage": "light_gbm",
        "cirrhosis": "vi_bnn",
    },
}

# Table 1 / Table 2 AUROC point estimates. Used only as a reconstruction check.
EXPECTED_AUROC = {
    "UMM": {
        "fibrosis": {
            "svm": 0.714, "rf": 0.859, "xgb": 0.821, "light_gbm": 0.799,
            "ffn": 0.825, "tab_transformer": 0.821, "vi_bnn": 0.842,
            "gandalf": 0.756,
        },
        "two_stage": {
            "svm": 0.786, "rf": 0.819, "xgb": 0.765, "light_gbm": 0.803,
            "ffn": 0.920, "tab_transformer": 0.895, "vi_bnn": 0.811,
            "gandalf": 0.811,
        },
        "cirrhosis": {
            "svm": 0.850, "rf": 0.823, "xgb": 0.736, "light_gbm": 0.800,
            "ffn": 0.891, "tab_transformer": 0.900, "vi_bnn": 0.791,
            "gandalf": 0.736,
        },
    },
    "MAINZ": {
        "fibrosis": {
            "svm": 0.835, "rf": 0.845, "xgb": 0.849, "light_gbm": 0.865,
            "ffn": 0.855, "tab_transformer": 0.845, "vi_bnn": 0.857,
            "gandalf": 0.601,
        },
        "two_stage": {
            "svm": 0.878, "rf": 0.910, "xgb": 0.918, "light_gbm": 0.925,
            "ffn": 0.885, "tab_transformer": 0.898, "vi_bnn": 0.917,
            "gandalf": 0.656,
        },
        "cirrhosis": {
            "svm": 0.858, "rf": 0.888, "xgb": 0.862, "light_gbm": 0.879,
            "ffn": 0.841, "tab_transformer": 0.891, "vi_bnn": 0.906,
            "gandalf": 0.656,
        },
    },
}

OUT_DIR = Path("outputs/clinical_utility")
N_BOOT = 1000
SEED = 42
AUC_TOLERANCE = 0.005
VI_BNN_POSTERIOR_SAMPLES = 200
DCA_THRESHOLDS = np.arange(0.05, 0.951, 0.01)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _tree_checkpoint_path(model_name: str, task: str) -> Path:
    path = Path(f"models/{model_name}/model_{task}.pickle")
    if path.exists():
        return path
    if model_name == "light_gbm":
        fallback = PRIMARY_LIGHTGBM_FALLBACK.get(task)
        if fallback is not None and fallback.exists():
            print(f"    NOTE: using 7-day LightGBM fallback: {fallback}")
            return fallback
    return path


def _load_ensemble(model_name: str, task: str):
    """Load the final ensemble without retraining.

    Classical models are pickle ensembles. Neural/tabular models are delegated
    to ``src.neural_loaders`` if available, matching the loader already used by
    the current recomputation scripts.
    """
    path = _tree_checkpoint_path(model_name, task)
    if path.exists():
        with open(path, "rb") as f:
            models = pickle.load(f)
        return list(models) if isinstance(models, (list, tuple)) else [models]

    try:
        from src.neural_loaders import load_any_ensemble, LOADERS
    except ImportError:
        try:
            from neural_loaders import load_any_ensemble, LOADERS
        except ImportError as exc:
            raise RuntimeError(
                "Could not import neural_loaders. Tree models can be evaluated, "
                "but MLP/TabTransformer/VI-BNN/GANDALF require the repository's "
                "neural_loaders.py."
            ) from exc

    if model_name not in LOADERS:
        raise RuntimeError(
            f"No loader registered for {model_name!r} in neural_loaders.LOADERS."
        )

    models = load_any_ensemble(
        model_name,
        task,
        model_dir=f"models/{model_name}",
    )
    return list(models) if isinstance(models, (list, tuple)) else [models]


# ---------------------------------------------------------------------------
# Probability inference
# ---------------------------------------------------------------------------

def _as_probability_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        # If already in [0,1], treat as positive-class probability; otherwise
        # interpret as a decision score/logit.
        if np.all(np.isfinite(arr)) and np.all((arr >= 0) & (arr <= 1)):
            q = arr
        else:
            q = expit(arr)
        return np.column_stack([1.0 - q, q])

    if arr.ndim == 2 and arr.shape[1] == 1:
        q0 = arr[:, 0]
        if np.all(np.isfinite(q0)) and np.all((q0 >= 0) & (q0 <= 1)):
            q = q0
        else:
            q = expit(q0)
        return np.column_stack([1.0 - q, q])

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected binary probability/logit output; got {arr.shape}.")

    rowsum = arr.sum(axis=1)
    is_prob = (
        np.all(np.isfinite(arr))
        and np.all(arr >= -1e-7)
        and np.all(arr <= 1.0 + 1e-7)
        and np.allclose(rowsum, 1.0, atol=1e-4, rtol=1e-4)
    )
    if is_prob:
        return arr

    # Treat two-column output as logits/log-probabilities.
    z = arr - np.max(arr, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def _extract_dataframe_probabilities(pred_df: pd.DataFrame) -> np.ndarray | None:
    if not isinstance(pred_df, pd.DataFrame):
        return None

    candidates = [
        ("0_probability", "1_probability"),
        ("0_prob", "1_prob"),
        ("probability_0", "probability_1"),
    ]
    for c0, c1 in candidates:
        if c0 in pred_df.columns and c1 in pred_df.columns:
            return pred_df[[c0, c1]].to_numpy(dtype=float)

    # Last-resort pattern used by some pytorch-tabular versions.
    prob_cols = [c for c in pred_df.columns if "probability" in str(c).lower()]
    if len(prob_cols) >= 2:
        prob_cols = sorted(prob_cols)[:2]
        return pred_df[prob_cols].to_numpy(dtype=float)
    return None


def _torch_member_proba(model, x: np.ndarray, vi_bnn: bool = False) -> np.ndarray:
    import torch

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    if hasattr(model, "eval"):
        model.eval()

    def _tensor_from_output(out):
        if torch.is_tensor(out):
            return out
        if isinstance(out, dict):
            for key in ("logits", "y_hat", "pred", "prediction", "predictions", "output"):
                if key in out and torch.is_tensor(out[key]):
                    return out[key]
            for value in out.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(out, (tuple, list)):
            for value in out:
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"Cannot extract tensor from {type(out).__name__} output.")

    with torch.no_grad():
        if vi_bnn:
            # Match the current manuscript loader's posterior averaging.
            probs = []
            for _ in range(VI_BNN_POSTERIOR_SAMPLES):
                out = _tensor_from_output(model(xt))
                probs.append(_as_probability_matrix(out.detach().cpu().numpy()))
            return np.mean(probs, axis=0)

        out = _tensor_from_output(model(xt))
        return _as_probability_matrix(out.detach().cpu().numpy())


def _predict_member_proba(model, x, model_name: str, df_cols) -> np.ndarray:
    x = np.asarray(x)

    if hasattr(model, "predict_proba"):
        return _as_probability_matrix(np.asarray(model.predict_proba(x)))

    if hasattr(model, "decision_function"):
        return _as_probability_matrix(np.asarray(model.decision_function(x)))

    # PyTorch/Lightning models.
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return _torch_member_proba(model, x, vi_bnn=(model_name == "vi_bnn"))
    except ImportError:
        pass

    # pytorch-tabular / GANDALF style predictor.
    if hasattr(model, "predict"):
        try:
            pred = model.predict(pd.DataFrame(x, columns=df_cols))
        except Exception:
            pred = model.predict(x)

        df_p = _extract_dataframe_probabilities(pred) if isinstance(pred, pd.DataFrame) else None
        if df_p is not None:
            return _as_probability_matrix(df_p)

        arr = np.asarray(pred)
        # Do NOT silently calibrate class labels as probabilities.
        if arr.ndim == 1 and np.all(np.isin(np.unique(arr), [0, 1])):
            raise ValueError(
                f"{model_name} .predict() returned class labels, not probabilities. "
                "The native loader must expose probability output for calibration/DCA."
            )
        return _as_probability_matrix(arr)

    raise TypeError(
        f"Cannot obtain probabilities from {model_name} member of type {type(model).__name__}."
    )


def _ensemble_proba(models, xs, model_name: str, df_cols) -> np.ndarray:
    """Member i on imputation i, then soft-average probabilities."""
    xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
    if not xs:
        raise ValueError("No imputed feature matrices supplied.")

    member_probas = []
    for i, model in enumerate(models):
        x_i = xs[i] if i < len(xs) else xs[0]
        member_probas.append(_predict_member_proba(model, x_i, model_name, df_cols))

    shapes = {p.shape for p in member_probas}
    if len(shapes) != 1:
        raise ValueError(f"Ensemble members returned incompatible shapes: {shapes}")

    p = np.mean(member_probas, axis=0)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Non-positive ensemble probability row sum encountered.")
    return p / row_sums


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------

def _safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibration_intercept(y, p) -> float:
    """Calibration-in-the-large with slope fixed at 1; ideal value = 0."""
    y = np.asarray(y, dtype=float)
    lp = _safe_logit(p)
    if len(np.unique(y)) < 2:
        return np.nan

    def score(alpha):
        return float(np.sum(expit(alpha + lp) - y))

    # Wide bounds comfortably cover realistic prevalence shifts.
    try:
        return float(brentq(score, -30.0, 30.0))
    except Exception:
        return np.nan


def _calibration_joint(y, p):
    """Joint logistic recalibration y ~ intercept + slope * logit(p)."""
    y = np.asarray(y, dtype=float)
    lp = _safe_logit(p)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    def nll(theta):
        alpha, beta = theta
        z = alpha + beta * lp
        # Stable Bernoulli negative log-likelihood.
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    try:
        res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
        if not res.success and not np.all(np.isfinite(res.x)):
            return np.nan, np.nan
        return float(res.x[0]), float(res.x[1])
    except Exception:
        return np.nan, np.nan


def _calibration_metrics(y, p):
    y = np.asarray(y).astype(int).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    recal_intercept, slope = _calibration_joint(y, p)
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": _calibration_intercept(y, p),
        "calibration_slope": slope,
        "recalibration_intercept_joint": recal_intercept,
        "mean_predicted_risk": float(np.mean(p)),
        "observed_prevalence": float(np.mean(y)),
    }


def _bootstrap_calibration(y, p, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    vals = {"brier_score": [], "calibration_intercept": [], "calibration_slope": []}
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = _calibration_metrics(y[idx], p[idx])
        for key in vals:
            if np.isfinite(m[key]):
                vals[key].append(m[key])

    out = {}
    for key, arr in vals.items():
        if arr:
            out[f"{key}_ci_low"] = float(np.percentile(arr, 2.5))
            out[f"{key}_ci_high"] = float(np.percentile(arr, 97.5))
        else:
            out[f"{key}_ci_low"] = np.nan
            out[f"{key}_ci_high"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def _net_benefit(y, p, thresholds):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    n = len(y)
    rows = []

    for pt in thresholds:
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        rows.append((float(pt), float(nb)))
    return rows


def _decision_curve_dataframe(y, model_probabilities: dict[str, np.ndarray]):
    y = np.asarray(y).astype(int).ravel()
    prevalence = float(np.mean(y))

    df = pd.DataFrame({"threshold_probability": DCA_THRESHOLDS})
    df["treat_none"] = 0.0
    df["treat_all"] = (
        prevalence
        - (1.0 - prevalence)
        * (df["threshold_probability"] / (1.0 - df["threshold_probability"]))
    )

    for model_name, p in model_probabilities.items():
        nb = _net_benefit(y, p, DCA_THRESHOLDS)
        df[model_name] = [x[1] for x in nb]
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_calibration_plot(y, p, model_label, task, cohort, stem):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()

    # The UMM test set is only n=31; five quantile bins are more defensible than
    # ten. MAINZ uses ten bins.
    n_bins = 5 if len(y) < 100 else 10
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    met = _calibration_metrics(y, p)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=model_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event proportion")
    ax.set_title(f"{TASK_LABEL[task]} – {cohort}")
    ax.grid(axis="both", alpha=0.2)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.03,
        0.97,
        (
            f"n = {len(y)}\n"
            f"Brier = {met['brier_score']:.3f}\n"
            f"Intercept = {met['calibration_intercept']:.2f}\n"
            f"Slope = {met['calibration_slope']:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_decision_curve_plot(dca_df, best_label, best_key, task, cohort, stem):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(
        dca_df["threshold_probability"],
        dca_df[best_key],
        linewidth=2.0,
        label=best_label,
    )
    ax.plot(
        dca_df["threshold_probability"],
        dca_df["treat_all"],
        linestyle="--",
        linewidth=1.4,
        label="Treat all",
    )
    ax.plot(
        dca_df["threshold_probability"],
        dca_df["treat_none"],
        linestyle=":",
        linewidth=1.4,
        label="Treat none",
    )
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"{TASK_LABEL[task]} – {cohort}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        from src.preprocess import prepare_data
    except ImportError:
        from preprocess import prepare_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    best_rows = []

    for task in TASKS:
        print(f"\n=== {TASK_LABEL[task]} ({task}) ===")

        # Prepare once unscaled and once scaled, only if needed.
        data_unscaled = prepare_data(task, False, False)
        data_scaled = None

        # Collect per-cohort probabilities for DCA and prediction export.
        cohort_probs = {"UMM": {}, "MAINZ": {}}
        cohort_y = {}

        for model_label, model_name in MODELS:
            print(f"  {model_label} ...")

            try:
                models = _load_ensemble(model_name, task)
            except Exception as exc:
                warnings.warn(f"Skipping {model_name}/{task}: {exc}")
                continue

            if model_name in SCALING_MODELS:
                if data_scaled is None:
                    data_scaled = prepare_data(task, False, True)
                data = data_scaled
            else:
                data = data_unscaled

            # prepare_data returns:
            # 0 xs_train, 1 ys_train, 2 xs_val, 3 ys_val,
            # 4 xs_test,  5 ys_test, 6 xs_mainz, 7 ys_mainz, 8 df_cols
            df_cols = data[8]

            for cohort, x_idx, y_idx in (("UMM", 4, 5), ("MAINZ", 6, 7)):
                y = np.asarray(data[y_idx][0]).astype(int).ravel()
                try:
                    pmat = _ensemble_proba(models, data[x_idx], model_name, df_cols)
                except Exception as exc:
                    warnings.warn(f"Prediction failed for {model_name}/{task}/{cohort}: {exc}")
                    continue

                p = np.asarray(pmat[:, 1], dtype=float)
                auc = float(roc_auc_score(y, p))
                expected = EXPECTED_AUROC.get(cohort, {}).get(task, {}).get(model_name, np.nan)
                auc_diff = auc - expected if np.isfinite(expected) else np.nan

                if np.isfinite(expected) and abs(auc_diff) > AUC_TOLERANCE:
                    warnings.warn(
                        f"AUROC reconstruction mismatch for {model_name}/{task}/{cohort}: "
                        f"computed {auc:.3f}, manuscript {expected:.3f} (diff {auc_diff:+.3f}). "
                        "Do not use calibration/DCA for this model until the inference "
                        "convention/checkpoint is reconciled."
                    )

                cal = _calibration_metrics(y, p)
                cal_ci = _bootstrap_calibration(y, p)

                row = {
                    "task": task,
                    "task_label": TASK_LABEL[task],
                    "cohort": cohort,
                    "model": model_name,
                    "model_label": model_label,
                    "n": int(len(y)),
                    "events": int(np.sum(y)),
                    "prevalence": float(np.mean(y)),
                    "auroc_reconstructed": auc,
                    "auroc_manuscript": expected,
                    "auroc_difference": auc_diff,
                    "auroc_matches_manuscript": bool(
                        (not np.isfinite(expected)) or abs(auc_diff) <= AUC_TOLERANCE
                    ),
                    **cal,
                    **cal_ci,
                }
                metrics_rows.append(row)

                cohort_probs[cohort][model_name] = p
                cohort_y[cohort] = y

                # Individual calibration plot for every successfully reconstructed model.
                _save_calibration_plot(
                    y,
                    p,
                    model_label,
                    task,
                    cohort,
                    OUT_DIR / f"calibration_{task}_{cohort.lower()}_{model_name}",
                )

        # Export patient-level probabilities and DCA data separately per cohort.
        for cohort in ("UMM", "MAINZ"):
            if cohort not in cohort_y or not cohort_probs[cohort]:
                continue

            y = cohort_y[cohort]
            pred_df = pd.DataFrame({
                "row_index": np.arange(len(y)),
                "outcome": y,
            })
            for model_name, p in cohort_probs[cohort].items():
                pred_df[f"p_{model_name}"] = p
            pred_df.to_csv(
                OUT_DIR / f"predictions_{task}_{cohort.lower()}.csv",
                index=False,
            )

            dca = _decision_curve_dataframe(y, cohort_probs[cohort])
            dca.to_csv(
                OUT_DIR / f"decision_curve_{task}_{cohort.lower()}.csv",
                index=False,
            )

            best_key = MANUSCRIPT_BEST[cohort][task]
            if best_key in cohort_probs[cohort]:
                label_map = dict((key, label) for label, key in MODELS)
                _save_decision_curve_plot(
                    dca,
                    label_map.get(best_key, best_key),
                    best_key,
                    task,
                    cohort,
                    OUT_DIR / f"decision_curve_{task}_{cohort.lower()}",
                )

    metrics = pd.DataFrame(metrics_rows)
    if metrics.empty:
        raise SystemExit("No models could be evaluated.")

    metrics.to_csv(OUT_DIR / "calibration_metrics_all.csv", index=False)

    # Compact table containing the same best model per task/cohort as the current
    # manuscript performance narrative.
    for cohort in ("UMM", "MAINZ"):
        for task in TASKS:
            best_key = MANUSCRIPT_BEST[cohort][task]
            hit = metrics[
                (metrics["cohort"] == cohort)
                & (metrics["task"] == task)
                & (metrics["model"] == best_key)
            ]
            if not hit.empty:
                best_rows.append(hit.iloc[0].to_dict())

    best = pd.DataFrame(best_rows)
    best.to_csv(OUT_DIR / "calibration_metrics_manuscript_best.csv", index=False)

    # A concise terminal summary for copy/paste back into ChatGPT.
    cols = [
        "task_label", "cohort", "model_label", "n", "events",
        "auroc_reconstructed", "auroc_manuscript", "brier_score",
        "calibration_intercept", "calibration_slope",
    ]
    print("\n=== Manuscript-best calibration summary ===")
    if not best.empty:
        print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("No manuscript-best rows could be reconstructed.")

    print(f"\nOutputs written to: {OUT_DIR.resolve()}")
    print(
        "\nIMPORTANT: Before manuscript use, inspect 'auroc_matches_manuscript'. "
        "Any False row indicates that the loaded checkpoint/inference convention does "
        "not reproduce Tables 1/2 and should not be interpreted for calibration/DCA."
    )


if __name__ == "__main__":
    # Allows both `python src/clinical_utility_from_checkpoints.py` and
    # `python -m src.clinical_utility_from_checkpoints` from the project root.
    os.chdir(Path(__file__).resolve().parent)
    main()
