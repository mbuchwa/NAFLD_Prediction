"""
Derive task-specific biomarker rankings for reduced-feature model selection
using ONLY the UMM training partition.

This script is deliberately separate from shap_publication_figures.py:
- Table 4 / Figures 7-10 explain models on UMM test + MAINZ.
- This script performs FEATURE SELECTION for Table 5 using UMM training data only.

No UMM test data or MAINZ data are used for feature selection.
"""

import os
import pickle
from pathlib import Path
import os
import numpy as np
import pandas as pd
import shap

try:
    from src.preprocess import prepare_data
    from src.utils.ger_eng_dict import dict_germ_eng
except ImportError:
    from preprocess import prepare_data
    from utils.ger_eng_dict import dict_germ_eng


OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "fibrosis",
    "two_stage",
    "cirrhosis",
    # "three_stage",   # uncomment if you also want a ranking for this task
]

TASK_LABELS = {
    "fibrosis": "Moderate fibrosis",
    "two_stage": "Severe fibrosis",
    "cirrhosis": "Cirrhosis",
    "three_stage": "Three-stage",
}

# The window sweep moved this checkpoint out of models/light_gbm/.
PRIMARY_LIGHTGBM_FALLBACK = {
    "two_stage": Path(
        "models/light_gbm_window/pre7_post0/model_two_stage.pickle"
    ),
}


def lightgbm_checkpoint(task):
    path = Path(f"models/light_gbm/model_{task}.pickle")

    if path.exists():
        return path

    fallback = PRIMARY_LIGHTGBM_FALLBACK.get(task)

    if fallback is not None and fallback.exists():
        print(
            f"  {path} missing; using prespecified 7-day checkpoint:\n"
            f"  {fallback}"
        )
        return fallback

    raise FileNotFoundError(
        f"No LightGBM checkpoint found for task '{task}'."
    )


def as_array(shap_values):
    """
    Convert older SHAP list output to ndarray.

    Binary:
        (n_samples, n_features)

    Multiclass:
        (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    return np.asarray(shap_values)


def global_shap_importance(model, x):
    """
    Return one global mean-|SHAP| value per feature.

    Binary:
        mean absolute SHAP over patients.

    Multiclass:
        mean absolute SHAP over patients AND classes.
    """
    explainer = shap.TreeExplainer(model)
    sv = as_array(explainer.shap_values(x))

    if sv.ndim == 3:
        return np.abs(sv).mean(axis=(0, 2))

    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0)

    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")


def main():
    all_rows = []

    print("\nDevelopment-only LightGBM SHAP feature selection")
    print("================================================")
    print("Feature selection uses UMM TRAINING data only.")
    print("UMM test and MAINZ are not used.\n")

    for task in TASKS:
        print(f"\n=== {TASK_LABELS[task]} ===")

        (
            xs_train,
            ys_train,
            xs_val,
            ys_val,
            xs_test,
            ys_test,
            xs_pro,
            ys_pro,
            df_cols_de,
        ) = prepare_data(task, False, False)

        feature_names = [
            dict_germ_eng.get(c, c)
            for c in df_cols_de
        ]

        checkpoint = lightgbm_checkpoint(task)

        print(f"Checkpoint: {checkpoint}")

        with open(checkpoint, "rb") as fh:
            models = pickle.load(fh)

        if len(models) != len(xs_train):
            raise ValueError(
                f"Expected one LightGBM model per imputed training set, "
                f"but found {len(models)} models and {len(xs_train)} training imputations "
                f"for task '{task}'."
            )

        print(f"Ensemble members / training imputations: {len(models)}")
        print(
            "Training samples per imputation: "
            + ", ".join(str(len(x)) for x in xs_train)
        )

        per_member = []

        # IMPORTANT:
        # Model i is explained on the corresponding UMM TRAINING imputation i.
        # No validation, held-out UMM test, or MAINZ observations enter the
        # feature-selection ranking.
        for i, (model, x_train_i) in enumerate(zip(models, xs_train)):
            x_train_i = np.asarray(x_train_i)

            importance = global_shap_importance(
                model,
                x_train_i,
            )

            per_member.append(importance)

            print(
                f"  ensemble member {i + 1}/{len(models)} "
                f"on training imputation {i + 1} done "
                f"(n={len(x_train_i)})"
            )

        per_member = np.vstack(per_member)

        mean_abs = per_member.mean(axis=0)
        sd_abs = per_member.std(axis=0, ddof=1)

        order = np.argsort(mean_abs)[::-1]

        result = pd.DataFrame({
            "task": task,
            "task_label": TASK_LABELS[task],
            "feature": np.asarray(feature_names)[order],
            "rank": np.arange(1, len(feature_names) + 1),
            "mean_abs_shap": mean_abs[order],
            "sd_abs_shap": sd_abs[order],
        })

        result["selected_top3"] = result["rank"] <= 3

        all_rows.append(result)

        print("\nTop 3 selected biomarkers:")
        print(
            result.loc[
                result["rank"] <= 3,
                ["rank", "feature", "mean_abs_shap", "sd_abs_shap"],
            ].to_string(index=False)
        )

    all_df = pd.concat(all_rows, ignore_index=True)

    all_path = OUTDIR / "shap_feature_selection_development_all.csv"
    top3_path = OUTDIR / "shap_feature_selection_development_top3.csv"

    all_df.to_csv(all_path, index=False)
    all_df[all_df["selected_top3"]].to_csv(
        top3_path,
        index=False,
    )

    print("\n================================================")
    print("Saved:")
    print(f"  {all_path}")
    print(f"  {top3_path}")

    print("\nFINAL TOP-3 FEATURE SETS")
    print("========================")

    for task in TASKS:
        subset = all_df[
            (all_df["task"] == task)
            & (all_df["rank"] <= 3)
        ]

        features = ", ".join(subset["feature"].tolist())

        print(
            f"{TASK_LABELS[task]}: {features}"
        )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()