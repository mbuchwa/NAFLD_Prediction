"""Generate reporting summary CSV artifacts from documented pipeline settings."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_preprocessing_summary() -> pd.DataFrame:
    rows = [
        {
            "sequence_order": 1,
            "stage": "cleaning",
            "implementation_summary": "Temporal filtering to nearest pre-biopsy lab, age derivation, string cleanup, censored-value handling (default: set to NaN), decimal normalization, and invalid token replacement.",
            "source": "src/preprocess.py::preprocess(), clean_df(), handle_operator()",
        },
        {
            "sequence_order": 2,
            "stage": "missing-data handling",
            "implementation_summary": "Rows with missing target (Micro) are removed; rows with >70% missing features are dropped via thresholded row-wise filtering.",
            "source": "src/preprocess.py::preprocess(), drop_rows_with_high_missing_data()",
        },
        {
            "sequence_order": 3,
            "stage": "imputation",
            "implementation_summary": "Multiple Imputation by Chained Equations (IterativeImputer) with sample_posterior=True; 10 imputed datasets are created with random_state in [0..9] and min_value=0.",
            "source": "src/preprocess.py::mice()",
        },
        {
            "sequence_order": 4,
            "stage": "scaling/encoding",
            "implementation_summary": "Target is encoded via categorize_micro() to task labels; optional StandardScaler is fit on training imputation and applied to downstream splits.",
            "source": "src/preprocess.py::preprocess(), categorize_micro()",
        },
        {
            "sequence_order": 5,
            "stage": "outlier policy",
            "implementation_summary": "No explicit winsorization/clipping/IQR outlier removal is implemented; extreme or censored operator values are handled through censored-value policy in cleaning.",
            "source": "src/preprocess.py::clean_df(), handle_operator()",
        },
    ]
    return pd.DataFrame(rows)


def build_hyperparameter_search_summary() -> pd.DataFrame:
    rows = [
        {
            "model": "SVM",
            "tuned_params": "degree",
            "ranges": "degree: [2, 3, 4]",
            "search_strategy": "RandomizedSearchCV with PredefinedSplit",
            "optimization_metric": "neg_log_loss",
            "number_of_trials": 10,
        },
        {
            "model": "RandomForest",
            "tuned_params": "n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap",
            "ranges": "n_estimators: 10..100 step 10; max_depth: None or 2..10; min_samples_split: {2,5,10}; min_samples_leaf: {1,2,4}; bootstrap: {True,False}",
            "search_strategy": "RandomizedSearchCV with PredefinedSplit",
            "optimization_metric": "neg_log_loss",
            "number_of_trials": 10,
        },
        {
            "model": "XGBoost",
            "tuned_params": "max_depth, learning_rate, subsample",
            "ranges": "max_depth: 1..39; learning_rate: linspace(0.5,0.01,5); subsample: linspace(1.0,0.3,5)",
            "search_strategy": "RandomizedSearchCV (5-fold CV)",
            "optimization_metric": "neg_log_loss",
            "number_of_trials": 10,
        },
        {
            "model": "LightGBM",
            "tuned_params": "max_depth, learning_rate, subsample",
            "ranges": "max_depth: 1..39; learning_rate: linspace(0.5,0.01,5); subsample: linspace(1.0,0.3,5)",
            "search_strategy": "RandomizedSearchCV (5-fold CV)",
            "optimization_metric": "accuracy (three_stage) or neg_mean_squared_error (binary tasks)",
            "number_of_trials": 10,
        },
        {
            "model": "Feedforward NN",
            "tuned_params": "lr, dropout, num_layers, hidden_dim",
            "ranges": "lr: linspace(0.005,0.0005,5); dropout: linspace(0.3,0.0,5); num_layers: 3..6; hidden_dim: {32,64,128,256}",
            "search_strategy": "Manual random search with KFold CV",
            "optimization_metric": "mean validation accuracy",
            "number_of_trials": 10,
        },
        {
            "model": "VI-BNN",
            "tuned_params": "lr, dropout, num_layers, hidden_dim",
            "ranges": "lr: linspace(0.01,0.001,2); dropout: linspace(0.2,0.0,2); num_layers: 1..2; hidden_dim: {32,64}",
            "search_strategy": "Manual random search using hold-out validation",
            "optimization_metric": "validation accuracy",
            "number_of_trials": 5,
        },
        {
            "model": "MCMC-BNN",
            "tuned_params": "num_layers, hidden_dim",
            "ranges": "num_layers: 1..2; hidden_dim: {32,64}",
            "search_strategy": "Manual random search with hold-out validation",
            "optimization_metric": "classification report accuracy",
            "number_of_trials": 5,
        },
        {
            "model": "GANDALF",
            "tuned_params": "learning_rate, gflu_dropout, gflu_stages",
            "ranges": "learning_rate: linspace(0.01,0.0001,5); gflu_dropout: linspace(0.5,0.0,5); gflu_stages: 2..6",
            "search_strategy": "Manual random search with KFold CV",
            "optimization_metric": "mean validation accuracy",
            "number_of_trials": 30,
        },
        {
            "model": "TabTransformer",
            "tuned_params": "lr, dropout, num_layers, hidden_dim",
            "ranges": "lr: linspace(0.005,0.0001,5); dropout: linspace(0.5,0.0,5); num_layers: 2..6; hidden_dim: {32,64,128,256}",
            "search_strategy": "Manual random search with KFold CV",
            "optimization_metric": "mean validation accuracy",
            "number_of_trials": 30,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_summary_path = output_dir / "preprocessing_summary.csv"
    hyperparameter_summary_path = output_dir / "hyperparameter_search_summary.csv"

    build_preprocessing_summary().to_csv(preprocessing_summary_path, index=False)
    build_hyperparameter_search_summary().to_csv(hyperparameter_summary_path, index=False)

    print(f"Wrote {preprocessing_summary_path}")
    print(f"Wrote {hyperparameter_summary_path}")


if __name__ == "__main__":
    main()
