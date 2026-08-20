#!/usr/bin/env python3
"""Read-only prerequisite check for the repository's primary pipeline.

This module deliberately inspects paths and import specifications only.  In
particular, it does not import third-party packages: importing packages such as
PyTorch can initialize hardware or otherwise have avoidable side effects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable


REQUIRED_IMPORTS = {
    # Cohort preparation, analysis, and classical models.
    "numpy": "NumPy",
    "pandas": "pandas",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "matplotlib": "Matplotlib",
    "seaborn": "seaborn",
    "openpyxl": "openpyxl",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    # Neural models and the canonical checkpoint/SHAP workflow.
    "torch": "PyTorch",
    "pyro": "Pyro",
    "pytorch_lightning": "PyTorch Lightning",
    "torchmetrics": "TorchMetrics",
    "tab_transformer_pytorch": "tab-transformer-pytorch",
    "pytorch_tabular": "PyTorch Tabular",
    "shap": "SHAP",
    "te2rules": "TE2Rules",
    "tqdm": "tqdm",
}

# These support deliberately disabled branches or non-primary utilities.
OPTIONAL_IMPORTS = {
    "imblearn": "imbalanced-learn (SMOTE branch)",
}
AUXILIARY_IMPORTS = {
    "requests": "Requests (download/network utilities)",
    "yaml": "PyYAML (environment/config tooling)",
}

DATA_SPLITS = ("train", "val", "test", "prospective")
MODEL_IDS = (
    "svm",
    "rf",
    "xgb",
    "light_gbm",
    "ffn",
    "vi_bnn",
    "gandalf",
    "tab_transformer",
)
ARTIFACT_SUFFIXES = (".pickle", ".pth", ".pt", ".ckpt")


def repository_root() -> Path:
    """Locate the checkout independently of the caller's working directory."""
    candidate = Path(__file__).resolve().parent
    for directory in (candidate, *candidate.parents):
        if (directory / "README.md").is_file() and (directory / "src").is_dir():
            return directory
    raise RuntimeError("could not locate repository root from checker path")


def check_imports(label: str, imports: dict[str, str]) -> list[str]:
    """Report module discoverability without importing any package."""
    missing: list[str] = []
    print(f"\n{label} dependencies:")
    for module, description in imports.items():
        available = importlib.util.find_spec(module) is not None
        print(f"  [{'OK' if available else 'MISSING'}] {description} ({module})")
        if not available:
            missing.append(module)
    return missing


def count_files(directory: Path, patterns: Iterable[str]) -> int:
    """Count matching regular files without exposing potentially sensitive names."""
    if not directory.is_dir():
        return 0
    return sum(1 for pattern in patterns for path in directory.glob(pattern) if path.is_file())


def report_data(root: Path) -> None:
    data = root / "data"
    print("\nData locations (private data may be intentionally absent):")
    print(f"  [{'OK' if data.is_dir() else 'WARNING'}] data/ directory")
    raw_count = count_files(data, ("*.xlsx", "*.xls"))
    print(
        "  [WARNING] controlled raw workbooks are absent"
        if raw_count == 0
        else f"  [OK] controlled raw workbook candidates: {raw_count} (names withheld)"
    )
    for split in DATA_SPLITS:
        clean = data / f"preprocessed_no_mice_{split}"
        imputed = data / f"preprocessed_mice_fib_{split}"
        clean_count = count_files(clean, ("*.csv",))
        imputed_count = count_files(imputed, ("*.csv",))
        print(
            f"  [{'OK' if clean.is_dir() else 'WARNING'}] clean {split} directory; "
            f"candidate tables: {clean_count}"
        )
        print(
            f"  [{'OK' if imputed.is_dir() else 'WARNING'}] imputed {split} directory; "
            f"candidate tables: {imputed_count}"
        )


def report_models(root: Path) -> None:
    models = root / "src" / "models"
    reserved = root / "src" / "checkpoints"
    print("\nModel/checkpoint locations (checkpoints may be intentionally absent):")
    print(f"  [{'OK' if models.is_dir() else 'WARNING'}] canonical src/models/ directory")
    for model_id in MODEL_IDS:
        directory = models / model_id
        candidates = count_files(directory, (f"*{suffix}" for suffix in ARTIFACT_SUFFIXES))
        # GANDALF may save a checkpoint as a directory with a .pth suffix.
        if directory.is_dir():
            candidates += sum(1 for path in directory.glob("*.pth") if path.is_dir())
        state = "OK" if candidates else "WARNING"
        print(f"  [{state}] {model_id}: {candidates} candidate artifact(s)")
    print(
        f"  [INFO] reserved legacy src/checkpoints/ directory: "
        f"{'present (not searched by the pipeline)' if reserved.is_dir() else 'absent'}"
    )


def main() -> int:
    try:
        root = repository_root()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Repository root: {root}")
    print(f"Python version: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        print("[WARNING] canonical environment uses Python 3.11")

    missing_required = check_imports("Required", REQUIRED_IMPORTS)
    missing_optional = check_imports("Optional", OPTIONAL_IMPORTS)
    missing_auxiliary = check_imports("Auxiliary", AUXILIARY_IMPORTS)
    report_data(root)
    report_models(root)

    print("\nSummary:")
    if missing_optional:
        print(f"  [WARNING] optional dependencies missing: {len(missing_optional)}")
    if missing_auxiliary:
        print(f"  [WARNING] auxiliary dependencies missing: {len(missing_auxiliary)}")
    if missing_required:
        print(f"  [FAIL] required dependencies missing: {len(missing_required)}")
        return 1
    print("  [PASS] all required software is discoverable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
