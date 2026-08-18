"""Compatibility entry point for the archived dataset-variance exploration."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parents[1] / "misc/exploratory/variance_test_datasets.py",
        run_name="__main__",
    )
