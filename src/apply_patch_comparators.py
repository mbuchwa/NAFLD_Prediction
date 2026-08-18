"""Compatibility entry point for the archived comparator migration script."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parents[1] / "misc/legacy/apply_patch_comparators.py",
        run_name="__main__",
    )
