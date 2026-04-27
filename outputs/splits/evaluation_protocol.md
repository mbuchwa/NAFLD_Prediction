# Evaluation Protocol

This project follows a strict split protocol to prevent leakage from held-out data.

## Split flow

`Train (CV inside) -> Val -> Test (single final pass)`

## Rules

1. **Train**: model fitting and hyperparameter search happen here.
2. **CV inside Train/Val only**: any fold-based selection is restricted to the training workflow and uses `xs_train`/`xs_val`.
3. **Val**: used for tuning decisions and early stopping as part of model development.
4. **Test**: used exactly once per evaluation run (no StratifiedKFold or any test re-splitting).
5. **Prospective cohort**: evaluated once as an external single-pass check, separate from train/val/test optimization.
