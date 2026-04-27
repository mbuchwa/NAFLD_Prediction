# Missingness Assumptions Report Template (Short)

## Dataset / Extraction Context
- Dataset version:
- Extraction date:
- Analyst:
- Cohort definition:

## Missingness Mechanism Assessment (MCAR / MAR / MNAR)
Document likely mechanisms for high-missingness features from `missingness_profile.csv`.

| Feature | Missingness % (before cleaning) | Likely Mechanism (MCAR/MAR/MNAR) | Rationale / Evidence | Planned Handling |
|---|---:|---|---|---|
| example_feature | 0.0 | MCAR | Replace with cohort-specific rationale. | Impute / Exclude / Model-based |

## Features Flagged >20% Missingness
- List flagged features (`flag_gt_20pct_before` and/or `flag_gt_20pct_after`):
- Clinical or operational explanation:
- Bias risk and downstream model implications:

## Justification for >70% Row Exclusion Threshold
- Threshold used in preprocessing: rows with >70% missing values are excluded.
- Why this threshold is reasonable for this cohort:
- Sensitivity analysis plan (e.g., 60% / 80% thresholds):
- Impact on cohort size and representativeness:

## Validation / Sensitivity Checks
- Compare model performance with and without high-missingness rows:
- Check whether exclusions disproportionately affect subgroups:
- Record final decision and sign-off:
