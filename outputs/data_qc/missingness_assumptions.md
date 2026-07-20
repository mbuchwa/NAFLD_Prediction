# Missingness Assumptions Report Template (Short)

## Dataset / Extraction Context
- Dataset version:
- Extraction date:
- Analyst:
- Cohort definition:

## Missingness Mechanism Assessment (MCAR / MAR / MNAR)
Document likely mechanisms for high-missingness features from `missingness_profile.csv`.

| Feature | Missingness % (before cleaning) | Missingness % (after cleaning) | Likely Mechanism (MCAR/MAR/MNAR) | Rationale / Evidence | Planned Handling |
|---|---:|---:|---|---|---|
| example_feature | 0.0 | 0.0 | MCAR | Measurement appears unrelated to patient status or observed covariates; replace with cohort-specific evidence. | Impute / Exclude / Model-based |
| optional_or_specialty_lab | TBD | TBD | MAR | Missingness may reflect ordering patterns explained by clinic, disease severity, or other observed labs. | Consider imputation plus sensitivity checks. |
| suspected_unrecorded_due_to_value | TBD | TBD | MNAR | Missingness may depend on the unobserved value itself, assay limits, or selective documentation. | Consider exclusion, indicator features, or sensitivity analysis. |

## Features Flagged >20% Missingness
- List flagged features (`flag_gt_20pct_before` and/or `flag_gt_20pct_after`):
- Clinical or operational explanation:
- Bias risk and downstream model implications:
- Decision rule for retention/exclusion:

## Justification for >70% Row Exclusion Threshold
- Threshold used in preprocessing: rows with >70% missing values are excluded before imputation.
- Rationale: patients above this threshold have too little observed laboratory evidence for reliable multivariate imputation and may represent incomplete/duplicate exports rather than analytically usable encounters.
- Why this threshold is reasonable for this cohort:
- Sensitivity analysis plan (e.g., 60% / 80% thresholds):
- Impact on cohort size and representativeness:

## Validation / Sensitivity Checks
- Compare model performance with and without high-missingness rows:
- Check whether exclusions disproportionately affect subgroups:
- Record final decision and sign-off:
