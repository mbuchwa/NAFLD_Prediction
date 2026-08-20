#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

usage() {
    cat <<'USAGE'
Usage: scripts/run_primary_pipeline.sh <stage>

Verified stages:
  train             TRAINS models using the reviewed constants already present
                    in src/run_all_train_experiments.py.
  evaluate          Reads checkpoints and evaluates them using the reviewed
                    constants already present in src/run_all_tests.py.
  three-stage       Reads checkpoints and recomputes the ordinal results table.
  shap              Reads checkpoints and creates canonical publication SHAP
                    attributions and figures (computationally substantial).
  reduced           Reads full and reduced checkpoints and recomputes Table 5.
  clinical-utility  Reads checkpoints and creates calibration/decision-curve
                    outputs.
  tables            Reads checkpoints and recomputes the canonical binary tables.
  figures           Reads checkpoints and prepared data to create publication
                    figures, then checks table/figure consistency.

There is intentionally no "preprocess" stage: the repository has no current,
standalone preprocessing CLI; preprocessing is performed by the configured
training/evaluation entry points. There is intentionally no "all" stage because
the training and evaluation runners' checked-in configurations are not a
matched authoritative full-pipeline configuration. Review their constants and
RETRAINING_PLAYBOOK.md before invoking train or evaluate. This wrapper never
edits those constants or selects a model/task configuration.
USAGE
}

heading() {
    printf '\n================================================================\n'
    printf '%s\n' "$1"
    printf '%s\n' "$2"
    printf '================================================================\n\n'
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

# The supported module entry points below normalize themselves to src when
# needed. Launching them from the repository root preserves their documented
# `python -m src.<module>` working-directory contract.
cd -- "${REPO_ROOT}"

case "$1" in
    train)
        heading "TRAIN" "TRAINS MODELS: expensive optimization and ensemble fitting; uses existing reviewed script constants unchanged."
        "${PYTHON}" -m src.run_all_train_experiments
        ;;
    evaluate)
        heading "EVALUATE" "READS CHECKPOINTS: does not fit models; uses existing reviewed script constants unchanged."
        "${PYTHON}" -m src.run_all_tests
        ;;
    three-stage)
        heading "THREE-STAGE" "READS CHECKPOINTS: recomputes ordinal metrics and Table 3; does not fit models."
        "${PYTHON}" -m src.recompute_three_stage
        ;;
    shap)
        heading "SHAP" "READS CHECKPOINTS: generates canonical attribution/reporting outputs; does not fit models."
        "${PYTHON}" -m src.shap_publication_figures
        ;;
    reduced)
        heading "REDUCED" "READS CHECKPOINTS: compares full/reduced models and recomputes Table 5; does not fit models."
        "${PYTHON}" -m src.recompute_reduced_tables
        ;;
    clinical-utility)
        heading "CLINICAL UTILITY" "READS CHECKPOINTS: computes calibration and decision-curve outputs; does not fit models."
        "${PYTHON}" -m src.clinical_utility_from_checkpoints
        ;;
    tables)
        heading "TABLES" "READS CHECKPOINTS: recomputes canonical binary Tables 1-2; does not fit models."
        "${PYTHON}" -m src.recompute_tables
        ;;
    figures)
        heading "FIGURES" "READS CHECKPOINTS: creates publication figures and verifies table/figure agreement; does not fit models."
        "${PYTHON}" -m src.make_publication_figures
        "${PYTHON}" -m src.check_table_figure_consistency
        ;;
    -h|--help|help)
        usage
        ;;
    preprocess)
        printf 'Error: no verified standalone preprocessing entry point exists; preprocessing is integrated into configured training/evaluation.\n' >&2
        exit 2
        ;;
    all)
        printf 'Error: "all" is intentionally unavailable until a matched authoritative training/evaluation configuration is established.\n' >&2
        exit 2
        ;;
    *)
        printf 'Error: unknown stage: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac
