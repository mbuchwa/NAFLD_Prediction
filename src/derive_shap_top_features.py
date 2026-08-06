"""
derive_shap_top_features.py
===========================
Reads the SHAP values produced by the full-biomarker run and writes the top-3
biomarkers per task to a JSON file, so the reduced-feature run does not depend
on a hand-edited dict inside preprocess.py.

Run from:  src/   ->   python -m src.derive_shap_top_features
Input:     outputs/figures/shap_all_features.csv   (written by shap_publication_figures.py)
Output:    outputs/shap_top_features.json

WHY A FILE AND NOT AN EDIT
--------------------------
The hard-coded dict in preprocess.py carries the German column names of the
2024 run. After every retraining the ranking can change, and a stale dict makes
the reduced models silently use different biomarkers than the SHAP table
reports. Reading from a file that is regenerated together with the SHAP values
keeps the two in step, and the JSON is a checkable artefact for the appendix.

NAME MAPPING
------------
The SHAP CSV holds ENGLISH names (translated in shap_publication_figures.py via
dict_germ_eng), while preprocess.py selects on the GERMAN column names. This
script inverts dict_germ_eng and verifies every derived name against the actual
column list returned by prepare_data, so a mapping gap fails here rather than
inside the training sweep.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

SHAP_CSV = Path('outputs/figures/shap_all_features.csv')
OUT_JSON = Path('outputs/shap_top_features.json')
TOP_N = 3
COHORT = 'UMM'          # rank on the development cohort, not the external one

# Model whose ranking defines the reduced set, per task. Keep in sync with
# BEST_MODEL_PER_TASK in shap_publication_figures.py.
BEST_MODEL_PER_TASK = {
    'fibrosis': 'XGBoost',
    'two_stage': 'XGBoost',
    'cirrhosis': 'XGBoost',
    'three_stage': 'XGBoost',
}


def main():
    if not SHAP_CSV.exists():
        raise SystemExit(f'{SHAP_CSV} not found — run shap_publication_figures.py first.')

    try:
        from src.preprocess import prepare_data
        from src.utils.ger_eng_dict import dict_germ_eng
    except ImportError:
        from preprocess import prepare_data
        from utils.ger_eng_dict import dict_germ_eng

    eng_to_ger = {}
    for ger, eng in dict_germ_eng.items():
        eng_to_ger.setdefault(eng, ger)

    df = pd.read_csv(SHAP_CSV)
    tasks = [t for t in BEST_MODEL_PER_TASK if t in set(df.task)]
    print(f'Tasks in {SHAP_CSV.name}: {sorted(set(df.task))}')

    # actual column names, to validate against
    cols = list(prepare_data(tasks[0], False, False)[8])
    print(f'Columns from prepare_data: {len(cols)}')

    out, problems = {}, []
    for task in tasks:
        model = BEST_MODEL_PER_TASK[task]
        sub = df[(df.task == task) & (df.cohort == COHORT) & (df.model == model)]
        if sub.empty:
            problems.append(f'{task}: no rows for model={model}, cohort={COHORT}')
            continue
        top = sub.sort_values('rank').head(TOP_N)

        german = []
        for _, r in top.iterrows():
            g = eng_to_ger.get(r.feature, r.feature)
            if g not in cols:
                problems.append(f'{task}: "{r.feature}" -> "{g}" is not a column '
                                f'in prepare_data output')
            german.append(g)
        out[task] = german
        print(f'  {task:12s} ({model}): ' +
              ', '.join(f'{r.feature} [{r.mean_abs_shap:.3f}]' for _, r in top.iterrows()))
        print(f'  {"":12s}  -> {german}')

    if problems:
        print('\nPROBLEMS — fix these before training the reduced models:')
        for p in problems:
            print(f'  - {p}')
        print('\nMost likely cause: a biomarker missing from dict_germ_eng, so the\n'
              'English name passes through unchanged and does not match a column.')
        sys.exit(1)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n-> {OUT_JSON}')
    print('\nPaste-ready fallback for preprocess.py:\n')
    print('            shap_top_features = {')
    for k, v in out.items():
        print(f'                {k!r}: {v!r},')
    print('            }')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
