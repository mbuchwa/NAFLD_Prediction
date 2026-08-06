"""
check_split_stratification.py
=============================
Answers two questions in one run:

  1. Is the stratify patch actually active?
  2. If it is, does it survive the filtering that happens AFTER the split?

Run from:  src/   ->   python check_split_stratification.py

BACKGROUND
----------
prepare_data splits the RAW frame (654 rows) and only then calls preprocess()
on each partition, which applies the pre-biopsy window, the Micro dropna and the
>70%-missing rule. Roughly 53% of rows are removed after the split.

Stratifying the raw frame therefore controls the grade mix BEFORE filtering, not
after. If the removal rate differs by grade -- and it does, the seven-day window
enriches F4 -- the analytic partitions can still drift apart. This script shows
the distribution at both stages so you can see which of the two is the problem.
"""

import inspect
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

GRADES = [0, 1, 2, 3, 4]


def _dist(series, label, n_total=None):
    s = pd.to_numeric(series, errors='coerce')
    counts = {g: int((s == g).sum()) for g in GRADES}
    n = int(s.notna().sum())
    adv = counts[4]
    print(f'  {label:22s} n={n:4d}  ' +
          '  '.join(f'F{g}:{counts[g]:3d}' for g in GRADES) +
          f'   F4 {100 * adv / n:5.1f}%' if n else f'  {label}: empty')
    return counts, n


def main():
    try:
        from src import preprocess as pp
    except ImportError:
        import preprocess as pp

    # ---------------------------------------------------------------- 1 -----
    print('=' * 74)
    print('1  Is the stratify patch active?')
    print('=' * 74)
    src = inspect.getsource(pp.prepare_data)
    has_strata = '_strata' in src or 'stratify=' in src
    print(f'  "stratify=" found in prepare_data : {has_strata}')
    for line in src.splitlines():
        if 'train_test_split' in line or 'stratify' in line or '_strata' in line:
            print(f'    | {line.strip()[:92]}')
    if not has_strata:
        print('\n  => The patch is NOT in the file that Python imports. Check that you\n'
              '     edited src/preprocess.py and not a copy, and that no stale\n'
              '     __pycache__ is being used (delete src/__pycache__).')
        return

    # ---------------------------------------------------------------- 2 -----
    print('\n' + '=' * 74)
    print('2  Grade distribution before and after the post-split filtering')
    print('=' * 74)

    df = pd.read_excel('../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx')
    df2 = pd.read_excel('../data/202403 Lap und Histo Daten von Ines Tuschner.xlsx')
    df2 = df2[['HbA1c (%)', 'Glucose in plasma (mg/dL)', 'LDL- Cholesterin (mg/dL)']]
    df = pd.concat([df, df2], axis=1)
    print(f'\n  raw frame: {len(df)} rows')
    _dist(df['Micro'], 'raw (all)')

    from sklearn.model_selection import train_test_split
    strata = pd.to_numeric(df['Micro'], errors='coerce')
    strata = strata.where(strata.between(0, 4), -1).fillna(-1).astype(int)

    for name, kw in (('UNstratified', {}), ('stratified', {'stratify': strata})):
        tv, te = train_test_split(df, test_size=0.1, random_state=42, **kw)
        kw2 = {'stratify': strata.loc[tv.index]} if kw else {}
        tr, va = train_test_split(tv, test_size=0.2, random_state=42, **kw2)
        print(f'\n  --- {name}, BEFORE filtering ---')
        for lbl, part in (('train', tr), ('val', va), ('test', te)):
            _dist(part['Micro'], lbl)

        print(f'  --- {name}, AFTER filtering (analytic cohort) ---')
        for lbl, part in (('train', tr), ('val', va), ('test', te)):
            f = part.copy()
            f, _ = pp.temporal_filter_pre_biopsy_labs(f, summary_output_path=None)
            f = pp.calculate_age(f)
            f, _ = pp.clean_df(f)
            f = f.astype(float).dropna(subset=['Micro'])
            f = pp.drop_rows_with_high_missing_data(f)
            _dist(f['Micro'], lbl)

    print('\n' + '=' * 74)
    print('Reading the output')
    print('=' * 74)
    print('  If "stratified / BEFORE" is balanced but "stratified / AFTER" is not,\n'
          '  the split order is the problem, not the stratify call: the filtering\n'
          '  removes ~53% of rows per partition at a grade-dependent rate. The fix\n'
          '  is to determine the analytic cohort first and split that, which also\n'
          '  removes the patient-level leakage between partitions.')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
