"""
plot_prebiopsy_days.py
======================
Distribution of the interval (in days) between blood draw and liver biopsy for
the UMM cohort, plus the QC counts you need for the Methods section.

Place in:  src/plot_prebiopsy_days.py
Run from:  src/    ->   python plot_prebiopsy_days.py

Reads the raw UMM excel directly (independent of the pipeline), so it reflects
the timing situation *before* any filtering. Positive x = lab drawn BEFORE
biopsy (the desired direction); 0 = same day; negative = post-biopsy (leakage).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BIOPSY_COL = 'LAP-Termin'
LAB_COL = 'Blutentnahme'
WINDOW_PRE = 7          # the pre-biopsy window used in the paper (days)
WINDOW_POST = 0
CLIP_DAYS = 60         # x-axis clip for readability (outliers reported separately)


def load_umm():
    df = pd.read_excel('../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx')
    return df


def main():
    out_dir = Path('outputs/data_qc')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_umm()
    if BIOPSY_COL not in df.columns or LAB_COL not in df.columns:
        raise SystemExit(f'Columns {BIOPSY_COL!r}/{LAB_COL!r} not found. '
                         f'Available: {list(df.columns)}')

    biopsy = pd.to_datetime(df[BIOPSY_COL], errors='coerce')
    lab = pd.to_datetime(df[LAB_COL], errors='coerce')
    valid = biopsy.notna() & lab.notna()

    # days from lab to biopsy; positive = lab before biopsy (pre-biopsy)
    delta = (biopsy[valid].dt.normalize() - lab[valid].dt.normalize()).dt.days
    delta = delta.astype(float)

    n_valid = int(valid.sum())
    n_missing_dates = int((~valid).sum())
    same_day = int((delta == 0).sum())
    pre_in_window = int(((delta > 0) & (delta <= WINDOW_PRE)).sum())
    pre_outside = int((delta > WINDOW_PRE).sum())
    post_biopsy = int((delta < -WINDOW_POST).sum())  # lab AFTER biopsy = leakage
    within_final = int(((delta >= -WINDOW_POST) & (delta <= WINDOW_PRE)).sum())

    summary = {
        'n_records': int(len(df)),
        'n_with_both_dates': n_valid,
        'n_missing_a_date': n_missing_dates,
        'same_day (delta=0)': same_day,
        f'pre_biopsy_within_{WINDOW_PRE}d': pre_in_window,
        f'pre_biopsy_beyond_{WINDOW_PRE}d': pre_outside,
        'post_biopsy (leakage)': post_biopsy,
        f'kept_by_window[-{WINDOW_POST},+{WINDOW_PRE}]': within_final,
        'median_delta_days': float(np.median(delta)) if n_valid else None,
        'iqr_delta_days': [float(np.percentile(delta, 25)),
                           float(np.percentile(delta, 75))] if n_valid else None,
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'prebiopsy_days_summary.csv', index=False)
    print('Pre-biopsy timing summary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    # --- figure -------------------------------------------------------------
    clipped = delta[(delta >= -CLIP_DAYS) & (delta <= CLIP_DAYS)]
    plt.figure(figsize=(9, 5.5))
    bins = np.arange(-CLIP_DAYS - 0.5, CLIP_DAYS + 1.5, 1)
    plt.hist(clipped, bins=bins, color='#4C72B0', edgecolor='white', linewidth=0.3)
    # shade the accepted window
    plt.axvspan(-WINDOW_POST, WINDOW_PRE, color='green', alpha=0.12,
                label=f'Accepted window [-{WINDOW_POST}, +{WINDOW_PRE}] d')
    plt.axvline(0, color='black', linestyle='--', linewidth=1,
                label='Biopsy day (0)')
    plt.xlabel('Days between blood draw and biopsy  (positive = pre-biopsy)')
    plt.ylabel('Number of patients')
    plt.title('UMM cohort: lab-to-biopsy interval')
    plt.legend(loc='upper left')
    plt.tight_layout()
    fig_path = out_dir / 'prebiopsy_days_distribution.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f'\nFigure -> {fig_path}')

    n_outliers = int((np.abs(delta) > CLIP_DAYS).sum())
    if n_outliers:
        print(f'Note: {n_outliers} records with |delta| > {CLIP_DAYS} d clipped '
              f'from the plot (still counted in the summary).')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
