"""
Distribution of the interval between blood draw and liver biopsy in the
UMM cohort.

Positive x = laboratory measurement before biopsy.
0 = same-day measurement.
Negative x = laboratory measurement after biopsy.

Same-day measurements are considered eligible.
The final analytic cohort after all eligibility, timing, and missingness
criteria comprised n=304 patients.

Place in:  src/plot_prebiopsy_days.py
Run from project root:
    python -m src.plot_prebiopsy_days
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

BIOPSY_COL = "LAP-Termin"
LAB_COL = "Blutentnahme"

WINDOW_PRE = 7
INCLUDE_SAME_DAY = True

FINAL_ANALYTIC_N = 304

# Display range.
# Everything below/above is combined into an outlier bar.
DISPLAY_MIN = -3
DISPLAY_MAX = 30

# Colours chosen to match the manuscript style
COLOR_PRE = "#35577D"
COLOR_SAME_DAY = "#5A7F61"
COLOR_WINDOW = "#DCE8DF"
COLOR_OUTSIDE = "#AAB4BD"
COLOR_GRID = "#E6E6E6"


# ============================================================
# Data
# ============================================================

def load_umm():
    return pd.read_excel(
        "../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx"
    )


def main():

    out_dir = Path("outputs/data_qc")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_umm()

    if BIOPSY_COL not in df.columns or LAB_COL not in df.columns:
        raise SystemExit(
            f"Columns {BIOPSY_COL!r}/{LAB_COL!r} not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Calculate interval in calendar days
    # --------------------------------------------------------

    biopsy = pd.to_datetime(
        df[BIOPSY_COL],
        errors="coerce"
    )

    lab = pd.to_datetime(
        df[LAB_COL],
        errors="coerce"
    )

    valid = biopsy.notna() & lab.notna()

    # Positive = laboratory measurement BEFORE biopsy
    #
    # normalize() deliberately evaluates calendar days:
    #   same calendar day -> 0
    #   one day before    -> +1
    #   one day after     -> -1
    delta = (
        biopsy[valid].dt.normalize()
        - lab[valid].dt.normalize()
    ).dt.days.astype(int)

    # --------------------------------------------------------
    # QC summary
    # --------------------------------------------------------

    n_records = len(df)
    n_valid = int(valid.sum())
    n_missing_dates = int((~valid).sum())

    same_day = int((delta == 0).sum())

    pre_in_window = int(
        ((delta > 0) & (delta <= WINDOW_PRE)).sum()
    )

    pre_outside_window = int(
        (delta > WINDOW_PRE).sum()
    )

    post_biopsy = int(
        (delta < 0).sum()
    )

    raw_0_to_7 = int(
        ((delta >= 0) & (delta <= WINDOW_PRE)).sum()
    )

    summary = {
        "n_records": n_records,
        "n_with_both_dates": n_valid,
        "n_missing_a_date": n_missing_dates,
        "same_day": same_day,
        f"pre_biopsy_1_to_{WINDOW_PRE}d": pre_in_window,
        f"pre_biopsy_beyond_{WINDOW_PRE}d": pre_outside_window,
        "post_biopsy": post_biopsy,
        f"raw_records_0_to_{WINDOW_PRE}d": raw_0_to_7,
        "final_analytic_cohort": FINAL_ANALYTIC_N,
        "median_delta_days": float(np.median(delta)),
        "iqr_delta_days": [
            float(np.percentile(delta, 25)),
            float(np.percentile(delta, 75)),
        ],
    }

    pd.DataFrame([summary]).to_csv(
        out_dir / "prebiopsy_days_summary.csv",
        index=False
    )

    print("\nPre-biopsy timing summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ========================================================
    # Prepare plotting counts
    # ========================================================

    # Counts for each visible integer day
    visible_days = np.arange(
        DISPLAY_MIN,
        DISPLAY_MAX + 1
    )

    counts = pd.Series(delta).value_counts()

    visible_counts = np.array([
        int(counts.get(day, 0))
        for day in visible_days
    ])

    # Aggregate observations outside plotting range
    n_left_outside = int(
        (delta < DISPLAY_MIN).sum()
    )

    n_right_outside = int(
        (delta > DISPLAY_MAX).sum()
    )

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(11.5, 4.8)
    )

    # --------------------------------------------------------
    # Accepted analysis window
    # --------------------------------------------------------

    # Half-day extension makes the shaded area line up exactly
    # with the integer-day bars 0,...,7.
    ax.axvspan(
        -0.5,
        WINDOW_PRE + 0.5,
        color=COLOR_WINDOW,
        alpha=0.55,
        zorder=0
    )

    # --------------------------------------------------------
    # Main bars
    # --------------------------------------------------------

    for day, count in zip(
        visible_days,
        visible_counts
    ):

        if day == 0:
            color = COLOR_SAME_DAY
        else:
            color = COLOR_PRE

        ax.bar(
            day,
            count,
            width=0.86,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3
        )

    # --------------------------------------------------------
    # Out-of-axis aggregate bars
    # --------------------------------------------------------

    left_x = DISPLAY_MIN - 1.2
    right_x = DISPLAY_MAX + 1.2

    ax.bar(
        left_x,
        n_left_outside,
        width=1.25,
        color=COLOR_OUTSIDE,
        edgecolor="white",
        linewidth=1.0,
        hatch="//",
        zorder=3
    )

    ax.bar(
        right_x,
        n_right_outside,
        width=1.25,
        color=COLOR_OUTSIDE,
        edgecolor="white",
        linewidth=1.0,
        hatch="//",
        zorder=3
    )

    # --------------------------------------------------------
    # Biopsy day
    # --------------------------------------------------------

    ax.axvline(
        0,
        color="#444444",
        linestyle="--",
        linewidth=1.4,
        zorder=4
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(
        left_x - 1.4,
        right_x + 1.4
    )

    # Similar tick spacing to your preferred figure
    normal_ticks = [
        -2,
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
    ]

    tick_positions = [
        left_x,
        *normal_ticks,
        right_x
    ]

    tick_labels = [
        f"<{DISPLAY_MIN}",
        *[str(x) for x in normal_ticks],
        f">{DISPLAY_MAX}",
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel(
        "Days from blood draw to biopsy  (positive = pre-biopsy)",
        fontsize=13
    )

    ax.set_ylabel(
        "Number of patients",
        fontsize=13
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    # --------------------------------------------------------
    # Grid / spines
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.8,
        alpha=0.8,
        zorder=0
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_handles = [

        Patch(
            facecolor=COLOR_WINDOW,
            edgecolor="none",
            alpha=0.55,
            label=(
                f"Analysis window: 0–{WINDOW_PRE} days "
                f"(final analytic cohort n = {FINAL_ANALYTIC_N})"
            )
        ),

        Patch(
            facecolor=COLOR_PRE,
            edgecolor="none",
            label="Pre-biopsy draw"
        ),

        Patch(
            facecolor=COLOR_SAME_DAY,
            edgecolor="none",
            label=f"Same-day draw (n = {same_day})"
        ),

        Patch(
            facecolor=COLOR_OUTSIDE,
            edgecolor="white",
            hatch="//",
            label=(
                f"Outside axis "
                f"(n = {n_left_outside + n_right_outside})"
            )
        ),

        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            linewidth=1.4,
            label="Biopsy day"
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=11.5,
        handlelength=1.5
    )

    # No title: cleaner for publication figure;
    # the information belongs in the figure caption.

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png_path = (
        out_dir /
        "prebiopsy_days_distribution.png"
    )

    pdf_path = (
        out_dir /
        "prebiopsy_days_distribution.pdf"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nPNG -> {png_path}")
    print(f"PDF -> {pdf_path}")

    print(
        f"Outside left (<{DISPLAY_MIN} d): "
        f"{n_left_outside}"
    )

    print(
        f"Outside right (>{DISPLAY_MAX} d): "
        f"{n_right_outside}"
    )


if __name__ == "__main__":
    os.chdir(
        Path(__file__).resolve().parent
    )
    main()